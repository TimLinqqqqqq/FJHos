import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix
import logging

# 設定 logging
logging.basicConfig(
    filename='/home/u3861345/preprocess_and_model_training/log_data/training_model.log',   # 設置日誌文件路徑
    level=logging.INFO,                          # 設置日誌級別
    format='%(asctime)s - %(levelname)s - %(message)s',  # 設置日誌格式
    datefmt='%Y-%m-%d %H:%M:%S'
)

class SimpleModel512(nn.Module):
    def __init__(self):
        super(SimpleModel512, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        
        # 計算展平後的尺寸，根據 (512, 512, 128) 的資料大小
        self.fc1 = nn.Linear(32 * 128 * 128 * 32, 128)  # 根據資料計算
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SimpleModel128(nn.Module):
    def __init__(self):
        super(SimpleModel128, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)

        # 動態計算展平後的大小
        self.flattened_size = self.get_flattened_size((1, 128, 128, 128))
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 1)

    def get_flattened_size(self, input_shape):
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.pool(torch.relu(self.conv1(x)))
            print(f"Shape after conv1 and pool: {x.shape}")
            x = self.pool(torch.relu(self.conv2(x)))
            print(f"Shape after conv2 and pool: {x.shape}")
            return x.numel()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        print(f"Shape after conv1 and pool: {x.shape}")
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        print(f"Shape after conv2 and pool: {x.shape}")
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(train_loader, model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            inputs = batch["vol"]
            labels = batch["label"].clone().detach().float()

            optimizer.zero_grad()
            outputs = model(inputs)
            
            outputs = outputs.view(-1)
            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}')

def test_model(test_loader, model, criterion):
    model.eval()
    total_test_loss = 0.0
    correct_test = 0
    total_test = 0

    all_labels = []
    all_predictions = []
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["vol"]
            labels = batch["label"].clone().detach().float()

            # 使用 logging 來記錄測試標籤
            logging.info(f'Test Labels: {labels}, Type: {labels.dtype}, Unique Values: {labels.unique()}')

            outputs = model(inputs)
            outputs = outputs.view(-1)

            # 使用 logging 來記錄模型輸出
            logging.info(f'Raw outputs: {outputs}, Type: {outputs.dtype}')

            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            
            predicted = (outputs >= 0.5).float()
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            # 收集所有標籤和預測值以計算 F1 Score 和混淆矩陣
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            # 計算 TP, TN, FP, FN
            TP += ((predicted == 1) & (labels == 1)).sum().item()
            TN += ((predicted == 0) & (labels == 0)).sum().item()
            FP += ((predicted == 1) & (labels == 0)).sum().item()
            FN += ((predicted == 0) & (labels == 1)).sum().item()
        
        # 計算損失和準確率
        average_test_loss = total_test_loss / len(test_loader)
        test_accuracy = correct_test / total_test

        # logging.info(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}') 

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        # 計算 F1 Score 和混淆矩陣
        f1 = f1_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)        
        
        logging.info(f'Average Test Loss: {average_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        logging.info(f'Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}')
        logging.info(f'F1 Score: {f1:.4f}')
        logging.info(f'Confusion Matrix:\n{cm}')

    return test_accuracy  # 返回測試精度以便於交叉驗證中計算

if __name__ == "__main__":
    # transformed_train_data = torch.load('/home/u3861345/preprocess_and_model_training/PT/train_first_try.pt')
    # transformed_test_data = torch.load('/home/u3861345/preprocess_and_model_training/PT/test_first_try.pt')

    transformed_train_data = torch.load('/home/u3861345/preprocess_and_model_training/PT/train_resize_mask_and_patient.pt')
    transformed_test_data = torch.load('/home/u3861345/preprocess_and_model_training/PT/test_resize_mask_and_patient.pt')

    train_loader = DataLoader(transformed_train_data, batch_size=1)
    test_loader = DataLoader(transformed_test_data, batch_size=1)

    # model = SimpleModel512()
    model = SimpleModel128()
    criterion = nn.BCEWithLogitsLoss()  # 損失函數 用於二元分類問題
    optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-5)
    train_model(train_loader, model, criterion, optimizer, num_epochs=15)

    logging.info("Final Test on External Test Set:")
    test_model(test_loader, model, criterion)
