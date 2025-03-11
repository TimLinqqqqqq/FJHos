import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import KFold
import logging
from torch.nn import InstanceNorm3d

# 設定 logging
logging.basicConfig(
    filename='/home/u3861345/preprocess_and_model_training/log_data/training_model_resnet.log',   # 設置日誌文件路徑
    level=logging.INFO,                          # 設置日誌級別
    format='%(asctime)s - %(levelname)s - %(message)s',  # 設置日誌格式
    datefmt='%Y-%m-%d %H:%M:%S'
)

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()
        # 3D 卷積層 1
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        # 3D 卷積層 2
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 如果輸入與輸出通道不匹配，則使用 1x1 卷積進行降維
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # Max Pooling 層 1
        self.dropout1 = nn.Dropout(p=0.2)

        self.layer1 = self._make_layer(64, 64, stride=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # Max Pooling 層 2
        self.dropout2 = nn.Dropout(p=0.2)

        self.layer2 = self._make_layer(64, 128, stride=2)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=0.2)
        
        self.layer3 = self._make_layer(128, 256, stride=2)
        # self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(p=0.2)
        
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)
        # self.dropout5 = nn.Dropout(p=0.2)

        # self.fc = nn.Linear(512 * 16 * 16 * 4, num_classes) #給512,512,128用的
        self.fc = nn.Linear(512, num_classes)   #給128,128,128用的

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock3D(in_channels, out_channels, stride),
            BasicBlock3D(out_channels, out_channels, 1)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool1(out)   # 使用 Max Pooling
        out = self.dropout1(out)

        out = self.layer1(out)
        out = self.pool2(out)   # 使用 Max Pooling
        out = self.dropout2(out)

        out = self.layer2(out)
        out = self.pool3(out)  
        out = self.dropout3(out)

        out = self.layer3(out)
        # out = self.pool4(out)
        out = self.dropout4(out)

        out = self.layer4(out)
        out = self.pool5(out)
        # out = self.dropout5(out)

        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


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
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["vol"]
            labels = batch["label"].clone().detach().float()

            # 檢查標籤格式
            logging.info(f'Test Labels: {labels}, Type: {labels.dtype}, Unique Values: {labels.unique()}')

            outputs = model(inputs)
            outputs = outputs.view(-1)

            # 檢查模型輸出格式
            logging.info(f'Raw outputs: {outputs}, Type: {outputs.dtype}')

            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            
            predicted = (outputs >= 0.5).float()
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
            # 收集所有標籤和預測值，後續計算 F1 Score 和 confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        # 計算 F1 Score 和混淆矩陣
        f1 = f1_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)
        
        # 混淆矩陣中的 TP, TN, FP, FN
        TN, FP, FN, TP = cm.ravel()
        
        average_test_loss = total_test_loss / len(test_loader)
        test_accuracy = correct_test / total_test
        
        # 計算靈敏度和特異性
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        logging.info(f'Average Test Loss: {average_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        logging.info(f'Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}')
        logging.info(f'F1 Score: {f1:.4f}')
        logging.info(f'Confusion Matrix:\n{cm}')

if __name__ == "__main__":
    # transformed_train_data = torch.load('/home/u3861345/preprocess_and_model_training/PT/train_first_try.pt')
    # transformed_test_data = torch.load('/home/u3861345/preprocess_and_model_training/PT/test_first_try.pt')

    transformed_train_data = torch.load('/home/u3861345/preprocess_and_model_training/PT/train_resize_mask_and_patient.pt')
    transformed_test_data = torch.load('/home/u3861345/preprocess_and_model_training/PT/test_resize_mask_and_patient.pt')

    train_loader = DataLoader(transformed_train_data, batch_size=1)
    test_loader = DataLoader(transformed_test_data, batch_size=1)

    model = ResNet3D(num_classes=1)
    criterion = nn.BCEWithLogitsLoss()  # 損失函數 用於二元分類問題
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    train_model(train_loader, model, criterion, optimizer, num_epochs=100)

    logging.info("Final Test on External Test Set:")
    test_model(test_loader, model, criterion)
