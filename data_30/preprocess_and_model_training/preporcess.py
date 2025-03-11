from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    ScaleIntensityRanged,
    RandRotateD,         # 使用隨機旋轉
    RandAffineD,         # 使用隨機仿射變換
    RandZoomD,           # 使用隨機縮放
    RandGaussianNoiseD,  # 使用高斯雜訊
)
from monai.utils import set_determinism
import pandas as pd
import os
from glob import glob
import torch

def prepare(in_dir, label_file, pixdim=(1.0, 1.0, 1.0), a_min=-200, a_max=200, spatial_size=[512, 512, 128], cache=True):
    set_determinism(seed=0)

    labels_df = pd.read_csv(label_file)
    
    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes", "*.nii.gz")))
    path_test_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nii.gz")))

    train_files = [{"vol": image_name, "label": labels_df[labels_df['patient_id'] == os.path.basename(image_name).replace('.nii.gz', '')]['label'].values[0]} for image_name in path_train_volumes]
    test_files = [{"vol": image_name, "label": labels_df[labels_df['patient_id'] == os.path.basename(image_name).replace('.nii.gz', '')]['label'].values[0]} for image_name in path_test_volumes]
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["vol"]),
            EnsureChannelFirstD(keys=["vol"]),
            Spacingd(keys=["vol"], pixdim=pixdim, mode="bilinear"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            Resized(keys=["vol"], spatial_size=spatial_size),
            
            # 使用資料增強
            RandRotateD(keys=["vol"], range_x=(-15, 15), prob=0.5),   # 隨機旋轉
            RandAffineD(keys=["vol"], prob=0.5, rotate_range=(-0.1, 0.1), translate_range=(10, 10), scale_range=(0.9, 1.1)),  # 隨機仿射變換
            RandZoomD(keys=["vol"], min_zoom=0.9, max_zoom=1.1, prob=0.5),   # 隨機縮放
            RandGaussianNoiseD(keys=["vol"], mean=0.0, std=0.1, prob=0.5),  # 高斯雜訊
            
            ToTensord(keys=["vol"]),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["vol"]),
            EnsureChannelFirstD(keys=["vol"]),
            Spacingd(keys=["vol"], pixdim=pixdim, mode="bilinear"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            Resized(keys=["vol"], spatial_size=spatial_size),
            ToTensord(keys=["vol"]),
        ]
    )

    transformed_train_data = [train_transforms(item) for item in train_files]
    transformed_test_data = [test_transforms(item) for item in test_files]

    torch.save(transformed_train_data, '/home/u3861345/preprocess_and_model_training/PT/train_first_try.pt')
    torch.save(transformed_test_data, '/home/u3861345/preprocess_and_model_training/PT/test_first_try.pt')

if __name__ == "__main__":
    data_directory = "/home/u3861345/data_train_test/first_try_data"
    label_file = "/home/u3861345/data_train_test/first_try_data/labels.csv"
    prepare(in_dir=data_directory, label_file=label_file)
