import os
import random
from glob import glob
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import h5py
from monai.data import Dataset, CacheDataset
import monai.transforms as transforms
from monai.transforms import (
    Compose, EnsureChannelFirstd, LoadImaged, Resized, ToTensord, Spacingd,
    Orientationd, ScaleIntensityRanged, CropForegroundd,
)

def SegTHOR_dataset_3D(data_dir, split, output_size, intensity_range=[-384, 384]):
    patient_list = sorted(glob(os.path.join(data_dir, split, 'Patient*')))
    
    image_list = [glob(f"{patient}/Patient*.nii.gz")[0] for patient in patient_list]
    label_list = [glob(f"{patient}/GT*.nii.gz")[0] for patient in patient_list]

    _files = [{"image": img, "label": lbl} for img, lbl in zip(image_list, label_list)]

    common_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 2.5), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=intensity_range[0], a_max=intensity_range[1],
            b_min=0.0, b_max=1.0, clip=True
        ),
        ToTensord(keys=["image", "label"]),
    ]

    # Augmentation for training
    if split == 'train':
        train_transforms = common_transforms + [
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label", spatial_size=output_size,
                pos=2.0, neg=1.0, num_samples=2
            ),
            transforms.RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=50),
            transforms.RandShiftIntensityd(keys=["image"], offsets=30, prob=0.5),
            transforms.RandRotated(
                keys=["image", "label"], range_x=0.2, mode=["bilinear", "nearest"], prob=0.5
            ),
            transforms.RandAffined(
                keys=["image", "label"], prob=0.5, translate_range=[40],
                mode=["bilinear", "nearest"]
            ),
            Resized(keys=["image", "label"], spatial_size=output_size, mode=('bilinear', 'nearest')),
        ]
        _transforms = Compose(train_transforms)
    else:
        _transforms = Compose(common_transforms)

    if split == 'train':
        return Dataset(data=_files, transform=_transforms)
    else:
        return Dataset(data=_files, transform=_transforms)

In main.py:
train_dataset = SegTHOR_dataset_3D(data_dir="data/SegTHOR", split="train", output_size=[96,96,96], intensity_range = [-384,384])
    
train_loader = DataLoader(train_dataset,
                            batch_size=config.batch_size, 
                            shuffle=True,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            sampler=None)