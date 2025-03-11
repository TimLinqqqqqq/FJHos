import nibabel as nib
import numpy as np
import os
from monai.transforms import Resize

# 設定檔案路徑
mask_path = '/home/u3861345/data_50/fjdata_eso_dilated/51_eso_dilated.nii.gz'
patient_path = '/home/u3861345/data_50/fjdata_resampled/51_resampled.nii.gz'
output_dir = '/home/u3861345/data_50/data_train_test'

# 檢查檔案是否存在
if not os.path.exists(mask_path):
    raise FileNotFoundError(f"❌ Error: 找不到檔案 {mask_path}")

if not os.path.exists(patient_path):
    raise FileNotFoundError(f"❌ Error: 找不到檔案 {patient_path}")

# 確保輸出目錄存在
os.makedirs(output_dir, exist_ok=True)


# 確保輸出目錄存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 讀取 mask 和 patient 影像
mask = nib.load(mask_path)
patient = nib.load(patient_path)

# 取得影像數據陣列
data_mask = mask.get_fdata()
data_patient = patient.get_fdata()

# 確保影像形狀相同才能相乘
if data_mask.shape == data_patient.shape:
    # 進行逐元素相乘
    result = np.multiply(data_mask, data_patient)

    # 儲存 mask * patient 影像
    mix_output_path = os.path.join(output_dir, '51_mask_and_patient.nii.gz')
    result_img = nib.Nifti1Image(result, affine=mask.affine, header=mask.header)
    nib.save(result_img, mix_output_path)
    print(f"已保存相乘影像: {mix_output_path}")
else:
    raise ValueError("Error: Mask and Patient 影像形狀不匹配！")

# 讀取相乘後的影像
img = nib.load(mix_output_path)
data = img.get_fdata()

# **裁切非零區域**
non_zero_indices = np.argwhere(data != 0)
x_min, y_min, z_min = np.min(non_zero_indices, axis=0)
x_max, y_max, z_max = np.max(non_zero_indices, axis=0)
cropped_data = data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

# 儲存裁切後的影像
cropped_output_path = os.path.join(output_dir, '51_cropped_mask_and_patient.nii.gz')
cropped_img = nib.Nifti1Image(cropped_data, img.affine)
nib.save(cropped_img, cropped_output_path)
print(f"裁切範圍: x({x_min}, {x_max}), y({y_min}, {y_max}), z({z_min}, {z_max})")
print(f"已保存裁切影像: {cropped_output_path}")

# **Resize 與 Padding**
def resize_and_pad_slices(data, target_shape=(128, 128), target_slices=128):
    resizer = Resize(spatial_size=target_shape)
    resized_data = resizer(data)

    # 取得當前的切片數
    current_slices = resized_data.shape[0]

    # 如果切片數小於目標數，則補零
    if current_slices < target_slices:
        pad_width = ((0, target_slices - current_slices), (0, 0), (0, 0))
        resized_data = np.pad(resized_data, pad_width, mode='constant', constant_values=0)
    elif current_slices > target_slices:
        resized_data = resized_data[:target_slices, :, :]

    # 調整維度順序 (Z 軸放最後)
    resized_data = np.transpose(resized_data, (1, 2, 0))

    return resized_data

# 讀取裁切後的影像並轉換格式
cropped_img = nib.load(cropped_output_path)
cropped_data = cropped_img.get_fdata()

# 確保為 3D 影像
if cropped_data.ndim == 4:
    cropped_data = np.squeeze(cropped_data)

# Resize 影像至 (128,128,128)
resized_data = resize_and_pad_slices(cropped_data, target_shape=(128, 128), target_slices=128)

# 確保資料格式為 NumPy 陣列
if isinstance(resized_data, np.ndarray) is False:
    resized_data_np = resized_data.cpu().numpy()
else:
    resized_data_np = resized_data

# 儲存 Resize 後的影像
resized_output_path = os.path.join(output_dir, '51_resized_mask_and_patient.nii.gz')
resized_img = nib.Nifti1Image(resized_data_np, cropped_img.affine)
nib.save(resized_img, resized_output_path)

print(f"已將資料 Resize 成 {resized_data_np.shape}，並保存為 {resized_output_path}")
