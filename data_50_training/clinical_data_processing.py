import pandas as pd
import numpy as np

# 讀取 Excel 檔案，僅選取所需的 4 個欄位
clinical_data_path = "/home/u3861345/data_50/去識別化-食道癌-癌登資料20240207.xlsx"

usecols = ["腫瘤大小", "吸菸行為", "嚼檳榔行為", "喝酒行為"]

# 先讀取全部欄位，確認名稱是否正確
df = pd.read_excel(clinical_data_path, header=0)
df.columns = df.columns.str.strip().str.replace("\n", "")  # 移除多餘空格與換行符號

# 檢查所需的欄位是否存在
missing_cols = [col for col in usecols if col not in df.columns]
if missing_cols:
    raise ValueError(f"錯誤: Excel 檔案中找不到以下欄位: {missing_cols}")

# 只讀取指定欄位
df = df[usecols]

# **資料處理函數**
def ori_tumor_pro(value):
    """將腫瘤大小標準化到 0~1，假設最大為 150mm"""
    return np.clip(value / 150, 0, 1) if pd.notna(value) and value <= 150 else 70.0 / 150

def smoke_and_chew_pro(value):
    """將吸菸與嚼檳榔行為轉換為 0/1，預設未填者為 0"""
    return 0 if pd.isna(value) or int(value / 10000) in [0, 99] else 1

def drink_pro(value):
    """將喝酒行為轉換為 0/1，未填寫預設為喝酒 (1)"""
    if pd.isna(value) or value == 999:
        return 1
    return 0 if value <= 1 else 1

# **套用轉換函數**
df["腫瘤大小"] = df["腫瘤大小"].apply(ori_tumor_pro)
df["吸菸行為"] = df["吸菸行為"].apply(smoke_and_chew_pro)
df["嚼檳榔行為"] = df["嚼檳榔行為"].apply(smoke_and_chew_pro)
df["喝酒行為"] = df["喝酒行為"].apply(drink_pro)

# 儲存處理後的資料
output_path = "/home/u3861345/data_50/processed_clinical_data.xlsx"
df.to_excel(output_path, index=False)

print(f"✅ 處理後的資料已儲存至 {output_path}")
print(df)
