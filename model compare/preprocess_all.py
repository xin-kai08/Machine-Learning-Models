import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 設定你的資料夾路徑（請依實際修改）
BASE_PATH = r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\dataset\feature dim_4\hardware"

# 🔧 使用的欄位（可以刪掉其中一個）
SELECTED_FEATURES = ["current", "voltage",  "power", "temp_C"]

# 各分類資料夾對應標籤
LABEL_DIRS = {
    0: os.path.join(BASE_PATH, "normal"),
    1: os.path.join(BASE_PATH, "abnormal", "wire_rust"),
    2: os.path.join(BASE_PATH, "abnormal", "transformer_rust"),
    3: os.path.join(BASE_PATH, "abnormal", "transformer_overheating"),
}

# ✅ 快取根目錄（你想要的輸出位置）
CACHE_ROOT = os.path.join(BASE_PATH, "preprocessed")

# 建立 3D 快取資料
def generate_preprocessed_cache_3d(seq_lens, label_dirs,  cache_dir = CACHE_ROOT):
    os.makedirs(cache_dir, exist_ok=True)
    for seq_len in seq_lens:
        cache_X = os.path.join(cache_dir, f"X_seq{seq_len}_3d.npy")
        cache_y = os.path.join(cache_dir, f"y_seq{seq_len}_3d.npy")

        if os.path.exists(cache_X) and os.path.exists(cache_y):
            print(f"📥 已存在 3D 快取：seq_len={seq_len}，略過")
            continue

        all_seq, all_labels = [], []
        for label, folder in label_dirs.items():
            for fname in os.listdir(folder):
                if fname.endswith(".csv"):
                    path = os.path.join(folder, fname)
                    df = pd.read_csv(path)
                    try:
                        data = df[SELECTED_FEATURES].values.astype(np.float32)
                    except KeyError as e:
                        print(f"❌ 缺少欄位 {e}：{path}")
                        continue

                    num_chunks = len(data) // seq_len
                    chunks = [data[i * seq_len : (i + 1) * seq_len] for i in range(num_chunks)]
                    all_seq.extend(chunks)
                    all_labels.extend([label] * len(chunks))

        if not all_seq:
            print(f"⚠️ 無可用資料，跳過 seq_len={seq_len}")
            continue

        seq_arr = np.array(all_seq, dtype=np.float32)
        labels_arr = np.array(all_labels, dtype=np.int64)
        B, T, F = seq_arr.shape

        reshaped = seq_arr.reshape(-1, F)
        scaled = StandardScaler().fit_transform(reshaped).reshape(B, T, F)

        np.save(cache_X, scaled)
        np.save(cache_y, labels_arr)
        print(f"✅ 完成 3D 快取：seq_len={seq_len}（共 {B} 筆序列）")

# 建立 2D 快取資料
def generate_preprocessed_cache_2d(seq_lens, label_dirs,  cache_dir = CACHE_ROOT):
    os.makedirs(cache_dir, exist_ok=True)
    for seq_len in seq_lens:
        cache_X = os.path.join(cache_dir, f"X_seq{seq_len}_2d.npy")
        cache_y = os.path.join(cache_dir, f"y_seq{seq_len}_2d.npy")

        if os.path.exists(cache_X) and os.path.exists(cache_y):
            print(f"📥 已存在 2D 快取：seq_len={seq_len}，略過")
            continue

        all_features, all_labels = [], []
        for label, folder in label_dirs.items():
            for fname in os.listdir(folder):
                if fname.endswith(".csv"):
                    path = os.path.join(folder, fname)
                    df = pd.read_csv(path)
                    try:
                        data = df[SELECTED_FEATURES].values.astype(np.float32)
                    except KeyError as e:
                        print(f"❌ 缺少欄位 {e}：{path}")
                        continue

                    num_chunks = len(data) // seq_len
                    chunks = [data[i * seq_len : (i + 1) * seq_len] for i in range(num_chunks)]
                    for chunk in chunks:
                        flattened = chunk.flatten()  # 🌟 重點：直接攤平
                        all_features.append(flattened)
                        all_labels.append(label)

        if not all_features:
            print(f"⚠️ 無可用資料，跳過 seq_len={seq_len}")
            continue

        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)

        X_scaled = StandardScaler().fit_transform(X)

        np.save(cache_X, X_scaled)
        np.save(cache_y, y)
        print(f"✅ 完成 2D 快取：seq_len={seq_len}（共 {len(X_scaled)} 筆，shape: {X_scaled.shape}）")

# 主執行邏輯
if __name__ == "__main__":
    seq_lens = [10, 20, 30, 40]

    print("🔁 開始建立 3D 快取")
    generate_preprocessed_cache_3d(seq_lens, LABEL_DIRS)

    print("\n🔁 開始建立 2D 快取")
    generate_preprocessed_cache_2d(seq_lens, LABEL_DIRS)

    print("\n🎉 所有快取建立完成！")
