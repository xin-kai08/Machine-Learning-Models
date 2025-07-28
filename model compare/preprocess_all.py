import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# === 基本設定 ===
BASE_PATH = r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\dataset\feature dim_4\hardware"

SELECTED_FEATURES = ["current", "voltage", "power", "temp_C"]

LABEL_DIRS = {
    0: os.path.join(BASE_PATH, "normal"),
    1: os.path.join(BASE_PATH, "abnormal", "wire_rust"),
    2: os.path.join(BASE_PATH, "abnormal", "transformer_rust"),
    3: os.path.join(BASE_PATH, "abnormal", "transformer_overheating"),
}

CACHE_ROOT = os.path.join(BASE_PATH, "preprocessed_kfold")

def generate_kfold_preprocessed_cache(seq_lens, label_dirs, stride=1, k_folds=5, cache_dir=CACHE_ROOT):
    os.makedirs(cache_dir, exist_ok=True)
    for seq_len in seq_lens:
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

                    if len(data) < seq_len:
                        continue  # 略過過短資料

                    # === 滑動視窗切片 ===
                    chunks = [data[i:i + seq_len] for i in range(0, len(data) - seq_len + 1, stride)]
                    all_seq.extend(chunks)
                    all_labels.extend([label] * len(chunks))

        seq_arr = np.array(all_seq, dtype=np.float32)
        labels_arr = np.array(all_labels, dtype=np.int64)

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(seq_arr, labels_arr), 1):
            X_train, X_val = seq_arr[train_idx], seq_arr[val_idx]
            y_train, y_val = labels_arr[train_idx], labels_arr[val_idx]

            # === 做 train-only scaler ===
            B, T, F = X_train.shape
            train_reshaped = X_train.reshape(-1, F)
            scaler = StandardScaler().fit(train_reshaped)
            X_train_scaled = scaler.transform(train_reshaped).reshape(B, T, F)

            Bv, Tv, Fv = X_val.shape
            val_reshaped = X_val.reshape(-1, Fv)
            X_val_scaled = scaler.transform(val_reshaped).reshape(Bv, Tv, Fv)

            np.save(os.path.join(cache_dir, f"X_train_fold{fold_idx}_seq{seq_len}_3d.npy"), X_train_scaled)
            np.save(os.path.join(cache_dir, f"y_train_fold{fold_idx}_seq{seq_len}_3d.npy"), y_train)
            np.save(os.path.join(cache_dir, f"X_val_fold{fold_idx}_seq{seq_len}_3d.npy"), X_val_scaled)
            np.save(os.path.join(cache_dir, f"y_val_fold{fold_idx}_seq{seq_len}_3d.npy"), y_val)

            print(f"✅ Fold {fold_idx} done for seq_len={seq_len}, stride={stride} (train={len(train_idx)}, val={len(val_idx)})")

if __name__ == "__main__":
    seq_lens = [10, 20, 30, 40]
    strides = [1]

    for stride in strides:
        cache_dir = os.path.join(CACHE_ROOT, f"stride_{stride}")
        generate_kfold_preprocessed_cache(seq_lens, LABEL_DIRS, stride=stride, cache_dir=cache_dir)

    print("\n🎉 所有 K-Fold 前處理已完成！")