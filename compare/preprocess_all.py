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

CACHE_ROOT_3D = os.path.join(BASE_PATH, "preprocessed_kfold", "3D")
CACHE_ROOT_2D = os.path.join(BASE_PATH, "preprocessed_kfold", "2D")

def generate_kfold_preprocessed_cache(seq_lens, label_dirs, stride=1, k_folds=5):
    for seq_len in seq_lens:
        all_seq, all_labels = [], []
        all_features_2d, all_labels_2d = [], []

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

                    # === 3D 滑動視窗切片 ===
                    if len(data) >= seq_len:
                        chunks = [data[i:i + seq_len] for i in range(0, len(data) - seq_len + 1, stride)]
                        all_seq.extend(chunks)
                        all_labels.extend([label] * len(chunks))

                    # === 2D 非重疊摘要切片 ===
                    num_chunks = len(data) // seq_len
                    for i in range(num_chunks):
                        chunk = data[i * seq_len : (i + 1) * seq_len]
                        features = []
                        features.extend(np.mean(chunk, axis=0))
                        features.extend(np.std(chunk, axis=0))
                        features.extend(np.max(chunk, axis=0))
                        features.extend(np.min(chunk, axis=0))
                        all_features_2d.append(features)
                        all_labels_2d.append(label)

        # === 儲存 3D ===
        seq_arr = np.array(all_seq, dtype=np.float32)
        labels_arr = np.array(all_labels, dtype=np.int64)
        # === 儲存 2D ===
        arr_2d = np.array(all_features_2d, dtype=np.float32)
        labels_2d = np.array(all_labels_2d, dtype=np.int64)

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        cache_dir_3d = os.path.join(CACHE_ROOT_3D, f"stride_{stride}")
        os.makedirs(cache_dir_3d, exist_ok=True)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(seq_arr, labels_arr), 1):
            X_train, X_val = seq_arr[train_idx], seq_arr[val_idx]
            y_train, y_val = labels_arr[train_idx], labels_arr[val_idx]

            B, T, F = X_train.shape
            train_reshaped = X_train.reshape(-1, F)
            scaler = StandardScaler().fit(train_reshaped)
            X_train_scaled = scaler.transform(train_reshaped).reshape(B, T, F)

            Bv, Tv, Fv = X_val.shape
            val_reshaped = X_val.reshape(-1, Fv)
            X_val_scaled = scaler.transform(val_reshaped).reshape(Bv, Tv, Fv)

            np.save(os.path.join(cache_dir_3d, f"X_train_fold{fold_idx}_seq{seq_len}_3d.npy"), X_train_scaled)
            np.save(os.path.join(cache_dir_3d, f"y_train_fold{fold_idx}_seq{seq_len}_3d.npy"), y_train)
            np.save(os.path.join(cache_dir_3d, f"X_val_fold{fold_idx}_seq{seq_len}_3d.npy"), X_val_scaled)
            np.save(os.path.join(cache_dir_3d, f"y_val_fold{fold_idx}_seq{seq_len}_3d.npy"), y_val)

            print(f"✅ 3D Fold {fold_idx} done for seq_len={seq_len}, stride={stride} (train={len(train_idx)}, val={len(val_idx)})")

        cache_dir_2d = os.path.join(CACHE_ROOT_2D)
        os.makedirs(cache_dir_2d, exist_ok=True)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(arr_2d, labels_2d), 1):
            X_train_2d, X_val_2d = arr_2d[train_idx], arr_2d[val_idx]
            y_train_2d, y_val_2d = labels_2d[train_idx], labels_2d[val_idx]

            scaler = StandardScaler().fit(X_train_2d)
            X_train_scaled_2d = scaler.transform(X_train_2d)
            X_val_scaled_2d = scaler.transform(X_val_2d)

            np.save(os.path.join(cache_dir_2d, f"X_train_fold{fold_idx}_seq{seq_len}_2d.npy"), X_train_scaled_2d)
            np.save(os.path.join(cache_dir_2d, f"y_train_fold{fold_idx}_seq{seq_len}_2d.npy"), y_train_2d)
            np.save(os.path.join(cache_dir_2d, f"X_val_fold{fold_idx}_seq{seq_len}_2d.npy"), X_val_scaled_2d)
            np.save(os.path.join(cache_dir_2d, f"y_val_fold{fold_idx}_seq{seq_len}_2d.npy"), y_val_2d)

            print(f"✅ 2D Fold {fold_idx} done for seq_len={seq_len} (train={len(train_idx)}, val={len(val_idx)})")

if __name__ == "__main__":
    seq_lens = [5, 10, 15, 20, 30, 45]
    strides = [1]

    for stride in strides:
        generate_kfold_preprocessed_cache(seq_lens, LABEL_DIRS, stride=stride)

    print("\n🎉 所有 K-Fold 前處理已完成！")