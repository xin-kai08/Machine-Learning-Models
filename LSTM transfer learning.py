import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import joblib

from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. 一些基本設定（請依照你原本的模型調整）
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 4          # feature_dim_4 → [current, voltage, power, temp]
HIDDEN_DIM = 16        # ⚠ TODO: 改成你原本訓練用的值
NUM_LAYERS = 1         # ⚠ TODO: 改成你原本訓練用的值
NUM_CLASSES = 4        # 之前是 4 類：正常 + 三種異常

BATCH_SIZE = 16
MAX_SEQ_LEN = 15
STRIDE = 5
EPOCHS = 15            # fine-tune 不要太多，先試 3~5
LR = 1e-3              # 比原本小一點

# 這裡你可以決定「特殊樣本」視為哪一類
SPECIAL_LABEL = 1
SCALER_SAVE_PATH = r"C:/Users/boss9/OneDrive/桌面/專題/機器學習/result/20251115_scaler.pkl"

# -----------------------------
# 2. LSTM 模型（要跟你原本的一模一樣）
# -----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)

        out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden_dim)
        out = out[:, -1, :]              # 取最後一個時間點
        out = self.dropout(out)
        out = self.fc(out)               # (batch, num_classes)
        return out


# -----------------------------
# 3. Dataset：從資料夾讀 .csv 檔
# -----------------------------
class SequenceFolderDataset(Dataset):
    def __init__(self, folder_label_pairs, max_seq_len=15, stride=5):
        """
        folder_label_pairs: [(folder_path, label), ...]
        每個 folder 裡面放一堆 .csv，
        這裡會幫你切成 (max_seq_len, 4) 的小片段。
        """
        self.samples = []

        for folder, label in folder_label_pairs:
            pattern = os.path.join(folder, "*.csv")
            for path in glob.glob(pattern):
                df = pd.read_csv(path)

                current = df['current'].values
                voltage = df['voltage'].values
                power   = df['power'].values
                temp_C  = df['temp_C'].values

                seq = np.column_stack((current, voltage, power, temp_C))
                seq_len = seq.shape[0]

                for start in range(0, seq_len - max_seq_len + 1, stride):
                    end = start + max_seq_len
                    chunk = seq[start:end]              # (max_seq_len, 4)
                    self.samples.append((chunk.astype(np.float32), label))

        if not self.samples:
            raise RuntimeError(f"No csv files found in: {folder_label_pairs}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_np, label = self.samples[idx]
        x = torch.tensor(x_np, dtype=torch.float32)   # (seq_len, 4)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

# -----------------------------
# 4. 載入 base model 的權重
# -----------------------------
def load_base_model():
    model = LSTMClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    model_path = r"C:/Users/boss9/OneDrive/桌面/專題/機器學習/esp to python/樹莓派/20250819_fold_1_model.pth"
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("✅ Loaded base model from:", model_path)

    return model


# -----------------------------
# 5. 準備 fine-tune 資料
# -----------------------------
def build_datasets():
    base_root = r"C:/Users/boss9/OneDrive/桌面/專題/機器學習/dataset/feature dim_4/hardware"

    normal_dir   = os.path.join(base_root, "normal")
    abnormal_root = os.path.join(base_root, "abnormal")
    special_dir  = os.path.join(base_root, "特殊樣本")

    base_pairs = [
        (normal_dir, 0),
        (os.path.join(abnormal_root, "wire_rust"), 1),
        (os.path.join(abnormal_root, "transformer_rust"), 2),
        (os.path.join(abnormal_root, "transformer_overheating"), 3),
    ]

    special_pairs = [
        (special_dir, 1),
    ]

    base_dataset = SequenceFolderDataset(base_pairs, max_seq_len=MAX_SEQ_LEN, stride=STRIDE)
    special_dataset = SequenceFolderDataset(special_pairs, max_seq_len=MAX_SEQ_LEN, stride=STRIDE)

    # 可選：只取一部分 base，避免舊資料壓過特殊樣本
    from torch.utils.data import random_split, ConcatDataset
    base_len = len(base_dataset)
    keep_len = int(base_len * 0.5)  # 例如只用一半舊資料
    base_subset, _ = random_split(base_dataset, [keep_len, base_len - keep_len])

    finetune_dataset = ConcatDataset([base_subset, special_dataset])
    return finetune_dataset

# -----------------------------
# 6. 訓練 / fine-tune 函式
# -----------------------------
def train_finetune(model, dataset):
    model.to(DEVICE)

    # 1) 只訓練最後一層 fc（比較穩定）
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    # 2) 先把整個 dataset 抽出來，準備做 StandardScaler
    xs = []
    ys = []
    for x, y in dataset:  # x: (seq_len, 4), y: scalar
        xs.append(x.numpy())
        ys.append(int(y))

    xs = np.stack(xs, axis=0).astype(np.float32)  # (N, T, 4)
    ys = np.array(ys, dtype=np.int64)

    N, T, F = xs.shape
    # 3) 對所有 sample 做標準化 (samples * seq_len, features)
    scaler = StandardScaler()
    xs_2d = xs.reshape(-1, F)
    xs_2d = scaler.fit_transform(xs_2d)
    xs_scaled = xs_2d.reshape(N, T, F)

    # 4) 存成 pkl，之後推論要跟這顆 e10 模型一起用
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print("✅ Saved scaler to:", SCALER_SAVE_PATH)

    # 5) 把標準化後的資料包成新的 Dataset / DataLoader
    tensor_x = torch.tensor(xs_scaled, dtype=torch.float32)
    tensor_y = torch.tensor(ys, dtype=torch.long)
    finetune_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)

    dataloader = DataLoader(
        finetune_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )
    criterion = nn.CrossEntropyLoss()

    # 6) 訓練 loop
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / total
        acc = correct / total if total > 0 else 0.0
        print(f"[Epoch {epoch}/{EPOCHS}] loss = {avg_loss:.4f}, acc = {acc:.4f}")

    return model

# -----------------------------
# 7. 主程式
# -----------------------------
def main():
    model = load_base_model()
    finetune_dataset = build_datasets()
    model = train_finetune(model, finetune_dataset)

    # 存成新的 finetune 模型
    save_path = r"C:/Users/boss9/OneDrive/桌面/專題/機器學習/result/20251115_TLfinetune_e15_model.pth"
    torch.save(model.state_dict(), save_path)
    print("✅ Saved finetuned model to:", save_path)

if __name__ == "__main__":
    main()
