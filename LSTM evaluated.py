import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM  = 4
HIDDEN_DIM = 16      # 要跟你原本的一樣
NUM_LAYERS = 1
NUM_CLASSES = 4
MAX_SEQ_LEN = 15
STRIDE = 5
BATCH_SIZE = 64

# ---------- 跟原本一樣的 LSTM ----------
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
        b = x.size(0)
        h0 = torch.zeros(self.num_layers, b, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, b, self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ---------- 專門讀「特殊樣本」csv 的 Dataset ----------
class SpecialCSVSeqDataset(Dataset):
    def __init__(self, folder, max_seq_len=15, stride=5):
        self.samples = []
        pattern = os.path.join(folder, "*.csv")
        for path in glob.glob(pattern):
            df = pd.read_csv(path)

            current = df['current'].values
            voltage = df['voltage'].values
            power   = df['power'].values
            temp_C  = df['temp_C'].values

            seq = np.column_stack((current, voltage, power, temp_C))
            n = seq.shape[0]
            for start in range(0, n - max_seq_len + 1, stride):
                end = start + max_seq_len
                self.samples.append(seq[start:end].astype(np.float32))

        if not self.samples:
            raise RuntimeError(f"No csv in {folder}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)  # (T,4)
        return x

# ---------- 做一次標準化，兩個模型共用 ----------
def build_loader_special():
    special_dir = r"C:/Users/boss9/OneDrive/桌面/專題/機器學習/dataset/feature dim_4/hardware/特殊樣本"
    ds = SpecialCSVSeqDataset(special_dir, max_seq_len=MAX_SEQ_LEN, stride=STRIDE)

    # 把所有 sample 堆在一起做 scaler（跟剛剛 fine-tune 類似）
    all_x = torch.stack([ds[i] for i in range(len(ds))]).numpy()  # (N,T,4)
    N, T, F = all_x.shape
    scaler = StandardScaler()
    x2d = all_x.reshape(-1, F)
    x2d = scaler.fit_transform(x2d)
    all_x = x2d.reshape(N, T, F)

    tensor_x = torch.tensor(all_x, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor_x)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader

# ---------- 載入 model 並預測 ----------
def load_model(path):
    model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, dropout=0.0)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def predict_all(model, loader):
    preds = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(DEVICE)
            logits = model(x)
            p = torch.argmax(logits, dim=1)
            preds.extend(p.cpu().numpy().tolist())
    return np.array(preds, dtype=int)

def main():
    loader = build_loader_special()

    base_path = r"C:/Users/boss9/OneDrive/桌面/專題/機器學習/esp to python/樹莓派/20250819_fold_1_model.pth"
    ft_path   = r"C:/Users/boss9/OneDrive/桌面/專題/機器學習/result/2025_finetune_1112_e10_model.pth"

    base_model = load_model(base_path)
    ft_model   = load_model(ft_path)

    base_preds = predict_all(base_model, loader)
    ft_preds   = predict_all(ft_model, loader)

    print("🔹 base model 預測分佈（0=normal,1=wire,2=Tr rust,3=overheat）:")
    for c in range(NUM_CLASSES):
        print(f"  class {c}: {np.sum(base_preds==c)}")

    print("🔹 finetune model 預測分佈:")
    for c in range(NUM_CLASSES):
        print(f"  class {c}: {np.sum(ft_preds==c)}")

if __name__ == "__main__":
    main()
