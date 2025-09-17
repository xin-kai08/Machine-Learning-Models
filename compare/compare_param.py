import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

# === Dataset ===
class ChargeSequenceDataset3D(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# === LSTM 模型 ===
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout_rate=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最後一個時間步
        out = self.dropout(out)
        return self.fc(out)

# === 比較函式（支援滑動視窗） ===
def compare_lstm_param_sliding(param_name, param_values,
                               RESULT_DIR, suffix,
                               base_preprocessed_dir,
                               stride=5,
                               bs=16, lr=0.01, seq_len=15,
                               num_epochs=100, k_folds=5,
                               input_dim=4, hidden_dim=16, num_layers=1, num_classes=4):

    os.makedirs(RESULT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化圖
    plt_f1, plt_loss, plt_acc = plt.figure(figsize=(10,6)), plt.figure(figsize=(10,6)), plt.figure(figsize=(10,6))
    ax_f1, ax_loss, ax_acc = plt_f1.add_subplot(111), plt_loss.add_subplot(111), plt_acc.add_subplot(111)

    for val in param_values:
        cur_bs, cur_lr, cur_seq = bs, lr, seq_len
        if param_name == "batch_size":
            cur_bs = val
        elif param_name == "learning_rate":
            cur_lr = val
        elif param_name == "seq_len":
            cur_seq = val
        else:
            raise ValueError("param_name 必須是 'batch_size'、'learning_rate' 或 'seq_len'")

        f1_curves, acc_curves, loss_curves = [], [], []

        for fold_idx in range(1, k_folds+1):
            # 載入「滑動視窗」資料
            X_train = np.load(os.path.join(base_preprocessed_dir, f"X_train_fold{fold_idx}_seq{cur_seq}_{suffix}.npy"))
            y_train = np.load(os.path.join(base_preprocessed_dir, f"y_train_fold{fold_idx}_seq{cur_seq}_{suffix}.npy"))
            X_val = np.load(os.path.join(base_preprocessed_dir, f"X_val_fold{fold_idx}_seq{cur_seq}_{suffix}.npy"))
            y_val = np.load(os.path.join(base_preprocessed_dir, f"y_val_fold{fold_idx}_seq{cur_seq}_{suffix}.npy"))

            # 保證檔案是從「stride_x」資料夾讀的
            assert f"stride_{stride}" in base_preprocessed_dir, \
                "⚠️ 你的 base_preprocessed_dir 要指到滑動視窗 (stride) 的資料夾！"

            train_loader = DataLoader(ChargeSequenceDataset3D(X_train, y_train), batch_size=cur_bs, shuffle=True)
            val_loader = DataLoader(ChargeSequenceDataset3D(X_val, y_val), batch_size=cur_bs)

            model = LSTMClassifier(input_dim, hidden_dim, num_layers, num_classes).to(device)
            optimizer = optim.Adam(model.parameters(), lr=cur_lr)
            criterion = nn.CrossEntropyLoss()

            f1_list, acc_list, loss_list = [], [], []
            for epoch in range(num_epochs):
                # === train ===
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()

                # === validate ===
                model.eval()
                y_pred_epoch, y_true_epoch, total_loss = [], [], 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        out = model(xb)
                        loss = criterion(out, yb)
                        _, pred = torch.max(out, 1)
                        total_loss += loss.item() * yb.size(0)
                        y_pred_epoch.extend(pred.cpu().numpy())
                        y_true_epoch.extend(yb.cpu().numpy())

                acc = accuracy_score(y_true_epoch, y_pred_epoch)
                f1 = f1_score(y_true_epoch, y_pred_epoch, average='macro', zero_division=0)
                avg_loss = total_loss / len(y_true_epoch)

                acc_list.append(acc)
                f1_list.append(f1)
                loss_list.append(avg_loss)

                print(f"[{param_name}={val}] Fold {fold_idx} | Epoch {epoch+1}/{num_epochs} | "
                f"Loss={avg_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
                
            f1_curves.append(f1_list)
            acc_curves.append(acc_list)
            loss_curves.append(loss_list)

        # K-fold 平均
        avg_f1 = np.mean(f1_curves, axis=0)
        avg_acc = np.mean(acc_curves, axis=0)
        avg_loss = np.mean(loss_curves, axis=0)

        epochs = range(1, num_epochs+1)
        ax_f1.plot(epochs, avg_f1, label=f"{param_name}={val}")
        ax_acc.plot(epochs, avg_acc, label=f"{param_name}={val}")
        ax_loss.plot(epochs, avg_loss, label=f"{param_name}={val}")

    # === F1 圖 ===
    ax_f1.set_title(f"LSTM F1-score vs Epoch (stride={stride}) for Different {param_name}")
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("F1-score")
    ax_f1.legend()
    ax_f1.grid(True)
    plt_f1.savefig(os.path.join(RESULT_DIR, f"f1_curve_{param_name}_stride{stride}.png"), bbox_inches="tight")

    # === Accuracy 圖 ===
    ax_acc.set_title(f"LSTM Accuracy vs Epoch (stride={stride}) for Different {param_name}")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True)
    plt_acc.savefig(os.path.join(RESULT_DIR, f"acc_curve_{param_name}_stride{stride}.png"), bbox_inches="tight")

    # === Loss 圖 ===
    ax_loss.set_title(f"LSTM Loss vs Epoch (stride={stride}) for Different {param_name}")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)
    plt_loss.savefig(os.path.join(RESULT_DIR, f"loss_curve_{param_name}_stride{stride}.png"), bbox_inches="tight")

    plt.close("all")

compare_lstm_param_sliding(
    param_name="seq_len",              # 要比較的超參數 seq_len, batch_size, learning_rate
    param_values=[1, 3, 5, 7, 10, 15, 20],
    RESULT_DIR=r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\result\feature dim_4\hardware\compare_seq_len",
    suffix="3d",
    base_preprocessed_dir=r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\dataset\feature dim_4\hardware\preprocessed_kfold\3D\stride_5",
    stride=5,
    bs=16, lr=0.01, seq_len=10,
    num_epochs=100, k_folds=5
)