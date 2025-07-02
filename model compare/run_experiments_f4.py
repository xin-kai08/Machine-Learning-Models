"""
🚀 通用版 run_experiments.py 範例 🚀

✅ 支援模型：
    - LSTM
    - MLP
    - GRU
    - CNN1D
    - TimesNet

✅ 支援資料格式：
    - ChargeSequenceDataset3D → X_seq{seq_len}_3d.npy
    - ChargeSequenceDataset2D → X_seq{seq_len}_2d.npy

✅ 通用版 `train_and_search_model` 已整合：
    - log / tqdm / plot / confusion matrix / CSV → 一致風格
    - 一鍵跑 sweep → 可比較不同模型結果
    - 未來擴充新模型只需加 model_class + model_args 即可

✨ 專題 pipeline 高一致性版本 ✨
"""

import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# === Dataset ===
class ChargeSequenceDataset3D(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class ChargeSequenceDataset2D(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# === 通用版 train_and_search_model ===
def train_and_search_model(model_class, model_args, DatasetClass,
                           X_filename, y_filename,
                           batch_sizes, learning_rates, seq_lens,
                           RESULT_DIR,
                           num_epochs=100, k_folds=5, num_classes=4):

    PREPROCESSED_DIR = r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\dataset\feature dim_4\hardware\preprocessed"

    SUB_DIRS = [
        "accuracy_curves", "loss_curves", "f1_score_curves",
        "confusion_matrices", "train_accuracy", "train_loss", "logs"
    ]
    os.makedirs(RESULT_DIR, exist_ok=True)
    for sub in SUB_DIRS:
        os.makedirs(os.path.join(RESULT_DIR, sub), exist_ok=True)

    results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    total_combinations = len(batch_sizes) * len(learning_rates) * len(seq_lens)
    progress_bar = tqdm(total=total_combinations, desc="Total Sweep Progress")

    for bs in batch_sizes:
        for lr in learning_rates:
            for seq_len in seq_lens:
                print(f"\n🧪 BS={bs} | LR={lr} | SEQ={seq_len}")

                x_path = os.path.join(PREPROCESSED_DIR, X_filename.format(seq_len=seq_len))
                y_path = os.path.join(PREPROCESSED_DIR, y_filename.format(seq_len=seq_len))
                X = np.load(x_path)
                y = np.load(y_path)

                skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                acc_curves, loss_curves, f1_curves = [], [], []
                acc_train_curves, loss_train_curves = [], []
                all_y_true, all_y_pred = [], []

                t0 = time.time()

                log_path = os.path.join(RESULT_DIR, "logs", f"bs{bs}_lr{lr}_seq{seq_len}.txt")
                with open(log_path, "w") as log_file:

                    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                        X_train, X_val = X[train_idx], X[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]

                        train_loader = DataLoader(DatasetClass(X_train, y_train), batch_size=bs, shuffle=True)
                        val_loader = DataLoader(DatasetClass(X_val, y_val), batch_size=bs)

                        model = model_class(**model_args).to(device)
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        criterion = nn.CrossEntropyLoss()

                        acc_list, loss_list, f1_list = [], [], []
                        acc_train_list, loss_train_list = [], []

                        for epoch in range(num_epochs):
                            model.train()
                            total_loss_train, correct_train, total_train = 0, 0, 0
                            for xb, yb in train_loader:
                                xb, yb = xb.to(device), yb.to(device)
                                optimizer.zero_grad()
                                out = model(xb)
                                loss = criterion(out, yb)
                                loss.backward()
                                optimizer.step()

                                _, pred = torch.max(out, 1)
                                correct_train += (pred == yb).sum().item()
                                total_train += yb.size(0)
                                total_loss_train += loss.item() * xb.size(0)

                            acc_train = correct_train / total_train
                            loss_train = total_loss_train / total_train

                            acc_train_list.append(acc_train)
                            loss_train_list.append(loss_train)

                            model.eval()
                            correct, total, total_loss = 0, 0, 0
                            y_pred_epoch, y_true_epoch = [], []
                            with torch.no_grad():
                                for xb, yb in val_loader:
                                    xb, yb = xb.to(device), yb.to(device)
                                    out = model(xb)
                                    loss = criterion(out, yb)
                                    _, pred = torch.max(out, 1)
                                    correct += (pred == yb).sum().item()
                                    total += yb.size(0)
                                    total_loss += loss.item() * xb.size(0)
                                    y_pred_epoch.extend(pred.cpu().numpy())
                                    y_true_epoch.extend(yb.cpu().numpy())

                            acc = correct / total
                            avg_loss = total_loss / total
                            f1 = f1_score(y_true_epoch, y_pred_epoch, average='macro', zero_division=0)

                            acc_list.append(acc)
                            loss_list.append(avg_loss)
                            f1_list.append(f1)

                            if (epoch + 1) % 10 == 0:
                                print(f"    [Fold {fold+1}] Epoch {epoch+1}/{num_epochs} | Acc: {acc:.4f} | Loss: {avg_loss:.4f}")
                                log_file.write(f"[Fold {fold+1}] Epoch {epoch+1}/{num_epochs} | Acc: {acc:.4f} | Loss: {avg_loss:.4f}\n")

                        acc_curves.append(acc_list)
                        loss_curves.append(loss_list)
                        f1_curves.append(f1_list)
                        acc_train_curves.append(acc_train_list)
                        loss_train_curves.append(loss_train_list)
                        all_y_true.extend(y_true_epoch)
                        all_y_pred.extend(y_pred_epoch)

                    t1 = time.time()

                    def plot_metric_per_fold(fold_lists, metric_name, folder):
                        plt.figure(figsize=(10, 6))
                        for fold_idx, fold_metric in enumerate(fold_lists):
                            plt.plot(range(1, num_epochs + 1), fold_metric, label=f"Fold {fold_idx + 1}")
                        plt.title(f"Combined {metric_name.capitalize()} (BS={bs}, LR={lr}, SEQ={seq_len})")
                        plt.xlabel("Epoch")
                        plt.ylabel(metric_name.capitalize())
                        if metric_name != 'loss':
                            plt.ylim(0, 1.0)
                        plt.grid(True)
                        plt.legend()
                        path = os.path.join(RESULT_DIR, folder, f"bs{bs}_lr{lr}_seq{seq_len}_combined_{metric_name}.png")
                        plt.savefig(path, bbox_inches='tight')
                        plt.close()

                    plot_metric_per_fold(acc_curves, "accuracy", "accuracy_curves")
                    plot_metric_per_fold(loss_curves, "loss", "loss_curves")
                    plot_metric_per_fold(f1_curves, "f1_score", "f1_score_curves")
                    plot_metric_per_fold(acc_train_curves, "accuracy", "train_accuracy")
                    plot_metric_per_fold(loss_train_curves, "loss", "train_loss")

                    cm = confusion_matrix(all_y_true, all_y_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
                    disp.plot(cmap='Blues', values_format='d')
                    plt.title(f"Confusion Matrix\nBS={bs} LR={lr} SEQ={seq_len}")
                    plt.savefig(os.path.join(RESULT_DIR, "confusion_matrices", f"bs{bs}_lr{lr}_seq{seq_len}.png"), bbox_inches='tight')
                    plt.close()

                    final_result = {
                        'batch_size': bs, 'learning_rate': lr, 'seq_len': seq_len,
                        'final_acc': acc_list[-1], 'final_loss': loss_list[-1],
                        'precision': precision_score(all_y_true, all_y_pred, average='macro', zero_division=0),
                        'recall': recall_score(all_y_true, all_y_pred, average='macro', zero_division=0),
                        'f1_score': f1_score(all_y_true, all_y_pred, average='macro', zero_division=0),
                        'training_time_s': round(t1 - t0, 2)
                    }
                    if final_result['final_acc'] >= 1.0:
                        print(f"⚠️ 準確率為 1.0 的結果已跳過：BS={bs} LR={lr} SEQ={seq_len}")
                        continue
                    results.append(final_result)

                    print(f"\n📊 統計結果 (BS={bs}, LR={lr}, SEQ={seq_len}):")
                    print(f"  🔹 Final Accuracy : {final_result['final_acc']:.4f}")
                    print(f"  🔹 F1-score       : {final_result['f1_score']:.4f}")
                    print(f"  ⏱️  Training Time  : {final_result['training_time_s']} 秒")

                    log_file.write(f"\nFinal Result:\n")
                    log_file.write(f"  Final Accuracy: {final_result['final_acc']:.4f}\n")
                    log_file.write(f"  F1-score      : {final_result['f1_score']:.4f}\n")
                    log_file.write(f"  Training Time : {final_result['training_time_s']} 秒\n")

                progress_bar.update(1)

    progress_bar.close()

    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULT_DIR, "experiment_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n📄 All experiment results saved to {csv_path}")

    if not df.empty:
        best = df.loc[df['final_acc'].idxmax()]
        print(f"\n🏆 Best: BS={best['batch_size']} | LR={best['learning_rate']} | SEQ={best['seq_len']} | ACC={best['final_acc']:.4f}")
        return results, best
    else:
        print("\n❗ No valid results")
        return [], None


# === LSTMClassifier ===
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# === MLPClassifier ===
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# === GRUClassifier ===
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

# === CNN1DClassifier ===
class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        x = x.transpose(1, 2)  # (B, F, T)
        return self.model(x)

# === TimesNet ===
class TimesBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.skip_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1) if input_dim != hidden_dim else None
    def forward(self, x):
        residual = x.transpose(1, 2)  # (B, F, T)
        x = residual
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.skip_conv is not None:
            residual = self.skip_conv(residual)
        x = x + residual
        x = x.transpose(1, 2)  # (B, T, F)
        return x

class TimesNetClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.blocks = nn.ModuleList([TimesBlock(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.pool(x).squeeze(-1)  # (B, F)
        return self.fc(x)

# === model_args 範例 ===
lstm_args = {'input_dim': 4, 'hidden_dim': 64, 'num_layers': 1, 'num_classes': 4}
gru_args = {'input_dim': 4, 'hidden_dim': 64, 'num_layers': 1, 'num_classes': 4}
cnn_args = {'input_dim': 4, 'num_classes': 4}
timesnet_args = {'input_dim': 4, 'hidden_dim': 64, 'num_layers': 2, 'num_classes': 4}

# === 主程式呼叫範例 ===
if __name__ == "__main__":
    # 訓練參數設定
    batch_sizes = [8, 16, 32]
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
    seq_lens = [10, 20, 30, 40]
    num_epochs = 100

    # === LSTM ===
    # train_and_search_model(LSTMClassifier, lstm_args,
    #                        ChargeSequenceDataset3D,
    #                        "X_seq{seq_len}_3d.npy", "y_seq{seq_len}_3d.npy",
    #                        batch_sizes, learning_rates, seq_lens,
    #                        RESULT_DIR=r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\model compare\result\feature dim_4\hardware\LSTM",
    #                        num_epochs=num_epochs)

    # === MLP ===
    BASE_RESULT_DIR = r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\model compare\result\feature dim_4\hardware\MLP"

    for seq_len in seq_lens:
        mlp_args = {
            'input_dim': seq_len * 4,
            'hidden_dim': 128,
            'num_classes': 4
        }

        result_dir = os.path.join(BASE_RESULT_DIR, f"seq_{seq_len}")

        train_and_search_model(
            MLPClassifier,
            mlp_args,
            ChargeSequenceDataset2D,
            "X_seq{seq_len}_2d.npy",
            "y_seq{seq_len}_2d.npy",
            batch_sizes,
            learning_rates,
            [seq_len],  # 這邊固定 [seq_len] 就不會多餘
            RESULT_DIR=result_dir,
            num_epochs=num_epochs
        )

    # === GRU ===
    # train_and_search_model(GRUClassifier, gru_args,
    #                        ChargeSequenceDataset3D,
    #                        "X_seq{seq_len}_3d.npy", "y_seq{seq_len}_3d.npy",
    #                        batch_sizes, learning_rates, seq_lens,
    #                        RESULT_DIR=r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\model compare\result\feature dim_4\hardware\GRU",
    #                        num_epochs=num_epochs)

    # === CNN1D ===
    # train_and_search_model(CNN1DClassifier, cnn_args,
    #                        ChargeSequenceDataset3D,
    #                        "X_seq{seq_len}_3d.npy", "y_seq{seq_len}_3d.npy",
    #                        batch_sizes, learning_rates, seq_lens,
    #                        RESULT_DIR=r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\model compare\result\feature dim_4\hardware\CNN",
    #                        num_epochs=num_epochs)

    # === TimesNet ===
    # train_and_search_model(TimesNetClassifier, timesnet_args,
    #                        ChargeSequenceDataset3D,
    #                        "X_seq{seq_len}_3d.npy", "y_seq{seq_len}_3d.npy",
    #                        batch_sizes, learning_rates, seq_lens,
    #                        RESULT_DIR=r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\model compare\result\feature dim_4\hardware\TimesNet",
    #                        num_epochs=num_epochs)