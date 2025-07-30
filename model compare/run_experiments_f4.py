import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, accuracy_score, recall_score, f1_score
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC

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
                           RESULT_DIR, suffix:str,
                           num_epochs=100, k_folds=5, num_classes=4,
                           base_preprocessed_dir=None, trial=None):

    os.makedirs(RESULT_DIR, exist_ok=True)

    results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    total_combinations = len(batch_sizes) * len(learning_rates) * len(seq_lens)
    progress_bar = tqdm(total=total_combinations, desc="Total Sweep Progress")

    for bs in batch_sizes:
        for lr in learning_rates:
            for seq_len in seq_lens:
                print(f"\n🧪 BS={bs} | LR={lr} | SEQ={seq_len}")
                
                PREPROCESSED_DIR = base_preprocessed_dir

                acc_curves, loss_curves, f1_curves = [], [], []
                acc_train_curves, loss_train_curves = [], []
                all_y_true, all_y_pred = [], []

                t0 = time.time()

                for fold_idx in range(1, k_folds+1):
                    X_train = np.load(os.path.join(
                        PREPROCESSED_DIR, f"X_train_fold{fold_idx}_seq{seq_len}_{suffix}.npy"))
                    y_train = np.load(os.path.join(
                        PREPROCESSED_DIR, f"y_train_fold{fold_idx}_seq{seq_len}_{suffix}.npy"))
                    X_val = np.load(os.path.join(
                        PREPROCESSED_DIR, f"X_val_fold{fold_idx}_seq{seq_len}_{suffix}.npy"))
                    y_val = np.load(os.path.join(
                        PREPROCESSED_DIR, f"y_val_fold{fold_idx}_seq{seq_len}_{suffix}.npy"))

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
                            print(f"    [Fold {fold_idx}] Epoch {epoch+1}/{num_epochs} | Acc: {acc:.4f} | Loss: {avg_loss:.4f}")

                    acc_curves.append(acc_list)
                    loss_curves.append(loss_list)
                    f1_curves.append(f1_list)
                    acc_train_curves.append(acc_train_list)
                    loss_train_curves.append(loss_train_list)
                    all_y_true.extend(y_true_epoch)
                    all_y_pred.extend(y_pred_epoch)

                t1 = time.time()

                # === 曲線繪製 ===
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
                    path = os.path.join(RESULT_DIR, f"{metric_name}_curve.png")
                    plt.savefig(path, bbox_inches='tight')
                    plt.close()

                plot_metric_per_fold(acc_curves, "accuracy", RESULT_DIR)
                plot_metric_per_fold(loss_curves, "loss", RESULT_DIR)
                plot_metric_per_fold(f1_curves, "f1_score", RESULT_DIR)
                plot_metric_per_fold(acc_train_curves, "train_accuracy", RESULT_DIR)
                plot_metric_per_fold(loss_train_curves, "train_loss", RESULT_DIR)

                cm = confusion_matrix(all_y_true, all_y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(all_y_true))
                disp.plot(cmap='Blues', values_format='d')
                plt.title(f"Confusion Matrix\nBS={bs} LR={lr} SEQ={seq_len}")
                plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"), bbox_inches='tight')
                plt.close()

                final_result = {
                    'trial_number': trial.number,
                    'batch_size': bs,
                    'learning_rate': lr,
                    'seq_len': seq_len,
                    'dropout_rate': model_args.get('dropout_rate', None),
                    
                    # 模型專屬參數（通用處理）
                    'hidden_dim': model_args.get('hidden_dim', None),
                    'num_layers': model_args.get('num_layers', None),
                    'cnn_channels1': model_args.get('channels1', None),
                    'cnn_channels2': model_args.get('channels2', None),
                    'cnn_kernel_size': model_args.get('kernel_size', None),
                    'tf_heads': model_args.get('num_heads', None),
                    
                    # 最終結果
                    'final_acc': acc_list[-1],
                    'final_loss': loss_list[-1],
                    'precision': precision_score(all_y_true, all_y_pred, average='macro', zero_division=0),
                    'recall': recall_score(all_y_true, all_y_pred, average='macro', zero_division=0),
                    'f1_score': f1_score(all_y_true, all_y_pred, average='macro', zero_division=0),
                    'training_time_s': round(t1 - t0, 2)
                }
                results.append(final_result)
                
                csv_dir = os.path.dirname(RESULT_DIR)
                csv_path = os.path.join(csv_dir, "experiment_results.csv")
                if not os.path.exists(csv_path):
                    pd.DataFrame([final_result]).to_csv(csv_path, index=False)
                else:
                    pd.DataFrame([final_result]).to_csv(csv_path, mode='a', index=False, header=False)

                print(f"\n📊 統計結果 (BS={bs}, LR={lr}, SEQ={seq_len}):")
                print(f"  🔹 Final Accuracy : {final_result['final_acc']:.4f}")
                print(f"  🔹 F1-score       : {final_result['f1_score']:.4f}")
                print(f"  ⏱️  Training Time  : {final_result['training_time_s']} 秒")

                progress_bar.update(1)

    progress_bar.close()

    df = pd.DataFrame(results)
    print(f"\n📄 All experiment results saved to {csv_path}")

    if not df.empty:
        best = df.loc[df['final_acc'].idxmax()]
        print(f"\n🏆 Best: BS={best['batch_size']} | LR={best['learning_rate']} | SEQ={best['seq_len']} | ACC={best['final_acc']:.4f}")
        return results, best
    else:
        print("\n❗ No valid results")
        return [], None

def train_and_search_model_svm(model_class, model_args, DatasetClass,
                               X_filename, y_filename,
                               Cs, kernels, seq_lens,
                               RESULT_DIR, suffix:str,
                               k_folds=5, base_preprocessed_dir=None, trial=None):

    os.makedirs(RESULT_DIR, exist_ok=True)
    results = []

    for C in Cs:
        for kernel in kernels:
            for seq_len in seq_lens:
                all_y_true, all_y_pred = [], []

                acc_list, f1_list = [], []
                t0 = time.time()

                for fold_idx in range(1, k_folds+1):
                    X_train = np.load(os.path.join(base_preprocessed_dir, f"X_train_fold{fold_idx}_seq{seq_len}_{suffix}.npy"))
                    y_train = np.load(os.path.join(base_preprocessed_dir, f"y_train_fold{fold_idx}_seq{seq_len}_{suffix}.npy"))
                    X_val = np.load(os.path.join(base_preprocessed_dir, f"X_val_fold{fold_idx}_seq{seq_len}_{suffix}.npy"))
                    y_val = np.load(os.path.join(base_preprocessed_dir, f"y_val_fold{fold_idx}_seq{seq_len}_{suffix}.npy"))

                    model = model_class(C=C, kernel=kernel)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)

                    acc_list.append(accuracy_score(y_val, y_pred))
                    f1_list.append(f1_score(y_val, y_pred, average='macro', zero_division=0))
                    all_y_true.extend(y_val)
                    all_y_pred.extend(y_pred)

                t1 = time.time()

                result = {
                    'trial_number': trial.number if trial else -1,
                    'C': C,
                    'kernel': kernel,
                    'seq_len': seq_len,
                    'final_acc': np.mean(acc_list),
                    'f1_score': np.mean(f1_list),
                    'precision': precision_score(all_y_true, all_y_pred, average='macro', zero_division=0),
                    'recall': recall_score(all_y_true, all_y_pred, average='macro', zero_division=0),
                    'training_time_s': round(t1 - t0, 2)
                }

                results.append(result)

                # === 儲存 confusion matrix 圖到 trial 資料夾 ===
                if trial is not None:
                    cm = confusion_matrix(all_y_true, all_y_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(all_y_true))
                    disp.plot(cmap='Blues', values_format='d')
                    plt.title(f"Confusion Matrix\nSVM (C={C}, kernel={kernel}, seq={seq_len})")
                    os.makedirs(RESULT_DIR, exist_ok=True)
                    plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"), bbox_inches='tight')
                    plt.close()

    # === 儲存 CSV 到 RESULT_DIR ===
    csv_dir = os.path.dirname(RESULT_DIR)
    csv_path = os.path.join(csv_dir, "experiment_results.csv")

    df = pd.DataFrame(results)
    best = None
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', index=False, header=False)
    return results, best

def objective(trial, model_name, stride):
    # === 通用參數 ===
    bs = trial.suggest_categorical("batch_size", [8, 16, 32])
    lr = trial.suggest_categorical("learning_rate", [1e-1, 1e-2, 1e-3, 1e-4])
    seq_len = trial.suggest_categorical("seq_len", [10, 20, 30, 40])
    trial.set_user_attr("model", model_name)
    trial.set_user_attr("stride", stride)
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.0, 0.3])

    # === 根據模型選擇 Dataset 類型、suffix 以及模型超參數 ===
    if model_name == "MLP":
        hidden_dim = trial.suggest_categorical("mlp_hidden", [16, 32, 64])
        model_class = MLPClassifier
        model_args = {
            'input_dim': 16,
            'hidden_dim': hidden_dim,
            'num_classes': 4,
            'dropout_rate': dropout_rate
        }
        dataset_cls = ChargeSequenceDataset2D
        suffix = "2d"

    elif model_name == "SVM":
        C = trial.suggest_categorical("svm_C", [0.1, 1.0, 10.0])
        kernel = trial.suggest_categorical("svm_kernel", ["linear", "rbf"])
        seq_len = trial.suggest_categorical("seq_len", [10, 20, 30, 40])
        suffix = "2d"

        pre_dir = os.path.join(base_data_path, "2D")
        result_dir = os.path.join(base_result_path, "SVM", f"stride_{stride}", f"trial_{trial.number}")

        results, best = train_and_search_model_svm(
            model_class=SVMClassifier,
            model_args={'C': C, 'kernel': kernel},
            DatasetClass=None,  # 不使用 Dataset 物件
            X_filename="", y_filename="",
            Cs=[C], kernels=[kernel], seq_lens=[seq_len],
            RESULT_DIR=result_dir,
            suffix=suffix,
            base_preprocessed_dir=pre_dir,
            trial=trial
        )
        return best['final_acc'] if best is not None else 0.0

    # lstm_args = {'input_dim': 4, 'hidden_dim': 64, 'num_layers': 1, 'num_classes': 4}
    elif model_name == "LSTM":
        hidden_dim = trial.suggest_categorical("lstm_hidden", [16, 32, 64])
        num_layers = trial.suggest_categorical("lstm_layers", [1, 2, 3, 4])
        model_class = LSTMClassifier
        model_args = {
            'input_dim': 4,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_classes': 4,
            'dropout_rate': dropout_rate
        }
        dataset_cls = ChargeSequenceDataset3D
        suffix = "3d"

    # gru_args = {'input_dim': 4, 'hidden_dim': 64, 'num_layers': 1, 'num_classes': 4}
    elif model_name == "GRU":
        hidden_dim = trial.suggest_categorical("gru_hidden", [16, 32, 64])
        num_layers = trial.suggest_categorical("gru_layers", [1, 2, 3, 4])
        model_class = GRUClassifier
        model_args = {
            'input_dim': 4,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_classes': 4,
            'dropout_rate': dropout_rate
        }
        dataset_cls = ChargeSequenceDataset3D
        suffix = "3d"

    # cnn_args = {'input_dim': 4, 'num_classes': 4}
    elif model_name == "1D CNN":
        channels1 = trial.suggest_categorical("cnn_channels1", [16, 32, 64])
        channels2 = trial.suggest_categorical("cnn_channels2", [32, 64, 128])
        kernel_size = trial.suggest_categorical("cnn_kernel_size", [3, 5])

        model_class = CNN1DClassifier
        model_args = {
            'input_dim': 4,
            'num_classes': 4,
            'channels1': channels1,
            'channels2': channels2,
            'kernel_size': kernel_size,
            'dropout_rate': dropout_rate
        }
        dataset_cls = ChargeSequenceDataset3D
        suffix = "3d"

    # timesnet_args = {'input_dim': 4, 'hidden_dim': 64, 'num_layers': 2, 'num_classes': 4}
    elif model_name == "TimesNet":
        hidden_dim = trial.suggest_categorical("times_hidden", [16, 32, 64])
        num_layers = trial.suggest_categorical("times_layers", [1, 2, 3, 4])
        model_class = TimesNetClassifier
        model_args = {
            'input_dim': 4,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_classes': 4
        }
        dataset_cls = ChargeSequenceDataset3D
        suffix = "3d"

    # transformer_args = {'input_dim': 4, 'num_heads': 2, 'num_layers': 2, 'hidden_dim': 64, 'num_classes': 4}
    elif model_name == "Transformer":
        num_heads = trial.suggest_categorical("tf_heads", [2, 4])
        num_layers = trial.suggest_int("tf_layers", 1, 4)
        hidden_dim = trial.suggest_categorical("tf_hidden", [16, 32, 64])
        model_class = TransformerClassifier
        model_args = {
            'input_dim': 4,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'hidden_dim': hidden_dim,
            'num_classes': 4
        }
        dataset_cls = ChargeSequenceDataset3D
        suffix = "3d"

    # === 建立資料與結果路徑 ===
    if model_name == "MLP":
        pre_dir = os.path.join(base_data_path, "2D")
    else:
        pre_dir = os.path.join(base_data_path, "3D", f"stride_{stride}")
    result_dir = os.path.join(base_result_path, model_name, f"stride_{stride}", f"trial_{trial.number}")

    results, best = train_and_search_model(
        model_class=model_class,
        model_args=model_args,
        DatasetClass=dataset_cls,
        X_filename=f"X_seq{seq_len}_{suffix}.npy",
        y_filename=f"y_seq{seq_len}_{suffix}.npy",
        batch_sizes=[bs],
        learning_rates=[lr],
        seq_lens=[seq_len],
        RESULT_DIR=result_dir,
        suffix=suffix,
        num_epochs=num_epochs,
        base_preprocessed_dir=pre_dir,
        trial=trial
    )

    return best['final_acc'] if best is not None else 0.0

# === LSTMClassifier ===
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout_rate=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

# === MLPClassifier ===
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.view(x.size(0), -1)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input, got shape {x.shape}")
        return self.model(x)

# === SVMClassifier ===
class SVMClassifier:
    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel
        self.model = SVC(C=C, kernel=kernel)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
# === GRUClassifier ===
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout_rate=0.0):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

# === CNN1DClassifier ===
class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, channels1=16, channels2=32, kernel_size=3, dropout_rate=0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_dim, channels1, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels1, channels2, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels2, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
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

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, num_classes):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # or take x[:,0,:] if you add CLS token
        return self.fc(x)

# === 主程式呼叫範例 ===
if __name__ == "__main__":
    # 超參數搜尋設定
    num_epochs = 100
    feature_dim = 4
    n_trials = 10
    base_data_path = r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\dataset\feature dim_4\hardware\preprocessed_kfold"
    base_result_path = r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\model compare\result\feature dim_4\hardware\model"

    # 執行 Optuna 搜尋
    # 模型順序清單["LSTM", "MLP", "SVM", "GRU", "1D CNN", "TimesNet", "Transformer"]
    model_list = ["SVM"]
    stride_list = [10]
    
    for model_name in model_list:
        if model_name in ["MLP", "SVM"]:
            stride_used = [None]
        else:
            stride_used = stride_list

        for stride in stride_used:
             # === 清除舊資料 ===
            csv_path = os.path.join(base_result_path, model_name, f"stride_{stride}", "experiment_results.csv")
            if os.path.exists(csv_path):
                os.remove(csv_path)

            print(f"\n🚀 開始模型：{model_name}（stride={stride}）")

            # 傳入 stride 的 wrapped objective
            def wrapped_objective(trial):
                return objective(trial, model_name, stride)

            study = optuna.create_study(direction="maximize")
            study.optimize(wrapped_objective, n_trials=n_trials)

            print(f"✅ Best for {model_name} (stride={stride}): ACC={study.best_value:.4f}")