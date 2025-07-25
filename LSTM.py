# 0725後為滑動視窗

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import time

# 資料集根目錄
BASE_PATH = r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\dataset\feature dim_4\hardware"
RESULT_DIR = r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\result"

# 各分類資料夾設定
LABEL_DIRS = {
    0: os.path.join(BASE_PATH, "normal"),
    1: os.path.join(BASE_PATH, "abnormal/wire_rust"),
    2: os.path.join(BASE_PATH, "abnormal/transformer_rust"),
    3: os.path.join(BASE_PATH, "abnormal/transformer_overheating"),
}

# 設定參數
MAX_SEQ_LEN = 10
STRIDE = 1  # 每次滑動幾步

INPUT_DIM = 4
HIDDEN_DIM = 16
NUM_LAYERS = 4
NUM_CLASSES = len(LABEL_DIRS)
NUM_EPOCHS = 100

KFOLD_SPLITS = 5
SEED = 42

os.makedirs(RESULT_DIR, exist_ok=True)

# === 資料處理 ===
def process_file(file_path, label, sequences, labels, max_seq_len=MAX_SEQ_LEN):
    """
    讀取 CSV 檔案並切分為長度為 max_seq_len 的小段。
    :param file_path: CSV檔案路徑
    :param label: 資料標籤（例如：0、1、2、3）
    :param sequences: 儲存切分後片段的 list
    :param labels: 儲存對應標籤的 list
    """
    df = pd.read_csv(file_path)
    # 假設 CSV 中有 'current', 'voltage', 'power', 'temp_C' 四個欄位
    current = df['current'].values
    voltage = df['voltage'].values
    power = df['power'].values
    temp_C = df['temp_C'].values

    sequence = np.column_stack((current, voltage, temp_C, power))  # shape: (N, 4)
    seq_len = sequence.shape[0]
    for start in range(0, seq_len - max_seq_len + 1, STRIDE):
        end = start + max_seq_len
        chunk = sequence[start:end]
        sequences.append(chunk)
        labels.append(label)


def load_data():
    sequences = []
    labels = []
    for label, folder in LABEL_DIRS.items():
        for filename in os.listdir(folder):
            if filename.lower().endswith(".csv"):
                file_path = os.path.join(folder, filename)
                process_file(file_path, label=label, sequences=sequences, labels=labels, max_seq_len=MAX_SEQ_LEN)
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print("sequences shape:", sequences.shape)
    print("labels shape:", labels.shape)
    for i in range(NUM_CLASSES):
        print(f"Number of class {i} samples:", np.sum(labels == i))
    print("Unique labels:", np.unique(labels))
    return sequences, labels

def split_data(sequences, labels, test_size=0.2, val_size=0.1):
    """
    先切出測試集，再從剩餘資料中切出驗證集。
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        sequences, labels, test_size=test_size, stratify=labels, random_state=SEED)
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative, stratify=y_train_val, random_state=SEED)
    return X_train, X_val, X_test, y_train, y_val, y_test


# === 自訂 Dataset ===
class ChargingDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# === LSTM 模型定義 ===
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes=NUM_CLASSES, dropout_rate=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]   # 使用最後一個時間步的輸出
        out = self.dropout(out)
        out = self.fc(out)
        return out


# === 評估函數 ===
def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * x_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    avg_loss = running_loss / total
    avg_acc = correct / total if total > 0 else 0
    return avg_loss, avg_acc


# === K-fold 訓練流程 ===
def kfold_training(sequences, labels):
    dataset = ChargingDataset(sequences, labels)
    kfold = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fold_final_metrics = []
    all_folds_metrics = []  # 儲存每個 fold 的所有 epoch 指標 DataFrame

    fold_idx = 0
    for train_indices, test_indices in kfold.split(sequences, labels):
        fold_idx += 1
        print(f"\n=== Fold {fold_idx} / {KFOLD_SPLITS} ===")
        # 切分訓練與測試資料
        train_sequences, train_labels = sequences[train_indices], labels[train_indices]
        test_sequences, test_labels = sequences[test_indices], labels[test_indices]

        # --- 新增資料正規化處理 ---
        scaler = StandardScaler()
        # 將訓練資料從 3D 轉為 2D: (samples * seq_len, features)
        train_reshaped = train_sequences.reshape(-1, train_sequences.shape[-1])
        scaler.fit(train_reshaped)
        # 將訓練資料轉換回原本的形狀
        train_sequences = scaler.transform(train_reshaped).reshape(train_sequences.shape)

        # 測試資料僅使用 transform，避免資料洩漏
        test_reshaped = test_sequences.reshape(-1, test_sequences.shape[-1])
        test_sequences = scaler.transform(test_reshaped).reshape(test_sequences.shape)
        # --- 正規化處理結束 ---

        train_dataset = ChargingDataset(train_sequences, train_labels)
        test_dataset = ChargingDataset(test_sequences, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 建立模型（未訓練狀態）
        model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 訓練前：用初始模型在測試集上的表現作為基線（Pre-train Metrics）
        pre_test_loss, pre_test_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"Pre-training  | Test Loss: {pre_test_loss:.4f}, Test Acc: {pre_test_acc:.4f}")
        print("===========================================")

        # 記錄各 epoch 指標
        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []
        test_precision_list = []
        test_recall_list = []
        test_f1_list = []

        start_time = time.time()
        # 開始訓練
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total if total > 0 else 0
            epoch_test_loss, epoch_test_acc = evaluate_model(model, test_loader, criterion, device)

            train_loss_list.append(epoch_train_loss)
            train_acc_list.append(epoch_train_acc)
            test_loss_list.append(epoch_test_loss)
            test_acc_list.append(epoch_test_acc)

            # 計算其他指標（以 macro 平均）
            model.eval()
            preds = []
            trues = []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    outputs = model(x_batch)
                    _, predicted = torch.max(outputs, 1)
                    preds.extend(predicted.cpu().numpy())
                    trues.extend(y_batch.cpu().numpy())
            test_precision = precision_score(trues, preds, average='macro', zero_division=0)
            test_recall = recall_score(trues, preds, average='macro', zero_division=0)
            test_f1 = f1_score(trues, preds, average='macro', zero_division=0)

            test_precision_list.append(test_precision)
            test_recall_list.append(test_recall)
            test_f1_list.append(test_f1)

            print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
                  f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.4f} | "
                  f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

        end_time = time.time()
        print(f"⌛訓練時間：{end_time - start_time:.2f} 秒")

        # 訓練結束後，取得訓練後（Post-train）的測試集表現
        post_test_loss, post_test_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"Post-training | Test Loss: {post_test_loss:.4f}, Test Acc: {post_test_acc:.4f}")

        fold_final_metrics.append({
            'Fold': fold_idx,
            'Pre-train Loss': pre_test_loss,
            'Pre-train Accuracy': pre_test_acc,
            'Post-train Loss': post_test_loss,
            'Post-train Accuracy': post_test_acc,
            'Precision': test_precision_list[-1],
            'Recall': test_recall_list[-1],
            'F1-Score': test_f1_list[-1]
        })

        # 混淆矩陣 (使用所有類別)
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(x_batch)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
                trues.extend(y_batch.cpu().numpy())
        cm = confusion_matrix(trues, preds, labels=list(range(NUM_CLASSES)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(NUM_CLASSES)))
        plt.figure(figsize=(4,4))
        disp.plot(values_format='d', cmap='Blues')
        plt.title(f"Fold {fold_idx} - Confusion Matrix")
        cm_pdf_path = os.path.join(RESULT_DIR, f"fold_{fold_idx}_cm.pdf")
        cm_svg_path = os.path.join(RESULT_DIR, f"fold_{fold_idx}_cm.svg")
        plt.savefig(cm_pdf_path, bbox_inches='tight')
        plt.savefig(cm_svg_path, bbox_inches='tight')
        plt.show()

        # 儲存模型
        model_save_path = os.path.join(RESULT_DIR, f"fold_{fold_idx}_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model for fold {fold_idx} saved to {model_save_path}")

        # 將該 fold 的所有 epoch 指標記錄成 DataFrame
        epochs_range = range(1, NUM_EPOCHS+1)
        fold_metrics_df = pd.DataFrame({
            'Epoch': epochs_range,
            'Train Loss': train_loss_list,
            'Test Loss': test_loss_list,
            'Train Accuracy': train_acc_list,
            'Test Accuracy': test_acc_list,
            'Test Precision': test_precision_list,
            'Test Recall': test_recall_list,
            'Test F1-score': test_f1_list
        })
        fold_metrics_df['Fold'] = fold_idx
        all_folds_metrics.append(fold_metrics_df)

    final_metrics_df = pd.DataFrame(fold_final_metrics)
    print("\n=== Final Metrics Summary Across All Folds ===")
    print(final_metrics_df.to_string(index=False))
    return final_metrics_df, all_folds_metrics


def plot_metric_curves(all_folds_metrics):
    for fold_df in all_folds_metrics:
        fold = fold_df['Fold'].iloc[0]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Fold {fold} Metrics Curves', fontsize=16)
        epochs = fold_df['Epoch']

        # Train Loss
        axes[0, 0].plot(epochs, fold_df['Train Loss'], label='Train Loss')
        axes[0, 0].set_title('Train Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()

        # Test Loss
        axes[0, 1].plot(epochs, fold_df['Test Loss'], label='Test Loss', color='orange')
        axes[0, 1].set_title('Test Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()

        # Train Accuracy
        axes[0, 2].plot(epochs, fold_df['Train Accuracy'], label='Train Accuracy', color='green')
        axes[0, 2].set_title('Train Accuracy')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()

        # Test Accuracy
        axes[1, 0].plot(epochs, fold_df['Test Accuracy'], label='Test Accuracy', color='red')
        axes[1, 0].set_title('Test Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()

        # Test Precision
        axes[1, 1].plot(epochs, fold_df['Test Precision'], label='Test Precision', color='purple')
        axes[1, 1].set_title('Test Precision')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()

        # Test F1-score
        axes[1, 2].plot(epochs, fold_df['Test F1-score'], label='Test F1-score', color='brown')
        axes[1, 2].set_title('Test F1-score')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('F1-score')
        axes[1, 2].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_pdf = os.path.join(RESULT_DIR, f"fold_{fold}_metrics_curves.pdf")
        plot_svg = os.path.join(RESULT_DIR, f"fold_{fold}_metrics_curves.svg")
        plt.savefig(plot_pdf, bbox_inches='tight')
        plt.savefig(plot_svg, bbox_inches='tight')
        plt.close(fig)


def plot_overlaid_metrics(all_folds_metrics):
    # 假設所有 fold 的 epoch 數量相同
    epochs = all_folds_metrics[0]['Epoch']

    # 定義你想疊加繪製的指標
    metrics = {
        'Train Accuracy': 'Train Accuracy',
        'Test Accuracy': 'Test Accuracy',
        'Train Loss': 'Train Loss',
        'Test Loss': 'Test Loss',
        'Test Precision': 'Test Precision',
        'Test Recall': 'Test Recall',
        'Test F1-score': 'Test F1-score'
    }

    for metric_title, col in metrics.items():
        plt.figure(figsize=(10, 6))
        for fold_df in all_folds_metrics:
            fold = fold_df['Fold'].iloc[0]
            plt.plot(epochs, fold_df[col], label=f'Fold {fold}')
        plt.title(f'Combined {metric_title}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_title)
        plt.legend()
        plt.tight_layout()

        pdf_path = os.path.join(RESULT_DIR, f"combined_{col}.pdf")
        svg_path = os.path.join(RESULT_DIR, f"combined_{col}.svg")
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.savefig(svg_path, bbox_inches='tight')
        plt.show()
        plt.close()


def count_chunks_in_folder(folder_path, max_seq_len=MAX_SEQ_LEN):
    """
    計算指定資料夾內所有 CSV 檔案，依據 max_seq_len 切分後的總片段數量。
    """
    total_chunks = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            num_rows = len(df)
            chunks = num_rows // max_seq_len
            total_chunks += chunks
    return total_chunks

def best_result():
    results_csv_path = os.path.join(RESULT_DIR, "overall_experiment_log.csv")
    if not os.path.exists(results_csv_path):
        print("未發現網格搜尋結果 CSV，請先執行網格搜尋。")
        return None
    df = pd.read_csv(results_csv_path)
    # 假設以驗證準確率最高為最佳模型
    best_row = df.loc[df['post_train_acc'].idxmax()]
    return best_row

if __name__ == "__main__":
    # === ✅ 加這段，保證 feature dim_4 是乾淨的 ===
    import shutil
    CLEAR_RESULT_DIR_BEFORE_RUN = True

    if CLEAR_RESULT_DIR_BEFORE_RUN:
        print(f"[DEBUG] Clearing {RESULT_DIR} before run...")
        shutil.rmtree(RESULT_DIR, ignore_errors=True)
        os.makedirs(RESULT_DIR, exist_ok=True)

    # 直接設定變數即可
    args_check_data = False  # True 只檢查資料
    args_best = False        # True 只印最佳結果

    if args_best:
        best_model = best_result()
        if best_model is not None:
            print("最好的模型指標:")
            print(best_model)
    elif args_check_data:
        print(f"Max seq len = {MAX_SEQ_LEN}")
        for label, folder in LABEL_DIRS.items():
            count = count_chunks_in_folder(folder, max_seq_len=MAX_SEQ_LEN)
            print(f"Label {label} ({folder}): {count} chunks")
    else:
        # 超參數
        batch_size_values = [8, 16, 32]
        learning_rate_values = [1e-2, 1e-3, 1e-1, 1e-4]
        max_seq_len_values = [10, 20, 30, 40]
        stride_values = [1]

        # 儲存所有實驗結果記錄
        overall_experiment_logs = []
        # 記錄所有實驗的總結果（此 log 最後存成 CSV 檔）
        overall_results = []

        # 保存原始的 RESULT_DIR 值，方便還原
        original_RESULT_DIR = RESULT_DIR

        # 依照超參數組合進行迴圈
        for stride in stride_values:
            STRIDE = stride
            for bs in batch_size_values:
                for lr in learning_rate_values:
                    for seq in max_seq_len_values:
                        # 更新全域變數（注意：這裡是更新本模組內的變數）
                        BATCH_SIZE = bs
                        LEARNING_RATE = lr
                        MAX_SEQ_LEN = seq

                        # 為每組參數建立獨立儲存結果的資料夾
                        exp_id = f"bs_{bs}_lr_{lr}_seq_{seq}_stride_{stride}"
                        print(f"\n==== Running experiment: {exp_id} ====")
                        exp_result_dir = os.path.join(original_RESULT_DIR, exp_id)
                        os.makedirs(exp_result_dir, exist_ok=True)

                        # 暫時改寫 RESULT_DIR，讓後續的儲存檔案寫入此目錄
                        RESULT_DIR = exp_result_dir

                        # 載入資料（會根據 MAX_SEQ_LEN 切分資料）
                        all_sequences, all_labels = load_data()

                        # 執行 K-fold 訓練
                        final_metrics_df, all_folds_metrics = kfold_training(all_sequences, all_labels)

                        # 繪製指標曲線圖
                        plot_metric_curves(all_folds_metrics)
                        plot_overlaid_metrics(all_folds_metrics)

                        # 儲存本次實驗的最終指標 log
                        log_csv_path = os.path.join(RESULT_DIR, "final_metrics.csv")
                        final_metrics_df.to_csv(log_csv_path, index=False)
                        print(f"Final metrics logged at: {log_csv_path}")

                        # 將本次實驗資訊存入 overall_experiment_logs
                        overall_experiment_logs.append({
                            'experiment_id': exp_id,
                            'batch_size': bs,
                            'learning_rate': lr,
                            'max_seq_len': seq,
                            'final_metrics': final_metrics_df
                        })
                        # 也可將各折最終的平均值記錄下來（例如：Post-train Accuracy）
                        overall_results.append({
                            'experiment_id': exp_id,
                            'batch_size': bs,
                            'learning_rate': lr,
                            'max_seq_len': seq,
                            'pre_train_loss': final_metrics_df['Pre-train Loss'].mean(),
                            'pre_train_acc': final_metrics_df['Pre-train Accuracy'].mean(),
                            'post_train_loss': final_metrics_df['Post-train Loss'].mean(),
                            'post_train_acc': final_metrics_df['Post-train Accuracy'].mean(),
                            'precision': final_metrics_df['Precision'].mean(),
                            'recall': final_metrics_df['Recall'].mean(),
                            'f1_score': final_metrics_df['F1-Score'].mean()
                        })

                        RESULT_DIR = original_RESULT_DIR

        # 儲存所有實驗的總結果 log
        overall_log_path = os.path.join(RESULT_DIR, "overall_experiment_log.csv")
        overall_df = pd.DataFrame(overall_results)
        overall_df.to_csv(overall_log_path, index=False)
        print(f"Overall experiment log saved at: {overall_log_path}")