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
import time
import random
import joblib
import gc
import json
import shutil

# --- 1. 設定參數 ---
BASE_PATH = r"C:\Users\boss9\OneDrive\文件\專題\機器學習\dataset\feature dim_4\hardware"
RESULT_DIR = r"C:\Users\boss9\OneDrive\文件\專題\機器學習\result\pytorch\20260310"
os.makedirs(RESULT_DIR, exist_ok=True)

LABEL_DIRS = {
    0: os.path.join(BASE_PATH, "normal"),
    1: os.path.join(BASE_PATH, "abnormal/wire_rust"),
    2: os.path.join(BASE_PATH, "abnormal/transformer_rust"),
    3: os.path.join(BASE_PATH, "abnormal/transformer_overheating"),
}

SEGMENT_SIZE = 500   # 將 2500 筆的檔案切成 5 段，每段 500 筆
SAFETY_GAP = 50      # 段落間空出 50 筆，防止滑動視窗跨段重疊 (資料洩漏)

MAX_SEQ_LEN = 15
STRIDE = 1
INPUT_DIM = 4
HIDDEN_DIM = 16
NUM_LAYERS = 2
NUM_CLASSES = 4
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
def set_global_seed(seed=SEED):
    """確保實驗可重複性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# --- 2. 資料處理，加入虛擬段落邏輯 ---
def load_data():
    segments, labels = [], []
    for label, folder in LABEL_DIRS.items():
        if not os.path.exists(folder): continue
        for filename in os.listdir(folder):
            if filename.lower().endswith(".csv"):
                file_path = os.path.join(folder, filename)
                df = pd.read_csv(file_path)
                # 保留標題抓取
                seq_data = np.column_stack((df['current'].values, df['voltage'].values, df['power'].values, df['temp_C'].values))
                
                # 切分成虛擬段落
                start = 0
                while start + MAX_SEQ_LEN <= len(seq_data):
                    end = min(start + SEGMENT_SIZE, len(seq_data))
                    seg = seq_data[start:end]
                    if len(seg) >= MAX_SEQ_LEN:
                        segments.append(seg)
                        labels.append(label)
                    start = end + SAFETY_GAP
    return np.array(segments, dtype=object), np.array(labels)

# 將段落轉為滑動視窗 (訓練/驗證時呼叫)
def create_windows_from_segments(segments, labels, scaler, max_seq_len, stride):
    all_x, all_y = [], []
    for seg, lbl in zip(segments, labels):
        scaled_seg = scaler.transform(seg)
        for i in range(0, len(scaled_seg) - max_seq_len + 1, stride):
            all_x.append(scaled_seg[i : i + max_seq_len])
            all_y.append(lbl)
    return np.array(all_x, dtype=np.float32), np.array(all_y, dtype=np.int64)

# --- 4. 模型與 Dataset ---
class ChargingDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes=NUM_CLASSES, dropout_rate=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # 使用最後一個時間步的輸出
        out = self.dropout(out)
        return self.fc(out)

# === 5. 繪圖與評估函數 ===
def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            outputs = model(x_b)
            loss = criterion(outputs, y_b)
            running_loss += loss.item() * x_b.size(0)
            correct += (outputs.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
    return running_loss / total, correct / total

def evaluate_with_prf(model, loader, device, num_classes=NUM_CLASSES):
    """回傳：acc, precision, recall, f1, confusion_matrix"""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            preds.extend(model(x_b.to(device)).argmax(1).cpu().numpy())
            trues.extend(y_b.numpy())
    acc = (np.array(trues) == np.array(preds)).mean()
    p = precision_score(trues, preds, average='macro', zero_division=0)
    r = recall_score(trues, preds, average='macro', zero_division=0)
    f1 = f1_score(trues, preds, average='macro', zero_division=0)
    cm = confusion_matrix(trues, preds, labels=list(range(num_classes)))
    return acc, p, r, f1, cm

def plot_metric_curves(all_folds_metrics):
    """繪製各 Fold 的指標曲線圖"""
    for fold_df in all_folds_metrics:
        fold = fold_df['Fold'].iloc[0]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Fold {fold} Metrics Curves', fontsize=16)
        epochs = fold_df['Epoch']

        # 第一列：Loss 與 Precision
        axes[0, 0].plot(epochs, fold_df['Train Loss'], label='Train Loss')
        axes[0, 0].set_title('Train Loss'); axes[0, 0].legend()

        axes[0, 1].plot(epochs, fold_df['Test Loss'], label='Test Loss', color='orange')
        axes[0, 1].set_title('Test Loss'); axes[0, 1].legend()

        axes[0, 2].plot(epochs, fold_df['Train Precision'], label='Train Precision', color='green')
        axes[0, 2].set_title('Train Precision'); axes[0, 2].legend()
        
        # 第二列：Test Precision, Train F1, Test F1
        axes[1, 0].plot(epochs, fold_df['Test Precision'], label='Test Precision', color='red')
        axes[1, 0].set_title('Test Precision'); axes[1, 0].legend()
        
        axes[1, 1].plot(epochs, fold_df['Train F1-score'], label='Train F1-score', color='purple')
        axes[1, 1].set_title('Train F1-score'); axes[1, 1].legend()

        axes[1, 2].plot(epochs, fold_df['Test F1-score'], label='Test F1-score', color='brown')
        axes[1, 2].set_title('Test F1-score'); axes[1, 2].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(RESULT_DIR, f"fold_{fold}_metrics_curves.pdf"), bbox_inches='tight')
        plt.close()
        
def plot_overlaid_metrics(all_folds_metrics):
    """產出疊在一起的 Precision, Loss 與 F1-score 圖表"""
    metrics_to_plot = {
        "Test Precision": "combined_test_precision.pdf",
        "Test Loss": "combined_test_loss.pdf",
        "Test F1-score": "combined_test_f1.pdf"
    }
    
    for metric_name, file_name in metrics_to_plot.items():
        plt.figure(figsize=(10, 6))
        for fold_df in all_folds_metrics:
            fold = fold_df['Fold'].iloc[0]
            plt.plot(fold_df['Epoch'], fold_df[metric_name], label=f'Fold {fold}')
        
        plt.title(f'Combined {metric_name} Across Folds')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.split()[-1])
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULT_DIR, file_name), bbox_inches='tight')
        plt.close()

# === 6. K-fold 訓練流程 ===
def kfold_training(segments, labels):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_final_metrics, all_folds_metrics = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(segments, labels)):
        print(f"\n=== Fold {fold_idx+1} / 5 ===")
        train_segs, test_segs = segments[train_idx], segments[test_idx]
        
        scaler = StandardScaler(); scaler.fit(np.vstack(train_segs))
        X_train, y_train = create_windows_from_segments(train_segs, labels[train_idx], scaler, MAX_SEQ_LEN, STRIDE)
        X_test, y_test = create_windows_from_segments(test_segs, labels[test_idx], scaler, MAX_SEQ_LEN, STRIDE)

        train_loader = DataLoader(ChargingDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(ChargingDataset(X_test, y_test), batch_size=BATCH_SIZE)

        model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT_RATE).to(DEVICE)
        criterion = nn.CrossEntropyLoss(); optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_loss_l, test_loss_l, train_p_l, test_p_l, train_f1_l, test_f1_l = [], [], [], [], [], []
        
        for epoch in range(NUM_EPOCHS):
            model.train(); r_loss, corr, tot = 0, 0, 0
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad(); out = model(x_b); loss = criterion(out, y_b)
                loss.backward(); optimizer.step()
                r_loss += loss.item()*x_b.size(0); corr += (out.argmax(1)==y_b).sum().item(); tot += y_b.size(0)
            
            # 1. 計算測試集指標 (呼叫一次，拿回所有想要的資料)
            # evaluate_with_prf 回傳：acc, precision, recall, f1, cm
            v_loss, _ = evaluate_model(model, test_loader, criterion, DEVICE) # 拿 Loss
            _, test_p, _, test_f1, _ = evaluate_with_prf(model, test_loader, DEVICE) # 拿 P 和 F1

            # 2. 計算訓練集指標 (為了畫第六張圖與訓練 Precision)
            _, train_p, _, train_f1, _ = evaluate_with_prf(model, train_loader, DEVICE)

            # 3. 將資料存入清單
            train_loss_l.append(r_loss/tot)
            train_p_l.append(train_p)
            test_loss_l.append(v_loss)
            test_p_l.append(test_p)
            test_f1_l.append(test_f1)
            train_f1_l.append(train_f1)
            
            print(f"Epoch {epoch+1:2d} | Val Pre: {test_p:.4f} | Test F1: {test_f1:.4f} | Train F1: {train_f1:.4f}")

        fold_final_metrics.append({'Fold': fold_idx+1, 'Post-train Precision': test_p, 'F1-Score': test_f1})
        
        # 繪製混淆矩陣
        _, _, _, _, cm = evaluate_with_prf(model, test_loader, DEVICE)
        disp = ConfusionMatrixDisplay(cm).plot(cmap='Blues')
        plt.savefig(os.path.join(RESULT_DIR, f"fold_{fold_idx+1}_cm.svg"))
        plt.close()

        torch.save(model.state_dict(), os.path.join(RESULT_DIR, f"fold_{fold_idx+1}_model.pth"))
        joblib.dump(scaler, os.path.join(RESULT_DIR, f"fold_{fold_idx+1}_scaler.pkl"))

        # 封裝成 DataFrame
        all_folds_metrics.append(pd.DataFrame({
            'Epoch': range(1, NUM_EPOCHS+1), 
            'Fold': fold_idx+1, 
            'Train Loss': train_loss_l, 
            'Test Loss': test_loss_l, 
            'Train Precision': train_p_l,
            'Test Precision': test_p_l,
            'Test F1-score': test_f1_l,
            'Train F1-score': train_f1_l
        }))
        del model; torch.cuda.empty_cache(); gc.collect()

    return pd.DataFrame(fold_final_metrics), all_folds_metrics

def train_final_model(segments, labels):
    """訓練最終部署模型並儲存設定檔"""
    final_dir = os.path.join(RESULT_DIR, "final_train"); os.makedirs(final_dir, exist_ok=True)
    scaler = StandardScaler(); scaler.fit(np.vstack(segments))
    X_all, y_all = create_windows_from_segments(segments, labels, scaler, MAX_SEQ_LEN, STRIDE)
    loader = DataLoader(ChargingDataset(X_all, y_all), batch_size=BATCH_SIZE, shuffle=True)
    model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT_RATE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE); criterion = nn.CrossEntropyLoss()
    for epoch in range(NUM_EPOCHS):
        model.train()
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad(); criterion(model(x_b), y_b).backward(); optimizer.step()
    torch.save(model.state_dict(), os.path.join(final_dir, "final_model.pth"))
    joblib.dump(scaler, os.path.join(final_dir, "final_scaler.pkl"))
    with open(os.path.join(final_dir, "final_config.json"), "w") as f:
        json.dump({"MAX_SEQ_LEN": MAX_SEQ_LEN, "HIDDEN_DIM": HIDDEN_DIM, "NUM_LAYERS": NUM_LAYERS}, f)

def count_chunks_in_folder(folder_path, max_seq_len=MAX_SEQ_LEN, stride=STRIDE):
    """
    計算指定資料夾內所有 csv 檔案，依據 max_seq_len 與 stride（滑動步長）切分後的總片段數量。
    """
    total_chunks = 0
    if not isinstance(stride, int) or stride <= 0:
        stride = 1
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            num_rows = len(df)
            if num_rows < max_seq_len:
                chunks = 0
            else:
                chunks = 1 + (num_rows - max_seq_len) // stride
            total_chunks += chunks
    return total_chunks

if __name__ == "__main__":
    set_global_seed(SEED)
    print(f"[DEBUG] 開始載入資料並切分虛擬段落...")
    all_segments, all_labels = load_data()
    
    # 確認資料量
    for label, path in LABEL_DIRS.items():
        print(f"Label {label} 預計片段數: {count_chunks_in_folder(path)}")
    
    # 執行 K-fold 驗證
    final_metrics_df, all_folds_metrics = kfold_training(all_segments, all_labels)
    
    # 儲存最終摘要表格
    final_metrics_df.to_csv(os.path.join(RESULT_DIR, "kfold_final_metrics.csv"), index=False)
    print("\n=== K-fold 最終結果摘要 ===")
    print(final_metrics_df)
    
    # 繪製曲線圖
    plot_metric_curves(all_folds_metrics)
    plot_overlaid_metrics(all_folds_metrics)
    
    # 訓練最終部署模型
    train_final_model(all_segments, all_labels)