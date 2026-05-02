import os
import json
import torch
import optuna
import time
import math
import torch.fft
import gc
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.metrics import log_loss

# --- 1. 設定與路徑 ---
BASE_PATH = r"C:\Users\boss9\OneDrive\文件\專題\機器學習\dataset\feature dim_4\hardware"
RESULT_DIR = r"C:\Users\boss9\OneDrive\文件\專題\機器學習\result\20260420"
os.makedirs(RESULT_DIR, exist_ok=True)

# 定義標籤路徑
LABEL_DIRS = {
    0: os.path.join(BASE_PATH, "normal"),
    1: os.path.join(BASE_PATH, "abnormal/wire_rust"),
    2: os.path.join(BASE_PATH, "abnormal/transformer_rust"),
    3: os.path.join(BASE_PATH, "abnormal/transformer_overheating"),
}

LOG_CSV = os.path.join(RESULT_DIR, "optuna_model_comparison.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# --- 2. 模型定義與參數註冊器 (模組化核心) ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers,
            dropout=0, 
            batch_first=True, 
            bidirectional=bidirectional
        )
        
        # 如果是雙向，輸出會是 hidden_dim * 2
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        # 取最後一個時間步
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        # h_n shape: (num_layers, batch, hidden_dim)
        out, _ = self.gru(x)
        # 取最後一層的最後一個時間步
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out) 

class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, out_channels=64, kernel_size=3, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels, num_classes)
    def forward(self, x):
        # 輸入預期: (Batch, Channels, Seq_Len)
        x = self.conv1(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    def forward(self, x):
        return self.net(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 創建一個足夠長的 PE 矩陣 (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 使用 register_buffer，它會跟隨模型移動到 GPU/CPU，但不會被視為要訓練的參數
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        # 動態根據當前 x 的長度 (Seq_Len) 進行切片
        x = x + self.pe[:, :x.size(1), :]
        return x
    
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dim_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, dim_model)
        # 加入位置編碼（支援動態長度）
        self.pos_encoder = PositionalEncoding(dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(dim_model, num_classes)
    def forward(self, x):
        x = self.embedding(x) # (B, L, F) -> (B, L, D)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :]) # 取最後一個時間步進行分類

class TimesBlock(nn.Module):
    def __init__(self, seq_len, top_k, d_model, d_ff, num_kernels, dropout):
        super().__init__()
        self.top_k = top_k
        self.seq_len = seq_len
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_ff, kernel_size=num_kernels, padding=num_kernels//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(d_ff, d_model, kernel_size=num_kernels, padding=num_kernels//2)
        )
    def forward(self, x):
        B, T, D = x.size()
        # 1. 快速傅立葉變換 (FFT) 找週期
        xf = torch.fft.rfft(x, dim=1)
        frequency_list = torch.abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, self.top_k)
        top_list = top_list.detach().cpu().numpy()
        top_list = np.clip(top_list, 1, T // 2)
        period = T // top_list
        res = []
        for i in range(self.top_k):
            p = period[i]
            # 2. 1D 轉 2D (Folding)
            # 這裡需要 Padding 確保能被整除
            if T % p != 0:
                length = ((T // p) + 1) * p
                padding = torch.zeros([B, length - T, D]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x
            
            out = out.reshape(B, length // p, p, D).permute(0, 3, 1, 2) # (B, D, 2D_H, 2D_W)
            
            # 3. 2D 卷積提取特徵
            out = self.conv(out)
            
            # 4. 2D 轉回 1D (Unfolding)
            out = out.permute(0, 2, 3, 1).reshape(B, length, D)
            res.append(out[:, :T, :])
            
        # 5. 權重融合
        res = torch.stack(res, dim=-1)
        # 這裡簡化為平均融合
        out = torch.mean(res, dim=-1)
        return out + x # 殘差連接

class TimesNetClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, top_k, d_model, d_ff, num_kernels, num_classes, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Linear(input_dim, d_model)
        self.model = TimesBlock(seq_len, top_k, d_model, d_ff, num_kernels, dropout)
        self.fc = nn.Linear(d_model, num_classes)
    def forward(self, x):
        # x: (B, L, F)
        x = self.embedding(x)
        x = self.model(x)
        x = x.mean(1) # 池化收尾
        return self.fc(x)
    
def get_model_params(trial):
    model_type = trial.suggest_categorical("model_type", ["GRU"])          #["LSTM", "GRU", "CNN1D", "MLP", "Transformer", "TimesNet", "SVM"]
    
    params = {}
    if model_type == "LSTM":
        params["seq_len"] = trial.suggest_categorical("lstm_seq_len", [7])
        params["batch_size"] = trial.suggest_categorical("lstm_batch_size", [128])
        params["lr"] = trial.suggest_float("lstm_lr", 5e-2, log=True)
        params["hidden_dim"] = trial.suggest_categorical("lstm_hidden", [64])
        params["num_layers"] = trial.suggest_categorical("lstm_layers", [3])
        params["dropout"] = trial.suggest_categorical("lstm_dropout", [0.5])

    elif model_type == "GRU":
        params["seq_len"] = trial.suggest_categorical("gru_seq_len", [15, 20, 25])
        params["batch_size"] = trial.suggest_categorical("gru_batch_size", [64, 128])
        params["lr"] = trial.suggest_float("gru_lr", 5e-3, 5e-2, log=True)
        params["hidden_dim"] = trial.suggest_categorical("gru_hidden", [32, 64])
        params["num_layers"] = trial.suggest_categorical("gru_layers", [1, 2])
        params["dropout"] = trial.suggest_categorical("gru_dropout", [0.4, 0.5])

    elif model_type == "CNN1D":
        params["seq_len"] = trial.suggest_categorical("cnn_seq_len", [10, 20])
        params["batch_size"] = trial.suggest_categorical("cnn_batch_size", [64, 128])
        params["lr"] = trial.suggest_categorical("cnn_lr", [1e-1, 1e-2])
        params["out_channels"] = trial.suggest_categorical("cnn_channels", [32])
        params["kernel_size"] = trial.suggest_categorical("cnn_kernel", [3, 5])
        params["dropout"] = trial.suggest_categorical("cnn_dropout", [0.3])

    elif model_type == "MLP":
        params["seq_len"] = trial.suggest_categorical("mlp_seq_len", [10, 20])
        params["batch_size"] = trial.suggest_categorical("mlp_batch_size", [64, 128])
        params["lr"] = trial.suggest_categorical("mlp_lr", [1e-1, 1e-2])
        params["hidden_dim"] = trial.suggest_categorical("mlp_hidden", [16])
        params["dropout"] = trial.suggest_categorical("mlp_dropout", [0.3])

    elif model_type == "Transformer":
        params["seq_len"] = trial.suggest_categorical("trans_seq_len", [10, 20])
        params["batch_size"] = trial.suggest_categorical("trans_batch_size", [64, 128])
        params["lr"] = trial.suggest_categorical("trans_lr", [1e-1, 1e-2])
        params["dim_model"] = trial.suggest_categorical("trans_dim", [32])
        params["nhead"] = trial.suggest_categorical("trans_nhead", [2])
        params["num_layers"] = trial.suggest_categorical("trans_layers", [2])
        params["dropout"] = trial.suggest_categorical("trans_dropout", [0.3])

    elif model_type == "TimesNet":
        params["seq_len"] = trial.suggest_categorical("times_seq_len", [10, 20])
        params["batch_size"] = trial.suggest_categorical("times_batch_size", [64, 128])
        params["lr"] = trial.suggest_categorical("times_lr", [1e-1, 1e-2])
        params["top_k"] = trial.suggest_categorical("times_topk", [1])
        params["d_model"] = trial.suggest_categorical("times_dim", [16])
        params["d_ff"] = params["d_model"] * 2
        params["num_kernels"] = trial.suggest_int("times_kernels", 4, 8)
        params["dropout"] = trial.suggest_categorical("times_dropout", [0.3])

    elif model_type == "SVM":
        params["seq_len"] = trial.suggest_categorical("svm_seq_len", [10, 20])
        params["C"] = trial.suggest_categorical("svm_C",[0.1])
        params["kernel"] = trial.suggest_categorical("svm_kernel", ["rbf", "linear"])
        params["gamma"] = trial.suggest_categorical("svm_gamma", ["scale", "auto"])

    return model_type, params

# --- 3. 資料處理模組 ---
class ChargingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def load_data(label_dirs, segment_size, safety_gap, min_len):
    """透過參數傳入設定，不依賴外部全域變數"""
    segments, labels = [], []
    for label, folder in label_dirs.items():
        if not os.path.exists(folder): continue
        for filename in os.listdir(folder):
            if filename.lower().endswith(".csv"):
                file_path = os.path.join(folder, filename)
                df = pd.read_csv(file_path)
                # 抓取特徵欄位
                seq_data = np.column_stack((df['current'].values, df['voltage'].values, 
                                            df['power'].values, df['temp_C'].values))
                
                start = 0
                while start + min_len <= len(seq_data):
                    end = min(start + segment_size, len(seq_data))
                    seg = seq_data[start:end]
                    if len(seg) >= min_len:
                        segments.append(seg)
                        labels.append(label)
                    start = end + safety_gap
    return np.array(segments, dtype=object), np.array(labels)

def create_windows_from_segments(segments, labels, scaler, max_seq_len, stride=1):
    """根據指定的 max_seq_len 產生滑動視窗"""
    all_x, all_y = [], []
    for seg, lbl in zip(segments, labels):
        scaled_seg = scaler.transform(seg)
        for i in range(0, len(scaled_seg) - max_seq_len + 1, stride):
            all_x.append(scaled_seg[i : i + max_seq_len])
            all_y.append(lbl)
    return np.array(all_x, dtype=np.float32), np.array(all_y, dtype=np.int64)

# --- 4. 訓練與評估引擎 ---
def run_train_fold(model, train_loader, val_loader, lr, epochs=100, adapter=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_f1, best_loss = 0, float('inf')

    for epoch in range(epochs):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            if adapter: x_b = adapter(x_b)
            
            optimizer.zero_grad()
            outputs = model(x_b)
            
            # --- 通用安全性檢查：使用 outputs 的維度，不再依賴 model.fc ---
            if torch.max(y_b) >= outputs.shape[1]:
                raise ValueError(f"發現非法標籤: {torch.max(y_b).item()}，"
                                 f"但模型輸出維度僅有 {outputs.shape[1]}。")
            
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss, all_preds, all_targets = 0, [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                if adapter: x_b = adapter(x_b)
                outputs = model(x_b)
                total_loss += criterion(outputs, y_b).item()
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_targets.extend(y_b.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        if f1 > best_f1:
            best_f1, best_loss = f1, avg_loss
            
    return best_loss, best_f1

# --- 5. Optuna 目標函數 ---
def objective(trial, all_segments, all_labels):
    start_time = time.time()
    model_type, params = get_model_params(trial)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []
    
    # 動態取得類別數，避免硬編碼 4 導致的越界問題
    num_classes = len(np.unique(all_labels))

    for fold_idx, (t_idx, v_idx) in enumerate(kfold.split(all_segments, all_labels)):
        # 標準化與視窗切割
        scaler = StandardScaler().fit(np.vstack(all_segments[t_idx]))
        X_train, y_train = create_windows_from_segments(all_segments[t_idx], all_labels[t_idx], scaler, params['seq_len'])
        X_val, y_val = create_windows_from_segments(all_segments[v_idx], all_labels[v_idx], scaler, params['seq_len'])

        if model_type == "SVM":
            X_train_2d = X_train.reshape(len(X_train), -1)
            X_val_2d = X_val.reshape(len(X_val), -1)
            model = SVC(C=params['C'], kernel=params['kernel'], gamma=params['gamma'], probability=True)
            model.fit(X_train_2d, y_train)
            preds = model.predict(X_val_2d)
            f1 = f1_score(y_val, preds, average='macro', zero_division=0)
            loss = log_loss(y_val, model.predict_proba(X_val_2d), labels=np.arange(num_classes))
        else:
            train_loader = DataLoader(ChargingDataset(X_train, y_train), batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(ChargingDataset(X_val, y_val), batch_size=params['batch_size'])
            
            adapter = None
            if model_type == "LSTM":
                model = LSTMClassifier(4, params['hidden_dim'], params['num_layers'], num_classes, params['dropout']).to(DEVICE)
            elif model_type == "GRU":
                model = GRUClassifier(4, params['hidden_dim'], params['num_layers'], num_classes, params['dropout']).to(DEVICE)
            elif model_type == "CNN1D":
                model = CNN1DClassifier(4, num_classes, params['out_channels'], params['kernel_size'], params['dropout']).to(DEVICE)
                adapter = lambda x: x.transpose(1, 2)
            elif model_type == "MLP":
                model = MLPClassifier(4 * params['seq_len'], params['hidden_dim'], num_classes, params['dropout']).to(DEVICE)
                adapter = lambda x: x.view(x.size(0), -1)
            elif model_type == "Transformer":
                if params['dim_model'] % params['nhead'] != 0: raise optuna.exceptions.TrialPruned()
                model = TransformerClassifier(4, num_classes, params['dim_model'], params['nhead'], params['num_layers'], params['dropout']).to(DEVICE)
            elif model_type == "TimesNet":
                model = TimesNetClassifier(4, params['seq_len'], params['top_k'], params['d_model'], num_classes, params['dropout']).to(DEVICE)
            
            loss, f1 = run_train_fold(model, train_loader, val_loader, lr=params['lr'], adapter=adapter)

        # 存入 fold 結果，包含 "fold" 這個 Key
        fold_results.append({"fold": fold_idx, "loss": loss, "f1": f1})

    # (B) 計算總訓練時間 (秒)
    end_time = time.time()
    duration = end_time - start_time

    # (C) 整理最終數據
    best_fold = max(fold_results, key=lambda x: x['f1'])
    avg_f1 = np.mean([f['f1'] for f in fold_results])
    avg_loss = np.mean([f['loss'] for f in fold_results])

    log_entry = {
        "Trial_No": trial.number,
        "Model": model_type,
        "Train_Time_Sec": round(duration, 2),
        "Best_Fold": best_fold['fold'],
        "Best_F1": best_fold['f1'],
        "Best_Loss": best_fold['loss'],
        "Avg_F1": avg_f1,
        "Avg_Loss": avg_loss,
        "Params": json.dumps(params)
    }
    
    # 寫入 CSV
    df = pd.DataFrame([log_entry])
    df.to_csv(LOG_CSV, mode='a', index=False, header=not os.path.exists(LOG_CSV))
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return avg_loss, avg_f1

# --- 6. 啟動實驗 ---
if __name__ == "__main__":
    # 使用參數傳遞數值，解決函式內部的黃線問題
    print("正在載入資料...")
    all_segments, all_labels = load_data(
        label_dirs=LABEL_DIRS, 
        segment_size=500, 
        safety_gap=50, 
        min_len=30 # 這裡設為尋優範圍的最大值，保證資料長度足夠
    )
    
    error_indices = np.where(all_labels >= 4)[0]
    if len(error_indices) > 0:
        print(f"!!! 警告：發現非法標籤 !!!")
        print(f"在索引 {error_indices} 處發現了標籤值: {all_labels[error_indices]}")
    else:
        print("所有標籤檢查正常 (0-3)")
        
    study = optuna.create_study(directions=["minimize", "maximize"])
    study.optimize(lambda trial: objective(trial, all_segments, all_labels), n_trials=40)
    print(f"優化完成，結果儲存至: {LOG_CSV}")