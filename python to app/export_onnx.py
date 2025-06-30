import torch
import torch.nn as nn

# === 模型結構：LSTMClassifier（從原始 lstm_f4.py 中擷取）===
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes=6, dropout_rate=0.3):
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
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# === 模型設定（與訓練時保持一致） ===
INPUT_DIM = 4         # 每筆資料有 4 個特徵
HIDDEN_DIM = 16
NUM_LAYERS = 4
NUM_CLASSES = 6
MAX_SEQ_LEN = 10

# === 載入模型並讀取訓練好的權重 ===
model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES)
model.load_state_dict(torch.load("0627_fold_1_model.pth", map_location=torch.device('cpu')))
model.eval()

# === 建立假資料用來匯出（batch_size=1, seq_len=10, input_dim=4）===
dummy_input = torch.randn(1, MAX_SEQ_LEN, INPUT_DIM)

# === 匯出成 ONNX 檔案 ===
torch.onnx.export(
    model,
    dummy_input,
    "0627_lstm_model_fold1.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size', 1: 'seq_len'},
        'output': {0: 'batch_size'}
    }
)

print("✅ 模型已成功匯出為 lstm_model.onnx")
