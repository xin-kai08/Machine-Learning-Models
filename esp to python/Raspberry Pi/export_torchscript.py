# export_torchscript.py
import torch
from model import LSTMClassifier  # 這會用到你的 model.py

# === 參數（跟訓練時要一致）===
INPUT_DIM = 4
HIDDEN_DIM = 16
NUM_LAYERS = 4
NUM_CLASSES = 4

# === 載入原始 PyTorch 模型 ===
model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES)
model.load_state_dict(torch.load("0626_fold_5_model.pth", map_location="cpu"))
model.eval()

print("✅ 已載入 .pth 權重")

# === 建立範例輸入（假設序列長度是 10）===
example_input = torch.rand(1, 10, INPUT_DIM)  # (batch_size, seq_len, input_dim)

# === 將模型轉成 TorchScript ===
traced_script_module = torch.jit.trace(model, example_input)

# === 儲存為 .pt ===
traced_script_module.save("0626_fold_5_model_scripted.pt")

print("🎉 已成功匯出 TorchScript 檔案：0626_fold_5_model_scripted.pt")
