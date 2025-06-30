import numpy as np
import joblib
import onnxruntime as ort

# === 載入 scaler ===
scaler = joblib.load("0627_scaler_fold1.pkl")

# === 準備一筆 raw feature ===
raw_input = np.array([
    [0.988,	4.171,	34.2,	4.120948],
    [0.648,	4.171,	34.2,	2.702808],
    [0.956,	4.171,	34.2,	3.987476],
    [0.947,	4.171,	34.2,	3.949937],
    [0.821,	4.171,	34.2,	3.424391],
    [0.952,	4.171,	34.2,	3.970792],
    [0.939,	4.171,	34.2,	3.916569],
    [0.956,	4.171,	34.2,	3.987476],
    [0.944,	4.171,	34.2,	3.937424],
    [0.953,	4.171,	34.2,	3.974963]
])

# === 正規化 ===
scaled_input = scaler.transform(raw_input)

# === ONNX input shape: (batch_size, seq_len, input_dim) ===
onnx_input = scaled_input.reshape(1, 10, 4).astype(np.float32)

# === 使用 ONNX Runtime 推論 ===
ort_session = ort.InferenceSession("0627_lstm_model_fold1.onnx")
inputs = {ort_session.get_inputs()[0].name: onnx_input}
outputs = ort_session.run(None, inputs)

idx2label = {
    0: "正常",
    1: "充電線生鏽",
    2: "變壓器生鏽",
    3: "手機過熱",
    4: "電線剝皮",
    5: "電線彎折"
}

# === 取出輸出 logits ===
logits = outputs[0]  # shape: (batch_size, num_classes)

# === 計算每筆的預測類別 ===
pred_idx = np.argmax(logits, axis=1)[0]
pred_label = idx2label[pred_idx]

print(f"✅ 預測類別: {pred_idx} → {pred_label}")
print(f"📊 類別機率 (logits): {logits}")