import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# ============================
# 路徑設定（依你的環境調整）
# ============================
BASE_ROOT = r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\dataset\feature dim_4\hardware"
NORMAL_DIR = os.path.join(BASE_ROOT, "normal")
ABNORMAL_DIR = os.path.join(BASE_ROOT, "abnormal")
SPECIAL_DIR = os.path.join(BASE_ROOT, "特殊樣本")

WIRE_RUST_DIR = os.path.join(ABNORMAL_DIR, "wire_rust")
TRANS_RUST_DIR = os.path.join(ABNORMAL_DIR, "transformer_rust")
TRANS_OVER_DIR = os.path.join(ABNORMAL_DIR, "transformer_overheating")

# 這是你用 Keras 版 LSTM 訓練出來的 base model
BASE_MODEL_PATH = r"C:\Users\boss9\Downloads\fold_1_model.keras"

# 遷移學習後模型輸出位置
OUTPUT_MODEL_PATH = r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\result\20251130_TLfinetune_model.keras"
os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)

# ============================
# 超參數
# ============================
MAX_SEQ_LEN = 15
STRIDE = 5
INPUT_DIM = 4
NUM_CLASSES = 4

LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 16
BASE_KEEP_RATIO = 0.5   # 舊資料只保留 50%，跟 PyTorch TL 一樣的概念:contentReference[oaicite:2]{index=2}


# ============================
# Sliding Window 讀資料
# ============================
def read_folder(folder, label):
    """
    從指定資料夾讀取所有 CSV，做 sliding window：
    每個 chunk 形狀 = (MAX_SEQ_LEN, 4)
    回傳 X: (N, T, 4), y: (N,)
    """
    all_chunks = []
    all_labels = []

    for fname in os.listdir(folder):
        if not fname.lower().endswith(".csv"):
            continue

        path = os.path.join(folder, fname)
        df = pd.read_csv(path)

        # 欄位順序與訓練 LSTM 時保持一致：current, voltage, power, temp_C :contentReference[oaicite:3]{index=3}
        current = df["current"].values
        voltage = df["voltage"].values
        power = df["power"].values
        temp_C = df["temp_C"].values

        seq = np.column_stack((current, voltage, power, temp_C)).astype(np.float32)
        L = seq.shape[0]

        for start in range(0, L - MAX_SEQ_LEN + 1, STRIDE):
            end = start + MAX_SEQ_LEN
            chunk = seq[start:end]  # (T, 4)
            all_chunks.append(chunk)
            all_labels.append(label)

    if not all_chunks:
        return np.empty((0, MAX_SEQ_LEN, INPUT_DIM), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.stack(all_chunks, axis=0)
    y = np.array(all_labels, dtype=np.int64)
    return X, y


def build_finetune_dataset():
    """
    建立遷移學習用資料：
    - base 資料：normal + 3 abnormal，隨機保留 50%
    - special 資料：全部特殊樣本 (label=1)
    """
    X0, y0 = read_folder(NORMAL_DIR, 0)
    X1, y1 = read_folder(WIRE_RUST_DIR, 1)
    X2, y2 = read_folder(TRANS_RUST_DIR, 2)
    X3, y3 = read_folder(TRANS_OVER_DIR, 3)

    Xs, ys = read_folder(SPECIAL_DIR, 1)  # SPECIAL_LABEL = 1，跟 PyTorch 版一致:contentReference[oaicite:4]{index=4}

    # 合併 base 四類
    X_base_all = np.concatenate([X0, X1, X2, X3], axis=0)
    y_base_all = np.concatenate([y0, y1, y2, y3], axis=0)

    # 隨機打散，保留前 BASE_KEEP_RATIO 部分
    idx = np.random.permutation(len(X_base_all))
    X_base_all = X_base_all[idx]
    y_base_all = y_base_all[idx]

    keep_base = int(len(X_base_all) * BASE_KEEP_RATIO)
    X_base = X_base_all[:keep_base]
    y_base = y_base_all[:keep_base]

    # 舊資料 + 特殊樣本
    X_final = np.concatenate([X_base, Xs], axis=0)
    y_final = np.concatenate([y_base, ys], axis=0)

    print("Base all shape:", X_base_all.shape)
    print("Special shape:", Xs.shape)
    print("Final finetune data:", X_final.shape, y_final.shape)
    print("Label distribution:", np.bincount(y_final))
    return X_final, y_final


# ============================
# 建立「含 Normalization」的遷移學習模型
# ============================
def build_finetune_model_with_norm(base_model: keras.Model, X_raw: np.ndarray):
    """
    base_model：你原本訓練好的 LSTM Keras 模型（不含 Normalization）
    X_raw：raw data，shape = (N, T, F)，用來讓 Normalization.adapt()
    """
    # 1. 不要凍 LSTM，全部一起 fine-tune（LR 調小）
    for layer in base_model.layers:
        layer.trainable = True

    # 2. Normalization 層
    norm_layer = layers.Normalization(axis=-1)
    norm_layer.adapt(X_raw)

    # 3. 接到 base_model 前面
    raw_inputs = keras.Input(shape=(MAX_SEQ_LEN, INPUT_DIM), name="raw_input")
    x = norm_layer(raw_inputs)
    outputs = base_model(x)

    finetune_model = keras.Model(inputs=raw_inputs, outputs=outputs, name="lstm_with_norm")

    # 4. compile，用比較小的 learning rate
    finetune_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    finetune_model.summary()
    return finetune_model

# ============================
# 主流程：載入 base model → 建新 model（含 Norm）→ fine-tune → 匯出 .keras & .tflite
# ============================
def main():
    print("=== 1. 載入 base Keras LSTM 模型 ===")
    base_model = keras.models.load_model(BASE_MODEL_PATH)
    base_model.summary()

    print("\n=== 2. 準備遷移學習資料（raw, 未標準化） ===")
    X_raw, y = build_finetune_dataset()   # shape = (N, T, 4)

    print("\n=== 3. 建立含 Normalization 的遷移學習模型 ===")
    finetune_model = build_finetune_model_with_norm(base_model, X_raw)

    print("\n=== 4. 開始 fine-tune（吃 raw data，模型內部自己做標準化） ===")
    early_stop = EarlyStopping(
        monitor="val_loss",        # 看驗證集 loss
        patience=10,               # 連續 10 個 epoch 沒進步就停
        restore_best_weights=True  # 回到 val_loss 最好的那一版權重
    )
    finetune_model.fit(
        X_raw,
        y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        validation_split=0.2,      # 20% 當驗證集
        callbacks=[early_stop],
    )

    print("\n=== 5. 儲存遷移學習後模型 (.keras) ===")
    finetune_model.save(OUTPUT_MODEL_PATH)
    print("Saved finetuned model with normalization to:", OUTPUT_MODEL_PATH)

    print("\n=== 6. 匯出 TFLite（含 Normalization） ===")
    converter = tf.lite.TFLiteConverter.from_keras_model(finetune_model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,   # 一般 TFLite op
        tf.lite.OpsSet.SELECT_TF_OPS      # 少量 TF op（含這種 LSTM/TensorList 相關）
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()
    tflite_path = OUTPUT_MODEL_PATH.replace(".keras", ".tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("Saved TFLite model to:", tflite_path)

if __name__ == "__main__":
    main()