import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# =====================
# 路徑設定（請依你的實際路徑調整）
# =====================
SPECIAL_DIR = r"C:/Users/boss9/OneDrive/桌面/專題/機器學習/dataset/feature dim_4/hardware/特殊樣本"

# 這兩個是你剛剛 TL+Normalization 存出來的檔案
KERAS_MODEL_PATH = r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\result\20251130_TLfinetune_model.keras"
TFLITE_MODEL_PATH = r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\result\20251130_TLfinetune_model.tflite"

MAX_SEQ_LEN = 15
STRIDE = 5
NUM_CLASSES = 4
BATCH_SIZE = 64

# =====================
# 讀取「特殊樣本」並做 sliding window
# =====================
def load_special_sequences():
    all_chunks = []
    pattern = os.path.join(SPECIAL_DIR, "*.csv")
    for path in glob.glob(pattern):
        df = pd.read_csv(path)

        # 欄位順序：current, voltage, power, temp_C
        current = df["current"].values
        voltage = df["voltage"].values
        power   = df["power"].values
        temp_C  = df["temp_C"].values

        seq = np.column_stack((current, voltage, power, temp_C)).astype(np.float32)
        n = seq.shape[0]
        for start in range(0, n - MAX_SEQ_LEN + 1, STRIDE):
            end = start + MAX_SEQ_LEN
            all_chunks.append(seq[start:end])

    if not all_chunks:
        raise RuntimeError(f"No csv files found in {SPECIAL_DIR}")

    X = np.stack(all_chunks, axis=0)  # (N, 15, 4)
    print("Special samples shape:", X.shape)
    return X

# =====================
# 用 Keras 模型預測（含 Normalization）
# =====================
def predict_with_keras(model, X):
    logits = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    preds = np.argmax(logits, axis=1)
    return preds

# =====================
# 用 TFLite 模型預測
# =====================
def predict_with_tflite(tflite_path, X):
    with open(tflite_path, "rb") as f:
        tflite_model = f.read()

    # 用 model_content，而不是 model_path
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    preds = []
    for i in range(X.shape[0]):
        x = X[i:i+1].astype(np.float32)  # shape (1, 15, 4)
        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()
        logits = interpreter.get_tensor(output_details["index"])[0]
        preds.append(int(np.argmax(logits)))

    return np.array(preds, dtype=int)

# =====================
# 主程式
# =====================
def main():
    # 1. 載入資料（raw，沒標準化；Normalization 在模型裡）
    X = load_special_sequences()

    # 2. Keras 模型預測
    print("\n=== Keras finetune+Norm 模型預測 ===")
    keras_model = keras.models.load_model(KERAS_MODEL_PATH)
    keras_preds = predict_with_keras(keras_model, X)
    for c in range(NUM_CLASSES):
        print(f"  class {c}: {np.sum(keras_preds == c)}")

    # 3. TFLite 模型預測
    print("\n=== TFLite 模型預測（同一顆模型轉出的 .tflite） ===")
    print("TFLite path =", TFLITE_MODEL_PATH)
    print("os.path.exists ->", os.path.exists(TFLITE_MODEL_PATH))
    tflite_preds = predict_with_tflite(TFLITE_MODEL_PATH, X)
    for c in range(NUM_CLASSES):
        print(f"  class {c}: {np.sum(tflite_preds == c)}")

    # 4. 順便看一下兩者是否一致
    same = np.sum(keras_preds == tflite_preds)
    print(f"\nKeras vs TFLite 預測完全相同的樣本數：{same} / {len(X)}")

if __name__ == "__main__":
    main()
