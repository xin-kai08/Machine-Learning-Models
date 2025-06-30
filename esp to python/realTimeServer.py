import time
from threading import Lock, Thread

import joblib
import numpy as np
import pandas as pd
import serial
import torch
from flask import Flask, jsonify
from flask_cors import CORS
import ctypes
import winsound
from model import LSTMClassifier

# === 參數設定 ===
COM_PORT = "COM14"
BAUD_RATE = 9600
MAX_SEQ_LEN = 10
INPUT_DIM = 4
HIDDEN_DIM = 16
NUM_LAYERS = 4
NUM_CLASSES = 4
TEMP_THRESHOLD = 40

# === 載入模型與 scaler ===
model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES)
model.load_state_dict(
    torch.load(
        r"C:\Users\boss9\OneDrive\桌面\專題\機器學習\esp to python\0626_fold_5_model.pth",
        map_location=torch.device("cpu"),
    )
)
model.eval()
scaler = joblib.load("0626_scaler_fold5.pkl")

LABEL_NAMES = {
    0: "正常",
    1: "充電線生鏽",
    2: "變壓器生鏽",
    3: "變壓器過熱",
}
feature_names = ["current", "voltage", "power", "temp_C"]

# === 緩衝區與狀態 ===
sequence_buffer = []
raw_sequence_buffer = []
prev_predicted = 0
pause_reading = False

# === Flask 伺服器 ===
app = Flask(__name__)
CORS(app)  # ✅ 保證跨域，給 APP 用
result_lock = Lock()
latest_result = {}

@app.route("/get_result")
def get_result():
    with result_lock:
        return jsonify(latest_result if latest_result else {})

def update_result(predicted, label, input_sequence, feature_names):
    with result_lock:
        latest_result["predicted"] = int(predicted)
        latest_result["label"] = label
        latest_result["sequence"] = [
            {name: float(val) for name, val in zip(feature_names, row)}
            for row in input_sequence
        ]
        latest_result["timestamp"] = time.time()

def serial_inference_loop():
    ser = serial.Serial(COM_PORT, BAUD_RATE)
    print(f"✅ Serial 已連接：{COM_PORT} @ {BAUD_RATE} baud")

    global sequence_buffer, raw_sequence_buffer, prev_predicted, pause_reading

    while True:
        try:
            if pause_reading:
                time.sleep(0.1)
                continue

            line = ser.readline().decode("utf-8").strip()
            if not line:
                continue

            try:
                input_data = list(map(float, line.split(",")))
            except Exception as e:
                print(f"❌ 數據解析錯誤: {line} → {e}")
                continue

            if len(input_data) != len(feature_names):
                print(f"❌ 數據長度錯誤: {input_data}")
                continue

            input_df = pd.DataFrame([input_data], columns=feature_names)
            scaler_cols = getattr(scaler, "feature_names_in_", feature_names)
            input_df = input_df[scaler_cols]
            input_scaled = scaler.transform(input_df.values)[0]
            sequence_buffer.append(input_scaled)
            raw_sequence_buffer.append(input_data)

            print(f"📥 收到第 {len(sequence_buffer)} 筆: {input_data}")

            if len(sequence_buffer) == MAX_SEQ_LEN:
                input_tensor = torch.tensor([sequence_buffer], dtype=torch.float32)
                with torch.no_grad():
                    output = model(input_tensor)
                    predicted = torch.argmax(output, dim=1).item()
                    label = LABEL_NAMES.get(predicted, f"未知類別 {predicted}")

                    # === 雙向溫度後處理 ===
                    try:
                        temp_idx = feature_names.index("temp_C")
                        last_temp = raw_sequence_buffer[-1][temp_idx]
                        if label == "變壓器過熱" and last_temp < TEMP_THRESHOLD:
                            print(f"⚠️ 過熱校正 → 溫度 {last_temp:.1f}°C 未達門檻，改判正常")
                            predicted = 0
                            label = "過熱修正"
                        elif label == "正常" and last_temp >= TEMP_THRESHOLD:
                            print(f"⚠️ 正常校正 → 溫度 {last_temp:.1f}°C 超門檻，強制過熱")
                            predicted = 3
                            label = "過熱（溫度校正）"
                    except Exception as e:
                        print(f"⚠️ 後處理失敗：{e}")

                    result_sequence = raw_sequence_buffer.copy()

                    # === 終端輸出 ===
                    if predicted != 0 and predicted != prev_predicted:
                        print(f"⚠️⚠️⚠️ 新異常：{predicted} → {label}")
                    elif predicted != 0:
                        print(f"🔄 異常持續中：{predicted} → {label}")
                    else:
                        print("✅ 正常狀態")

                    # === 跳窗邏輯（僅新異常）
                    if predicted != 0 and predicted != prev_predicted:
                        pause_reading = True
                        ser.close()
                        print("🛑 Serial 已關閉，暫停接收")

                        sequence_buffer.clear()
                        raw_sequence_buffer.clear()
                        with result_lock:
                            latest_result.clear()

                        winsound.Beep(1000, 500)
                        ctypes.windll.user32.MessageBoxW(
                            0, f"偵測到異常：{label}", "⚠️ 系統警告", 0x30
                        )

                        ser.open()
                        ser.reset_input_buffer()
                        print("✅ Serial 已重新開啟")
                        pause_reading = False

                    prev_predicted = 0 if predicted == 0 else predicted

                    update_result(predicted, label, result_sequence, scaler_cols)

                sequence_buffer.clear()
                raw_sequence_buffer.clear()

            time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n🛑 使用者中斷")
            break
        except Exception as e:
            print(f"⚠️ 發生錯誤：{e}")

if __name__ == "__main__":
    t = Thread(target=serial_inference_loop)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=8080)