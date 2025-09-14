import time
import os
import csv
from threading import Lock, Thread

import joblib
import numpy as np
import pandas as pd
import serial
import serial.tools.list_ports
import torch
from flask import Flask, jsonify
from flask_cors import CORS

# === DEBUG 模式開關，從環境變數讀取 ===
DEBUG = os.environ.get("DEBUG", "False") == "True"

# === 參數設定 ===
BAUD_RATE = 9600
MAX_SEQ_LEN = 15  # 序列長度
INPUT_DIM = 4
STRIDE = 5
TEMP_THRESHOLD = 35
RESULTS_PATH = "/home/pi/machine_learning/results.csv"

LABEL_NAMES = {
    0: "正常",
    1: "充電線生鏽",
    2: "變壓器生鏽",
    3: "變壓器過熱",
}
feature_names = ["current", "voltage", "power", "temp_C"]

# === 載入模型與 Scaler ===
model = torch.jit.load("20250819_fold_4_model_scripted.pt")
model.eval()
if DEBUG:
    print("Model loaded.")

scaler = joblib.load("0626_scaler_fold5.pkl")
if DEBUG:
    print("Scaler loaded.")

# === Flask 設定 ===
app = Flask(__name__)
CORS(app)
result_lock = Lock()
latest_result = {}

# === USB 與推論狀態變數 ===
usb_connected = False       # 是否連接 USB 裝置
inference_active = False    # 是否正在推論中

# === API: 讓 APP 端讀取最新推論結果與狀態 ===
@app.route("/get_result")
def get_result():
    with result_lock:
        # Case 1: USB未連接
        if not usb_connected and not inference_active:
            return jsonify({
                "predicted": 0,
                "label": "USB未連接",
                "timestamp": time.time(),
                "sequence": [{"current": 0, "voltage": 0, "power": 0, "temp_C": 0}]
            })
        # Case 2: USB 有接 & 最近功率為0 => 未充電
        elif usb_connected and latest_power_zero:
            return jsonify({
                "predicted": 0,
                "label": "未充電",
                "timestamp": time.time(),
                "sequence": [{"current": 0, "voltage": 0, "power": 0, "temp_C": 0}]
            })
        # Case 3: 有最新推論結果
        else:
            return jsonify({
                "predicted": latest_result.get("predicted", 0),
                "label": latest_result.get("label", "未知"),
                "timestamp": latest_result.get("timestamp", time.time()),
                "sequence": latest_result.get("sequence", [])
            })

# === 寫入結果到 CSV ===
def write_to_csv(timestamp, predicted, label, input_sequence):
    file_exists = os.path.isfile(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "event", "predicted", "label"] + feature_names)
        for row in input_sequence:
            writer.writerow([timestamp, "", predicted, label] + list(row))

# === 記錄裝置連接或斷線事件 ===
def log_device_event(event_type):
    timestamp = time.time()
    file_exists = os.path.isfile(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "event", "predicted", "label"] + feature_names)
        writer.writerow([timestamp, event_type, "", "", "", "", "", ""])

# === 更新結果給 API 用 ===
def update_result(predicted, label, input_sequence, feature_names):
    timestamp = time.time()
    with result_lock:
        latest_result["predicted"] = int(predicted)
        latest_result["label"] = label
        latest_result["timestamp"] = timestamp
        latest_result["sequence"] = [
            {name: float(val) for name, val in zip(feature_names, row)}
            for row in input_sequence
        ]
    write_to_csv(timestamp, predicted, label, input_sequence)

# === 自動搜尋 USB 埠口 ===
def find_serial_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "USB" in p.device or "ACM" in p.device:
            if DEBUG:
                print(f"Found port: {p.device} {p.description}")
            return p.device
    return None

# === 串口推論主循環 + 自動搜尋埠號 + 斷線安全重連 ===
def serial_inference_loop():
    global usb_connected, inference_active, latest_power_zero
    latest_power_zero = False
    prev_predicted = 0
    POWER_THRESHOLD = 0.1  # 功率門檻
    ZERO_COUNT_LIMIT = 30  # 累積多少次為 0 才 idle
    zero_count = 0

    try:
        while True:
            port = find_serial_port()
            if not port:
                usb_connected = False
                inference_active = False
                if DEBUG:
                    print("No device connected, retry in 5 sec...")
                time.sleep(5)
                continue

            ser = None  # 先宣告
            try:
                ser = serial.Serial(port, BAUD_RATE)
                usb_connected = True
                # === 插拔後冷啟動緩衝 ===
                time.sleep(1)
                if DEBUG:
                    print("Serial connected, stabilizing for 0.5 sec...")
                    
                if DEBUG:
                    print(f"Serial connected: {port} @ {BAUD_RATE}")
                log_device_event("USB_CONNECTED")

                sequence_buffer = []
                raw_sequence_buffer = []

                while True:
                    try:
                        line = ser.readline().decode("utf-8").strip()
                        if not line:
                            continue

                        parts = line.split(",")
                        if len(parts) != len(feature_names):
                            if DEBUG:
                                print(f"Data format error: {line}")
                            continue

                        try:
                            input_data = list(map(float, parts))
                        except ValueError as e:
                            if DEBUG:
                                print(f"ValueError: {line} | {e}")
                            continue

                        power = input_data[2]

                        if power > POWER_THRESHOLD:
                            inference_active = True  # 有功率，啟動推論
                            latest_power_zero = False  # 有功率 => 重置
                            zero_count = 0

                            df = pd.DataFrame([input_data], columns=feature_names)
                            scaler_cols = getattr(scaler, "feature_names_in_", feature_names)
                            df = df[scaler_cols]
                            input_scaled = scaler.transform(df.values)[0]

                            sequence_buffer.append(input_scaled)
                            raw_sequence_buffer.append(input_data)

                            if DEBUG:
                                print(f"Data received {len(sequence_buffer)}: {input_data}")

                            if len(sequence_buffer) >= MAX_SEQ_LEN:
                                input_tensor = torch.tensor([np.array(sequence_buffer[-MAX_SEQ_LEN:])], dtype=torch.float32)
                                with torch.no_grad():
                                    output = model(input_tensor)
                                    predicted = torch.argmax(output, dim=1).item()
                                    label = LABEL_NAMES.get(predicted, f"Unknown {predicted}")

                                    temp_idx = feature_names.index("temp_C")
                                    last_temp = raw_sequence_buffer[-1][temp_idx]
                                    if DEBUG:
                                        print(f"Temp check: {last_temp:.1f}°C")

                                    # 過熱校正
                                    if label == "變壓器過熱" and last_temp < TEMP_THRESHOLD:
                                        if DEBUG:
                                            print("Overheat corrected to normal")
                                        predicted = 0
                                        label = "過熱修正"
                                    elif label == "正常" and last_temp >= TEMP_THRESHOLD:
                                        if DEBUG:
                                            print("Normal corrected to overheat")
                                        predicted = 3
                                        label = "過熱（溫度校正）"

                                    if predicted != 0 and predicted != prev_predicted:
                                        if DEBUG:
                                            print(f"New anomaly: {predicted} -> {label}")
                                    elif predicted != 0:
                                        if DEBUG:
                                            print(f"Anomaly continuing: {predicted} -> {label}")
                                    else:
                                        if DEBUG:
                                            print("Status: Normal")

                                    prev_predicted = 0 if predicted == 0 else predicted

                                    update_result(predicted, label, raw_sequence_buffer.copy(), scaler_cols)
                                    if DEBUG:
                                        print("Result updated and saved")

                                sequence_buffer = sequence_buffer[STRIDE:]
                                raw_sequence_buffer = raw_sequence_buffer[STRIDE:]

                        else:
                            # 功率只要是 0，序列立刻清空
                            sequence_buffer.clear()
                            raw_sequence_buffer.clear()

                            latest_power_zero = True
                            
                            if not inference_active:
                                # 已經是 idle 狀態，就不再累積
                                zero_count = ZERO_COUNT_LIMIT
                                if DEBUG:
                                    print("功率為零，暫停推論")
                            else:
                                zero_count += 1
                                latest_power_zero = True
                                if DEBUG:
                                    print(f"功率為零次數: {zero_count}/{ZERO_COUNT_LIMIT}")

                                if zero_count >= ZERO_COUNT_LIMIT:
                                    inference_active = False  # Idle 狀態
                                    zero_count = ZERO_COUNT_LIMIT  # 固定值，不再累積
                                    sequence_buffer.clear()
                                    raw_sequence_buffer.clear()
                                    if DEBUG:
                                        print("功率長時間為零，暫停推論（串口保持連線）")
                                    time.sleep(1)
                                    continue


                        time.sleep(0.5)

                    except serial.SerialException:
                        if ser:
                            ser.close()
                        usb_connected = False
                        inference_active = False
                        log_device_event("USB_DISCONNECTED")
                        if DEBUG:
                            print("Serial disconnected, retrying...")
                        break

            except serial.SerialException:
                if ser:
                    ser.close()
                usb_connected = False
                inference_active = False
                log_device_event("USB_DISCONNECTED")
                if DEBUG:
                    print("Serial exception, retrying...")
                time.sleep(1)

    except KeyboardInterrupt:
        if DEBUG:
            print("Stopped by user or shutdown")

# === 執行 ===
if __name__ == "__main__":
    t = Thread(target=serial_inference_loop)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=8080)