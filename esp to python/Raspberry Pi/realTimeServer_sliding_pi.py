import time
import os
import csv
import json
from threading import Lock, Thread

import joblib
import numpy as np
import pandas as pd
import serial
import serial.tools.list_ports
import torch
import paho.mqtt.client as mqtt

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

# === MQTT 設定 ===
MQTT_BROKER = "41d94ebb0f8e4a538cec6b931a1c2c27.s1.eu.hivemq.cloud"  # 或你的 HiveMQ broker 地址
MQTT_PORT = 8883
MQTT_TOPIC_PREDICTION = "iot/charger/prediction"  # 預測結果主題
MQTT_TOPIC_SENSOR_DATA = "iot/charger/sensor"    # 感測器數據主題
MQTT_TOPIC_STATUS = "iot/charger/status"         # 狀態主題
MQTT_CLIENT_ID = "raspberry_pi_charger"
MQTT_USERNAME="NUTNee"
MQTT_PASSWORD="NUTNee1234"

# === 載入模型與 Scaler ===
model = torch.jit.load("20250819_fold_4_model_scripted.pt")
model.eval()
if DEBUG:
    print("Model loaded.")

scaler = joblib.load("0626_scaler_fold5.pkl")
if DEBUG:
    print("Scaler loaded.")

# === MQTT 客戶端設定 ===
mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)
mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
mqtt_client.tls_set()  # 啟用 TLS/SSL
mqtt_connected = False
result_lock = Lock()
latest_result = {}

# === USB 與推論狀態變數 ===
usb_connected = False       # 是否連接 USB 裝置
inference_active = False    # 是否正在推論中

# === MQTT 回調函數 ===
def on_mqtt_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        mqtt_connected = True
        if DEBUG:
            print("Connected to MQTT broker")
        # 發送連接狀態
        publish_status("MQTT_CONNECTED")
    else:
        mqtt_connected = False
        if DEBUG:
            print(f"Failed to connect to MQTT broker, return code {rc}")

def on_mqtt_disconnect(client, userdata, rc):
    global mqtt_connected
    mqtt_connected = False
    if DEBUG:
        print("Disconnected from MQTT broker")

def on_mqtt_publish(client, userdata, mid):
    if DEBUG:
        print(f"Message {mid} published successfully")

# === 發布狀態到 MQTT ===
def publish_status(status):
    if mqtt_connected:
        status_data = {
            "timestamp": time.time(),
            "status": status,
            "usb_connected": usb_connected,
            "inference_active": inference_active
        }
        try:
            mqtt_client.publish(MQTT_TOPIC_STATUS, json.dumps(status_data))
            if DEBUG:
                print(f"Status published: {status}")
        except Exception as e:
            if DEBUG:
                print(f"Failed to publish status: {e}")

# === 發布預測結果到 MQTT ===
def publish_prediction(predicted, label):
    if mqtt_connected:
        timestamp = time.time()
        prediction_data = {
            "timestamp": timestamp,
            "predicted": int(predicted),
            "label": label
        }
        try:
            mqtt_client.publish(MQTT_TOPIC_PREDICTION, json.dumps(prediction_data))
            if DEBUG:
                print(f"Prediction published: {predicted} -> {label}")
        except Exception as e:
            if DEBUG:
                print(f"Failed to publish prediction: {e}")

# === 發布感測器數據到 MQTT ===
def publish_sensor_data(sensor_values, feature_names):
    if mqtt_connected:
        timestamp = time.time()
        sensor_data = {
            "timestamp": timestamp,
            "current": float(sensor_values[0]),
            "voltage": float(sensor_values[1]), 
            "power": float(sensor_values[2]),
            "temp_C": float(sensor_values[3])
        }
        try:
            mqtt_client.publish(MQTT_TOPIC_SENSOR_DATA, json.dumps(sensor_data))
            if DEBUG:
                print(f"Sensor data published: {sensor_data}")
        except Exception as e:
            if DEBUG:
                print(f"Failed to publish sensor data: {e}")

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
    
    # 發布狀態變化到 MQTT
    publish_status(event_type)

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
        
    # 寫入 CSV
    write_to_csv(timestamp, predicted, label, input_sequence)
    
    # 發布預測結果到專用主題
    publish_prediction(predicted, label)

# === 自動搜尋 USB 埠口 ===
def find_serial_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "USB" in p.device or "ACM" in p.device:
            if DEBUG:
                print(f"Found port: {p.device} {p.description}")
            return p.device
    return None

# === MQTT 連接管理 ===
def mqtt_connection_loop():
    global mqtt_connected
    
    # 設置回調函數
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_disconnect = on_mqtt_disconnect
    mqtt_client.on_publish = on_mqtt_publish
    
    while True:
        if not mqtt_connected:
            try:
                if DEBUG:
                    print("Attempting to connect to MQTT broker...")
                mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
                mqtt_client.loop_start()
                time.sleep(2)  # 等待連接建立
            except Exception as e:
                if DEBUG:
                    print(f"MQTT connection failed: {e}")
                time.sleep(5)  # 等待5秒後重試
        else:
            time.sleep(10)  # MQTT 已連接，每10秒檢查一次

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
                        
                        # 發布感測器數據到專用主題
                        publish_sensor_data(input_data, feature_names)

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
        publish_status("SYSTEM_SHUTDOWN")
        mqtt_client.disconnect()

# === 執行 ===
if __name__ == "__main__":
    # 啟動 MQTT 連接管理線程
    mqtt_thread = Thread(target=mqtt_connection_loop)
    mqtt_thread.daemon = True
    mqtt_thread.start()
    
    # 啟動串口推論線程
    serial_thread = Thread(target=serial_inference_loop)
    serial_thread.daemon = True
    serial_thread.start()
    
    try:
        # 主線程保持運行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        publish_status("SYSTEM_SHUTDOWN")
        mqtt_client.disconnect()
        print("Shutdown complete")