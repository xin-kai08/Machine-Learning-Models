import serial
import time
import csv
import os
from datetime import datetime

# === 設定 ===
COM_PORT = 'COM16'          # 替換成你 ESP32 的實際序列埠
BAUD_RATE = 9600
SAVE_PATH = "data_log.csv"  # 儲存資料的 CSV 檔案

# === 初始化序列埠 ===
ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
print(f"✅ 已連線 ESP32 @ {COM_PORT}")
print(f"📁 資料將儲存在：{os.path.abspath(SAVE_PATH)}")

# === 等待使用者啟動 ===
input("👉 請按 Enter 開始錄製（Ctrl+C 可手動結束）...")

# === 開始紀錄 ===
with open(SAVE_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "current", "voltage", "power", "temp_C"])  # CSV 表頭

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            try:
                data = list(map(float, line.split(',')))
                if len(data) == 4:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    writer.writerow([timestamp] + data)
                    print(f"📥 收到: {[timestamp] + data}")
            except Exception as e:
                print(f"❌ 解析錯誤: {line} → {e}")
    except KeyboardInterrupt:
        print("\n🛑 使用者手動中止錄製。")

ser.close()
