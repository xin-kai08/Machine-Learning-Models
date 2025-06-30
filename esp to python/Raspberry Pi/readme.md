# 📌 Raspberry Pi LSTM Real-Time Inference System

# 手動執行 終端機關了就停了
cd ~/machine_learning
source venv/bin/activate
python3 realTimeServer_pi.py


# 開機自動執行
sudo nano /etc/systemd/system/realtime.service

# DEBUG=True 終端看的到輸出 DEBUG=False 終端看不到輸出
[Unit]
Description=My LSTM RealTime Server
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/machine_learning
Environment="DEBUG=True"
ExecStart=/home/pi/machine_learning/venv/bin/python -u /home/pi/machine_learning/realTimeServer_pi.py
Restart=always

[Install]
WantedBy=multi-user.target

# 啟用
sudo systemctl daemon-reload
sudo systemctl enable realtime
sudo systemctl start realtime

# 查看執行狀態
sudo systemctl status realtime

# 監控 realtime 的即時狀態
journalctl -u realtime -f

# 暫停執行
sudo systemctl stop realtime

# 取消開機啟動
sudo systemctl disable realtime

# 重新啟動服務（改 DEBUG 記得要先 daemon-reload）
sudo systemctl daemon-reload
sudo systemctl restart realtime