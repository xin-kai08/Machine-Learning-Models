#include <Wire.h>
#include <Adafruit_INA219.h>
#include <Adafruit_MLX90614.h>

Adafruit_INA219 ina219;
Adafruit_MLX90614 mlx = Adafruit_MLX90614();

void setup() {
  // === 初始化 Serial 與 INA219 ===
  Serial.begin(9600);
  Wire.begin();
  ina219.begin();
  mlx.begin();
}

void loop() {
  // === 讀 INA219 最新電壓/電流/功率 ===
  float busVoltage = ina219.getBusVoltage_V();  // 系統電壓 (V)
  float current_mA = ina219.getCurrent_mA();    // 電流 (mA)
  float power_mW   = ina219.getPower_mW();      // 功率 (mW)
  float temp = mlx.readObjectTempC();           // 物體溫度 (°C)

  // === 串口傳送: 電流(A), 電壓(V), 功率(W), 溫度(C) ===
  Serial.print(current_mA / 1000.0, 6);  // 換成 A，保留 6 位小數
  Serial.print(",");
  Serial.print(busVoltage, 6);           // 保留 6 位小數
  Serial.print(",");
  Serial.print(power_mW / 1000.0, 6);    // 換成 W，保留 6 位小數
  Serial.print(",");
  Serial.println(temp, 2);               // 溫度保留 2 位小數

  // === 每秒送一次，持續輪詢 ===
  delay(1000);
}
