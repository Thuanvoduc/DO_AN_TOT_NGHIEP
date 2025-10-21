#include <WiFi.h>
#include <Firebase_ESP_Client.h>

// Thư viện hỗ trợ (có sẵn trong Firebase ESP Client)
#include "addons/TokenHelper.h"
#include "addons/RTDBHelper.h"

// ================= Cấu hình WiFi =================
#define WIFI_SSID "Su"
#define WIFI_PASSWORD "sususu2003"

// ================= Cấu hình Firebase =================
#define DATABASE_URL "https://do-an-tot-nghiep-9ac13-default-rtdb.firebaseio.com/"  
#define DATABASE_SECRET "s6z18jPvvTSFJnlUg7lZtRWvePBwAC5ZduKAW4w9"   // 🔑 lấy trong Firebase console

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

void setup() {
  Serial.begin(115200);

  // Kết nối WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("🔌 Đang kết nối WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(300);
  }
  Serial.println("\n✅ Đã kết nối WiFi");

  // Thiết lập Firebase
  config.database_url = DATABASE_URL;
  config.signer.tokens.legacy_token = DATABASE_SECRET;  // Dùng secret để auth

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  Serial.println("🔍 Bắt đầu ghi dữ liệu...");

  // Test ghi dữ liệu
  if (Firebase.RTDB.setString(&fbdo, "/test/message", "Hello ESP32")) {
    Serial.println("✅ Ghi dữ liệu thành công!");
  } else {
    Serial.println("❌ Lỗi: " + fbdo.errorReason());
  }
}

void loop() {
}
