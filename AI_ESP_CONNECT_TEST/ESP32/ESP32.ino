#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "Su";       // 🔹 Nhập Wi-Fi của bạn
const char* password = "sususu2003";

WebServer server(80); // Tạo web server cổng 80

void handleRoot() {
  server.send(200, "text/plain", "ESP32 is online!");
}

void handleSend() {
  if (server.hasArg("msg")) {
    String msg = server.arg("msg");
    Serial.print("📩 Nhận được dữ liệu từ Python: ");
    Serial.println(msg);
    String response = "ESP32 đã nhận được: " + msg;
    server.send(200, "text/plain", response);
  } else {
    server.send(400, "text/plain", "Thiếu tham số msg!");
  }
}

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  Serial.println("\nĐang kết nối Wi-Fi...");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\n✅ Kết nối Wi-Fi thành công!");
  Serial.print("📡 ESP32 IP: ");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/send", handleSend);
  server.begin();
  Serial.println("🌐 Server đã khởi động!");
}

void loop() {
  server.handleClient();
}
