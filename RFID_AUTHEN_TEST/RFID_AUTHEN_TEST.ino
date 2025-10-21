#include <SPI.h>
#include <MFRC522.h>
#include <WiFi.h>
#include <Firebase_ESP_Client.h>
#include "addons/TokenHelper.h"
#include "addons/RTDBHelper.h"
#include "time.h"
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#define SS_PIN 5
#define RST_PIN 2
#define BUZZER_PIN 25                // buzzer pin
MFRC522 rfid(SS_PIN, RST_PIN);       // setup RFID
LiquidCrystal_I2C lcd(0x27, 16, 2);  // setup LCD

// WiFi
#define WIFI_SSID "NHAM COFFEE"
#define WIFI_PASSWORD "nhamquan10"

// Firebase
#define API_KEY "AIzaSyDML_o7tVQOf7wrzdA3NasklY5Wb3cPCjo"
#define DATABASE_URL "https://do-an-tot-nghiep-9ac13-default-rtdb.firebaseio.com/"

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

// NTP config
const char* ntpServer = "time.google.com";
const long gmtOffset_sec = 7 * 3600;  // GMT+7 cho Việt Nam
const int daylightOffset_sec = 0;

// Biến toàn cục của NTP
unsigned long lastNtpSync = 0;
const unsigned long ntpInterval = 30 * 60 * 1000;  // 30 phút (30 * 60 * 1000 ms)

// Biến toàn cục của LCD
unsigned long lastDisplayTime = 0;
bool showingMessage = false;

// hàm của buzzer
void beep(int duration) {
  digitalWrite(BUZZER_PIN, LOW);
  delay(duration);
  digitalWrite(BUZZER_PIN, HIGH);
}

// hàm format thời gians
String getTime() {
  time_t now;
  struct tm timeinfo;
  int retry = 0;

  time(&now);
  localtime_r(&now, &timeinfo);

  while (now < 100000 && retry < 5) {  // nếu chưa có thời gian hợp lệ
    Serial.println("⏳ Chưa lấy được NTP, thử lại...");
    configTime(7 * 3600, 0, "time.google.com");
    delay(2000);  // chờ NTP cập nhật
    time(&now);
    retry++;
  }

  if (now < 100000) {
    return "NTP Error";
  }

  char buf[20];
  strftime(buf, sizeof(buf), "%d-%m-%Y %H:%M:%S", &timeinfo);
  return String(buf);
}

// hàm hiển thị giao diện chờ của LCD
void showDefaultScreen() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Xin moi quet the");
  lcd.setCursor(0, 1);
  lcd.print("----------------");
  showingMessage = false;
}

// hàm hiển thị thông tin thẻ trên LCD
void showMessage(String line1, String line2) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(line1);
  lcd.setCursor(0, 1);
  lcd.print(line2);
  lastDisplayTime = millis();
  showingMessage = true;
}

// hàm chuyển chuỗi từ có dấu sang không dấu
String removeVietnameseTones(String str) {
  str.replace("á", "a"); str.replace("à", "a"); str.replace("ả", "a"); str.replace("ã", "a"); str.replace("ạ", "a");
  str.replace("ă", "a"); str.replace("ắ", "a"); str.replace("ằ", "a"); str.replace("ẳ", "a"); str.replace("ẵ", "a"); str.replace("ặ", "a");
  str.replace("â", "a"); str.replace("ấ", "a"); str.replace("ầ", "a"); str.replace("ẩ", "a"); str.replace("ẫ", "a"); str.replace("ậ", "a");

  str.replace("đ", "d");

  str.replace("é", "e"); str.replace("è", "e"); str.replace("ẻ", "e"); str.replace("ẽ", "e"); str.replace("ẹ", "e");
  str.replace("ê", "e"); str.replace("ế", "e"); str.replace("ề", "e"); str.replace("ể", "e"); str.replace("ễ", "e"); str.replace("ệ", "e");

  str.replace("í", "i"); str.replace("ì", "i"); str.replace("ỉ", "i"); str.replace("ĩ", "i"); str.replace("ị", "i");

  str.replace("ó", "o"); str.replace("ò", "o"); str.replace("ỏ", "o"); str.replace("õ", "o"); str.replace("ọ", "o");
  str.replace("ô", "o"); str.replace("ố", "o"); str.replace("ồ", "o"); str.replace("ổ", "o"); str.replace("ỗ", "o"); str.replace("ộ", "o");
  str.replace("ơ", "o"); str.replace("ớ", "o"); str.replace("ờ", "o"); str.replace("ở", "o"); str.replace("ỡ", "o"); str.replace("ợ", "o");

  str.replace("ú", "u"); str.replace("ù", "u"); str.replace("ủ", "u"); str.replace("ũ", "u"); str.replace("ụ", "u");
  str.replace("ư", "u"); str.replace("ứ", "u"); str.replace("ừ", "u"); str.replace("ử", "u"); str.replace("ữ", "u"); str.replace("ự", "u");

  str.replace("ý", "y"); str.replace("ỳ", "y"); str.replace("ỷ", "y"); str.replace("ỹ", "y"); str.replace("ỵ", "y");

  // Viết hoa
  str.replace("Á", "A"); str.replace("À", "A"); str.replace("Ả", "A"); str.replace("Ã", "A"); str.replace("Ạ", "A");
  str.replace("Ă", "A"); str.replace("Ắ", "A"); str.replace("Ằ", "A"); str.replace("Ẳ", "A"); str.replace("Ẵ", "A"); str.replace("Ặ", "A");
  str.replace("Â", "A"); str.replace("Ấ", "A"); str.replace("Ầ", "A"); str.replace("Ẩ", "A"); str.replace("Ẫ", "A"); str.replace("Ậ", "A");

  str.replace("Đ", "D");

  str.replace("É", "E"); str.replace("È", "E"); str.replace("Ẻ", "E"); str.replace("Ẽ", "E"); str.replace("Ẹ", "E");
  str.replace("Ê", "E"); str.replace("Ế", "E"); str.replace("Ề", "E"); str.replace("Ể", "E"); str.replace("Ễ", "E"); str.replace("Ệ", "E");

  str.replace("Í", "I"); str.replace("Ì", "I"); str.replace("Ỉ", "I"); str.replace("Ĩ", "I"); str.replace("Ị", "I");

  str.replace("Ó", "O"); str.replace("Ò", "O"); str.replace("Ỏ", "O"); str.replace("Õ", "O"); str.replace("Ọ", "O");
  str.replace("Ô", "O"); str.replace("Ố", "O"); str.replace("Ồ", "O"); str.replace("Ổ", "O"); str.replace("Ỗ", "O"); str.replace("Ộ", "O");
  str.replace("Ơ", "O"); str.replace("Ớ", "O"); str.replace("Ờ", "O"); str.replace("Ở", "O"); str.replace("Ỡ", "O"); str.replace("Ợ", "O");

  str.replace("Ú", "U"); str.replace("Ù", "U"); str.replace("Ủ", "U"); str.replace("Ũ", "U"); str.replace("Ụ", "U");
  str.replace("Ư", "U"); str.replace("Ứ", "U"); str.replace("Ừ", "U"); str.replace("Ử", "U"); str.replace("Ữ", "U"); str.replace("Ự", "U");

  str.replace("Ý", "Y"); str.replace("Ỳ", "Y"); str.replace("Ỷ", "Y"); str.replace("Ỹ", "Y"); str.replace("Ỵ", "Y");

  return str;
}

String shortenName(String fullName) {
  fullName.trim();  // loại bỏ khoảng trắng thừa
  int start = 0;
  int idx = fullName.indexOf(' ');
  String parts[10]; // mảng chứa tối đa 10 từ
  int count = 0;

  // Tách tên thành các từ
  while (idx >= 0) {
    parts[count++] = fullName.substring(start, idx);
    start = idx + 1;
    idx = fullName.indexOf(' ', start);
  }
  parts[count++] = fullName.substring(start); // thêm từ cuối cùng

  // Nếu <= 3 chữ thì đảo lại cho hợp format
  if (count <= 3) {
    String result = parts[count - 1];
    for (int i = 0; i < count - 1; i++) {
      result += " " + parts[i];
    }
    return result;
  }

  // Nếu > 3 chữ → chỉ lấy 3 chữ (Tên + Họ + Đệm đầu)
  String result = parts[count - 1];   // Tên chính (cuối)
  result += " " + parts[0];           // Họ (đầu tiên)
  result += " " + parts[1];           // Đệm (đầu tiên)
  
  return result;
}

// hàm setup
void setup() {
  Serial.begin(115200);
  SPI.begin();
  rfid.PCD_Init();

  // khởi tạo LCD
  lcd.init();
  lcd.backlight();

  // khởi tạo buzzer
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, HIGH);

  // khởi tạo wifi và kiểm tra
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println("\n✅ Đã kết nối WiFi");

  // Khởi tạo NTP và kiểm tra (retry cho tới khi sync được)
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  struct tm timeinfo;
  while (!getLocalTime(&timeinfo)) {
    Serial.println("❌ Failed to obtain time, retrying...");
    delay(2000);  // đợi 2s rồi thử lại
  }
  Serial.println("✅ Time synced!");

  // Khởi tạo và Đăng nhập FIREBASE dưới dạng Anonymous (chỉ chạy sau khi có time)
  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;
  Firebase.signUp(&config, &auth, "", "");
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  Serial.println("✅ Ready to read RFID...");
  showDefaultScreen();

}

// vòng lặp chính
void loop() {
  // Nếu đang hiển thị thông điệp và quá 3 giây thì trả về mặc định
  if (showingMessage && millis() - lastDisplayTime > 3000) {
    showDefaultScreen();
  }

  // 🔄 Sync lại NTP định kỳ
  if (millis() - lastNtpSync > ntpInterval) {
    Serial.println("🔄 Sync lại NTP...");
    configTime(7 * 3600, 0, "time.google.com");  // GMT+7 (VN)
    lastNtpSync = millis();
  }

  // Nếu Firebase chưa sẵn sàng thì bỏ qua vòng lặp này
  if (!Firebase.ready()) {
    Serial.println("⚠️ Firebase chưa sẵn sàng, bỏ qua vòng lặp...");
    delay(500);
    return;
  }

  // Kiểm tra thẻ RFID
  if (!rfid.PICC_IsNewCardPresent() || !rfid.PICC_ReadCardSerial()) return;

  // Lấy UID
  String uid = "";
  for (byte i = 0; i < rfid.uid.size; i++) {
    uid += String(rfid.uid.uidByte[i], HEX);
  }
  uid.toUpperCase();

  beep(100);

  String timeNow = getTime();
  Serial.println("🔑 UID: " + uid);
  Serial.println("⏰ Time: " + timeNow);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("UID:");
  lcd.print(uid);

  lcd.setCursor(0, 1);
  lcd.print(timeNow);


  // Kiểm tra UID đã có trong DB chưa
  String statusPath = "/RFID/" + uid + "/lastStatus"; // đường dẫn lấy dữ liệu của last status
  String namePath = "/USER/" + uid + "/name"; // đường dẫn lấy dữ liệu name của thẻ nếu có trên hệ thống
  if (Firebase.RTDB.getString(&fbdo, statusPath)) {
    // UID đã tồn tại → xử lý bình thường
    String lastStatus = fbdo.stringData();

    // Xác định trạng thái mới
    String newStatus = (lastStatus == "Len-xe") ? "Xuong-xe" : "Len-xe";

    // Epoch âm làm key
    time_t now;
    time(&now);
    long negativeKey = -now;

    String logPath = "/RFID/" + uid + "/accessLog/" + String(negativeKey);

    // Tạo JSON gồm cả status và time
    FirebaseJson json;
    json.set("status", newStatus);
    json.set("time", timeNow);

    // Ghi vào accessLog
    if (Firebase.RTDB.setJSON(&fbdo, logPath, &json)) {
      Serial.println("✅ Thêm log: " + newStatus);
    } else {
      Serial.println("❌ Lỗi: " + fbdo.errorReason());
    }

    // Cập nhật lastStatus
    Firebase.RTDB.setString(&fbdo, statusPath, newStatus);

    // Lấy dữ liệu name của USER
    Firebase.RTDB.getString(&fbdo, namePath.c_str());
    String studentName = fbdo.stringData();
    String lcdName = removeVietnameseTones(studentName);
    String lcdFormatName = shortenName(lcdName);

    beep(100);
    delay(100);
    beep(100);  // kêu 2 lần

    // Hiển thị lên LCD
    showMessage("" + lcdFormatName, "Status:" + newStatus);

  } else {
    // UID chưa tồn tại → đưa vào Pending
    String pendingPath = "/Pending/" + uid;

    FirebaseJson json;
    json.set("time", timeNow);
    json.set("status", "Chua-duyet");

    if (Firebase.RTDB.setJSON(&fbdo, pendingPath, &json)) {
      Serial.println("⚠️ UID mới, đưa vào Pending chờ duyệt: " + uid);
    } else {
      Serial.println("❌ Lỗi Pending: " + fbdo.errorReason());
    }

    beep(500);  // kêu dài hơn

    // UID mới
    showMessage("UID moi", "Cho duyet...");
  }

  delay(2000);
}
