#include <WebServer.h>
#include <WiFi.h>
#include <esp32cam.h>

const char* WIFI_SSID = "YoshiYoshi Coffee & Tea";
const char* WIFI_PASS = "camonquykhach";

WebServer server(80);

// Dùng các độ phân giải chuẩn của OV2640
static auto RES_LO  = esp32cam::Resolution::find(320, 240);   // QVGA
static auto RES_MID = esp32cam::Resolution::find(640, 480);   // VGA
static auto RES_HI  = esp32cam::Resolution::find(800, 600);   // SVGA

enum Mode { LO, MID, HI };
static Mode curMode = MID;   // mode hiện tại (nhớ để không đổi lại vô ích)

static void serveJpg() {
  auto frame = esp32cam::capture();
  if (frame == nullptr) {
    Serial.println("CAPTURE FAIL");
    server.send(503, "text/plain", "capture fail");
    return;
  }
  Serial.printf("CAP OK %dx%d %dB\n", frame->getWidth(), frame->getHeight(), (int)frame->size());
  server.sendHeader("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
  server.setContentLength(frame->size());
  server.send(200, "image/jpeg");
  WiFiClient client = server.client();
  frame->writeTo(client);
}

static bool setMode(Mode m) {
  if (m == curMode) return true;              // đang đúng rồi → khỏi đổi
  const auto& res = (m==LO? RES_LO : m==MID? RES_MID : RES_HI);
  if (!esp32cam::Camera.changeResolution(res)) {
    return false;
  }
  curMode = m;
  return true;
}

static void handleJpgLo()  { if (!setMode(LO))  Serial.println("SET-LO-RES FAIL");  serveJpg(); }
static void handleJpgMid() { if (!setMode(MID)) Serial.println("SET-MID-RES FAIL"); serveJpg(); }
static void handleJpgHi()  { if (!setMode(HI))  Serial.println("SET-HI-RES FAIL");  serveJpg(); }

void setup() {
  Serial.begin(115200);
  Serial.println();

  using namespace esp32cam;
  Config cfg;
  cfg.setPins(pins::AiThinker);
  cfg.setResolution(RES_MID);   // mặc định chạy mid (VGA)
  cfg.setBufferCount(2);        // 2 hoặc 3 nếu còn RAM/PSRAM
  cfg.setJpeg(70);              // cân bằng chất lượng/dung lượng (thử 60–85 tuỳ mạng)
  bool ok = Camera.begin(cfg);
  Serial.println(ok ? "CAMERA OK" : "CAMERA FAIL");

  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);         // giảm độ trễ mạng
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) delay(250);

  Serial.print("http://"); Serial.println(WiFi.localIP());
  Serial.println("  /cam-lo.jpg");
  Serial.println("  /cam-mid.jpg");
  Serial.println("  /cam-hi.jpg");

  server.on("/cam-lo.jpg",  handleJpgLo);
  server.on("/cam-mid.jpg", handleJpgMid);
  server.on("/cam-hi.jpg",  handleJpgHi);
  server.begin();
}

void loop() {
  server.handleClient();
}
