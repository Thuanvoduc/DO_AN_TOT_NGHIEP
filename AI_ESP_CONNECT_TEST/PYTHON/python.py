import requests
import time

# ⚠️ Thay bằng IP của ESP32 in ra trong Serial Monitor
ESP32_IP = "172.16.10.86"
URL = f"http://{ESP32_IP}/send"

def send_message(msg):  
    try:
        response = requests.get(URL, params={"msg": msg}, timeout=3)
        print("📨 Gửi:", msg)
        print("📬 ESP32 phản hồi:", response.text)
    except Exception as e:
        print("❌ Lỗi khi gửi dữ liệu:", e)

if __name__ == "__main__":
    # LED 1
    # send_message("P1:DETECTED")
    # send_message("P1:UNDETECTED")
    # send_message("P1:DROWSINESS")
    # LED 2
    # send_message("P2:DETECTED")
    # send_message("P2:UNDETECTED")
    # send_message("P2:DROWSINESS")
    # LED 3
    # send_message("P3:DETECTED")
    send_message("P3:UNDETECTED")
    # send_message("P3:DROWSINESS")

