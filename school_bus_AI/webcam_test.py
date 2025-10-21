import cv2

# Mở camera mặc định (0 = webcam laptop, 1 = camera USB ngoài)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không mở được camera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không nhận được frame từ camera")
        break

    # Hiển thị hình ảnh
    cv2.imshow("Webcam Test", frame)

    # Nhấn phím q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
