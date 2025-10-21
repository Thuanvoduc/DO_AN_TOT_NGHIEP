import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import time

# =========================
# CẤU HÌNH ESP32-CAM STREAM
# =========================
url = "http://192.168.1.39/cam-hi.jpg"   # 👉 thay <ESP32-CAM-IP> bằng IP thực tế

# Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# =========================
# HÀM TÍNH EAR
# =========================
def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    # Chuyển landmark index thành pixel
    pts = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in eye_indices]

    # EAR công thức
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    ear = (A + B) / (2.0 * C)
    return ear, pts

# Landmark index mắt trong FaceMesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # 6 điểm mắt trái
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # 6 điểm mắt phải

# Ngưỡng EAR
EAR_THRESH = 0.25
EAR_CONSEC_FRAMES = 15

# Bộ đếm nhắm mắt
counter = 0

# =========================
# VÒNG LẶP LIVE STREAM
# =========================
while True:
    try:
        # Lấy frame từ ESP32-CAM
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipe xử lý
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # EAR cho 2 mắt
                left_ear, left_pts = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
                right_ear, right_pts = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0

                # Vẽ contour mắt
                for pt in left_pts + right_pts:
                    cv2.circle(frame, pt, 2, (0, 255, 0), -1)

                # Kiểm tra ngưỡng EAR
                if ear < EAR_THRESH:
                    counter += 1
                    if counter >= EAR_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS DETECTED!", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    counter = 0

                # Hiển thị EAR
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

        cv2.imshow("Phát Hiện Ngủ Gật", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Lỗi:", e)
        time.sleep(0.1)
        continue

cv2.destroyAllWindows()