import cv2
import mediapipe as mp
import urllib.request
import numpy as np
import math

# ESP32-CAM URL
url = 'http://192.168.1.5/cam-hi.jpg'

# Khởi tạo mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Các điểm landmark cho mắt trái và phải (theo MediaPipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]  
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean_distance(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(landmarks, eye_points, w, h):
    p1 = (int(landmarks[eye_points[0]].x * w), int(landmarks[eye_points[0]].y * h))
    p2 = (int(landmarks[eye_points[1]].x * w), int(landmarks[eye_points[1]].y * h))
    p3 = (int(landmarks[eye_points[2]].x * w), int(landmarks[eye_points[2]].y * h))
    p4 = (int(landmarks[eye_points[3]].x * w), int(landmarks[eye_points[3]].y * h))
    p5 = (int(landmarks[eye_points[4]].x * w), int(landmarks[eye_points[4]].y * h))
    p6 = (int(landmarks[eye_points[5]].x * w), int(landmarks[eye_points[5]].y * h))

    # EAR công thức: (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    EAR = (vertical1 + vertical2) / (2.0 * horizontal)
    return EAR

while True:
    # Đọc ảnh từ ESP32-CAM
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgnp, -1)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Tính EAR cho 2 mắt
            left_EAR = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
            right_EAR = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
            ear = (left_EAR + right_EAR) / 2.0

            # Ngưỡng EAR, nếu nhỏ hơn coi như mắt nhắm
            if ear < 0.25:
                cv2.putText(frame, "Eyes Closed", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "Eyes Open", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Live Transmission with Eye Blink Detection", frame)
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
