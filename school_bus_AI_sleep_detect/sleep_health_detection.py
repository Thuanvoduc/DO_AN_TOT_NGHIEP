import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import time

# ========= CẤU HÌNH =========
URL = "http://192.168.1.174/cam-hi.jpg"   # <-- đổi IP của bạn
TIMEOUT_SEC = 2.0                       # timeout khi đọc frame qua HTTP

# Yêu cầu phải CẢ HAI tín hiệu (mắt lờ đờ + ngáp) mới coi là "Mệt mỏi"
REQUIRE_BOTH_FOR_DROWSY = True

# Ngưỡng & thời gian
EAR_SLEEP = 0.20            # mắt nhắm hẳn (rất thấp)
EAR_DROWSY = 0.27           # mắt lờ đờ (thấp nhưng chưa nhắm)
MAR_YAWN  = 0.35            # ngáp (tỉ lệ môi mở theo bề ngang miệng)
CLOSED_EYES_TIME = 2.0      # (giây) nhắm mắt liên tục -> ngủ gật
DROWSY_HOLD_SEC  = 1.0      # (giây) mắt lờ đờ phải kéo dài
YAWN_HOLD_SEC    = 0.6      # (giây) miệng mở lớn (ngáp) phải kéo dài

# ========= MEDIAPIPE =========
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Chỉ số landmark cho mắt (MediaPipe Face Mesh)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]           # (p1,p2,p3,p4,p5,p6)
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# Chỉ số landmark cho miệng (đơn giản, bền vững)
# Dùng đỉnh môi trên/dưới (13,14) và hai khoé miệng (61,291)
MOUTH_TOP     = 13
MOUTH_BOTTOM  = 14
MOUTH_LEFT    = 61
MOUTH_RIGHT   = 291

def eye_aspect_ratio(landmarks, eye_idx, W, H):
    # Lấy 6 điểm mắt theo pixel
    pts = [(landmarks[i].x * W, landmarks[i].y * H) for i in eye_idx]
    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in pts]
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4) + 1e-6
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(landmarks, W, H):
    # MAR = khoảng cách dọc (13-14) / khoảng cách ngang (61-291)
    top = np.array([landmarks[MOUTH_TOP].x * W, landmarks[MOUTH_TOP].y * H])
    bot = np.array([landmarks[MOUTH_BOTTOM].x * W, landmarks[MOUTH_BOTTOM].y * H])
    left = np.array([landmarks[MOUTH_LEFT].x * W, landmarks[MOUTH_LEFT].y * H])
    right = np.array([landmarks[MOUTH_RIGHT].x * W, landmarks[MOUTH_RIGHT].y * H])
    vertical = np.linalg.norm(top - bot)
    horizontal = np.linalg.norm(left - right) + 1e-6
    return vertical / horizontal

# Bộ đếm thời gian trạng thái
t_eye_closed_start = None
t_drowsy_start = None
t_yawn_start = None

def seconds_since(t0):
    return (time.time() - t0) if t0 is not None else 0.0

# ========= VÒNG LẶP =========
cv2.namedWindow("Driver State", cv2.WINDOW_AUTOSIZE)

while True:
    try:
        # Đọc frame từ ESP32-CAM (ảnh JPEG)
        resp = urllib.request.urlopen(URL, timeout=TIMEOUT_SEC)
        data = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status = "Binh thuong"

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            lm = face.landmark

            # EAR (trung bình 2 mắt)
            left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  W, H)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE, W, H)
            EAR = (left_ear + right_ear) / 2.0

            # MAR (ngáp)
            MAR = mouth_aspect_ratio(lm, W, H)

            # Bounding box khuôn mặt cho đẹp tự nhiên
            xs = [int(p.x * W) for p in lm]
            ys = [int(p.y * H) for p in lm]
            x1, y1 = max(min(xs), 0), max(min(ys), 0)
            x2, y2 = min(max(xs), W-1), min(max(ys), H-1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)

            # ---- Logic ngủ gật (nhắm hẳn) ----
            if EAR < EAR_SLEEP:
                if t_eye_closed_start is None:
                    t_eye_closed_start = time.time()
                if seconds_since(t_eye_closed_start) >= CLOSED_EYES_TIME:
                    status = "Ngu gat"
            else:
                t_eye_closed_start = None  # reset khi không nhắm hẳn

            # ---- Logic mệt mỏi (mắt lờ đờ +/hoặc ngáp) ----
            # Mắt lờ đờ
            if EAR_SLEEP <= EAR < EAR_DROWSY:
                if t_drowsy_start is None:
                    t_drowsy_start = time.time()
            else:
                t_drowsy_start = None

            # Ngáp
            if MAR > MAR_YAWN:
                if t_yawn_start is None:
                    t_yawn_start = time.time()
            else:
                t_yawn_start = None

            drowsy_ok = (seconds_since(t_drowsy_start) >= DROWSY_HOLD_SEC)
            yawn_ok   = (seconds_since(t_yawn_start)   >= YAWN_HOLD_SEC)

            if status != "Ngu gat":  # chỉ xét mệt khi chưa ngủ gật
                if REQUIRE_BOTH_FOR_DROWSY:
                    if drowsy_ok and yawn_ok:
                        status = "Met moi"
                else:
                    if drowsy_ok or yawn_ok:
                        status = "Met moi"

            # Overlay thông tin
            cv2.putText(frame, f"EAR:{EAR:.2f}  MAR:{MAR:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(
                frame,
                f"Trang thai: {status}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255) if status == "Ngu gat" else (0, 165, 255) if status == "Met moi" else (0, 200, 0),
                2
            )

        cv2.imshow("Driver State", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        # Mạng/HTTP trục trặc: đợi nhẹ và thử lại
        print("Warn:", e)
        time.sleep(0.05)
        continue

cv2.destroyAllWindows()
