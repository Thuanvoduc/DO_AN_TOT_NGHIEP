import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import requests
from queue import Queue, Empty
from collections import deque

# ================== CẤU HÌNH ==================
URL_SNAPSHOT      = "http://192.168.1.174/cam-mid.jpg"  # <-- đổi IP/endpoint của bạn
CONNECT_TIMEOUT   = 1.5
READ_TIMEOUT      = 2.5
TARGET_FETCH_FPS  = 6

# Xử lý & hiển thị
PROCESS_WIDTH     = 416
DISPLAY_WIDTH     = 800
DISPLAY_FPS       = 15
SHOW_WINDOW       = True
DRAW_OVERLAY      = True

# Mediapipe & theo dõi
MAX_TRACKS        = 3
ASSIGN_DIST_RATIO = 0.2
LOST_SEC          = 1.0
MESH_MIN_INTERVAL = 0.12   # ~8Hz → mượt mà nhưng không khoá UI

# ====== Ngưỡng phát hiện (VỪA PHẢI) ======
# Mắt (EAR)
EAR_SLEEP        = 0.21     # nhắm mắt "đủ sâu"
CLOSED_EYES_TIME = 1.2      # phải nhắm liên tục ≥ 1.2s mới "Ngu gat"

# Lờ đờ/ngáp (để ra "Met moi", không ảnh hưởng "Ngu gat")
EAR_DROWSY       = 0.27
MAR_YAWN         = 0.35
DROWSY_HOLD_SEC  = 1.0
YAWN_HOLD_SEC    = 0.6

# Đầu nghiêng (roll) – moderate
TILT_DEG_SLEEP   = 22.0     # |roll| ≥ 22°
TILT_HOLD_SEC    = 0.8      # giữ ≥ 0.8s → "Ngu gat"

# Wake-up (thoát 'Ngu gat' khi đã tỉnh)
WAKE_EAR         = 0.27     # mở mắt rõ ràng
WAKE_TILT_DEG    = 8.0      # đầu tương đối thẳng
WAKE_HOLD_SEC    = 0.4

# ========== Giữ ID ổn định (CHỐNG NHẢY P#) ==========
REACQUIRE_SEC      = 1.2   # trong ~1.2s sau khi mất dấu, ưu tiên ghép lại đúng track
REACQ_DIST_RATIO   = 0.35  # ngưỡng xa hơn cho re-acquire (tính theo cạnh ngắn khung)
STICKY_IOU_BONUS   = 0.50  # 0..1, IoU cao -> giảm điểm khoảng cách (dính hơn)

# ================== FETCHER (snapshot, non-blocking) ==================
class SnapshotFetcher:
    def __init__(self, url, target_fps=6, connect_to=1.5, read_to=2.5):
        self.url = url
        self.min_dt = 1.0 / max(1, int(target_fps))
        self.connect_to = connect_to
        self.read_to    = read_to
        self.sess = requests.Session()
        self.sess.headers.update({
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
            "User-Agent": "opencv-python"
        })
        self._lock = threading.Lock()
        self._frame = None
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        last = 0.0
        while not self._stop:
            dt = time.time() - last
            if dt < self.min_dt:
                time.sleep(self.min_dt - dt)
            try:
                r = self.sess.get(self.url, timeout=(self.connect_to, self.read_to))
                if r.status_code == 200:
                    arr = np.frombuffer(r.content, dtype=np.uint8)
                    frm = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # decode full để hiển thị sắc nét
                    if frm is not None:
                        with self._lock:
                            self._frame = frm
                        last = time.time()
            except Exception:
                time.sleep(0.05)  # lỗi mạng tạm thời → bỏ khung

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self):
        self._stop = True
        try: self._t.join(timeout=1.0)
        except: pass
        try: self.sess.close()
        except: pass

# ================== MEDIAPIPE FACE MESH (worker riêng) ==================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=MAX_TRACKS,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmarks
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 61, 291
# hai khoé mắt ngoài để ước tính roll
EYE_OUTER_LEFT_IDX  = 33
EYE_OUTER_RIGHT_IDX = 263

def eye_aspect_ratio(landmarks, eye_idx, W, H):
    pts = [(landmarks[i].x * W, landmarks[i].y * H) for i in eye_idx]
    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in pts]
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4) + 1e-6
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(landmarks, W, H):
    top = np.array([landmarks[MOUTH_TOP].x * W, landmarks[MOUTH_TOP].y * H])
    bot = np.array([landmarks[MOUTH_BOTTOM].x * W, landmarks[MOUTH_BOTTOM].y * H])
    left = np.array([landmarks[MOUTH_LEFT].x * W, landmarks[MOUTH_LEFT].y * H])
    right = np.array([landmarks[MOUTH_RIGHT].x * W, landmarks[MOUTH_RIGHT].y * H])
    return np.linalg.norm(top - bot) / (np.linalg.norm(left - right) + 1e-6)

def head_roll_deg(landmarks, W, H):
    pL = np.array([landmarks[EYE_OUTER_LEFT_IDX].x * W,  landmarks[EYE_OUTER_LEFT_IDX].y * H])
    pR = np.array([landmarks[EYE_OUTER_RIGHT_IDX].x * W, landmarks[EYE_OUTER_RIGHT_IDX].y * H])
    dy = pR[1] - pL[1]
    dx = (pR[0] - pL[0]) + 1e-6
    return float(np.degrees(np.arctan2(dy, dx)))

def seconds_since(t0): 
    return 0.0 if t0 is None else (time.time() - t0)

# ========== IoU cho bbox (dùng trong ghép track) ==========
def bbox_iou(b1, b2):
    """IoU của 2 bbox (x1,y1,x2,y2). Trả 0.0 nếu thiếu."""
    if b1 is None or b2 is None:
        return 0.0
    x11,y11,x12,y12 = b1
    x21,y21,x22,y22 = b2
    xi1, yi1 = max(x11,x21), max(y11,y21)
    xi2, yi2 = min(x12,x22), min(y12,y22)
    iw, ih = max(0, xi2-xi1), max(0, yi2-yi1)
    inter = iw*ih
    a1 = max(0, x12-x11) * max(0, y12-y11)
    a2 = max(0, x22-x21) * max(0, y22-y21)
    union = a1 + a2 - inter + 1e-6
    return float(inter/union)

# ====== tracks & logic ======
tracks = {
    tid: {
        "active": False,
        "cx": None, "cy": None,
        # timers
        "t_eye_closed_start": None,
        "t_drowsy_start": None,
        "t_yawn_start": None,
        "t_wake_start": None,
        "t_tilt_start": None,
        # state
        "status": "Trống",
        "last_seen": 0.0,
        # last metrics
        "EAR_last": None, "MAR_last": None, "ROLL_last": None,
        # giữ ID ổn định
        "vx": 0.0, "vy": 0.0,      # vận tốc ước lượng (pixel/frame)
        "last_bbox": None          # bbox trước để tính IoU
    } for tid in range(1, MAX_TRACKS + 1)
}

def reset_track(tid, deactivate=False, cx=None, cy=None):
    tr = tracks[tid]
    tr["t_eye_closed_start"] = None
    tr["t_drowsy_start"] = None
    tr["t_yawn_start"] = None
    tr["t_wake_start"] = None
    tr["t_tilt_start"] = None
    tr["status"] = "Trống" if deactivate else "Binh thuong"
    tr["EAR_last"] = None
    tr["MAR_last"] = None
    tr["ROLL_last"] = None
    tr["cx"], tr["cy"] = cx, cy
    tr["vx"], tr["vy"] = 0.0, 0.0
    tr["last_bbox"] = None
    tr["last_seen"] = time.time()
    tr["active"] = not deactivate

# ========== GHÉP TRACK ỔN ĐỊNH (có dự đoán + re-acquire + IoU bonus) ==========
def assign_faces_to_tracks(dets, W, H):
    """
    dets: list[{cx,cy,EAR,MAR,ROLL,bbox,area}]
    - Ưu tiên ghép vào track đang active gần nhất (theo vị trí DỰ ĐOÁN + IoU bonus).
    - Nếu không được: thử ghép vào track vừa mất (inactive nhưng last_seen gần).
    - Nếu vẫn không: bật ô trống.
    - Track active không được gán và mất quá lâu -> chuyển inactive (không reset cứng, để re-acquire).
    """
    assigned = []
    used_tids = set()
    now = time.time()

    dist_th_active = ASSIGN_DIST_RATIO * min(W, H)
    dist_th_reacq  = REACQ_DIST_RATIO  * min(W, H)

    # Ưu tiên mặt to trước
    dets = sorted(dets, key=lambda d: d["area"], reverse=True)

    for det in dets:
        cx, cy = det["cx"], det["cy"]

        # 1) Ghép vào track đang active (ưu tiên cao)
        best_tid, best_score = None, 1e9
        for tid, tr in tracks.items():
            if tid in used_tids or not tr["active"] or tr["cx"] is None:
                continue
            # dự đoán vị trí kế tiếp
            px = tr["cx"] + tr["vx"]
            py = tr["cy"] + tr["vy"]
            d  = np.hypot(cx - px, cy - py)
            iou = bbox_iou(det["bbox"], tr["last_bbox"])
            # điểm hiệu dụng: giảm theo IoU (bbox chồng lấn -> dính hơn)
            score = d * (1.0 - STICKY_IOU_BONUS * iou)
            if score < best_score:
                best_score, best_tid = score, tid

        if best_tid is not None and best_score <= dist_th_active:
            used_tids.add(best_tid)
            assigned.append((best_tid, det))
            continue

        # 2) Re-acquire: ghép vào track vừa mất gần đây (inactive)
        best_tid, best_d = None, 1e9
        for tid, tr in tracks.items():
            if tid in used_tids or tr["active"] or tr["cx"] is None:
                continue
            if (now - tr["last_seen"]) > REACQUIRE_SEC:
                continue
            d = np.hypot(cx - tr["cx"], cy - tr["cy"])
            if d < best_d:
                best_d, best_tid = d, tid

        if best_tid is not None and best_d <= dist_th_reacq:
            # hồi sinh track mà KHÔNG reset timer/status
            tr = tracks[best_tid]
            tr["active"] = True
            used_tids.add(best_tid)
            assigned.append((best_tid, det))
            continue

        # 3) Bật ô trống (nếu có) - ưu tiên ô trống lâu nhất
        oldest_age, free_tid = -1, None
        for tid, tr in tracks.items():
            if tid in used_tids:
                continue
            if not tr["active"]:
                age = now - tr["last_seen"]
                if age > oldest_age:
                    oldest_age, free_tid = age, tid
        if free_tid is not None:
            reset_track(free_tid, deactivate=False, cx=cx, cy=cy)
            used_tids.add(free_tid)
            assigned.append((free_tid, det))
        # hết ô thì bỏ qua det thừa

    # 4) Cập nhật vị trí/van tốc/bbox cho track đã gán
    for tid, det in assigned:
        tr = tracks[tid]
        if tr["cx"] is not None:
            dx = det["cx"] - tr["cx"]
            dy = det["cy"] - tr["cy"]
            alpha = 0.7
            tr["vx"] = alpha * dx + (1 - alpha) * tr["vx"]
            tr["vy"] = alpha * dy + (1 - alpha) * tr["vy"]
        tr["cx"], tr["cy"] = det["cx"], det["cy"]
        tr["last_bbox"] = det["bbox"]
        tr["last_seen"] = now

    # 5) Track active không được gán -> chuyển inactive nếu mất quá lâu
    for tid, tr in tracks.items():
        if tr["active"] and tid not in [t for t, _ in assigned]:
            if now - tr["last_seen"] > LOST_SEC:
                tr["active"] = False
                # không reset timers/status để có thể re-acquire đúng P#
    return assigned

# ================== Worker Mediapipe (queue size=1) ==================
class FaceMeshWorker:
    def __init__(self, process_width=416, mesh_min_interval=0.12):
        self.q = Queue(maxsize=1)
        self.result = None
        self.r_lock = threading.Lock()
        self.stop = False
        self.process_width = process_width
        self.mesh_min_interval = mesh_min_interval
        self.last_mesh_ts = 0.0
        self.smooth_ear = {tid: deque(maxlen=5) for tid in tracks}
        self.smooth_mar = {tid: deque(maxlen=5) for tid in tracks}
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def submit(self, frame):
        try:
            if self.q.full():
                self.q.get_nowait()
            self.q.put_nowait(frame)
        except Exception:
            pass

    def get_result(self):
        with self.r_lock:
            return None if self.result is None else self.result.copy()

    def close(self):
        self.stop = True
        try: self.t.join(timeout=1.0)
        except: pass

    def _run(self):
        while not self.stop:
            try:
                frame = self.q.get(timeout=0.2)
            except Empty:
                continue

            # Resize xử lý nhanh
            H0, W0 = frame.shape[:2]
            if W0 > self.process_width:
                new_h = int(H0 * self.process_width / W0)
                proc = cv2.resize(frame, (self.process_width, new_h), interpolation=cv2.INTER_AREA)
            else:
                proc = frame
            H, W = proc.shape[:2]

            now_ts = time.time()
            if (now_ts - self.last_mesh_ts) < self.mesh_min_interval:
                continue

            # Mediapipe
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            detections = []
            if res and res.multi_face_landmarks:
                for face in res.multi_face_landmarks:
                    lm = face.landmark
                    left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  W, H)
                    right_ear = eye_aspect_ratio(lm, RIGHT_EYE, W, H)
                    EAR  = float((left_ear + right_ear) / 2.0)
                    MAR  = float(mouth_aspect_ratio(lm, W, H))
                    ROLL = float(head_roll_deg(lm, W, H))

                    xs = [int(p.x * W) for p in lm]
                    ys = [int(p.y * H) for p in lm]
                    x1, y1 = max(min(xs), 0), max(min(ys), 0)
                    x2, y2 = min(max(xs), W - 1), min(max(ys), H - 1)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    area = max(1, (x2 - x1) * (y2 - y1))

                    detections.append({"EAR": EAR, "MAR": MAR, "ROLL": ROLL,
                                       "bbox": (x1, y1, x2, y2),
                                       "cx": cx, "cy": cy, "area": area})

            pairs = assign_faces_to_tracks(detections, W, H) if detections else []

            overlay = []
            for tid, det in pairs:
                EAR, MAR, ROLL = det["EAR"], det["MAR"], det["ROLL"]
                x1, y1, x2, y2  = det["bbox"]
                tr = tracks[tid]

                # ===== NHẮM MẮT đủ lâu -> Ngu gat =====
                if EAR < EAR_SLEEP:
                    if tr["t_eye_closed_start"] is None:
                        tr["t_eye_closed_start"] = now_ts
                    if seconds_since(tr["t_eye_closed_start"]) >= CLOSED_EYES_TIME:
                        tr["status"] = "Ngu gat"
                        tr["t_wake_start"] = None
                else:
                    tr["t_eye_closed_start"] = None

                # ===== NGHIÊNG ĐẦU đủ lâu -> Ngu gat =====
                if abs(ROLL) >= TILT_DEG_SLEEP:
                    if tr["t_tilt_start"] is None:
                        tr["t_tilt_start"] = now_ts
                    if seconds_since(tr["t_tilt_start"]) >= TILT_HOLD_SEC:
                        tr["status"] = "Ngu gat"
                        tr["t_wake_start"] = None
                else:
                    tr["t_tilt_start"] = None

                # ===== MET MOI (tuỳ chọn) =====
                if EAR_SLEEP <= EAR < EAR_DROWSY:
                    if tr["t_drowsy_start"] is None:
                        tr["t_drowsy_start"] = now_ts
                else:
                    tr["t_drowsy_start"] = None

                if MAR > MAR_YAWN:
                    if tr["t_yawn_start"] is None:
                        tr["t_yawn_start"] = now_ts
                else:
                    tr["t_yawn_start"] = None

                d_ok = (seconds_since(tr["t_drowsy_start"]) >= DROWSY_HOLD_SEC)
                y_ok = (seconds_since(tr["t_yawn_start"])   >= YAWN_HOLD_SEC)

                if tr["status"] != "Ngu gat":
                    tr["status"] = "Met moi" if (d_ok and y_ok) else "Binh thuong"
                else:
                    # ===== WAKE-UP: mở mắt rõ + đầu thẳng lại đủ lâu =====
                    if (EAR >= WAKE_EAR) and (abs(ROLL) <= WAKE_TILT_DEG):
                        if tr["t_wake_start"] is None:
                            tr["t_wake_start"] = now_ts
                        if seconds_since(tr["t_wake_start"]) >= WAKE_HOLD_SEC:
                            tr["status"] = "Binh thuong"
                            tr["t_wake_start"]   = None
                            tr["t_drowsy_start"] = None
                            tr["t_yawn_start"]   = None
                    else:
                        tr["t_wake_start"] = None

                # lưu chỉ số mượt hiển thị
                tr["EAR_last"]  = EAR
                tr["MAR_last"]  = MAR
                tr["ROLL_last"] = ROLL
                self.smooth_ear[tid].append(EAR)
                self.smooth_mar[tid].append(MAR)
                ear_show = float(np.mean(self.smooth_ear[tid]))
                mar_show = float(np.mean(self.smooth_mar[tid]))

                overlay.append({
                    "tid": tid,
                    "bbox": (x1, y1, x2, y2),
                    "status": tr["status"],
                    "EAR": ear_show,
                    "MAR": mar_show
                })

            self.last_mesh_ts = now_ts
            with self.r_lock:
                self.result = {"ts": now_ts, "proc_size": (W, H), "overlay": overlay}

# ================== MAIN LOOP (UI không bị chặn) ==================
cv2.setUseOptimized(True)
# cv2.setNumThreads(2)

fetcher = SnapshotFetcher(URL_SNAPSHOT, target_fps=TARGET_FETCH_FPS,
                          connect_to=CONNECT_TIMEOUT, read_to=READ_TIMEOUT)
worker  = FaceMeshWorker(process_width=PROCESS_WIDTH, mesh_min_interval=MESH_MIN_INTERVAL)

if SHOW_WINDOW:
    cv2.namedWindow("Driver State (snapshot, 3-person, moderate)", cv2.WINDOW_AUTOSIZE)

last_disp = 0.0

try:
    while True:
        frame = fetcher.read()
        if frame is None:
            if SHOW_WINDOW:
                blank = np.zeros((240, 320, 3), np.uint8)
                cv2.putText(blank, "No frame - check snapshot URL", (12, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.imshow("Driver State (snapshot, 3-person, moderate)", blank)
                cv2.waitKey(1)
            continue

        # luôn đẩy frame mới nhất cho worker (drop cũ)
        worker.submit(frame)

        # Cập nhật cửa sổ theo FPS mục tiêu (không khoá vòng lặp)
        now = time.time()
        if SHOW_WINDOW and (now - last_disp) >= (1.0 / DISPLAY_FPS):
            last_disp = now
            H0, W0 = frame.shape[:2]
            if W0 != DISPLAY_WIDTH:
                disp_h = int(H0 * DISPLAY_WIDTH / W0)
                display = cv2.resize(frame, (DISPLAY_WIDTH, disp_h), interpolation=cv2.INTER_LINEAR)
            else:
                display = frame.copy()
            dispH, dispW = display.shape[:2]

            res = worker.get_result()
            if res and DRAW_OVERLAY:
                Wp, Hp = res["proc_size"]
                sx, sy = (dispW / float(Wp), dispH / float(Hp))
                for item in res["overlay"]:
                    x1, y1, x2, y2 = item["bbox"]
                    X1, Y1, X2, Y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                    st = item["status"]
                    color = (0, 0, 255) if st == "Ngu gat" else (0, 165, 255) if st == "Met moi" else (0, 200, 0)
                    cv2.rectangle(display, (X1, Y1), (X2, Y2), color, 2)
                    cv2.putText(display, f"P{item['tid']} {st}", (X1, max(20, Y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Driver State (snapshot, 3-person, moderate)", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    pass
finally:
    worker.close()
    fetcher.close()
    face_mesh.close()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
