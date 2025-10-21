import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import requests
import tensorflow as tf
from queue import Queue, Empty
from collections import deque
from pathlib import Path
import os

# ========== EMOTION RUNTIME ==========
EMO_MODE = os.environ.get("EMO_MODE", "ON").upper()
print(f"[BOOT] EMO_MODE = {EMO_MODE}")

# ========== MODEL PATH RESOLUTION ==========
BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "emotion_model.h5",
    BASE_DIR / "emotion_model.keras",
    BASE_DIR / "emotion_model",
    BASE_DIR / "models" / "emotion_model.h5",
    BASE_DIR / "models" / "emotion_model.keras",
    BASE_DIR / "models" / "emotion_model",
]
def find_model_path():
    for p in MODEL_CANDIDATES:
        if p.exists(): return str(p)
    return None
MODEL_PATH = find_model_path()

# ========== EMOTION SERVICE ==========
class EmotionService:
    def __init__(self, model_path=MODEL_PATH, input_size=(48, 48), labels=None, custom_objects=None):
        self.path = model_path
        self.input_size = input_size
        self.labels = labels or ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
        self.custom_objects = custom_objects or {}
        self.model = None
        self.ready = False
        self.error = None
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            print("[EMOTION] resolved model path:", self.path)
            if not self.path:
                raise FileNotFoundError("Emotion model not found at candidates.")
            self.model = tf.keras.models.load_model(self.path, compile=False, custom_objects=self.custom_objects)
            self.ready = True
            print("[EMOTION] model loaded OK")
        except Exception as e:
            self.error = repr(e)
            self.ready = False
            print("[EMOTION] load failed:", self.error)

    def predict(self, face_roi):
        if not self.ready or face_roi is None or face_roi.size == 0:
            return None
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, self.input_size, interpolation=cv2.INTER_AREA)
            x = (resized / 255.0).astype(np.float32)
            x = np.reshape(x, (1, self.input_size[0], self.input_size[1], 1))
            pred = self.model.predict(x, verbose=0)
            return self.labels[int(np.argmax(pred))]
        except Exception:
            return None

# ========== CONFIG ==========
URL_SNAPSHOT      = "http://172.16.10.59/cam-mid.jpg"  # đổi đúng endpoint
CONNECT_TIMEOUT   = 3.0
READ_TIMEOUT      = 6.0
TARGET_FETCH_FPS  = 6

# ESP32 endpoint
ESP32_IP = "172.16.10.86"               # <— NHỚ đổi đúng IP hiện tại
URL_SEND = f"http://{ESP32_IP}/send"    # <— đổi route/port nếu firmware khác

# Hiển thị
PROCESS_WIDTH     = 416
DISPLAY_WIDTH     = 800
DISPLAY_FPS       = 15
SHOW_WINDOW       = True
DRAW_OVERLAY      = True

# Track/mesh
MAX_TRACKS        = 3
ASSIGN_DIST_RATIO = 0.25   # ↑ mượt hơn khi frame lớn
LOST_SEC          = 1.0
MESH_MIN_INTERVAL = 0.12

# Ngưỡng/logic
EAR_SLEEP_ABS     = 0.21
EAR_DROWSY_ABS    = 0.27
CLOSED_EYES_TIME  = 1.8
EAR_SLEEP_RATIO   = 0.60
EAR_DROWSY_RATIO  = 0.80
BASELINE_MIN      = 0.22
BASELINE_MAX      = 0.40
BASELINE_ALPHA    = 0.05

MAR_YAWN          = 0.35
DROWSY_HOLD_SEC   = 1.0
YAWN_HOLD_SEC     = 0.6

TILT_DEG_SLEEP    = 22.0
TILT_HOLD_SEC     = 1.0
ROLL_VEL_MAX      = 120.0

WAKE_EAR          = 0.27
WAKE_TILT_DEG     = 8.0
WAKE_HOLD_SEC     = 0.4

REACQUIRE_SEC     = 1.2
REACQ_DIST_RATIO  = 0.35
STICKY_IOU_BONUS  = 0.35   # ↓ bớt dính khi IoU cao

EMOTION_CHECK_INTERVAL = 0.5
TILT_NEEDS_EYE_CLOSED = True
WAKE_EAR_MARGIN       = 0.01
UNSLEEP_GRACE_SEC     = 0.6
STUCK_CLEAR_SEC       = 1.2
BLINK_REFRACT_SEC     = 0.7

# ========== STATUS BUS: điều độ HTTP tới ESP32 ==========
class StatusBus:
    """
    Gửi trạng thái P1..P3 điều độ, chống spam/timeout.
    - aggregate=False: gửi "P1: DETECTED" / "P2: UNDETECTED" / "P3: DROWSINESS" (định dạng cũ).
    - aggregate=True: gửi 1 snapshot "STATE P1=...;P2=...;P3=..." (nếu firmware hỗ trợ).
    """
    def __init__(self, url_send, aggregate=False, period=0.60, tx_gap_ms=250):
        self.url = url_send
        self.AGGREGATE = aggregate
        self.PERIOD = float(period)
        self.TX_GAP = tx_gap_ms / 1000.0
        self._lock = threading.Lock()
        self._stop = False
        self.state = {1: "UNDETECTED", 2: "UNDETECTED", 3: "UNDETECTED"}
        self._last_sent = None
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def set_state(self, tid, label):
        if tid not in (1,2,3): return
        with self._lock:
            self.state[tid] = label

    def close(self):
        self._stop = True
        try: self._t.join(timeout=1.0)
        except: pass

    def _build_snapshot(self):
        with self._lock:
            s1 = self.state.get(1, "UNDETECTED")
            s2 = self.state.get(2, "UNDETECTED")
            s3 = self.state.get(3, "UNDETECTED")
            snap = (s1, s2, s3)
            msg = f"STATE P1={s1};P2={s2};P3={s3}"
        return snap, msg

    def _safe_get(self, params):
        # retry 3 lần, tách timeout connect/read, đóng kết nối sau mỗi request
        for attempt in range(3):
            try:
                r = requests.get(
                    self.url,
                    params=params,
                    timeout=(1.5, 3.0),                # connect=1.5s, read=3s
                    headers={"Connection": "close"},
                    allow_redirects=False
                )
                print("→ TX", r.url, "| HTTP", r.status_code, "| RX:", getattr(r, "text", "")[:120])
                return True, getattr(r, "text", "")
            except Exception as e:
                print(f"❌ TX attempt {attempt+1} error:", e)
                time.sleep(0.25 * (attempt + 1))       # backoff nhẹ
        return False, None

    def _run(self):
        backoff = 0.0
        while not self._stop:
            t0 = time.time()
            if self.AGGREGATE:
                snap, msg = self._build_snapshot()
                if snap != self._last_sent:
                    ok, _ = self._safe_get({"msg": msg})
                    backoff = 0.0 if ok else min(1.0, max(0.2, backoff + 0.2))
                    if ok: self._last_sent = snap
            else:
                with self._lock:
                    items = list(self.state.items())
                for tid, label in items:
                    text = f"P{tid}: {label}"
                    ok, _ = self._safe_get({"msg": text})
                    time.sleep(self.TX_GAP)
                    backoff = 0.0 if ok else min(1.0, max(0.2, backoff + 0.2))
            dt = time.time() - t0
            time.sleep(max(0.0, self.PERIOD + backoff - dt))

# ========== FETCHER (snapshot, chống cache) ==========
class SnapshotFetcher:
    def __init__(self, url, target_fps=6, connect_to=3.0, read_to=6.0):
        self.base_url = url
        self.min_dt = 1.0 / max(1, int(target_fps))
        self.connect_to = connect_to
        self.read_to    = read_to
        self._lock = threading.Lock()
        self._frame = None
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        print("[FETCHER] started")

    def _url(self):
        return f"{self.base_url}?t={int(time.time()*1000)}"  # cache-buster

    def _run(self):
        last = 0.0
        while not self._stop:
            dt = time.time() - last
            if dt < self.min_dt:
                time.sleep(self.min_dt - dt)
            try:
                r = requests.get(self._url(),
                                 timeout=(self.connect_to, self.read_to),
                                 headers={
                                     "Accept": "image/jpeg",
                                     "Cache-Control": "no-cache, no-store, must-revalidate",
                                     "Pragma": "no-cache",
                                     "Connection": "close",
                                     "User-Agent": "opencv-python"
                                 })
                if r.status_code != 200:
                    print(f"[FETCHER] HTTP {r.status_code}")
                    time.sleep(0.2); continue
                arr = np.frombuffer(r.content, dtype=np.uint8)
                frm = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frm is None:
                    print(f"[FETCHER] imdecode failed (len={len(r.content)})")
                    time.sleep(0.2); continue
                with self._lock:
                    self._frame = frm
                last = time.time()
            except requests.Timeout:
                print("[FETCHER] timeout -> check IP/endpoint"); time.sleep(0.2)
            except requests.ConnectionError as e:
                print("[FETCHER] conn error:", e); time.sleep(0.5)
            except Exception as e:
                print("[FETCHER] error:", repr(e)); time.sleep(0.2)

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self):
        self._stop = True
        try: self._t.join(timeout=1.0)
        except: pass

# ========== MEDIAPIPE ==========
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=MAX_TRACKS,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 61, 291
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

# ========== GHÉP TRACK ỔN ĐỊNH ==========
def bbox_iou(b1, b2):
    if b1 is None or b2 is None:
        return 0.0
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2
    xi1, yi1 = max(x11, x21), max(y11, y21)
    xi2, yi2 = min(x12, x22), min(y12, y22)
    iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = iw * ih
    a1 = max(0, x12 - x11) * max(0, y12 - y11)
    a2 = max(0, x22 - x21) * max(0, y22 - y21)
    union = a1 + a2 - inter + 1e-6
    return float(inter / union)

tracks = {
    tid: {
        "active": False,
        "cx": None, "cy": None,
        "t_eye_closed_start": None,
        "t_drowsy_start": None,
        "t_yawn_start": None,
        "t_wake_start": None,
        "t_tilt_start": None,
        "status": "Trống",
        "last_seen": 0.0,
        "EAR_last": None, "MAR_last": None, "ROLL_last": None,
        "vx": 0.0, "vy": 0.0,
        "last_bbox": None,
        "ear_filt": None, "roll_filt": None, "roll_vel_filt": 0.0,
        "ear_open_baseline": None,
        "last_logic_ts": None,
        "t_last_emotion_check": 0.0,
        "emotion_status": "Unknown",
        "_t_last_sleep_cue": None,
        "_t_last_blink": 0.0
    } for tid in range(1, MAX_TRACKS + 1)
}

def reset_track(tid, deactivate=False, cx=None, cy=None):
    tr = tracks[tid]
    tr.update({
        "t_eye_closed_start": None, "t_drowsy_start": None, "t_yawn_start": None,
        "t_wake_start": None, "t_tilt_start": None, "_t_last_sleep_cue": None,
        "_t_last_blink": 0.0, "status": "Trống" if deactivate else "Binh thuong",
        "EAR_last": None, "MAR_last": None, "ROLL_last": None,
        "cx": cx, "cy": cy, "vx": 0.0, "vy": 0.0, "last_bbox": None,
        "ear_filt": None, "roll_filt": None, "roll_vel_filt": 0.0,
        "ear_open_baseline": None, "last_logic_ts": None,
        "last_seen": time.time(), "active": (not deactivate)
    })

def assign_faces_to_tracks(dets, W, H):
    assigned = []
    used_tids = set()
    now = time.time()
    dist_th_active = ASSIGN_DIST_RATIO * min(W, H)
    dist_th_reacq  = REACQ_DIST_RATIO  * min(W, H)
    dets = sorted(dets, key=lambda d: d["area"], reverse=True)

    for det in dets:
        cx, cy = det["cx"], det["cy"]

        # 1) Ưu tiên track active + dự đoán vị trí + IoU bonus
        best_tid, best_score = None, 1e9
        for tid, tr in tracks.items():
            if tid in used_tids or not tr["active"] or tr["cx"] is None:
                continue
            px = tr["cx"] + tr["vx"]
            py = tr["cy"] + tr["vy"]
            d  = np.hypot(cx - px, cy - py)
            iou = bbox_iou(det["bbox"], tr["last_bbox"])
            score = d * (1.0 - STICKY_IOU_BONUS * iou)
            if score < best_score:
                best_score, best_tid = score, tid
        if best_tid is not None and best_score <= dist_th_active:
            used_tids.add(best_tid)
            assigned.append((best_tid, det))
            continue

        # 2) Re-acquire
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
            tr = tracks[best_tid]; tr["active"] = True
            used_tids.add(best_tid)
            assigned.append((best_tid, det))
            continue

        # 3) Bật ô trống lâu nhất
        oldest_age, free_tid = -1, None
        for tid, tr in tracks.items():
            if tid in used_tids: continue
            if not tr["active"]:
                age = now - tr["last_seen"]
                if age > oldest_age:
                    oldest_age, free_tid = age, tid
        if free_tid is not None:
            reset_track(free_tid, deactivate=False, cx=cx, cy=cy)
            used_tids.add(free_tid)
            assigned.append((free_tid, det))

    # 4) Cập nhật vị trí/vận tốc/bbox
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

    # 5) Inactive nếu mất lâu
    for tid, tr in tracks.items():
        if tr["active"] and tid not in [t for t, _ in assigned]:
            if now - tr["last_seen"] > LOST_SEC:
                tr["active"] = False
    return assigned

# ========== WORKER ==========
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

        # Emotion wiring
        if EMO_MODE == "ON":
            self.emotion = EmotionService(MODEL_PATH)
        elif EMO_MODE == "SAFE":
            class EmotionSafeProxy:
                def __init__(self, path):
                    self.svc = EmotionService(path); self.ready = False; self.error = None
                def predict(self, face_roi):
                    try:
                        self.ready = bool(self.svc and self.svc.ready); self.error = self.svc.error
                        if not self.ready: return None
                        return self.svc.predict(face_roi)
                    except Exception as e:
                        self.ready, self.error = False, repr(e); return None
            self.emotion = EmotionSafeProxy(MODEL_PATH)
        else:
            self.emotion = type("Off", (), {"ready": False, "error":"OFF", "predict": staticmethod(lambda *_: None)})()

        # Trục B (sức khỏe)
        self.win_sec = 60.0
        self.samples = {tid: deque() for tid in tracks}
        self.events  = {tid: {"blinks": deque(), "yawns": deque()} for tid in tracks}
        self.closed_flag = {tid: False for tid in tracks}
        self.S_disp = {tid: 85.0 for tid in tracks}

        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def submit(self, frame):
        try:
            if self.q.full(): self.q.get_nowait()
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

    def compute_window_metrics(self, tid, ear_sleep_thr):
        dq = self.samples[tid]
        if len(dq) < 2:
            return {"perclos": 0.0, "blink_rpm": 0.0, "yawn_rpm": 0.0, "stable_pct": 1.0}
        ts0 = dq[0][0]; ts1 = dq[-1][0]; dur = max(1e-3, ts1 - ts0)
        low_time = 0.0; stable_time = 0.0
        for i in range(1, len(dq)):
            t0, EAR0, _, R0 = dq[i-1]; t1, EAR1, _, R1 = dq[i]
            dt = t1 - t0
            if min(EAR0, EAR1) < ear_sleep_thr: low_time += dt
            if abs((R0 + R1)/2.0) < 12.0: stable_time += dt
        perclos = low_time / dur
        stable_pct = stable_time / dur
        blink_rpm = len(self.events[tid]["blinks"]) * (60.0 / self.win_sec)
        yawn_rpm  = len(self.events[tid]["yawns"])  * (60.0 / self.win_sec)
        return {"perclos": perclos, "blink_rpm": blink_rpm, "yawn_rpm": yawn_rpm, "stable_pct": stable_pct}

    def health_score_from_metrics(self, tid, metrics, emotion):
        S = 85.0
        S -= min(45.0, 100.0 * max(0.0, metrics["perclos"] - 0.15))
        br = metrics["blink_rpm"]
        if br < 8:   S -= (8 - br) * 1.5
        if br > 25:  S -= (br - 25) * 1.0
        S -= min(20.0, metrics["yawn_rpm"] * 6.0)
        S -= max(0.0, (0.6 - metrics["stable_pct"]) * 10.0)
        emo_pen = {"Sad":6, "Fear":6, "Angry":4, "Surprise":2, "Neutral":0, "Happy":0, "Emotion:OFF":0}
        S -= emo_pen.get(emotion, 0)
        S = float(max(0.0, min(100.0, S)))
        self.S_disp[tid] = 0.3 * S + 0.7 * self.S_disp[tid]
        if self.S_disp[tid] >= 75: health = "Tot"
        elif self.S_disp[tid] >= 55: health = "On"
        elif self.S_disp[tid] >= 40: health = "Met moi"
        else: health = "Khong tot"
        return self.S_disp[tid], health

    def safety_risk_level(self, state):
        return 3 if state == "Ngu gat" else 0

    def _run(self):
        while not self.stop:
            try:
                frame = self.q.get(timeout=0.2)
            except Empty:
                continue

            # Resize nhanh
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

            # Gán track
            pairs = assign_faces_to_tracks(detections, W, H) if detections else []

            overlay = []
            for tid, det in pairs:
                EAR, MAR, ROLL = det["EAR"], det["MAR"], det["ROLL"]
                x1, y1, x2, y2  = det["bbox"]
                tr = tracks[tid]

                # Lọc mềm + vận tốc quay
                dt = now_ts - tr["last_logic_ts"] if tr["last_logic_ts"] else 1/8.0
                dt = max(0.05, min(0.5, dt))
                a_ear, a_roll, a_vel = 0.4, 0.5, 0.6
                tr["ear_filt"]  = EAR  if tr["ear_filt"]  is None else (a_ear*EAR  + (1-a_ear)*tr["ear_filt"])
                tr["roll_filt"] = ROLL if tr["roll_filt"] is None else (a_roll*ROLL + (1-a_roll)*tr["roll_filt"])
                roll_vel_inst = (tr["roll_filt"] - (tr["ROLL_last"] if tr["ROLL_last"] is not None else tr["roll_filt"])) / dt
                tr["roll_vel_filt"] = a_vel*roll_vel_inst + (1-a_vel)*tr["roll_vel_filt"]

                EARf  = tr["ear_filt"]
                ROLLf = tr["roll_filt"]

                # Baseline mắt mở
                if tr["status"] != "Ngu gat" and EARf is not None and (abs(ROLLf) < 12.0):
                    if (tr["ear_open_baseline"] is None) and (EARf > 0.24):
                        tr["ear_open_baseline"] = EARf
                    elif tr["ear_open_baseline"] is not None and EARf > 0.22:
                        tr["ear_open_baseline"] = (1-BASELINE_ALPHA)*tr["ear_open_baseline"] + BASELINE_ALPHA*EARf
                        tr["ear_open_baseline"] = max(BASELINE_MIN, min(BASELINE_MAX, tr["ear_open_baseline"]))

                if tr["ear_open_baseline"] is not None:
                    ear_sleep_thr  = max(EAR_SLEEP_ABS,  tr["ear_open_baseline"]*EAR_SLEEP_RATIO)
                    ear_drowsy_thr = max(EAR_DROWSY_ABS, tr["ear_open_baseline"]*EAR_DROWSY_RATIO)
                else:
                    ear_sleep_thr  = EAR_SLEEP_ABS
                    ear_drowsy_thr = EAR_DROWSY_ABS

                # Trục A: Ngủ gật / Bình thường
                eyes_closed_raw = (EAR < ear_sleep_thr)
                tilt_big        = (abs(ROLLf) >= TILT_DEG_SLEEP)
                tilt_ok         = (tilt_big and (not TILT_NEEDS_EYE_CLOSED or eyes_closed_raw))

                t_last_blink = tr.get("_t_last_blink", 0.0)
                blink_refract_ok = (now_ts - t_last_blink) >= BLINK_REFRACT_SEC

                if eyes_closed_raw and blink_refract_ok:
                    if tr["t_eye_closed_start"] is None:
                        tr["t_eye_closed_start"] = now_ts
                else:
                    tr["t_eye_closed_start"] = None

                if tilt_ok:
                    if tr["t_tilt_start"] is None: tr["t_tilt_start"] = now_ts
                else:
                    tr["t_tilt_start"] = None

                sleep_by_eye  = (seconds_since(tr["t_eye_closed_start"]) >= CLOSED_EYES_TIME) if tr["t_eye_closed_start"] else False
                sleep_by_tilt = (seconds_since(tr["t_tilt_start"])      >= TILT_HOLD_SEC)    if tr["t_tilt_start"]      else False

                if sleep_by_eye or sleep_by_tilt:
                    tr["status"] = "Ngu gat"; tr["_t_last_sleep_cue"] = now_ts; tr["t_wake_start"] = None
                else:
                    wake_thr = max(WAKE_EAR, ear_sleep_thr + WAKE_EAR_MARGIN)
                    wake_pose_ok = (abs(ROLLf) <= WAKE_TILT_DEG)
                    wake_eye_ok  = (EARf >= wake_thr)
                    if tr["status"] == "Ngu gat":
                        if wake_eye_ok and wake_pose_ok:
                            if tr["t_wake_start"] is None: tr["t_wake_start"] = now_ts
                            if seconds_since(tr["t_wake_start"]) >= max(WAKE_HOLD_SEC, UNSLEEP_GRACE_SEC):
                                tr["status"] = "Binh thuong"; tr["t_wake_start"] = None
                        else:
                            tr["t_wake_start"] = None
                        no_more_cue = (not eyes_closed_raw) and (not tilt_big)
                        t_last_cue  = tr.get("_t_last_sleep_cue", now_ts)
                        if no_more_cue and (now_ts - t_last_cue) > STUCK_CLEAR_SEC:
                            tr["status"] = "Binh thuong"; tr["t_wake_start"] = None
                    else:
                        tr["status"] = "Binh thuong"

                # Sự kiện cho metrics
                if MAR > MAR_YAWN:
                    if tr["t_yawn_start"] is None: tr["t_yawn_start"] = now_ts
                else:
                    if tr["t_yawn_start"] is not None and seconds_since(tr["t_yawn_start"]) >= YAWN_HOLD_SEC:
                        self.events[tid]["yawns"].append(now_ts)
                    tr["t_yawn_start"] = None

                # Emotion
                emotion = tr.get("emotion_status", "Unknown")
                if (now_ts - tr.get("t_last_emotion_check", 0.0)) > EMOTION_CHECK_INTERVAL:
                    label = None
                    if getattr(self.emotion, "ready", False):
                        x1c = max(0, min(x1, W-1)); x2c = max(0, min(x2, W-1))
                        y1c = max(0, min(y1, H-1)); y2c = max(0, min(y2, H-1))
                        if x2c > x1c and y2c > y1c:
                            face_roi = proc[y1c:y2c, x1c:x2c]
                            label = self.emotion.predict(face_roi)
                    elif getattr(self.emotion, "error", None):
                        label = "Emotion:OFF"
                    if label:
                        emotion = label; tr["emotion_status"] = emotion
                    tr["t_last_emotion_check"] = now_ts

                # Mẫu 60s & blink
                self.samples[tid].append((now_ts, tr["ear_filt"], MAR, tr["roll_filt"]))
                while self.samples[tid] and now_ts - self.samples[tid][0][0] > self.win_sec:
                    self.samples[tid].popleft()

                ear_thr_blink = max(ear_sleep_thr, 0.75*ear_drowsy_thr)
                if tr["ear_filt"] < ear_thr_blink and not self.closed_flag[tid]:
                    self.closed_flag[tid] = True; tr["_blink_start"] = now_ts
                elif tr["ear_filt"] >= ear_thr_blink and self.closed_flag[tid]:
                    dur = now_ts - tr.get("_blink_start", now_ts)
                    if 0.08 <= dur <= 0.45:
                        self.events[tid]["blinks"].append(now_ts); tr["_t_last_blink"] = now_ts
                    self.closed_flag[tid] = False

                for key in ("blinks", "yawns"):
                    dq = self.events[tid][key]
                    while dq and now_ts - dq[0] > self.win_sec: dq.popleft()

                # Sức khỏe (trục B)
                metrics = self.compute_window_metrics(tid, ear_sleep_thr)
                S_disp, health = self.health_score_from_metrics(tid, metrics, emotion)

                # Lưu overlay
                tr["EAR_last"]  = EAR
                tr["MAR_last"]  = MAR
                tr["ROLL_last"] = tr["roll_filt"]
                tr["last_logic_ts"] = now_ts
                self.smooth_ear[tid].append(EAR)
                self.smooth_mar[tid].append(MAR)
                ear_show = float(np.mean(self.smooth_ear[tid]))
                mar_show = float(np.mean(self.smooth_mar[tid]))

                overlay.append({
                    "tid": tid,
                    "bbox": (x1, y1, x2, y2),
                    "status": tr["status"],
                    "EAR": ear_show,
                    "MAR": mar_show,
                    "emotion": emotion,
                    "S": float(S_disp),
                    "health": health,
                    "perclos60": metrics["perclos"],
                    "blink_rpm": metrics["blink_rpm"],
                    "yawn_rpm": metrics["yawn_rpm"],
                    "risk": self.safety_risk_level(tr["status"])
                })

            self.last_mesh_ts = now_ts
            with self.r_lock:
                self.result = {"ts": now_ts, "proc_size": (W, H), "overlay": overlay}

# ========== MAIN LOOP ==========
cv2.setUseOptimized(True)

fetcher = SnapshotFetcher(URL_SNAPSHOT, target_fps=TARGET_FETCH_FPS,
                          connect_to=CONNECT_TIMEOUT, read_to=READ_TIMEOUT)
worker  = FaceMeshWorker(process_width=PROCESS_WIDTH, mesh_min_interval=MESH_MIN_INTERVAL)

# BUS gửi trạng thái: dùng định dạng "P#: LABEL" (firmware cũ)
status_bus = StatusBus(URL_SEND, aggregate=False, period=0.60, tx_gap_ms=250)

if SHOW_WINDOW:
    cv2.namedWindow("Driver State (snapshot, 3-person, robust)", cv2.WINDOW_AUTOSIZE)

last_disp = 0.0

try:
    while True:
        frame = fetcher.read()
        if frame is None:
            if SHOW_WINDOW:
                blank = np.zeros((240, 320, 3), np.uint8)
                cv2.putText(blank, "No frame - check snapshot URL", (12, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(blank, time.strftime("%H:%M:%S"), (12, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.imshow("Driver State (snapshot, 3-person, robust)", blank)
                cv2.waitKey(1)
            continue

        worker.submit(frame)

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
                try:
                    Wp, Hp = res["proc_size"]
                    sx, sy = (dispW / float(Wp), dispH / float(Hp))
                    present_now = set()

                    for item in res["overlay"]:
                        x1, y1, x2, y2 = item["bbox"]
                        x1 = max(0, min(x1, Wp-1)); x2 = max(0, min(x2, Wp-1))
                        y1 = max(0, min(y1, Hp-1)); y2 = max(0, min(y2, Hp-1))
                        if x2 <= x1 or y2 <= y1: continue
                        X1, Y1, X2, Y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)

                        tid = int(item.get("tid", 0))
                        st = item.get("status","Binh thuong")
                        emotion = item.get("emotion","")
                        S_disp = int(round(item.get("S", 85.0)))
                        health = item.get("health","On")
                        color = (0,0,255) if st == "Ngu gat" else (0,200,0)

                        present_now.add(tid)

                        cv2.rectangle(display, (X1, Y1), (X2, Y2), color, 2)
                        cv2.putText(display, f"P{tid} {st}", (X1, max(24, Y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        if emotion:
                            cv2.putText(display, emotion, (X1, Y2 + 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                        cv2.putText(display, f"S:{S_disp}  {health}",
                                    (X1, Y2 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        cv2.putText(display, time.strftime("%H:%M:%S"), (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                        # Cập nhật BUS (không gửi trực tiếp)
                        label = "DROWSINESS" if st == "Ngu gat" else "DETECTED"
                        status_bus.set_state(tid, label)

                    # P nào mất → UNDETECTED
                    for tid in tracks.keys():
                        if tid not in present_now:
                            status_bus.set_state(tid, "UNDETECTED")

                except Exception as e:
                    print("[DRAW] error:", repr(e))

            cv2.imshow("Driver State (snapshot, 3-person, robust)", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    pass
finally:
    worker.close()
    fetcher.close()
    try: face_mesh.close()
    except: pass
    try: status_bus.close()
    except: pass
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
