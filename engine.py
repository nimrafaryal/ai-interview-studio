"""
engine.py  -  Proctoring core (camera brain)

What it watches (all free, all offline, runs on your laptop CPU):
  * Face + eyes + head pose     -> MediaPipe Face Landmarker  (models/face_landmarker.task)
  * Phone / book / extra person -> YOLOv8n                    (models/yolov8n.pt)

Speed design (v3):
  The face check (~15 ms) and the object check (YOLO, ~60-150 ms) run in
  SEPARATE threads (see devices.py). The slow YOLO never delays the face check
  or the video, and each always works on the newest camera frame, so there's
  no lag. A phone is flagged after being seen for ~0.3-0.5 s.
"""

import os
import time
import threading
from collections import deque

import cv2
import mediapipe as mp

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
SNAP_DIR = os.path.join(HERE, "snapshots")

try:
    import winsound  # Windows only - silently skipped elsewhere
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

# --- MediaPipe landmark indices -----------------------------------------
LEFT_IRIS, RIGHT_IRIS = 468, 473
LEFT_EYE_H, RIGHT_EYE_H = (33, 133), (362, 263)
LEFT_EYE_V, RIGHT_EYE_V = (159, 145), (386, 374)
NOSE, FACE_L, FACE_R, FOREHEAD, CHIN = 1, 234, 454, 10, 152

# --- Penalties (points taken off the 100 integrity score) ----------------
#  name: (points, cooldown_seconds)
EVENTS = {
    "Looking away":       (3, 6.0),
    "Face not visible":   (6, 6.0),
    "Multiple people":    (20, 10.0),
    "Phone detected":     (15, 8.0),
    "Book / notes":       (8, 10.0),
}
# COCO id -> (event name, minimum confidence).
# A phone held in a hand is often labelled "remote" by YOLO, so both count as a phone.
YOLO_CLASSES = {
    67: ("Phone detected", 0.25),   # cell phone
    65: ("Phone detected", 0.30),   # remote (usually a phone in hand)
    73: ("Book / notes", 0.40),     # book
}
COCO_PERSON = 0
YOLO_SIZE = 480             # input size: smaller = faster; 480 still sees a phone easily
WINDOW = 4                  # look at the last 4 YOLO passes...
HITS_NEEDED = 2             # ...and flag if the object was in at least 2 of them
CALIB_FRAMES = 25
MAX_EVENTS = 300


def _ratio(v, a, b):
    lo, hi = (a, b) if a <= b else (b, a)
    if hi - lo < 1e-6:
        return 0.5
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


class Proctor:
    def __init__(self, enable_alarm=True):
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=os.path.join(MODELS, "face_landmarker.task")),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=2,                       # 2 -> lets us spot a second face
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)

        import torch
        from ultralytics import YOLO
        try:   # leave a couple of CPU cores for the camera, face check and browser
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
        except Exception:
            pass
        self.yolo = YOLO(os.path.join(MODELS, "yolov8n.pt"))
        import numpy as np
        self.yolo(np.zeros((480, 640, 3), np.uint8), imgsz=YOLO_SIZE, verbose=False)  # warm-up
        os.makedirs(SNAP_DIR, exist_ok=True)

        self.lock = threading.RLock()
        self._t0 = time.monotonic()
        self._last_ts = -1
        self.yolo_ms = 0.0
        self.face_ms = 0.0

        # tunable settings
        self.enable_alarm = enable_alarm
        self.look_away_seconds = 1.5
        self.no_face_seconds = 2.0
        self.object_scale = 1.0     # <1 = more sensitive, >1 = stricter
        self.person_conf = 0.60
        self.set_sensitivity("Normal")
        self.recalibrate()
        self.reset()

    # ------------------------------------------------------------------ setup
    def set_sensitivity(self, level):
        mult = {"Strict": 0.75, "Normal": 1.0, "Relaxed": 1.35}.get(level, 1.0)
        self.tol_eye = 0.12 * mult
        self.tol_yaw = 0.08 * mult
        self.tol_pitch = 0.08 * mult

    def reset(self):
        """Clear score + events (keeps calibration)."""
        with self.lock:
            self.penalty = 0
            self.events = []
            self.look_away_start = None
            self.no_face_start = None
            self._last_logged = {}
            self._streak = {}
            self._hist = {}
            self._boxes = []          # last YOLO boxes, drawn on every frame
            self._box_until = 0.0
            self.last_objects = []
            self._attn_frames = 0
            self._seen_frames = 0
            self._last_alarm_penalty = 0
            self.status = "Starting..."
            self.status_color = (90, 200, 90)
            self.iris = []
            self.monitoring = False   # only log events while True
            self.session = time.time()

    def recalibrate(self):
        with self.lock:
            self.calibrated = False
            self._calib = []
            self.neutral = (0.5, 0.5, 0.5, 0.5)

    # --------------------------------------------------------------- readouts
    @property
    def integrity(self):
        return max(0, 100 - self.penalty)

    @property
    def attention_pct(self):
        if self._seen_frames == 0:
            return 100.0
        return 100.0 * self._attn_frames / self._seen_frames

    @staticmethod
    def risk_of(integrity):
        if integrity >= 80:
            return "LOW"
        if integrity >= 50:
            return "MEDIUM"
        return "HIGH"

    def snapshot(self):
        with self.lock:
            return {
                "integrity": self.integrity,
                "penalty": self.penalty,
                "risk": self.risk_of(self.integrity),
                "events": list(self.events),
                "status": self.status,
                "calibrated": self.calibrated,
                "calib_progress": len(self._calib) / CALIB_FRAMES,
                "attention": self.attention_pct,
                "objects": list(self.last_objects),
                "monitoring": self.monitoring,
                "session": self.session,
                "yolo_ms": round(self.yolo_ms),
                "face_ms": round(self.face_ms),
            }

    # ---------------------------------------------------------------- logging
    def _log(self, name, frame=None):
        """Call with self.lock held."""
        if not self.monitoring:
            return False
        pts, cooldown = EVENTS[name]
        now = time.time()
        if now - self._last_logged.get(name, 0) < cooldown:
            return False
        self._last_logged[name] = now
        self.penalty += pts
        snap = None
        if frame is not None:
            snap = os.path.join(SNAP_DIR, f"{int(now * 1000)}_{name.split()[0].lower()}.jpg")
            try:
                cv2.imwrite(snap, frame)
            except Exception:
                snap = None
        self.events.append({"time": time.strftime("%H:%M:%S"), "t": now, "event": name,
                            "points": pts, "integrity": self.integrity, "snapshot": snap})
        if len(self.events) > MAX_EVENTS:
            self.events = self.events[-MAX_EVENTS:]
        if pts >= 8 or self.penalty - self._last_alarm_penalty >= 15:
            self._last_alarm_penalty = self.penalty
            self._alarm()
        return True

    def _alarm(self):
        if not (self.enable_alarm and _HAS_WINSOUND):
            return
        path = os.path.join(MODELS, "alarm.wav")

        def _play():
            try:
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                pass
        threading.Thread(target=_play, daemon=True).start()

    def _window_hit(self, key, present):
        h = self._hist.setdefault(key, deque(maxlen=WINDOW))
        h.append(bool(present))
        return sum(h) >= HITS_NEEDED

    # ------------------------------------------------------------------ gaze
    @staticmethod
    def _metrics(lm):
        hx = (_ratio(lm[LEFT_IRIS].x, lm[LEFT_EYE_H[0]].x, lm[LEFT_EYE_H[1]].x) +
              _ratio(lm[RIGHT_IRIS].x, lm[RIGHT_EYE_H[0]].x, lm[RIGHT_EYE_H[1]].x)) / 2
        vy = (_ratio(lm[LEFT_IRIS].y, lm[LEFT_EYE_V[0]].y, lm[LEFT_EYE_V[1]].y) +
              _ratio(lm[RIGHT_IRIS].y, lm[RIGHT_EYE_V[0]].y, lm[RIGHT_EYE_V[1]].y)) / 2
        yaw = _ratio(lm[NOSE].x, lm[FACE_L].x, lm[FACE_R].x)
        pitch = _ratio(lm[NOSE].y, lm[FOREHEAD].y, lm[CHIN].y)
        return hx, vy, yaw, pitch

    def _decide(self, m):
        hx, vy, yaw, pitch = m
        nhx, nvy, nyaw, npitch = self.neutral
        dev = {
            "Eyes sideways": abs(hx - nhx) / self.tol_eye,
            "Eyes up/down": abs(vy - nvy) / (self.tol_eye * 1.6),   # blinks are noisy
            "Head turned": abs(yaw - nyaw) / self.tol_yaw,
            "Head tilted": abs(pitch - npitch) / self.tol_pitch,
        }
        worst = max(dev, key=dev.get)
        return (dev[worst] > 1.0), worst

    # ------------------------------------------------ step 1: face (fast thread)
    def process_face(self, frame):
        """frame: BGR, mirrored. Runs MediaPipe and updates gaze state."""
        t = time.perf_counter()
        h, w = frame.shape[:2]
        ts = int((time.monotonic() - self._t0) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)   # no lock held here
        faces = [f for f in (result.face_landmarks or []) if len(f) > RIGHT_IRIS]
        self.face_ms = 0.8 * self.face_ms + 0.2 * (time.perf_counter() - t) * 1000

        with self.lock:
            if self.monitoring:
                self._seen_frames += 1
            color = (90, 200, 90)
            if faces:
                self.no_face_start = None
                lm = faces[0]
                m = self._metrics(lm)
                self.iris = [(int(lm[i].x * w), int(lm[i].y * h)) for i in (LEFT_IRIS, RIGHT_IRIS)]
                if len(faces) > 1:
                    self._streak["faces2"] = self._streak.get("faces2", 0) + 1
                    if self._streak["faces2"] >= 4:
                        self._log("Multiple people", frame)
                else:
                    self._streak["faces2"] = 0

                if not self.calibrated:
                    self._calib.append(m)
                    self.status = f"Calibrating {len(self._calib)}/{CALIB_FRAMES}"
                    color = (255, 200, 60)
                    if len(self._calib) >= CALIB_FRAMES:
                        self.neutral = tuple(sum(v) / len(self._calib) for v in zip(*self._calib))
                        self.calibrated = True
                    self.look_away_start = None
                else:
                    away, label = self._decide(m)
                    if away:
                        self.status = label
                        color = (60, 170, 255)
                        if self.look_away_start is None:
                            self.look_away_start = time.time()
                        elif time.time() - self.look_away_start > self.look_away_seconds:
                            color = (60, 60, 255)
                            self._log("Looking away", frame)
                    else:
                        self.status = "Focused"
                        self.look_away_start = None
                        if self.monitoring:
                            self._attn_frames += 1
            else:
                self.iris = []
                self.look_away_start = None
                self.status = "Face not visible"
                color = (60, 60, 255)
                if self.no_face_start is None:
                    self.no_face_start = time.time()
                elif time.time() - self.no_face_start > self.no_face_seconds:
                    self._log("Face not visible", frame)
            self.status_color = color

    # --------------------------------------------- step 2: objects (own thread)
    def process_objects(self, frame):
        t = time.perf_counter()
        ids = [COCO_PERSON] + list(YOLO_CLASSES)
        min_conf = min(c for _, c in YOLO_CLASSES.values()) * min(1.0, self.object_scale)
        results = self.yolo(frame, conf=min_conf, classes=ids, verbose=False, imgsz=YOLO_SIZE)
        self.yolo_ms = 0.8 * self.yolo_ms + 0.2 * (time.perf_counter() - t) * 1000
        area_total = frame.shape[0] * frame.shape[1]
        persons, seen, boxes, objs = 0, set(), [], []
        for r in results:
            for b in r.boxes:
                cid, conf = int(b.cls[0]), float(b.conf[0])
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                area = (x2 - x1) * (y2 - y1) / area_total
                if cid == COCO_PERSON:
                    if conf >= self.person_conf and area > 0.05:
                        persons += 1
                    continue
                name, need = YOLO_CLASSES[cid]
                if conf >= need * self.object_scale and 0.002 < area < 0.6:
                    seen.add(name)
                    short = name.split()[0]
                    boxes.append((x1, y1, x2, y2, f"{short} {conf:.0%}"))
                    objs.append(f"{short} {conf:.0%}")
        with self.lock:
            now = time.time()
            if boxes:
                self._boxes, self.last_objects, self._box_until = boxes, objs, now + 0.6
            elif now > self._box_until:
                self._boxes, self.last_objects = [], []
            if self._window_hit("persons2", persons > 1):
                self._log("Multiple people", frame)
            for name in {n for n, _ in YOLO_CLASSES.values()}:
                if self._window_hit(name, name in seen):
                    self._log(name, frame)

    # ------------------------------------------------------- overlays for video
    def draw(self, frame):
        with self.lock:
            boxes, iris = list(self._boxes), list(self.iris)
            status, color, integ = self.status, self.status_color, self.integrity
        for p in iris:
            cv2.circle(frame, p, 3, (80, 255, 120), -1)
        for (x1, y1, x2, y2, label) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 255), 3)
            cv2.putText(frame, label, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (60, 60, 255), 2)
        w = frame.shape[1]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 34), (20, 20, 24), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.circle(frame, (16, 17), 6, color, -1)
        cv2.putText(frame, status, (30, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (240, 240, 240), 1, cv2.LINE_AA)
        txt = f"Integrity {integ}"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(frame, txt, (w - tw - 12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (240, 240, 240), 1, cv2.LINE_AA)
        return frame

    # backwards-compatible single call (used by tests)
    def process_frame(self, frame):
        self.process_face(frame)
        self.process_objects(frame)
        return self.draw(frame)