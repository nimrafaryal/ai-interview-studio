# """
# engine.py  —  Proctoring core

# Fixes over the previous version:
#   * False positives (phantom person/phone): higher confidence + a detection must
#     persist for N consecutive YOLO runs before it's logged.
#   * Water bottle / notes now actually watched (they were never in the list before).
#   * Gaze: calibrated, multi-axis attention (eyes sideways, eyes up/down, head turn)
#     with a short, tunable delay instead of horizontal-only + a long 2s wait.

# The Proctor object is created once and stored in st.session_state.
# """

# import os
# import time
# import threading

# import cv2
# import mediapipe as mp
# from ultralytics import YOLO

# try:
#     import winsound
#     _HAS_WINSOUND = True
# except ImportError:
#     _HAS_WINSOUND = False


# # --- Landmark indices (MediaPipe FaceMesh) ------------------------------
# LEFT_IRIS, RIGHT_IRIS = 468, 473
# LEFT_EYE_H, RIGHT_EYE_H = (33, 133), (362, 263)   # horizontal corners
# LEFT_EYE_V, RIGHT_EYE_V = (159, 145), (386, 374)  # top / bottom lids
# NOSE, FACE_L, FACE_R, FOREHEAD, CHIN = 1, 234, 454, 10, 152

# # --- Scoring ------------------------------------------------------------
# SCORE_LOOK_AWAY = 5
# SCORE_NO_FACE = 8
# SCORE_PHONE = 15
# SCORE_BOTTLE = 8
# SCORE_BOOK = 8
# SCORE_MULTI_PERSON = 20

# # COCO class id -> (event name, points).  Only these object classes are used.
# WATCH = {
#     67: ("Phone Detected", SCORE_PHONE),        # cell phone
#     39: ("Bottle Detected", SCORE_BOTTLE),      # bottle
#     73: ("Book/Notes Detected", SCORE_BOOK),    # book
# }
# COCO_PERSON = 0

# # --- Timing / thresholds (defaults; several are tunable at runtime) ------
# NO_FACE_SECONDS = 1.2
# EVENT_COOLDOWN = 3.0
# YOLO_EVERY_N = 5
# ALARM_DELTA = 15
# CALIB_FRAMES = 20            # frames of "looking at screen" used to calibrate
# DET_STREAK_NEEDED = 2        # object must appear in this many YOLO runs in a row
# MAX_EVENTS = 500


# def _ratio(v, a, b):
#     """Position of v between a and b, order-independent, clamped to [0,1]."""
#     lo, hi = (a, b) if a <= b else (b, a)
#     if hi - lo < 1e-6:
#         return 0.5
#     return max(0.0, min(1.0, (v - lo) / (hi - lo)))


# class Proctor:
#     def __init__(self, yolo_model="yolov8n.pt",
#                  face_model="face_landmarker.task", enable_alarm=True):
#         self.enable_alarm = enable_alarm

#         base = mp.tasks.BaseOptions(model_asset_path=face_model)
#         opts = mp.tasks.vision.FaceLandmarkerOptions(
#             base_options=base,
#             running_mode=mp.tasks.vision.RunningMode.VIDEO,
#             num_faces=1,
#         )
#         self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
#         self.yolo = YOLO(yolo_model)

#         os.makedirs("snapshots", exist_ok=True)

#         # runtime-tunable settings
#         self.look_away_seconds = 0.7
#         self.object_conf = 0.40      # YOLO confidence floor for objects
#         self.person_conf = 0.60      # stricter bar for counting a person
#         self.set_sensitivity("Normal")

#         self.reset()

#     # -------------------------------------------------------------------
#     def set_sensitivity(self, level):
#         mult = {"Tight": 0.7, "Normal": 1.0, "Loose": 1.4}.get(level, 1.0)
#         self.tol_eye = 0.10 * mult
#         self.tol_yaw = 0.06 * mult
#         self.tol_pitch = 0.06 * mult

#     def reset(self):
#         self.score = 0
#         self.events = []
#         self.frame_count = 0
#         self.timestamp_ms = 0
#         self.last_alarm_score = 0
#         self.look_away_start = None
#         self.no_face_start = None
#         self._last_logged = {}
#         self._streak = {}
#         self.recalibrate()

#     def recalibrate(self):
#         self.calibrated = False
#         self._calib = []
#         self.neutral = (0.5, 0.5, 0.5, 0.5)   # fallback if never calibrated

#     # -------------------------------------------------------------------
#     def _log(self, name, points):
#         now = time.time()
#         if now - self._last_logged.get(name, 0) < EVENT_COOLDOWN:
#             return False
#         self._last_logged[name] = now
#         self.score += points
#         self.events.append((name, self.score, time.strftime("%H:%M:%S")))
#         if len(self.events) > MAX_EVENTS:
#             self.events = self.events[-MAX_EVENTS:]
#         return True

#     def _alarm(self):
#         if not (self.enable_alarm and _HAS_WINSOUND):
#             return
#         def _play():
#             try:
#                 winsound.PlaySound("alarm.wav", winsound.SND_FILENAME)
#             except Exception:
#                 pass
#         threading.Thread(target=_play, daemon=True).start()

#     def _streak_bump(self, key, present):
#         self._streak[key] = self._streak.get(key, 0) + 1 if present else 0
#         return self._streak[key]

#     # -------------------------------------------------------------------
#     def _metrics(self, lm):
#         """Four attention numbers, each ~0.5 when centred."""
#         hx = (_ratio(lm[LEFT_IRIS].x,  lm[LEFT_EYE_H[0]].x,  lm[LEFT_EYE_H[1]].x) +
#               _ratio(lm[RIGHT_IRIS].x, lm[RIGHT_EYE_H[0]].x, lm[RIGHT_EYE_H[1]].x)) / 2
#         vy = (_ratio(lm[LEFT_IRIS].y,  lm[LEFT_EYE_V[0]].y,  lm[LEFT_EYE_V[1]].y) +
#               _ratio(lm[RIGHT_IRIS].y, lm[RIGHT_EYE_V[0]].y, lm[RIGHT_EYE_V[1]].y)) / 2
#         yaw = _ratio(lm[NOSE].x, lm[FACE_L].x, lm[FACE_R].x)
#         pitch = _ratio(lm[NOSE].y, lm[FOREHEAD].y, lm[CHIN].y)
#         return hx, vy, yaw, pitch

#     def _decide(self, m):
#         hx, vy, yaw, pitch = m
#         nhx, nvy, nyaw, npitch = self.neutral
#         dev = {
#             "EYES SIDEWAYS": abs(hx - nhx) / self.tol_eye,
#             "EYES UP/DOWN":  abs(vy - nvy) / self.tol_eye,
#             "HEAD TURNED":   abs(yaw - nyaw) / self.tol_yaw,
#             "HEAD TILT":     abs(pitch - npitch) / self.tol_pitch,
#         }
#         worst = max(dev, key=dev.get)
#         if dev[worst] > 1.0:
#             return True, worst
#         return False, "CENTER"

#     # -------------------------------------------------------------------
#     def process_frame(self, frame):
#         self.frame_count += 1
#         self.timestamp_ms += 33
#         h, w = frame.shape[:2]

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#         result = self.landmarker.detect_for_video(mp_img, self.timestamp_ms)

#         status = "NO FACE"

#         if result.face_landmarks and len(result.face_landmarks[0]) > RIGHT_IRIS:
#             lm = result.face_landmarks[0]
#             self.no_face_start = None
#             m = self._metrics(lm)

#             for idx in (LEFT_IRIS, RIGHT_IRIS):
#                 cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)),
#                            3, (0, 255, 0), -1)

#             if not self.calibrated:
#                 self._calib.append(m)
#                 status = f"CALIBRATING {len(self._calib)}/{CALIB_FRAMES}"
#                 if len(self._calib) >= CALIB_FRAMES:
#                     self.neutral = tuple(sum(v) / len(self._calib)
#                                          for v in zip(*self._calib))
#                     self.calibrated = True
#                 self.look_away_start = None
#             else:
#                 away, label = self._decide(m)
#                 status = label
#                 if away:
#                     if self.look_away_start is None:
#                         self.look_away_start = time.time()
#                     elif time.time() - self.look_away_start > self.look_away_seconds:
#                         self._log("Looking Away", SCORE_LOOK_AWAY)
#                 else:
#                     self.look_away_start = None
#         else:
#             self.look_away_start = None
#             if self.no_face_start is None:
#                 self.no_face_start = time.time()
#             elif time.time() - self.no_face_start > NO_FACE_SECONDS:
#                 self._log("No Face Detected", SCORE_NO_FACE)

#         if self.frame_count % YOLO_EVERY_N == 0:
#             self._run_yolo(frame)

#         if self.score - self.last_alarm_score >= ALARM_DELTA:
#             self._alarm()
#             self.last_alarm_score = self.score

#         self._draw_hud(frame, status)
#         return frame, self.score, self.events

#     # -------------------------------------------------------------------
#     def _run_yolo(self, frame):
#         ids = [COCO_PERSON] + list(WATCH.keys())
#         results = self.yolo(frame, conf=self.object_conf, classes=ids, verbose=False)

#         area_total = frame.shape[0] * frame.shape[1]
#         person_count = 0
#         seen = set()

#         for r in results:
#             for b in r.boxes:
#                 cid = int(b.cls[0])
#                 conf = float(b.conf[0])
#                 x1, y1, x2, y2 = map(int, b.xyxy[0])
#                 area = (x2 - x1) * (y2 - y1) / area_total

#                 if cid == COCO_PERSON:
#                     if conf >= self.person_conf and area > 0.03:
#                         person_count += 1
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
#                     continue

#                 if cid in WATCH and 0.004 < area < 0.6:
#                     seen.add(cid)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, f"{self.yolo.names[cid]} {conf:.2f}",
#                                 (x1, max(y1 - 6, 12)),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

#         # temporal consistency: only fire after a short streak of detections
#         if self._streak_bump("multi_person", person_count > 1) >= DET_STREAK_NEEDED:
#             self._log("Multiple Persons", SCORE_MULTI_PERSON)

#         for cid, (name, pts) in WATCH.items():
#             if self._streak_bump(cid, cid in seen) >= DET_STREAK_NEEDED:
#                 if self._log(name, pts):
#                     cv2.imwrite(f"snapshots/{name.split()[0].lower()}_"
#                                 f"{int(time.time())}.jpg", frame)

#     # -------------------------------------------------------------------
#     def _draw_hud(self, frame, status):
#         if self.score < 20:
#             risk, color = "LOW", (0, 200, 0)
#         elif self.score < 50:
#             risk, color = "MEDIUM", (0, 200, 255)
#         else:
#             risk, color = "HIGH", (0, 0, 255)
#         cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
#         cv2.putText(frame, f"{status}  |  Score: {self.score}  |  {risk} RISK",
#                     (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

"""
engine.py  —  Proctoring core

Fixes over the previous version:
  * False positives (phantom person/phone): higher confidence + a detection must
    persist for N consecutive YOLO runs before it's logged.
  * Water bottle / notes now actually watched (they were never in the list before).
  * Gaze: calibrated, multi-axis attention (eyes sideways, eyes up/down, head turn)
    with a short, tunable delay instead of horizontal-only + a long 2s wait.

The Proctor object is created once and stored in st.session_state.
"""

import os
import time
import threading

import cv2
import mediapipe as mp
from ultralytics import YOLO

try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False


# --- Landmark indices (MediaPipe FaceMesh) ------------------------------
LEFT_IRIS, RIGHT_IRIS = 468, 473
LEFT_EYE_H, RIGHT_EYE_H = (33, 133), (362, 263)   # horizontal corners
LEFT_EYE_V, RIGHT_EYE_V = (159, 145), (386, 374)  # top / bottom lids
NOSE, FACE_L, FACE_R, FOREHEAD, CHIN = 1, 234, 454, 10, 152

# --- Scoring ------------------------------------------------------------
SCORE_LOOK_AWAY = 5
SCORE_NO_FACE = 8
SCORE_PHONE = 15
SCORE_BOTTLE = 8
SCORE_BOOK = 8
SCORE_MULTI_PERSON = 20

# COCO class id -> (event name, points).  Only these object classes are used.
WATCH = {
    67: ("Phone Detected", SCORE_PHONE),        # cell phone
    39: ("Bottle Detected", SCORE_BOTTLE),      # bottle
    73: ("Book/Notes Detected", SCORE_BOOK),    # book
}
COCO_PERSON = 0

# --- Timing / thresholds (defaults; several are tunable at runtime) ------
NO_FACE_SECONDS = 1.2
EVENT_COOLDOWN = 3.0
YOLO_EVERY_N = 5
ALARM_DELTA = 15
CALIB_FRAMES = 20            # frames of "looking at screen" used to calibrate
DET_STREAK_NEEDED = 2        # object must appear in this many YOLO runs in a row
MAX_EVENTS = 500


def _ratio(v, a, b):
    """Position of v between a and b, order-independent, clamped to [0,1]."""
    lo, hi = (a, b) if a <= b else (b, a)
    if hi - lo < 1e-6:
        return 0.5
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


class Proctor:
    def __init__(self, yolo_model="yolov8n.pt",
                 face_model="face_landmarker.task", enable_alarm=True):
        self.enable_alarm = enable_alarm

        base = mp.tasks.BaseOptions(model_asset_path=face_model)
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
        self.yolo = YOLO(yolo_model)

        os.makedirs("snapshots", exist_ok=True)

        # runtime-tunable settings
        self.look_away_seconds = 0.7
        self.object_conf = 0.40      # YOLO confidence floor for objects
        self.person_conf = 0.60      # stricter bar for counting a person
        self.set_sensitivity("Normal")

        self.reset()

    # -------------------------------------------------------------------
    def set_sensitivity(self, level):
        mult = {"Tight": 0.7, "Normal": 1.0, "Loose": 1.4}.get(level, 1.0)
        self.tol_eye = 0.10 * mult
        self.tol_yaw = 0.06 * mult
        self.tol_pitch = 0.06 * mult

    def reset(self):
        self.score = 0
        self.events = []
        self.frame_count = 0
        self.timestamp_ms = 0
        self.last_alarm_score = 0
        self.look_away_start = None
        self.no_face_start = None
        self._last_logged = {}
        self._streak = {}
        self.recalibrate()

    def recalibrate(self):
        self.calibrated = False
        self._calib = []
        self.neutral = (0.5, 0.5, 0.5, 0.5)   # fallback if never calibrated

    # -------------------------------------------------------------------
    def _log(self, name, points):
        now = time.time()
        if now - self._last_logged.get(name, 0) < EVENT_COOLDOWN:
            return False
        self._last_logged[name] = now
        self.score += points
        self.events.append((name, self.score, time.strftime("%H:%M:%S")))
        if len(self.events) > MAX_EVENTS:
            self.events = self.events[-MAX_EVENTS:]
        return True

    def _alarm(self):
        if not (self.enable_alarm and _HAS_WINSOUND):
            return
        def _play():
            try:
                winsound.PlaySound("alarm.wav", winsound.SND_FILENAME)
            except Exception:
                pass
        threading.Thread(target=_play, daemon=True).start()

    def _streak_bump(self, key, present):
        self._streak[key] = self._streak.get(key, 0) + 1 if present else 0
        return self._streak[key]

    # -------------------------------------------------------------------
    def _metrics(self, lm):
        """Four attention numbers, each ~0.5 when centred."""
        hx = (_ratio(lm[LEFT_IRIS].x,  lm[LEFT_EYE_H[0]].x,  lm[LEFT_EYE_H[1]].x) +
              _ratio(lm[RIGHT_IRIS].x, lm[RIGHT_EYE_H[0]].x, lm[RIGHT_EYE_H[1]].x)) / 2
        vy = (_ratio(lm[LEFT_IRIS].y,  lm[LEFT_EYE_V[0]].y,  lm[LEFT_EYE_V[1]].y) +
              _ratio(lm[RIGHT_IRIS].y, lm[RIGHT_EYE_V[0]].y, lm[RIGHT_EYE_V[1]].y)) / 2
        yaw = _ratio(lm[NOSE].x, lm[FACE_L].x, lm[FACE_R].x)
        pitch = _ratio(lm[NOSE].y, lm[FOREHEAD].y, lm[CHIN].y)
        return hx, vy, yaw, pitch

    def _decide(self, m):
        hx, vy, yaw, pitch = m
        nhx, nvy, nyaw, npitch = self.neutral
        dev = {
            "EYES SIDEWAYS": abs(hx - nhx) / self.tol_eye,
            "EYES UP/DOWN":  abs(vy - nvy) / self.tol_eye,
            "HEAD TURNED":   abs(yaw - nyaw) / self.tol_yaw,
            "HEAD TILT":     abs(pitch - npitch) / self.tol_pitch,
        }
        worst = max(dev, key=dev.get)
        if dev[worst] > 1.0:
            return True, worst
        return False, "CENTER"

    # -------------------------------------------------------------------
    def process_frame(self, frame):
        self.frame_count += 1
        self.timestamp_ms += 33
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_img, self.timestamp_ms)

        status = "NO FACE"

        if result.face_landmarks and len(result.face_landmarks[0]) > RIGHT_IRIS:
            lm = result.face_landmarks[0]
            self.no_face_start = None
            m = self._metrics(lm)

            for idx in (LEFT_IRIS, RIGHT_IRIS):
                cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)),
                           3, (0, 255, 0), -1)

            if not self.calibrated:
                self._calib.append(m)
                status = f"CALIBRATING {len(self._calib)}/{CALIB_FRAMES}"
                if len(self._calib) >= CALIB_FRAMES:
                    self.neutral = tuple(sum(v) / len(self._calib)
                                         for v in zip(*self._calib))
                    self.calibrated = True
                self.look_away_start = None
            else:
                away, label = self._decide(m)
                status = label
                if away:
                    if self.look_away_start is None:
                        self.look_away_start = time.time()
                    elif time.time() - self.look_away_start > self.look_away_seconds:
                        self._log(label.title(), SCORE_LOOK_AWAY)
                else:
                    self.look_away_start = None
        else:
            self.look_away_start = None
            if self.no_face_start is None:
                self.no_face_start = time.time()
            elif time.time() - self.no_face_start > NO_FACE_SECONDS:
                self._log("No Face Detected", SCORE_NO_FACE)

        if self.frame_count % YOLO_EVERY_N == 0:
            self._run_yolo(frame)

        if self.score - self.last_alarm_score >= ALARM_DELTA:
            self._alarm()
            self.last_alarm_score = self.score

        self._draw_hud(frame, status)
        return frame, self.score, self.events

    # -------------------------------------------------------------------
    def _run_yolo(self, frame):
        ids = [COCO_PERSON] + list(WATCH.keys())
        results = self.yolo(frame, conf=self.object_conf, classes=ids, verbose=False)

        area_total = frame.shape[0] * frame.shape[1]
        person_count = 0
        seen = set()

        for r in results:
            for b in r.boxes:
                cid = int(b.cls[0])
                conf = float(b.conf[0])
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                area = (x2 - x1) * (y2 - y1) / area_total

                if cid == COCO_PERSON:
                    if conf >= self.person_conf and area > 0.03:
                        person_count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
                    continue

                if cid in WATCH and 0.004 < area < 0.6:
                    seen.add(cid)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{self.yolo.names[cid]} {conf:.2f}",
                                (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # temporal consistency: only fire after a short streak of detections
        if self._streak_bump("multi_person", person_count > 1) >= DET_STREAK_NEEDED:
            self._log("Multiple Persons", SCORE_MULTI_PERSON)

        for cid, (name, pts) in WATCH.items():
            if self._streak_bump(cid, cid in seen) >= DET_STREAK_NEEDED:
                if self._log(name, pts):
                    cv2.imwrite(f"snapshots/{name.split()[0].lower()}_"
                                f"{int(time.time())}.jpg", frame)

    # -------------------------------------------------------------------
    def _draw_hud(self, frame, status):
        if self.score < 20:
            risk, color = "LOW", (0, 200, 0)
        elif self.score < 50:
            risk, color = "MEDIUM", (0, 200, 255)
        else:
            risk, color = "HIGH", (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, f"{status}  |  Score: {self.score}  |  {risk} RISK",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)