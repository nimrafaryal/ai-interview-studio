"""
devices.py  -  Camera (+ the old desktop microphone recorder, kept as a fallback).

Because the app and the candidate are on the SAME laptop, we read the webcam
and the mic directly (OpenCV + sounddevice). No WebRTC, no browser permission
dance, and proctoring keeps running no matter what the UI is doing.
"""

import os
import sys
import time
import wave
import threading

import cv2
import numpy as np

try:
    import sounddevice as sd
    HAS_AUDIO = True
except Exception:          # PortAudio missing etc.
    sd = None
    HAS_AUDIO = False

SR = 16000
SILENCE_LEVEL = 0.02       # peak below this = nothing was said


# =========================================================================
# CAMERA  (3 threads + a tiny live-video server)
# =========================================================================
#   capture thread : grabs frames as fast as the webcam gives them, keeps ONLY the newest
#   face thread    : gaze / face check on the newest frame (~15 ms) + draws the video
#   object thread  : YOLO phone/book/person check on the newest frame (~60-150 ms)
#   http server    : streams the video (MJPEG) and the live status (JSON) to the browser,
#                    so the page shows smooth video without Streamlit refreshing anything.
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Camera:
    def __init__(self, proctor, port=None, on_cmd=None, max_fps=15, yolo_gap=0.12):
        self.proctor = proctor
        self.on_cmd = on_cmd            # handles /cmd?... requests (used by vision_worker.py)
        self.max_fps = max_fps          # face check + video: cap so a weak laptop keeps up
        self.yolo_gap = yolo_gap        # min seconds between phone checks (leaves CPU for the UI)
        self._threads = []
        self._stop = threading.Event()
        self._frame, self._frame_id = None, 0
        self._flock = threading.Condition()
        self._jpeg, self._jpeg_id = None, 0
        self._jcond = threading.Condition()
        self.index = None
        self.error = None
        self.fps = 0.0
        self.port = None
        self._server = None
        self._start_server(port)

    # ------------------------------------------------------------- lifecycle
    @property
    def running(self):
        return bool(self._threads) and all(t.is_alive() for t in self._threads)

    def start(self, index=0):
        if self.running and self.index == index:
            return True
        self.stop()
        self.index, self.error = index, None
        self._stop.clear()
        self._ready = threading.Event()
        self._threads = [threading.Thread(target=f, daemon=True)
                         for f in (self._capture_loop, self._face_loop, self._object_loop)]
        for t in self._threads:
            t.start()
        self._ready.wait(timeout=8)
        return self.running and self.error is None

    def stop(self):
        self._stop.set()
        with self._flock:
            self._flock.notify_all()
        for t in self._threads:
            t.join(timeout=3)
        self._threads = []
        with self._jcond:
            self._jpeg = None

    def _open(self, index):
        if sys.platform.startswith("win"):
            for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
                cap = cv2.VideoCapture(index, api)
                if cap.isOpened():
                    return cap
        return cv2.VideoCapture(index)

    # --------------------------------------------------------------- threads
    def _capture_loop(self):
        cap = self._open(self.index)
        if not cap.isOpened():
            self.error = (f"Could not open camera {self.index}. Close other apps using it "
                          "(Zoom, Teams, Camera) or try another camera number.")
            self._ready.set()
            self._stop.set()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # never hand us old, queued frames
        cap.set(cv2.CAP_PROP_FPS, 30)
        self._ready.set()
        fails = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if fails > 50:
                        self.error = "Camera stopped sending frames."
                        break
                    time.sleep(0.02)
                    continue
                fails = 0
                frame = cv2.flip(frame, 1)
                with self._flock:
                    self._frame, self._frame_id = frame, self._frame_id + 1
                    self._flock.notify_all()
        finally:
            cap.release()
            self._stop.set()

    def _latest(self, last_id, timeout=0.5):
        with self._flock:
            if self._frame_id == last_id:
                self._flock.wait(timeout)
            return self._frame, self._frame_id

    def _face_loop(self):
        last, t_prev = -1, time.time()
        while not self._stop.is_set():
            frame, fid = self._latest(last)
            if frame is None or fid == last:
                continue
            last = fid
            wait = 1.0 / self.max_fps - (time.time() - t_prev)
            if wait > 0:
                time.sleep(wait)
            try:
                self.proctor.process_face(frame.copy())
                shown = self.proctor.draw(frame.copy())
            except Exception as e:      # never let one bad frame kill the loop
                self.error = f"Proctor error: {e}"
                shown = frame
            ok, buf = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ok:
                with self._jcond:
                    self._jpeg, self._jpeg_id = buf.tobytes(), self._jpeg_id + 1
                    self._jcond.notify_all()
            now = time.time()
            self.fps = 0.85 * self.fps + 0.15 * (1.0 / max(now - t_prev, 1e-3))
            t_prev = now

    def _object_loop(self):
        last = -1
        while not self._stop.is_set():
            frame, fid = self._latest(last)
            if frame is None or fid == last:
                continue
            last = fid
            t = time.time()
            try:
                self.proctor.process_objects(frame.copy())
                rest = self.yolo_gap - (time.time() - t)
                if rest > 0:
                    time.sleep(rest)
            except Exception as e:
                self.error = f"Object detector error: {e}"
                time.sleep(0.5)

    # ----------------------------------------------------------------- output
    def jpeg(self):
        with self._jcond:
            return self._jpeg

    def wait_jpeg(self, last_id, timeout=1.0):
        with self._jcond:
            if self._jpeg_id == last_id:
                self._jcond.wait(timeout)
            return self._jpeg, self._jpeg_id

    # ------------------------------------------------------------ http server
    def _start_server(self, fixed_port=None):
        cam = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _cors(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Cache-Control", "no-store")

            def do_OPTIONS(self):
                self.send_response(204)
                self._cors()
                self.send_header("Access-Control-Allow-Methods", "GET")
                self.end_headers()

            def do_GET(self):
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self._cors()
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    last = -1
                    try:
                        while True:
                            jpg, jid = cam.wait_jpeg(last)
                            if jpg is None or jid == last:
                                time.sleep(0.05)
                                continue
                            last = jid
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    except Exception:
                        return
                elif self.path.startswith("/cmd"):
                    from urllib.parse import urlparse, parse_qs
                    args = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                    try:
                        out = cam.on_cmd(args) if cam.on_cmd else {"ok": False}
                    except Exception as e:
                        out = {"ok": False, "error": str(e)}
                    body = json.dumps(out).encode()
                    try:
                        self.send_response(200)
                        self._cors()
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(body)
                    except Exception:
                        return
                elif self.path.startswith("/state"):
                    s = cam.proctor.snapshot()
                    if "full=1" not in self.path:
                        s["events"] = [{k: e[k] for k in ("time", "event", "points")} for e in s["events"][-8:]] \
                            + [{"n": len(s["events"])}]
                    s["camera_ok"] = bool(cam.running and cam._jpeg is not None and not cam.error)
                    s["camera_running"] = cam.running
                    s["camera_error"] = cam.error
                    s["fps"] = round(cam.fps)
                    body = json.dumps(s).encode()
                    try:
                        self.send_response(200)
                        self._cors()
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(body)
                    except Exception:      # browser closed the connection - harmless
                        return
                else:
                    self.send_response(404)
                    self.end_headers()

        for port in ([fixed_port] if fixed_port else range(8502, 8520)):
            try:
                srv = ThreadingHTTPServer(("127.0.0.1", port), H)
                srv.daemon_threads = True
                break
            except OSError:
                continue
        else:
            return
        self._server, self.port = srv, port
        threading.Thread(target=srv.serve_forever, daemon=True).start()


# =========================================================================
# MICROPHONE
# =========================================================================
HOSTAPI_RANK = {"MME": 0, "Windows WASAPI": 1, "Windows DirectSound": 2}   # WDM-KS is skipped


def list_mics():
    """One entry per real microphone, using the most reliable Windows driver.
    (The same mic shows up 3-4 times in sounddevice; the 'WDM-KS' copy is the
    one that produces garbled audio, so we never offer it.)"""
    if not HAS_AUDIO:
        return []
    best = {}
    try:
        apis = sd.query_hostapis()
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) <= 0:
                continue
            api = apis[d["hostapi"]]["name"]
            if "WDM-KS" in api:
                continue
            name = d["name"].strip()
            if "mapper" in name.lower() or "primary sound capture" in name.lower():
                continue
            key = name[:28].lower()          # MME truncates names to ~31 chars
            rank = HOSTAPI_RANK.get(api, 5)
            if key not in best or rank < best[key][0]:
                best[key] = (rank, i, name)
    except Exception:
        pass
    return [(i, name) for _, i, name in sorted(best.values(), key=lambda t: t[1])]


def default_mic():
    try:
        return sd.default.device[0]
    except Exception:
        return None


def write_wav(path, data, sr=SR):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.astype(np.int16).tobytes())


def wav_peak(path):
    try:
        with wave.open(path, "rb") as wf:
            d = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return float(np.abs(d).max()) / 32768.0 if d.size else 0.0
    except Exception:
        return 0.0


def clean_audio(x, sr_in, sr_out=SR):
    """float32 mono at sr_in -> clean int16 at 16 kHz (what Whisper likes)."""
    x = x.astype(np.float32)
    x = x - x.mean()                                   # remove DC offset
    if sr_in != sr_out:
        try:
            from math import gcd
            from scipy.signal import resample_poly      # proper anti-aliased resampling
            g = gcd(int(sr_in), int(sr_out))
            x = resample_poly(x, sr_out // g, int(sr_in) // g).astype(np.float32)
        except Exception:
            n = int(len(x) * sr_out / sr_in)
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    try:                                               # cut low rumble / fan hum below 80 Hz
        from scipy.signal import butter, sosfilt
        x = sosfilt(butter(4, 80, "highpass", fs=sr_out, output="sos"), x).astype(np.float32)
    except Exception:
        pass
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak > 1e-4:
        x = x * min(0.9 / peak, 20.0)                  # normalise volume (quiet mics get louder)
    return np.clip(x * 32767, -32768, 32767).astype(np.int16), peak


class Recorder:
    """Non-blocking mic recorder with a live level meter.

    Records at the microphone's OWN sample rate (usually 44.1 or 48 kHz) and
    converts to 16 kHz afterwards. Forcing 16 kHz on Windows is what made the
    old recordings sound garbled/robotic."""

    def __init__(self):
        self._stream = None
        self._chunks = []
        self._lock = threading.Lock()
        self.level = 0.0
        self.device = None
        self.sr = SR
        self.overflows = 0
        self.started_at = None

    @property
    def recording(self):
        return self._stream is not None

    def _cb(self, indata, frames, t, status):
        if status and status.input_overflow:
            self.overflows += 1
        with self._lock:
            self._chunks.append(indata[:, 0].copy())
        pk = float(np.abs(indata).max())
        self.level = max(pk, self.level * 0.85)   # smooth decay for the meter

    def start(self, device=None):
        if not HAS_AUDIO:
            raise RuntimeError("sounddevice is not installed")
        self.stop(None)
        self._chunks, self.level, self.overflows = [], 0.0, 0
        try:
            self.sr = int(sd.query_devices(device, "input")["default_samplerate"]) or 48000
        except Exception:
            self.sr = 48000
        self._stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                      device=device, callback=self._cb,
                                      blocksize=int(self.sr * 0.05), latency="high")
        self._stream.start()
        self.device = device
        self.started_at = time.time()

    def stop(self, path):
        """Stop and (optionally) save. Returns (path or None, peak level 0-1)."""
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        with self._lock:
            chunks, self._chunks = self._chunks, []
        self.level = 0.0
        if not chunks or path is None:
            return None, 0.0
        raw = np.concatenate(chunks)
        data, peak = clean_audio(raw, self.sr)
        write_wav(path, data)
        return path, peak