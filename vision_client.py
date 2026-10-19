"""
vision_client.py  -  Starts vision_worker.py in the background and talks to it.

The app uses one Vision object for everything camera-related. It's a thin
remote control: every call is a quick local HTTP request.
"""

import atexit
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "vision_worker.log")


def _free_port(start=8502, end=8540):
    for p in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("No free port for the camera worker (8502-8539)")


class Vision:
    def __init__(self, timeout=180):
        self.port = _free_port()
        flags = 0
        if sys.platform.startswith("win"):
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._log = open(LOG, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "vision_worker.py"),
             "--port", str(self.port), "--parent", str(os.getpid())],
            cwd=HERE, stdout=self._log, stderr=subprocess.STDOUT, creationflags=flags)
        atexit.register(self.close)
        t0 = time.time()
        while time.time() - t0 < timeout:          # loading the AI models takes ~10-30 s
            if self.proc.poll() is not None:
                raise RuntimeError("The camera worker crashed while starting. "
                                   f"Details are in {LOG}")
            try:
                if self.cmd("ping", timeout=1).get("ok"):
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError(f"The camera worker did not start in time. See {LOG}")
        self.cam_index = None
        self._snap = {}

    # ------------------------------------------------------------- plumbing
    def _get(self, path, timeout=3):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=timeout) as r:
            return json.loads(r.read().decode())

    def cmd(self, a, timeout=3, **kw):
        q = "&".join([f"a={a}"] + [f"{k}={v}" for k, v in kw.items()])
        return self._get(f"/cmd?{q}", timeout=timeout)

    def close(self):
        try:
            self.cmd("quit", timeout=1)
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass

    @property
    def alive(self):
        return self.proc.poll() is None

    # --------------------------------------------------------------- camera
    def start(self, index=0):
        r = self.cmd("start", timeout=15, i=int(index))
        self.cam_index = int(index)
        return r.get("ok"), r.get("error")

    def stop(self):
        try:
            self.cmd("stop", timeout=5)
        except Exception:
            pass

    # -------------------------------------------------------------- proctor
    def snapshot(self, full=True):
        try:
            self._snap = self._get("/state?full=1" if full else "/state", timeout=3)
        except Exception:
            pass
        return self._snap

    def reset(self):
        self.cmd("reset")

    def recalibrate(self):
        self.cmd("recalibrate")

    def monitor(self, on):
        self.cmd("monitor", on=1 if on else 0)

    def configure(self, sens, delay, obj, alarm):
        self.cmd("set", sens=sens, delay=delay, obj=obj, alarm=1 if alarm else 0)