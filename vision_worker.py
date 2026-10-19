"""
vision_worker.py  -  The camera brain, running as its OWN program (process).

Why a separate process?  Face tracking + phone detection are heavy. When they
ran inside the Streamlit app they used up its processing time on slower laptops,
so the interview screen froze (buttons did nothing, "reading the question"
never ended). In a separate process they can't slow the interview screen down.

app.py starts this automatically - you never run it by hand.
It talks to the app over http://127.0.0.1:<port>:
    /stream   live video (MJPEG)       /state   live status (JSON)
    /cmd?a=start&i=0 | stop | reset | recalibrate | monitor&on=1 | set&... | ping | quit
"""

import argparse
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def lower_priority():
    """Let the browser + interview screen win when the CPU is busy."""
    try:
        import psutil
        p = psutil.Process()
        if sys.platform.startswith("win"):
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(5)
    except Exception:
        pass


def watch_parent(parent_pid):
    """Exit when the Streamlit app that started us is closed."""
    try:
        import psutil
    except Exception:
        return
    while True:
        time.sleep(2)
        if not psutil.pid_exists(parent_pid):
            os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--parent", type=int, default=0)
    a = ap.parse_args()

    lower_priority()
    if a.parent:
        threading.Thread(target=watch_parent, args=(a.parent,), daemon=True).start()

    from engine import Proctor
    from devices import Camera

    proctor = Proctor()
    state = {"cam": None}

    def on_cmd(q):
        cmd = q.get("a", "")
        cam = state["cam"]
        if cmd == "ping":
            return {"ok": True, "pid": os.getpid()}
        if cmd == "start":
            ok = cam.start(int(q.get("i", 0)))
            return {"ok": ok, "error": cam.error}
        if cmd == "stop":
            cam.stop()
            return {"ok": True}
        if cmd == "reset":
            proctor.reset()
            return {"ok": True}
        if cmd == "recalibrate":
            proctor.recalibrate()
            return {"ok": True}
        if cmd == "monitor":
            with proctor.lock:
                proctor.monitoring = q.get("on") == "1"
            return {"ok": True}
        if cmd == "set":
            if "sens" in q:
                proctor.set_sensitivity(q["sens"])
            if "delay" in q:
                proctor.look_away_seconds = float(q["delay"])
            if "obj" in q:
                proctor.object_scale = float(q["obj"])
            if "alarm" in q:
                proctor.enable_alarm = q["alarm"] == "1"
            return {"ok": True}
        if cmd == "quit":
            threading.Thread(target=lambda: (time.sleep(0.2), os._exit(0)), daemon=True).start()
            return {"ok": True}
        return {"ok": False, "error": f"unknown command {cmd}"}

    cam = Camera(proctor, port=a.port, on_cmd=on_cmd)
    state["cam"] = cam
    if not cam.port:
        print(f"vision_worker: port {a.port} is busy", flush=True)
        sys.exit(2)
    print(f"vision_worker ready on port {cam.port}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()