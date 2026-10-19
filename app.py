"""
AI Interview Studio  -  one app that replaces app.py / interview_app.py /
interview_exam.py / interview_pro.py / interview_f.py

    Run:  streamlit run app.py        (or double-click run.bat on Windows)

Flow:  1 Setup  ->  2 System check  ->  3 Interview  ->  4 Report
"""

import os
import json
import time
import base64
import html as _html
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from devices import Recorder, SILENCE_LEVEL, wav_peak
from report import build_html, score_color

HERE = os.path.dirname(os.path.abspath(__file__))
ANSWERS = os.path.join(HERE, "answers")
REPORTS = os.path.join(HERE, "reports")
SECRETS = os.path.join(HERE, ".streamlit", "secrets.toml")
os.makedirs(ANSWERS, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

st.set_page_config(page_title="AI Interview Studio", page_icon="🎙️", layout="wide",
                   initial_sidebar_state="collapsed")

# =========================================================================
# STYLE
# =========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea {font-family:'Inter',system-ui,sans-serif}
#MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden; height:0}
.block-container {padding-top:1.2rem; padding-bottom:2rem; max-width:1250px}
.card {color:#E8EAF0; background:#171A23; border:1px solid #262A36; border-radius:16px; padding:18px 20px; margin-bottom:14px}
.muted {color:#8A90A2; font-size:.88rem}
.h-title {font-size:1.9rem; font-weight:800; letter-spacing:-.02em; margin:0}
.brand {display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.1rem}
.brand .dot {width:12px; height:12px; border-radius:50%; background:linear-gradient(135deg,#7C5CFF,#22D3EE)}
.steps {display:flex; gap:8px; margin:6px 0 22px}
.step {flex:1; padding:8px 12px; border-radius:10px; background:#141720; border:1px solid #232733;
       color:#6B7186; font-size:.82rem; font-weight:600}
.step.on {border-color:#7C5CFF; color:#fff; background:linear-gradient(135deg,#7C5CFF33,#22D3EE18)}
.step.done {color:#9BE7B4; border-color:#1f5134}
.qcard {color:#F2F4F8; background:linear-gradient(160deg,#1B1F2B,#151821); border:1px solid #2A2F3D; border-radius:20px; padding:28px 30px}
.qmeta {color:#8A90A2; font-size:.85rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em}
.qtext {font-size:1.6rem; font-weight:700; line-height:1.35; margin:12px 0 4px}
.chip {display:inline-block; padding:3px 10px; border-radius:999px; font-size:.75rem; font-weight:700;
       background:#232838; color:#C9CEDC; margin-right:6px}
.live {display:inline-flex; align-items:center; gap:6px; font-size:.8rem; font-weight:700; color:#ff6b6b}
.live:before {content:''; width:8px; height:8px; border-radius:50%; background:#ff4d4d; animation:pulse 1.2s infinite}
@keyframes pulse {0%{opacity:1}50%{opacity:.25}100%{opacity:1}}
.cam img {width:100%; border-radius:14px; border:1px solid #262A36; display:block}
.cam-off {aspect-ratio:4/3; border-radius:14px; border:1px dashed #333a4a; display:flex; align-items:center;
          justify-content:center; color:#6B7186}
.meter {height:8px; border-radius:9px; background:#232733; overflow:hidden}
.meter > div {height:8px; border-radius:9px; transition:width .15s}
.big {font-size:2.4rem; font-weight:800; line-height:1}
.ev {display:flex; justify-content:space-between; font-size:.82rem; padding:5px 0; border-bottom:1px solid #222633}
.ev b {color:#ff8a8a; font-weight:700}
.check {display:flex; gap:10px; align-items:center; padding:9px 0; border-bottom:1px solid #222633}
.check .ic {width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center;
            font-size:.8rem; font-weight:800}
.ok .ic {background:#163d27; color:#5BE38C} .no .ic {background:#3d1c1c; color:#FF7A7A}
.wait .ic {background:#3a3216; color:#F5C451}
.kpi {text-align:left} .kpi .v {font-size:2.3rem; font-weight:800; line-height:1.05} .kpi .l {color:#8A90A2; font-size:.82rem}
.speaking {display:flex; gap:5px; align-items:flex-end; height:34px}
.speaking i {width:7px; border-radius:4px; background:#F5C451; animation:sp 0.9s infinite ease-in-out}
.speaking i:nth-child(2){animation-delay:.15s} .speaking i:nth-child(3){animation-delay:.3s} .speaking i:nth-child(4){animation-delay:.45s}
@keyframes sp {0%,100%{height:8px} 50%{height:34px}}
.rec-badge {display:inline-block; padding:6px 16px; border-radius:999px; font-weight:800;
            background:linear-gradient(135deg,#7C5CFF,#22D3EE); color:#fff}
div[data-testid="stExpander"] {border-radius:14px; border-color:#262A36}
[class*="st-key-autobegin"], [class*="st-key-giveup"] {display:none !important}
[class*="st-key-aihelper"] {position:absolute !important; width:0 !important; height:0 !important; overflow:hidden; opacity:0; pointer-events:none}
iframe[height="1"] {display:block}
</style>
""", unsafe_allow_html=True)


# =========================================================================
# SHARED RESOURCES (created once, survive reruns)
# =========================================================================
@st.cache_resource(show_spinner="Starting the camera AI in the background (first time takes ~20-40 s)...")
def _vision():
    from vision_client import Vision
    return Vision()


def get_vision():
    """Camera + proctoring run in a separate program (vision_worker.py) so they can
    never slow down or freeze the interview screen. Restarted automatically if it dies."""
    v = _vision()
    if not v.alive:
        _vision.clear()
        v = _vision()
    return v


@st.cache_resource
def get_recorder():
    return Recorder()


@st.cache_resource
def get_pool():
    return ThreadPoolExecutor(max_workers=2)


@st.cache_resource
def get_brain(key):
    from ai_brain import Brain
    return Brain(key)


def load_key():
    if st.session_state.get("api_key"):
        return st.session_state.api_key
    try:
        k = st.secrets.get("GROQ_API_KEY", "")
        if k:
            return k
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


def save_key(k):
    os.makedirs(os.path.dirname(SECRETS), exist_ok=True)
    with open(SECRETS, "w", encoding="utf-8") as f:
        f.write(f'GROQ_API_KEY = "{k.strip()}"\n')


DEFAULTS = dict(stage="setup", questions=[], idx=0, futures=[], results=[],
                phase="think", phase_start=0.0, phase_len=0.0, speak_nonce=0,
                started_at=None, summary=None, saved_dir=None, mic_ok=None)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
S = st.session_state


# =========================================================================
# SMALL UI HELPERS
# =========================================================================
def header():
    order = [("setup", "1 · Setup"), ("check", "2 · System check"),
             ("interview", "3 · Interview"), ("report", "4 · Report")]
    cur = [o[0] for o in order].index(S.stage)
    steps = "".join(
        f"<div class='step {'on' if i == cur else 'done' if i < cur else ''}'>"
        f"{'✓ ' if i < cur else ''}{lbl}</div>" for i, (_, lbl) in enumerate(order))
    st.markdown(f"<div class='brand'><span class='dot'></span>AI Interview Studio</div>"
                f"<div class='steps'>{steps}</div>", unsafe_allow_html=True)


def fmt(s):
    s = max(0, int(round(s)))
    return f"{s // 60}:{s % 60:02d}"


def risk_color(integ):
    return "#5BE38C" if integ >= 80 else "#F5C451" if integ >= 50 else "#FF7A7A"


def cam_html(jpeg):
    if not jpeg:
        return "<div class='cam-off'>Camera starting…</div>"
    b64 = base64.b64encode(jpeg).decode()
    return f"<div class='cam'><img src='data:image/jpeg;base64,{b64}'></div>"


def meter(level, color="#22D3EE"):
    w = min(100, level * 300)
    return f"<div class='meter'><div style='width:{w:.0f}%;background:{color}'></div></div>"


def ring(frac, label, color):
    deg = 360 * max(0.0, min(1.0, frac))
    return f"""
    <div style='display:flex;justify-content:center'>
      <div style='width:136px;height:136px;border-radius:50%;
        background:conic-gradient({color} {deg:.0f}deg,#232733 0deg);
        display:flex;align-items:center;justify-content:center'>
        <div style='width:112px;height:112px;border-radius:50%;background:#0E1016;
          display:flex;align-items:center;justify-content:center;
          font-size:1.9rem;font-weight:800;color:#fff'>{label}</div>
      </div>
    </div>"""


def _html_js(code):
    """Run a small invisible script (st.iframe on new Streamlit, components.html on older)."""
    import hashlib
    with st.container(key="aihelper_" + hashlib.md5(code.encode()).hexdigest()[:12]):
        if hasattr(st, "iframe"):
            st.iframe(code, height=1)
        else:
            components.html(code, height=0)


# -------------------------------------------------------------------------
# Browser controller. It is installed ONCE into the main page (not into a helper
# iframe), so its timers survive when Streamlit rebuilds the little iframes.
# Python just sends it idempotent commands; repeated commands are ignored.
# -------------------------------------------------------------------------
CONTROLLER_VERSION = "6"
CONTROLLER_JS = r"""
(function(){
  const W = window;
  if (W.__AI && W.__AI.v === "%V%") return;
  const done = new Set();
  const once = id => { if (done.has(id)) return false; done.add(id); return true; };
  const q = sel => W.document.querySelector(sel);
  const clickKey = k => { const b = q('.st-key-' + k + ' button'); if (b && !b.disabled) { b.click(); return true; } return false; };
  // the recorder that belongs to ONE question (each has its own key) - never touch another one
  const recBtn = (key, lbl) => q('.st-key-' + key + ' [data-testid="stAudioInputActionButton"][aria-label="' + lbl + '"]');
  const retry = (fn, ms, tries) => { let n = 0; const t = W.setInterval(() => { n++; if (fn() || n >= tries) W.clearInterval(t); }, ms); };
  const AI = { v: "%V%" };

  AI.speak = (id, text, beginKey, fallbackMs) => {
    if (!once("speak_" + id)) return;
    let fired = false;
    const go = () => { if (fired) return; fired = true; retry(() => clickKey(beginKey), 150, 40); };
    W.setTimeout(go, fallbackMs);                       // backup if the voice never says "finished"
    try {
      const s = W.speechSynthesis; if (!s) { W.setTimeout(go, 600); return; }
      s.cancel();
      const u = new W.SpeechSynthesisUtterance(text);
      u.rate = 1.0; u.lang = "en-US";
      const vs = s.getVoices().filter(v => v.lang && v.lang.startsWith("en"));
      const pref = vs.find(v => /natural|aria|jenny|google us|zira/i.test(v.name));
      if (pref) u.voice = pref;
      u.onend = () => W.setTimeout(go, 120);
      u.onerror = () => W.setTimeout(go, 120);
      AI._utt = u;                                      // keep a reference (Chrome bug)
      s.speak(u);
      const t0 = Date.now(); let heard = false;         // 2nd check: is the voice still talking?
      const poll = W.setInterval(() => {
        if (fired) { W.clearInterval(poll); return; }
        if (s.speaking) heard = true;
        const idle = !s.speaking && !s.pending;
        if (idle && (heard || Date.now() - t0 > 2500)) { W.clearInterval(poll); W.setTimeout(go, 120); }
      }, 250);
    } catch (e) { W.setTimeout(go, 600); }
  };
  AI.hush = id => { if (!once("hush_" + id)) return; try { W.speechSynthesis.cancel(); } catch (e) {} };

  AI.recStart = (key, nextKey, giveupKey, deadlineMs) => {
    if (!once("rec_" + key)) return;
    const alive = () => !!q('.st-key-' + key);                // still on this question?
    const stopNow = () => { const s = recBtn(key, "Stop recording"); if (s) s.click(); };
    // if the recording can't be saved within 5 s, tell the app to move on anyway
    const finish = () => { stopNow(); W.setTimeout(() => { if (alive()) retry(() => clickKey(giveupKey), 200, 20); }, 5000); };
    retry(() => {
      if (!alive()) return false;                              // widget not drawn yet
      if (recBtn(key, "Stop recording")) return true;          // already recording
      const b = recBtn(key, "Record"); if (b) { b.click(); return true; }
      return false;
    }, 120, 80);
    retry(() => {                                              // Next = stop THIS recorder instantly
      const nb = q('.st-key-' + nextKey + ' button');
      if (!nb) return false;
      if (nb.dataset.wiredFor !== key) {
        nb.dataset.wiredFor = key;
        nb.addEventListener("click", () => {
          if (nb.dataset.wiredFor !== key) return;
          finish();
          W.setTimeout(() => { nb.style.opacity = .6; nb.style.pointerEvents = "none";
            const p = nb.querySelector("p"); if (p) p.textContent = "Saving your answer…"; }, 0);
        }, true);
      }
      return true;
    }, 150, 80);
    const tick = W.setInterval(() => {                         // time is up -> stop and save
      if (!alive()) { W.clearInterval(tick); return; }
      if (Date.now() >= deadlineMs) { W.clearInterval(tick); finish(); }
    }, 250);
  };

  // smooth countdown for every element with class "ai-timer"
  const fmt = s => Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  W.setInterval(() => {
    W.document.querySelectorAll(".ai-timer").forEach(el => {
      const end = +el.dataset.end, total = +el.dataset.total, col = el.dataset.color;
      const left = Math.max(0, end - Date.now()), s = Math.ceil(left / 1000);
      const c = s <= 10 ? "#FF6B6B" : col;
      const r = el.querySelector(".ring"), t = el.querySelector(".in");
      if (r) r.style.background = `conic-gradient(${c} ${Math.min(1, left / total) * 360}deg,#232733 0deg)`;
      if (t) t.textContent = fmt(s);
    });
  }, 200);
  W.__AI = AI;
})();
""".replace("%V%", CONTROLLER_VERSION)


def browser_cmds(cmds):
    """Send commands to the page controller, e.g. [("speak", id, text, key, ms)].
    Always renders exactly one tiny iframe, so the layout never jumps."""
    calls = ";".join(f"W.__AI.{c[0]}({', '.join(json.dumps(x) for x in c[1:])})" for c in cmds)
    code = json.dumps(CONTROLLER_JS).replace("</", "<\\/")
    _html_js(f"""<script>(function(){{
      let W; try {{ W = window.parent; W.document; }} catch(e) {{ return; }}
      if (!W.__AI || W.__AI.v !== {json.dumps(CONTROLLER_VERSION)}) {{
        const s = W.document.createElement("script"); s.textContent = {code};
        W.document.head.appendChild(s);
      }}
      try {{ {calls} }} catch(e) {{ console.error(e); }}
    }})();</script>""")


def timer_html(deadline, total, color, caption):
    """Countdown ring; the page controller animates it (survives any redraw)."""
    left = max(0, deadline - time.time())
    deg = 360 * min(1.0, left / max(total, 1e-3))
    return (f"<div class='ai-timer' data-end='{deadline * 1000:.0f}' data-total='{total * 1000:.0f}' "
            f"data-color='{color}' style='display:flex;flex-direction:column;align-items:center;gap:8px'>"
            f"<div class='ring' style='width:136px;height:136px;border-radius:50%;display:flex;"
            f"align-items:center;justify-content:center;background:conic-gradient({color} {deg:.0f}deg,#232733 0deg)'>"
            f"<div class='in' style='width:112px;height:112px;border-radius:50%;background:#0E1016;display:flex;"
            f"align-items:center;justify-content:center;font-size:1.9rem;font-weight:800;color:#fff'>"
            f"{fmt(left)}</div></div><div class='muted'>{caption}</div></div>")


def read_secs(text):
    return max(4.0, len(text.split()) * 0.40 + 2.0)


def integrity_panel(snap, n_events=6):
    integ = snap["integrity"]
    col = risk_color(integ)
    evs = snap["events"][-n_events:][::-1]
    rows = "".join(f"<div class='ev'><span>{e['time']} · {e['event']}</span><b>-{e['points']}</b></div>"
                   for e in evs) or "<div class='muted' style='padding:6px 0'>All clear, no flags.</div>"
    st.markdown(f"""
    <div class='card'>
      <div style='display:flex;justify-content:space-between;align-items:flex-end'>
        <div><div class='muted'>Integrity</div><div class='big' style='color:{col}'>{integ}</div></div>
        <div style='text-align:right'><div class='muted'>Risk</div>
          <div style='font-weight:800;color:{col}'>{snap['risk']}</div></div>
      </div>
      <div style='margin:10px 0 12px'>{meter(integ / 300, col)}</div>
      <div class='muted' style='margin-bottom:4px'>Status: <b style='color:#fff'>{snap['status']}</b></div>
      {rows}
    </div>""", unsafe_allow_html=True)


def live_monitor(mode="interview", height=560):
    """Live video + live integrity panel, drawn by the BROWSER.
    Video comes straight from the camera thread as an MJPEG stream and the panel polls
    a small JSON endpoint 4x per second, so nothing here makes Streamlit refresh."""
    port = get_vision().port
    if not port:
        st.error("Live video server could not start (ports 8502-8519 busy).")
        return
    show_panel = "true" if mode == "interview" else "false"
    alarm = "true" if S.get("alarm", True) else "false"
    st.iframe(f"""<!doctype html><html><head><style>
    *{{box-sizing:border-box}} body{{margin:0;font-family:Inter,Segoe UI,system-ui,sans-serif;color:#E8EAF0;background:transparent}}
    .live{{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:700;color:#ff6b6b;margin-bottom:8px}}
    .live:before{{content:'';width:8px;height:8px;border-radius:50%;background:#ff4d4d;animation:p 1.2s infinite}}
    @keyframes p{{50%{{opacity:.25}}}}
    .vid{{position:relative;border-radius:14px;overflow:hidden;border:1px solid #262A36;background:#0b0d12;aspect-ratio:4/3}}
    .vid img{{width:100%;height:100%;object-fit:cover;display:block}}
    .off{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#6B7186;font-size:14px}}
    .flash{{position:absolute;inset:0;border:4px solid #ef4444;border-radius:14px;opacity:0;transition:opacity .3s;pointer-events:none}}
    .banner{{background:#7f1d1d;border:1px solid #ef4444;border-radius:12px;padding:9px 14px;font-weight:700;margin-top:10px;
             transform:translateY(-6px);opacity:0;transition:all .25s;height:0;padding-top:0;padding-bottom:0;overflow:hidden}}
    .banner.on{{opacity:1;transform:none;height:auto;padding-top:9px;padding-bottom:9px}}
    .card{{background:#171A23;border:1px solid #262A36;border-radius:14px;padding:14px 16px;margin-top:10px}}
    .row{{display:flex;justify-content:space-between;align-items:flex-end}}
    .muted{{color:#8A90A2;font-size:12.5px}} .big{{font-size:36px;font-weight:800;line-height:1}}
    .meter{{height:7px;border-radius:9px;background:#232733;overflow:hidden;margin:10px 0}}
    .meter div{{height:7px;border-radius:9px;transition:width .4s,background .4s}}
    .ev{{display:flex;justify-content:space-between;font-size:12.5px;padding:5px 0;border-bottom:1px solid #222633}}
    .ev b{{color:#ff8a8a}} .info{{font-size:13px;color:#C9CEDC;margin-top:8px;line-height:1.6}}
    .bar{{height:6px;border-radius:9px;background:#232733;overflow:hidden;margin-top:6px}} .bar div{{height:6px;background:#F5C451;transition:width .2s}}
    </style></head><body>
    <div class="live">LIVE PROCTORING</div>
    <div class="vid"><div class="off" id="off">Starting camera…</div>
      <img id="v" alt="" onload="document.getElementById('off').style.display='none'">
      <div class="flash" id="flash"></div></div>
    <div class="banner" id="banner"></div>
    <div id="panel"></div>
    <script>
    const PANEL={show_panel}, ALARM={alarm};
    const host=(()=>{{try{{return window.parent.location.hostname||"127.0.0.1"}}catch(e){{return "127.0.0.1"}}}})();
    const base="http://127.0.0.1:{port}";
    document.getElementById("v").src = base+"/stream?t="+Date.now();
    let seen=null, session=null, hideT=null;
    function beep(){{ if(!ALARM) return; try{{const A=window.AudioContext||window.webkitAudioContext;const c=new A();
      const o=c.createOscillator(),g=c.createGain();o.type="square";o.frequency.value=880;g.gain.value=.08;
      o.connect(g);g.connect(c.destination);o.start();setTimeout(()=>o.frequency.value=660,180);
      setTimeout(()=>{{o.stop();c.close()}},420)}}catch(e){{}} }}
    function col(i){{return i>=80?"#5BE38C":i>=50?"#F5C451":"#FF7A7A"}}
    function esc(s){{return String(s).replace(/[&<>]/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c]))}}
    async function tick(){{
      try{{
        const s = await (await fetch(base+"/state",{{cache:"no-store"}})).json();
        const n = s.events[s.events.length-1].n, evs = s.events.slice(0,-1);
        if(session!==s.session){{session=s.session; seen=n;}}
        if(seen!==null && n>seen && s.monitoring){{
          const e=evs[evs.length-1]; const b=document.getElementById("banner");
          b.textContent="🚨 "+e.event+"  (−"+e.points+")"; b.classList.add("on");
          document.getElementById("flash").style.opacity=1; beep();
          clearTimeout(hideT); hideT=setTimeout(()=>{{b.classList.remove("on");document.getElementById("flash").style.opacity=0}},3500);
        }}
        seen=n;
        const c=col(s.integrity);
        let h="";
        if(PANEL){{
          const rows=evs.slice(-6).reverse().map(e=>`<div class="ev"><span>${{e.time}} · ${{esc(e.event)}}</span><b>-${{e.points}}</b></div>`).join("")
                     ||'<div class="muted" style="padding:6px 0">All clear, no flags.</div>';
          h=`<div class="card"><div class="row"><div><div class="muted">Integrity</div><div class="big" style="color:${{c}}">${{s.integrity}}</div></div>
             <div style="text-align:right"><div class="muted">Risk</div><div style="font-weight:800;color:${{c}}">${{s.risk}}</div></div></div>
             <div class="meter"><div style="width:${{s.integrity}}%;background:${{c}}"></div></div>
             <div class="muted">Status: <b style="color:#fff">${{esc(s.status)}}</b></div>${{rows}}</div>`;
        }} else {{
          const objs=s.objects.length?s.objects.join(", "):"none";
          h=`<div class="info">Status: <b>${{esc(s.status)}}</b> · Objects seen: <b>${{esc(objs)}}</b> · ${{s.fps}} fps</div>`+
            (s.calibrated?"":`<div class="muted" style="margin-top:6px">Calibrating… look straight at the screen</div><div class="bar"><div style="width:${{Math.min(100,s.calib_progress*100)}}%"></div></div>`)+
            `<div class="muted" style="margin-top:6px">Test it: hold your phone up. A red box should appear within a second. Flags only count once the interview starts.</div>`;
          if(s.camera_error) h+=`<div class="info" style="color:#FF7A7A">${{esc(s.camera_error)}}</div>`;
        }}
        document.getElementById("panel").innerHTML=h;
      }}catch(e){{}}
      setTimeout(tick,250);
    }}
    tick();
    </script></body></html>""", height=height)


# =========================================================================
# 1. SETUP
# =========================================================================
def setup_page():
    st.markdown("<div class='h-title'>Set up your interview</div>"
                "<p class='muted'>The AI writes questions for the role, reads them aloud, "
                "listens to your spoken answers and watches the camera for integrity.</p>",
                unsafe_allow_html=True)
    left, right = st.columns([1.6, 1], gap="large")

    with right:
        key = load_key()
        box = st.container(border=True)
        box.markdown("**AI connection** · Groq (free)")
        with box:
            if key:
                ok, msg = S.get("key_status", (None, ""))
                if ok is None:
                    try:
                        ok, msg = get_brain(key).check_key()
                    except Exception as e:
                        ok, msg = False, str(e)
                    S.key_status = (ok, msg)
                if ok:
                    st.success("Connected and ready", icon="✅")
                else:
                    st.error(f"Key not working: {msg}")
            else:
                st.info("Paste a free Groq API key to start. Get one at console.groq.com/keys "
                        "(sign in with Google, no card needed).")
            with st.expander("Change / add API key", expanded=not key):
                new = st.text_input("Groq API key", type="password", placeholder="gsk_...")
                remember = st.checkbox("Remember on this laptop", value=True)
                if st.button("Save key", width="stretch") and new.strip():
                    S.api_key = new.strip()
                    if remember:
                        save_key(new)
                    S.pop("key_status", None)
                    st.rerun()

        st.markdown("""<div class='card'><b>How it works</b>
        <ol class='muted' style='margin:8px 0 0 18px;padding:0;line-height:1.7'>
        <li>Check camera + microphone</li><li>Look at the screen for 2 s to calibrate</li>
        <li>Each question is read aloud, then recording starts automatically</li>
        <li>Click <b>Next</b> when done (or the timer moves on)</li>
        <li>Get a scored report with feedback</li></ol></div>""", unsafe_allow_html=True)

    with left:
        c1, c2 = st.columns(2)
        name = c1.text_input("Candidate name", S.get("name", ""), placeholder="e.g. Nimra")
        role = c2.text_input("Role", S.get("role", "Junior Python Developer"))
        jd = st.text_area("Job description / focus (optional)", S.get("jd", ""), height=90,
                          placeholder="e.g. Django, REST APIs, SQL, teamwork")
        cv = st.text_area("Candidate skills / CV summary (optional, makes questions personal)",
                          S.get("cv", ""), height=90,
                          placeholder="e.g. BSCS final year, FYP on face recognition, knows Python, Flask, OpenCV")
        c3, c4, c5 = st.columns(3)
        n = c3.slider("Questions", 3, 10, S.get("n", 5))
        diff = c4.select_slider("Difficulty", ["easy", "mixed", "hard"], S.get("diff", "mixed"))
        secs = c5.select_slider("Time per answer", [45, 60, 90, 120, 180], S.get("secs", 90),
                                format_func=lambda s: fmt(s))
        st.caption(f"About {round(n * (secs + 10) / 60)} minutes in total.")

        if st.button("Generate questions  →", type="primary", width="stretch",
                     disabled=not (key and S.get("key_status", (False,))[0])):
            S.update(name=name or "Candidate", role=role, jd=jd, cv=cv, n=n, diff=diff, secs=secs)
            with st.spinner("The AI is writing your questions…"):
                qs = get_brain(key).generate_questions(role, jd, cv, n=n, difficulty=diff)
            get_vision()  # start the camera AI now (spinner shown)
            S.update(questions=qs, idx=0, futures=[], results=[], summary=None,
                     saved_dir=None, stage="check")
            st.rerun()


# =========================================================================
# 2. SYSTEM CHECK
# =========================================================================
def check_page():
    vis = get_vision()
    if S.get("monitor_on", True):
        vis.monitor(False)          # nothing counts until "Start interview"
        S.monitor_on = False
    st.markdown("<div class='h-title'>System check</div><p class='muted'>Sit centred, face the screen, "
                "good light in front of you (not behind). Look at the screen for 2 seconds to calibrate.</p>",
                unsafe_allow_html=True)
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        cc1, cc2 = st.columns([1, 1])
        idx = cc1.number_input("Camera number", 0, 5, S.get("cam_idx", 0),
                               help="0 = built-in webcam. Try 1 if you use an external camera.")
        st_now = vis.snapshot(full=False)
        if idx != vis.cam_index or not st_now.get("camera_running"):
            S.cam_idx = idx
            with st.spinner("Opening camera…"):
                ok, err = vis.start(int(idx))
            if not ok and err:
                st.error(err)
        if cc2.button("↻ Recalibrate gaze", width="stretch"):
            vis.recalibrate()
        live_monitor("check", height=600)

    with right:
        with st.container(border=True):
            st.markdown("**Microphone test**")
            st.caption("Click 🎙️, say a sentence like “Hello, my name is …”, then click ■. "
                       "Allow microphone access if the browser asks.")
            test = st.audio_input("Mic test", key=f"mic_test_{S.get('mic_test_n', 0)}",
                                  label_visibility="collapsed")
            if test is not None:
                data = test.getvalue()
                sig = hash(data)
                if S.get("mic_sig") != sig:
                    S.mic_sig = sig
                    path = os.path.join(ANSWERS, "mic_test.wav")
                    with open(path, "wb") as f:
                        f.write(data)
                    peak = wav_peak(path)
                    if peak < SILENCE_LEVEL:
                        S.mic_ok = False
                        S.mic_msg = (f"Only silence was recorded (level {peak:.3f}). Check that your mic "
                                     "isn't muted. To pick another mic, click the 🔒 icon next to "
                                     "localhost:8501 in the address bar → Microphone.")
                    else:
                        with st.spinner("Checking what the AI hears…"):
                            try:
                                heard = get_brain(load_key()).transcribe(path)
                            except Exception as e:
                                heard = f"(could not transcribe: {e})"
                        S.mic_ok = True
                        S.mic_msg = f"The AI heard: “{heard or '(nothing clear)'}”"
                if S.mic_ok:
                    st.success(S.mic_msg + "  \nIf that matches what you said, you're ready.")
                elif S.mic_ok is False:
                    st.error(S.mic_msg)
            if not S.get("mic_ok"):
                st.caption("Tip: a headset mic or sitting close to the laptop gives the best results.")

        with st.expander("⚙️ Proctoring settings"):
            sens = st.select_slider("Gaze sensitivity", ["Strict", "Normal", "Relaxed"],
                                    S.get("sens", "Normal"))
            delay = st.slider("Look-away grace time (s)", 0.5, 4.0, S.get("delay", 1.5), 0.1)
            obj = st.select_slider("Phone / book detection", ["Very sensitive", "Sensitive", "Normal", "Strict"],
                                   S.get("obj", "Sensitive"))
            alarm = st.toggle("Alarm sound on violations", S.get("alarm", True))
            conf = (sens, delay, obj, alarm)
            if S.get("vis_conf") != conf:
                S.vis_conf = conf
                vis.configure(sens, delay, {"Very sensitive": 0.7, "Sensitive": 0.85,
                                            "Normal": 1.0, "Strict": 1.3}[obj], alarm)
            S.update(sens=sens, delay=delay, alarm=alarm, obj=obj)

        checks_and_start()

    if st.button("← Back to setup"):
        vis.stop()
        S.stage = "setup"
        st.rerun()


@st.fragment(run_every=1.5)
def checks_and_start():
    snap = get_vision().snapshot(full=False)
    cam_ok = bool(snap.get("camera_ok"))
    face_ok = bool(snap.get("calibrated"))
    mic = S.mic_ok

    def row(state, title, sub):
        ic = {"ok": "✓", "no": "✕", "wait": "…"}[state]
        return (f"<div class='check {state}'><div class='ic'>{ic}</div><div><b>{title}</b>"
                f"<div class='muted'>{sub}</div></div></div>")
    st.markdown("<div class='card'><b>Ready checklist</b>" +
                row("ok" if cam_ok else "no", "Camera", "Live feed running" if cam_ok else "Not running") +
                row("ok" if face_ok else "wait", "Face calibrated", "Gaze baseline captured" if face_ok else "Look at the screen") +
                row("ok" if mic else "no" if mic is False else "wait", "Microphone",
                    "Voice detected" if mic else "Run the mic test") +
                row("ok", "AI interviewer", f"{len(S.questions)} questions ready") +
                "</div>", unsafe_allow_html=True)

    ready = cam_ok and face_ok
    if mic is not True:
        st.caption("Tip: test the mic first, otherwise your answers may be empty.")
    if st.button("Start interview  →", type="primary", width="stretch", disabled=not ready):
        start_interview()
        st.rerun(scope="app")


def start_interview():
    v = get_vision()
    v.reset()
    v.monitor(True)
    S.monitor_on = True
    S.report_snap = None
    S.update(stage="interview", idx=0, futures=[], results=[], started_at=time.time(),
             summary=None, saved_dir=None)
    enter_question(0)


# =========================================================================
# 3. INTERVIEW
# =========================================================================
def enter_question(i):
    q = S.questions[i]["question"]
    S.update(idx=i, phase="think", phase_start=time.time(),
             phase_len=read_secs(q) + 6.0,       # safety net if speech never reports "done"
             speak_nonce=S.speak_nonce + 1, stop_req=None)


def begin_answer():
    S.update(phase="answer", phase_start=time.time(), phase_len=float(S.secs), stop_req=None)


def submit_answer(audio=None):
    """Save the browser recording and send it to the AI in the background (non-blocking)."""
    i = S.idx
    q = S.questions[i]
    wav, peak = None, 0.0
    if audio is not None:
        data = audio.getvalue()
        if data:
            wav = os.path.join(ANSWERS, f"answer_{int(S.started_at)}_{i + 1}.wav")
            with open(wav, "wb") as f:
                f.write(data)
            peak = wav_peak(wav)
    item = {"n": i + 1, "question": q["question"], "topic": q.get("topic", ""),
            "difficulty": q.get("difficulty", ""), "ideal_points": q.get("ideal_points"),
            "wav": wav, "peak": peak,
            "duration": time.time() - S.phase_start if S.phase == "answer" else 0}
    brain = get_brain(load_key())
    S.futures.append(get_pool().submit(brain.process_answer, item, S.role, SILENCE_LEVEL))
    S.stop_req = None
    if i + 1 < len(S.questions):
        enter_question(i + 1)
        return False
    return True


def finish_interview():
    get_vision().monitor(False)
    S.monitor_on = False
    S.stage = "report"


def interview_page():
    left, right = st.columns([1.7, 1], gap="large")
    with right:
        live_monitor("interview", height=640)
    with left:
        question_panel()


@st.fragment
def question_panel():
    """Redraws ONLY when something happens (no polling): the browser controller presses
    hidden buttons when the question has been read, when time runs out, or when an
    answer couldn't be saved. The timer, video and integrity panel animate in the browser."""
    now = time.time()
    n = len(S.questions)

    def keys():
        b = f"{int(S.started_at)}_{S.idx}"
        return f"ans_{b}", f"autobegin_{b}_{S.speak_nonce}", f"giveup_{b}"
    rec_key, begin_key, giveup_key = keys()

    # ---- state changes FIRST, so the new screen is drawn in this same run ----
    if S.phase == "think":
        if st.session_state.get(begin_key):
            begin_answer()                       # the question has been read -> record
    elif S.phase == "answer":
        got = st.session_state.get(rec_key)
        if got is not None or st.session_state.get(giveup_key):
            if submit_answer(got):               # answer saved (or nothing came) -> next
                finish_interview()
                st.rerun(scope="app")
    rec_key, begin_key, giveup_key = keys()

    i = S.idx
    q = S.questions[i]
    last = i + 1 == n
    if S.phase == "think":
        st.button("begin", key=begin_key)        # hidden, pressed by the browser
    else:
        st.button("giveup", key=giveup_key)      # hidden, pressed by the browser

    # ---- question card ------------------------------------------------
    prog = "".join(
        f"<span style='display:inline-block;width:{100 / n:.2f}%;padding-right:4px'>"
        f"<span style='display:block;height:5px;border-radius:9px;background:"
        f"{'#7C5CFF' if k < i else '#22D3EE' if k == i else '#2A2F3D'}'></span></span>"
        for k in range(n))
    st.markdown(f"""
    <div class='qcard'>
      <div style='margin-bottom:16px'>{prog}</div>
      <div class='qmeta'>Question {i + 1} of {n}</div>
      <div class='qtext'>{_html.escape(q['question'])}</div>
      <div style='margin-top:10px'><span class='chip'>{_html.escape(str(q.get('topic', 'General')))}</span>
      <span class='chip'>{_html.escape(str(q.get('difficulty', '')).title())}</span></div>
    </div>""", unsafe_allow_html=True)

    st.write("")
    t1, t2 = st.columns([1, 1.4], gap="large")
    with t1:
        if S.phase == "think":
            st.markdown("<div style='height:136px;display:flex;flex-direction:column;align-items:center;"
                        "justify-content:center;gap:10px'><div class='speaking'><i></i><i></i><i></i><i></i></div>"
                        "<div class='muted' style='text-align:center'>The interviewer is reading<br>"
                        "the question…</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(timer_html(S.phase_start + S.phase_len, S.phase_len, "#22D3EE",
                                   "time left to answer"), unsafe_allow_html=True)
    with t2:
        if S.phase == "think":
            st.markdown("<div class='muted' style='margin:6px 0 10px'>🎙️ Recording will start "
                        "<b>automatically</b> as soon as the question has been read.</div>",
                        unsafe_allow_html=True)
            if st.button("Skip, answer now", type="primary", width="stretch"):
                begin_answer()
                st.rerun(scope="fragment")
            if st.button("🔁 Repeat question", width="stretch"):
                S.speak_nonce += 1
                st.rerun(scope="fragment")
        else:
            st.markdown("<span class='live'>RECORDING YOUR ANSWER</span>", unsafe_allow_html=True)
            st.audio_input("Your answer", key=rec_key, label_visibility="collapsed")
            st.caption("Speak clearly. When you're done, press **Next** (or ■ in the recorder).")
            lbl = "✅ Finish interview" if last else "Next question  →"
            # the browser stops the recorder the instant this is clicked; saving the
            # recording then moves to the next question automatically
            st.button(lbl, type="primary", width="stretch", key=f"next_{rec_key}")
        with st.popover("End interview early", width="stretch"):
            st.write("Unanswered questions will score 0.")
            if st.button("Yes, end now", type="primary"):
                if S.phase == "answer":
                    submit_answer(st.session_state.get(rec_key))
                finish_interview()
                st.rerun(scope="app")

    # ---- tell the browser what to do (one script slot, so the layout never jumps) ----
    if S.phase == "think":
        sid = f"{int(S.started_at)}_{i}_{S.speak_nonce}"
        fallback_ms = int((read_secs(q["question"]) + 2.5) * 1000)
        browser_cmds([("speak", sid, q["question"], begin_key, fallback_ms)])
    else:
        browser_cmds([("hush", rec_key),
                      ("recStart", rec_key, f"next_{rec_key}", giveup_key,
                       int((S.phase_start + S.phase_len) * 1000))])


# =========================================================================
# 4. REPORT
# =========================================================================
def report_page():
    vis = get_vision()
    if not S.get("report_snap"):
        S.report_snap = vis.snapshot(full=True)       # freeze the proctoring result
        vis.stop()                                     # camera off, free the CPU
    snap = S.report_snap

    # ---- collect background results --------------------------------------
    if not S.results and S.futures:
        bar = st.progress(0.0, "AI is transcribing and scoring your answers…")
        out = []
        for k, f in enumerate(S.futures):
            try:
                out.append(f.result(timeout=180))
            except Exception as e:
                out.append({"n": k + 1, "question": "?", "answer": "", "score": 0,
                            "verdict": "Error", "feedback": str(e), "strengths": [],
                            "improvements": [], "error": str(e)})
            bar.progress((k + 1) / len(S.futures), f"Scored {k + 1} of {len(S.futures)}")
        # unanswered questions count as 0
        for j in range(len(out), len(S.questions)):
            q = S.questions[j]
            out.append({"n": j + 1, "question": q["question"], "answer": "", "score": 0,
                        "verdict": "Skipped", "feedback": "Not answered (interview ended early).",
                        "strengths": [], "improvements": [], "communication": 0})
        S.results = out
        bar.empty()

    results = S.results
    overall = 10 * sum(r.get("score", 0) for r in results) / max(1, len(results))
    comm = 10 * sum(r.get("communication", 0) for r in results) / max(1, len(results))
    integ, attn = snap["integrity"], snap["attention"]

    if S.summary is None:
        with st.spinner("Writing the overall assessment…"):
            S.summary = get_brain(load_key()).summary(S.role, results, integ, attn) or {}
    summ = S.summary

    # ---- hero ------------------------------------------------------------
    st.markdown(f"<div class='h-title'>Interview report · {_html.escape(S.name)}</div>"
                f"<p class='muted'>{S.role} · {dt.datetime.now():%d %b %Y, %H:%M} · "
                f"{len(results)} questions</p>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl in ((k1, overall, "Answer quality"), (k2, comm, "Communication"),
                          (k3, integ, "Integrity"), (k4, attn, "Attention on screen")):
        col.markdown(f"<div class='card kpi'><div class='v' style='color:{score_color(val)}'>"
                     f"{val:.0f}<span style='font-size:1rem;color:#8A90A2'>/100</span></div>"
                     f"<div class='l'>{lbl}</div></div>", unsafe_allow_html=True)

    if summ:
        li = lambda xs: "".join(f"<li>{_html.escape(str(x))}</li>" for x in xs) or "<li>-</li>"
        st.markdown(f"""
        <div class='card'>
          <span class='rec-badge'>{_html.escape(str(summ.get('recommendation', '-')))}</span>
          <p style='margin:12px 0'>{_html.escape(str(summ.get('summary', '')))}</p>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:18px'>
            <div><b>Top strengths</b><ul class='muted'>{li(summ.get('top_strengths', []))}</ul></div>
            <div><b>Focus areas</b><ul class='muted'>{li(summ.get('focus_areas', []))}</ul></div>
          </div>
        </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📝 Answers", "🛡️ Integrity", "⬇️ Download"])

    with tab1:
        st.caption("Each card shows the question, what the candidate said, and the AI's evaluation.")
        for r in results:
            pct = r.get("score", 0) * 10
            col = score_color(pct)
            said = r.get("clean_answer") or r.get("answer") or ""
            ans_html = (_html.escape(said) if said else
                        "<i style='color:#8A90A2'>No answer was captured for this question.</i>")
            st.markdown(f"""
            <div class='card'>
              <div style='display:flex;justify-content:space-between;gap:12px;align-items:flex-start'>
                <div><div class='qmeta'>Question {r['n']}</div>
                  <div style='font-weight:700;font-size:1.08rem;margin-top:4px'>{_html.escape(r['question'])}</div></div>
                <div style='text-align:right;white-space:nowrap'>
                  <div style='font-size:1.6rem;font-weight:800;color:{col}'>{r.get('score', 0)}<span
                    style='font-size:.9rem;color:#8A90A2'>/10</span></div>
                  <div class='muted'>{_html.escape(str(r.get('verdict', '')))}</div></div>
              </div>
              <div class='qmeta' style='margin-top:14px'>Candidate's answer</div>
              <div style='border-left:3px solid #7C5CFF;background:#12141c;border-radius:8px;
                          padding:12px 14px;margin-top:6px;line-height:1.6'>{ans_html}</div>
              <div style='margin-top:12px'><b>Feedback:</b> {_html.escape(str(r.get('feedback', '')))}</div>
            </div>""", unsafe_allow_html=True)
            with st.expander(f"Details for Q{r['n']}: recording, exact transcript, strengths, model answer"):
                if r.get("error"):
                    st.warning(r["error"])
                if r.get("wav") and os.path.exists(r["wav"]):
                    st.audio(r["wav"])
                if r.get("answer"):
                    st.markdown("**Exact speech-to-text transcript (unedited):**")
                    st.code(r["answer"], language=None, wrap_lines=True)
                a, b = st.columns(2)
                a.markdown("**Strengths**\n" + ("\n".join(f"- {x}" for x in r.get("strengths", [])) or "-"))
                b.markdown("**To improve**\n" + ("\n".join(f"- {x}" for x in r.get("improvements", [])) or "-"))
                if r.get("model_answer"):
                    st.info(f"**Example strong answer:** {r['model_answer']}")

    events = snap["events"]
    with tab2:
        if not events:
            st.success("Clean session — no integrity flags were raised.")
        else:
            df = pd.DataFrame(events)
            start = S.started_at or df["t"].min()
            df["minute"] = ((df["t"] - start) / 60).round(2)
            c1, c2 = st.columns([1.3, 1])
            with c1:
                st.markdown("**Integrity over time**")
                line = pd.concat([pd.DataFrame({"minute": [0.0], "integrity": [100]}),
                                  df[["minute", "integrity"]]])
                st.line_chart(line, x="minute", y="integrity", height=240)
            with c2:
                st.markdown("**Flags by type**")
                st.bar_chart(df["event"].value_counts(), height=240, horizontal=True)
            st.dataframe(df[["time", "event", "points", "integrity"]], hide_index=True,
                         width="stretch")
            snaps = [e for e in events if e.get("snapshot") and os.path.exists(e["snapshot"])][-8:]
            if snaps:
                st.markdown("**Evidence snapshots**")
                cols = st.columns(4)
                for k, e in enumerate(snaps):
                    cols[k % 4].image(e["snapshot"], caption=f"{e['time']} · {e['event']}")

    meta = dict(name=S.name, role=S.role, date=f"{dt.datetime.now():%d %b %Y %H:%M}",
                overall=overall, integrity=integ, attention=attn)
    html_report = build_html(meta, results, events, summ)

    # auto-save once
    if not S.saved_dir:
        d = os.path.join(REPORTS, f"{dt.datetime.now():%Y%m%d_%H%M}_{''.join(c for c in S.name if c.isalnum()) or 'candidate'}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "report.html"), "w", encoding="utf-8") as f:
            f.write(html_report)
        pd.DataFrame(events).drop(columns=["t"], errors="ignore").to_csv(
            os.path.join(d, "proctoring_log.csv"), index=False)
        with open(os.path.join(d, "results.json"), "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "summary": summ, "results": results}, f, indent=2, default=str)
        S.saved_dir = d

    with tab3:
        st.markdown(f"Saved automatically to `{S.saved_dir}`")
        d1, d2, d3 = st.columns(3)
        d1.download_button("📄 Full report (HTML → print to PDF)", html_report.encode("utf-8"),
                           "interview_report.html", "text/html", width="stretch", type="primary")
        d2.download_button("📊 Proctoring log (CSV)",
                           pd.DataFrame(events).drop(columns=["t"], errors="ignore").to_csv(index=False).encode(),
                           "proctoring_log.csv", "text/csv", width="stretch")
        d3.download_button("🧾 Raw results (JSON)",
                           json.dumps({"meta": meta, "summary": summ, "results": results},
                                      indent=2, default=str).encode(),
                           "results.json", "application/json", width="stretch")

    st.divider()
    b1, b2, _ = st.columns([1, 1, 2])
    if b1.button("🔁 Retake (new questions)", width="stretch", type="primary"):
        with st.spinner("Writing new questions…"):
            S.questions = get_brain(load_key()).generate_questions(
                S.role, S.jd, S.cv, n=S.n, difficulty=S.diff)
        S.update(futures=[], results=[], summary=None, saved_dir=None, report_snap=None, stage="check")
        st.rerun()
    if b2.button("🏠 New setup", width="stretch"):
        S.update(futures=[], results=[], summary=None, saved_dir=None, report_snap=None, stage="setup")
        st.rerun()


# =========================================================================
header()
{"setup": setup_page, "check": check_page,
 "interview": interview_page, "report": report_page}[S.stage]()