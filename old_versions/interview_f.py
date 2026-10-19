"""
interview_exam.py  —  Timed full-screen interview (exam mode)
Run with:  streamlit run interview_exam.py

Key design decisions (these fix the hang + the camera dropping):
  1. The camera is rendered FIRST on every run, always with the same key and in
     the same column, so the WebRTC stream is never torn down. Permission is
     granted on the "camera check" screen BEFORE the interview begins, and the
     same live stream simply carries on into the interview.
  2. NOTHING blocking happens during the interview. Answers are only recorded
     and saved. Transcription + scoring (the slow Groq calls) run at the END on
     the report screen. That is why advancing a question is now instant.
  3. Timers use absolute deadlines; the main timer never resets.

Needs:  pip install streamlit-webrtc sounddevice streamlit-autorefresh
"""

import os
import json
import time
import wave

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from streamlit_autorefresh import st_autorefresh

from engine import Proctor
from interviewer import Interviewer
from voice import Transcriber

try:
    import sounddevice as sd
    _HAS_AUDIO = True
except Exception:
    _HAS_AUDIO = False

SR = 16000
SECONDS_PER_Q = 60
os.makedirs("answers", exist_ok=True)
RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

st.set_page_config(page_title="AI Interview", layout="wide")
st.markdown("""
<style>
#MainMenu, footer, header {visibility:hidden}
.block-container {padding-top:1rem}
</style>
""", unsafe_allow_html=True)

DEFAULTS = {"stage": "setup", "questions": [], "idx": 0, "pending": [],
            "results": [], "cur": -1, "sub": "reading", "nonce": 0}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def fmt(s):
    s = max(0, int(s))
    return f"{s // 60}:{s % 60:02d}"


def read_secs(text):
    return max(3.0, len(text.split()) * 0.42 + 1.5)


def speak_browser(text, key):
    """Browser reads the text aloud, once per key (no server TTS = no freeze)."""
    components.html(f"""
    <script>
    const k = "spoke_{key}";
    if (!window.sessionStorage.getItem(k)) {{
      try {{
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance({json.dumps(text)});
        u.rate = 1.0; u.lang = "en-US";
        window.speechSynthesis.speak(u);
        window.sessionStorage.setItem(k, "1");
      }} catch(e) {{}}
    }}
    </script>""", height=0)


# ---- mic (non-blocking) -----------------------------------------------
def start_rec():
    if not _HAS_AUDIO or "audio_stream" in st.session_state:
        return
    chunks = []
    st.session_state.audio_chunks = chunks
    s = sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                       device=st.session_state.get("mic_device"),
                       callback=lambda d, *a: chunks.append(d.copy()))
    s.start()
    st.session_state.audio_stream = s


def stop_rec(path):
    s = st.session_state.pop("audio_stream", None)
    if s is not None:
        s.stop()
        s.close()
    chunks = st.session_state.pop("audio_chunks", [])
    if not chunks:
        return None
    data = np.concatenate(chunks, axis=0)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(data.tobytes())
    return path


def clients():
    if "interviewer" not in st.session_state:
        try:
            st.session_state.interviewer = Interviewer()
            st.session_state.transcriber = Transcriber()
        except Exception as e:
            st.error(f"Could not connect to Groq. Is GROQ_API_KEY set? ({e})")
            st.stop()


# ---- microphone helpers -----------------------------------------------
def input_devices():
    """[(index, name), ...] for every device that can record."""
    if not _HAS_AUDIO:
        return []
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append((i, d["name"]))
    return out


def wav_level(path):
    """Peak level of a WAV, 0.0-1.0. Near 0 means we captured silence."""
    try:
        with wave.open(path, "rb") as wf:
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        if data.size == 0:
            return 0.0
        return float(np.abs(data).max()) / 32768.0
    except Exception:
        return 0.0


SILENCE_LEVEL = 0.02   # below this we treat the answer as "nothing was said"


# =======================================================================
# SETUP
# =======================================================================
def setup_stage():
    clients()
    st.subheader("Set up the interview")
    role = st.text_input("Role", "Junior Python Developer")
    jd = st.text_area("Job description / focus (optional)", "")
    n = st.slider("Number of questions", 3, 8, 3)
    diff = st.select_slider("Difficulty", ["easy", "mixed", "hard"], "mixed")
    st.info(f"{n} question(s) — 1 minute each — total {n} minute(s).")

    # ---------------- MICROPHONE CHECK ----------------
    st.divider()
    st.markdown("### Microphone check (do this first)")
    devs = input_devices()
    if not devs:
        st.error("No input device found. Is a microphone connected and enabled?")
    else:
        default_in = None
        try:
            default_in = sd.default.device[0]
        except Exception:
            pass
        labels = [f"[{i}] {name}" for i, name in devs]
        start_at = next((k for k, (i, _) in enumerate(devs) if i == default_in), 0)
        pick = st.selectbox("Microphone", labels, index=start_at)
        st.session_state.mic_device = devs[labels.index(pick)][0]

        if st.button("Test microphone (speak for 4 seconds)"):
            with st.spinner("Recording 4 seconds — say something now..."):
                try:
                    rec = sd.rec(int(4 * SR), samplerate=SR, channels=1,
                                 dtype="int16", device=st.session_state.mic_device)
                    sd.wait()
                    path = "answers/mic_test.wav"
                    with wave.open(path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(SR)
                        wf.writeframes(rec.tobytes())
                except Exception as e:
                    st.error(f"Recording failed: {e}")
                    path = None
            if path:
                lvl = wav_level(path)
                st.audio(open(path, "rb").read(), format="audio/wav")
                st.progress(min(lvl * 3, 1.0), f"Peak level: {lvl:.3f}")
                if lvl < SILENCE_LEVEL:
                    st.error(
                        "Silence detected — the interview will not capture your "
                        "answers. Fix this first:\n\n"
                        "1. Windows Settings → Privacy & security → Microphone → "
                        "turn ON 'Let desktop apps access your microphone'.\n"
                        "2. Pick a different microphone in the dropdown above.\n"
                        "3. Check the mic isn't muted / volume isn't at zero.")
                else:
                    with st.spinner("Transcribing your test..."):
                        try:
                            txt = st.session_state.transcriber.transcribe(path)
                        except Exception as e:
                            txt = f"(transcription failed: {e})"
                    st.success(f"Mic works. Whisper heard: “{txt}”")
    st.divider()

    if st.button("Continue to camera check", type="primary"):
        with st.spinner("Generating questions..."):
            iv = st.session_state.interviewer
            qs = iv.generate_questions(role, jd, n=n, difficulty=diff)
            t = 0
            while len(qs) < n and t < 2:
                qs += iv.generate_questions(role, jd, n=n - len(qs), difficulty=diff)
                t += 1
        qs = qs[:n]
        if not qs:
            st.error("No questions returned. Try again.")
            return
        if "proctor" not in st.session_state:
            with st.spinner("Loading proctoring models..."):
                st.session_state.proctor = Proctor()
        st.session_state.proctor.reset()
        st.session_state.update(role=role, questions=qs, idx=0, pending=[],
                                results=[], cur=-1, stage="ready")
        st.rerun()


# =======================================================================
# EXAM SCREEN  (shared by "ready" and "interview" so the camera NEVER remounts)
# =======================================================================
def exam_screen():
    st_autorefresh(interval=1000, key="tick")
    proctor = st.session_state.proctor
    qs = st.session_state.questions
    n = len(qs)
    now = time.time()

    left, center, right = st.columns([1.1, 2.3, 1.2])

    # ---------- CAMERA FIRST: rendered on every run, same key, same column ----
    with right:
        st.markdown("**● LIVE — monitoring**")
        ctx = webrtc_streamer(
            key="cam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=lambda f: av.VideoFrame.from_ndarray(
                proctor.process_frame(cv2.flip(f.to_ndarray(format="bgr24"), 1))[0],
                format="bgr24"),
            desired_playing_state=True,      # forced on, cannot be stopped
            async_processing=True,
        )
    live = bool(ctx and ctx.state.playing)

    # ---------- READY (camera permission) ----------
    if st.session_state.stage == "ready":
        with center:
            st.markdown("## Camera check")
            if live:
                st.success("Camera is on and monitoring. Look at the screen to "
                           "calibrate, then begin.")
            else:
                st.warning("Allow camera access in your browser prompt. "
                           "The camera must stay on for the whole interview.")
            st.write(f"**{n} question(s) · 1 minute each · {n} minute(s) total**")
            if st.button("Start interview", type="primary", disabled=not live):
                p = proctor                    # keep calibration, zero the score
                p.score, p.events = 0, []
                p._last_logged, p.last_alarm_score = {}, 0
                st.session_state.update(
                    interview_start=now, total_seconds=n * SECONDS_PER_Q,
                    idx=0, cur=-1, stage="interview")
                st.rerun()
        _monitor_panel(left, proctor)
        return

    # ---------- INTERVIEW ----------
    idx = st.session_state.idx
    total_left = st.session_state.total_seconds - (now - st.session_state.interview_start)

    # advance / end WITHOUT blocking (no Groq calls here — that's why it's instant)
    def save_and_advance(end=False):
        wav = f"answers/answer_{st.session_state.idx}.wav"
        stop_rec(wav)
        st.session_state.pending.append(
            {"question": qs[st.session_state.idx]["question"],
             "wav": wav if os.path.exists(wav) else None,
             "suspicion": proctor.score})
        st.session_state.idx += 1
        st.session_state.cur = -1
        if end or st.session_state.idx >= n:
            st.session_state.stage = "report"
        st.rerun()

    if total_left <= 0:
        save_and_advance(end=True)
    if idx >= n:
        st.session_state.stage = "report"
        st.rerun()

    q = qs[idx]
    qtext = q.get("question", "")

    # new question -> reading phase, then answering phase (mic starts after read)
    if st.session_state.cur != idx:
        st.session_state.update(cur=idx, sub="reading", q_present=now,
                                read_len=read_secs(qtext), nonce=0)

    if st.session_state.sub == "reading":
        if now - st.session_state.q_present >= st.session_state.read_len:
            start_rec()
            st.session_state.q_start = now
            st.session_state.sub = "answering"
            q_left = SECONDS_PER_Q
        else:
            q_left = SECONDS_PER_Q
    else:
        q_left = SECONDS_PER_Q - (now - st.session_state.q_start)
        if q_left <= 0:
            save_and_advance()

    with center:
        if st.session_state.sub == "reading":
            speak_browser(qtext, key=f"{idx}_{st.session_state.nonce}")

        reading = st.session_state.sub == "reading"
        color = "#d33" if (not reading and q_left <= 10) else "#111"
        big = "Reading…" if reading else fmt(q_left)
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:1rem;color:#666'>Total left "
            f"<b>{fmt(total_left)}</b></div>"
            f"<div style='font-size:2.8rem;font-weight:800;color:{color};"
            f"line-height:1.1'>{big}</div>"
            f"<div style='font-size:.85rem;color:#999'>Question {idx+1} of {n}</div>"
            f"<div style='font-size:1.55rem;font-weight:700;margin-top:1.1rem'>"
            f"{qtext}</div></div>", unsafe_allow_html=True)
        st.write("")
        b1, b2, b3 = st.columns(3)
        if b1.button("Repeat"):
            st.session_state.nonce += 1
            st.session_state.sub = "reading"
            st.session_state.q_present = now
            st.rerun()
        if b2.button("Submit & Next", type="primary"):
            save_and_advance()
        if b3.button("End interview"):
            save_and_advance(end=True)
        if not live:
            st.error("Camera is not running — please allow camera access.")
        if not _HAS_AUDIO:
            st.warning("Run: pip install sounddevice")

    _monitor_panel(left, proctor)


def _monitor_panel(col, proctor):
    with col:
        # live mic indicator — if this stays at 0 while you talk, the mic is dead
        if st.session_state.get("audio_chunks"):
            recent = st.session_state.audio_chunks[-6:]
            try:
                lvl = float(np.abs(np.concatenate(recent)).max()) / 32768.0
            except Exception:
                lvl = 0.0
            ok = lvl >= SILENCE_LEVEL
            st.markdown(
                f"<div style='font-size:.85rem;color:#666'>Mic "
                f"{'🎙️ hearing you' if ok else '⚠️ silent'}</div>",
                unsafe_allow_html=True)
            st.progress(min(lvl * 3, 1.0))

        s = proctor.score
        c = "#2e7d32" if s < 20 else "#f9a825" if s < 50 else "#c62828"
        st.markdown(
            f"<div style='font-size:.85rem;color:#666'>Danger score</div>"
            f"<div style='font-size:2.2rem;font-weight:800;color:{c}'>{s}</div>",
            unsafe_allow_html=True)
        st.markdown("**Monitoring activity**")
        if not proctor.events:
            st.caption("All clear.")
        for name, sc, t in reversed(proctor.events[-12:]):
            st.markdown(f"<div style='font-size:.82rem'>{t} — {name} "
                        f"<span style='color:#aaa'>({sc})</span></div>",
                        unsafe_allow_html=True)


# =======================================================================
# REPORT  — all the slow Groq work happens here, once, with a progress bar
# =======================================================================
def report_stage():
    proctor = st.session_state.proctor
    st.subheader("Interview report")

    if st.session_state.pending and not st.session_state.results:
        bar = st.progress(0.0, "Transcribing and scoring your answers...")
        tr = st.session_state.transcriber
        iv = st.session_state.interviewer
        total = len(st.session_state.pending)
        for i, p in enumerate(st.session_state.pending):
            answer = ""
            if p["wav"]:
                lvl = wav_level(p["wav"])
                if lvl < SILENCE_LEVEL:
                    answer = ""          # silence -> do NOT ask Whisper (it hallucinates)
                    st.warning(f"Q{i+1}: no speech was captured (mic level {lvl:.3f}).")
                else:
                    try:
                        answer = tr.transcribe(p["wav"])
                    except Exception:
                        answer = ""
            try:
                ev = iv.evaluate_answer(p["question"], answer,
                                        st.session_state.get("role", ""))
            except Exception:
                ev = {}
            st.session_state.results.append({
                "question": p["question"], "answer": answer,
                "score": ev.get("score", 0), "verdict": ev.get("verdict", ""),
                "feedback": ev.get("feedback", "")})
            bar.progress((i + 1) / total, f"Scored {i+1} of {total}")
        bar.empty()

    results = st.session_state.results
    avg = sum(r["score"] for r in results) / len(results) if results else 0
    risk = "LOW" if proctor.score < 20 else "MEDIUM" if proctor.score < 50 else "HIGH"
    c1, c2, c3 = st.columns(3)
    c1.metric("Average answer score", f"{avg:.1f} / 10")
    c2.metric("Final suspicion score", proctor.score)
    c3.metric("Integrity risk", risk)

    st.divider()
    for i, r in enumerate(results, 1):
        st.markdown(f"**Q{i}. {r['question']}**")
        st.write(f"*Answer:* {r['answer'] or '(none captured)'}")
        st.write(f"*Score:* {r['score']}/10 — {r['verdict']}")
        st.write(f"*Feedback:* {r['feedback']}")
        st.divider()

    st.subheader("Proctoring event log")
    if proctor.events:
        st.dataframe(pd.DataFrame(proctor.events, columns=["Event", "Score", "Time"]),
                     use_container_width=True)
    else:
        st.caption("No integrity events flagged.")

    lines = ["AI INTERVIEW REPORT", f"Role: {st.session_state.get('role','')}",
             f"Average answer score: {avg:.1f}/10",
             f"Integrity risk: {risk} (suspicion {proctor.score})", ""]
    for i, r in enumerate(results, 1):
        lines += [f"Q{i}: {r['question']}", f"  Answer: {r['answer']}",
                  f"  Score: {r['score']}/10 ({r['verdict']})",
                  f"  Feedback: {r['feedback']}", ""]
    st.download_button("Download report (.txt)", "\n".join(lines).encode(),
                       "interview_report.txt", "text/plain")

    if st.button("New interview"):
        for k in ("questions", "idx", "pending", "results", "role"):
            st.session_state.pop(k, None)
        for k, v in DEFAULTS.items():
            st.session_state.setdefault(k, v)
        st.session_state.stage = "setup"
        st.rerun()


# =======================================================================
stage = st.session_state.stage
if stage == "setup":
    setup_stage()
elif stage in ("ready", "interview"):
    exam_screen()
elif stage == "report":
    report_stage()