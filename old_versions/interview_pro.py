"""
interview_pro.py  —  Professional flow: camera ON for the whole interview
Run with:  streamlit run interview_pro.py

What changed vs interview_app.py:
  * The webcam runs continuously via streamlit-webrtc (a background thread),
    so proctoring never stops during the interview.
  * The main screen shows each question (text + spoken), and records answers
    with a non-blocking Start/Stop, so the UI never freezes.

Needs:  pip install streamlit-webrtc sounddevice
(engine.py, interviewer.py, voice.py and GROQ_API_KEY must already be set up.)
"""

import os
import wave

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from engine import Proctor
from interviewer import Interviewer
from voice import synth_to_file, Transcriber

try:
    import sounddevice as sd
    _HAS_AUDIO = True
except Exception:
    _HAS_AUDIO = False

SR = 16000
os.makedirs("answers", exist_ok=True)

RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

st.set_page_config(page_title="AI Interview (Pro)", layout="wide")
st.title("AI Interview + Live Proctoring")

# --- session defaults ---------------------------------------------------
for k, v in {"stage": "setup", "questions": [], "idx": 0,
             "results": [], "qstate": "ask", "played_idx": -1}.items():
    st.session_state.setdefault(k, v)


def _clients():
    if "interviewer" not in st.session_state:
        try:
            st.session_state.interviewer = Interviewer()
            st.session_state.transcriber = Transcriber()
        except Exception as e:
            st.error(f"Could not connect to Groq. Is GROQ_API_KEY set?  ({e})")
            st.stop()


# --- non-blocking mic recording ----------------------------------------
def start_recording():
    chunks = []
    st.session_state.audio_chunks = chunks

    def cb(indata, frames, t, status):
        chunks.append(indata.copy())

    stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16", callback=cb)
    stream.start()
    st.session_state.audio_stream = stream


def stop_recording(path):
    stream = st.session_state.pop("audio_stream", None)
    if stream is not None:
        stream.stop()
        stream.close()
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


# =======================================================================
# STAGE 1 — SETUP
# =======================================================================
def setup_stage():
    _clients()
    st.subheader("Set up the interview")
    role = st.text_input("Role", "Junior Python Developer")
    jd = st.text_area("Job description / focus (optional)", "")
    n = st.slider("Number of questions", 3, 8, 4)
    diff = st.select_slider("Difficulty", ["easy", "mixed", "hard"], "mixed")

    if st.button("Generate questions & start interview"):
        with st.spinner("Generating questions..."):
            qs = st.session_state.interviewer.generate_questions(
                role, jd, n=n, difficulty=diff)
        if not qs:
            st.error("No questions returned. Try again.")
            return
        st.session_state.update(role=role, questions=qs, idx=0,
                                results=[], qstate="ask", played_idx=-1)
        if "proctor" not in st.session_state:
            with st.spinner("Loading proctoring models..."):
                st.session_state.proctor = Proctor()
        st.session_state.proctor.reset()
        st.session_state.stage = "interview"
        st.rerun()


# =======================================================================
# STAGE 2 — INTERVIEW (camera always on)
# =======================================================================
def interview_stage():
    proctor = st.session_state.proctor
    qs = st.session_state.questions
    idx = st.session_state.idx

    if idx >= len(qs):
        st.session_state.stage = "report"
        st.rerun()

    # ---- continuous camera: this callback runs on EVERY frame ----
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        annotated, _, _ = proctor.process_frame(img)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    left, right = st.columns([1, 1])

    with left:
        st.caption("Camera stays on for the whole interview.")
        webrtc_streamer(
            key="proctor_cam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )

    with right:
        q = qs[idx]
        qtext = q.get("question", "")
        st.markdown(f"### Question {idx + 1} of {len(qs)}")
        st.markdown(f"## {qtext}")
        st.caption(f"Topic: {q.get('topic','?')} | "
                   f"Difficulty: {q.get('difficulty','?')}")

        # speak the question once when we arrive at it (+ repeat button)
        if st.session_state.played_idx != idx:
            path = synth_to_file(qtext, f"answers/q_{idx}.wav")
            st.audio(open(path, "rb").read(), format="audio/wav", autoplay=True)
            st.session_state.played_idx = idx
        if st.button("Repeat question"):
            path = synth_to_file(qtext, f"answers/q_{idx}.wav")
            st.audio(open(path, "rb").read(), format="audio/wav", autoplay=True)

        if not _HAS_AUDIO:
            st.warning("Run: pip install sounddevice")

        wav_path = f"answers/answer_{idx}.wav"
        qstate = st.session_state.qstate

        if qstate == "ask":
            if st.button("Start answering", type="primary"):
                start_recording()
                st.session_state.qstate = "recording"
                st.rerun()

        elif qstate == "recording":
            st.info("Recording your answer... speak, then click Stop.")
            if st.button("Stop answering", type="primary"):
                stop_recording(wav_path)
                st.session_state.qstate = "review"
                st.rerun()

        elif qstate == "review":
            if os.path.exists(wav_path):
                st.audio(open(wav_path, "rb").read(), format="audio/wav")
            c1, c2 = st.columns(2)
            if c1.button("Submit & score", type="primary"):
                with st.spinner("Transcribing + scoring..."):
                    answer = st.session_state.transcriber.transcribe(wav_path) \
                        if os.path.exists(wav_path) else ""
                    ev = st.session_state.interviewer.evaluate_answer(
                        qtext, answer, st.session_state.role)
                st.session_state.results.append({
                    "question": qtext, "answer": answer,
                    "score": ev.get("score", 0),
                    "verdict": ev.get("verdict", ""),
                    "feedback": ev.get("feedback", ""),
                    "suspicion_at_this_point": proctor.score,
                })
                st.session_state.idx += 1
                st.session_state.qstate = "ask"
                st.rerun()
            if c2.button("Re-record"):
                st.session_state.qstate = "ask"
                st.rerun()

    _sidebar(proctor)


def _sidebar(proctor):
    st.sidebar.header("Live integrity")
    s = proctor.score
    if s < 20:
        st.sidebar.success(f"LOW RISK — {s}")
    elif s < 50:
        st.sidebar.warning(f"MEDIUM RISK — {s}")
    else:
        st.sidebar.error(f"HIGH RISK — {s}")
    if st.sidebar.button("Recalibrate gaze"):
        proctor.recalibrate()
    st.sidebar.subheader("Recent events")
    if not proctor.events:
        st.sidebar.caption("None yet.")
    for name, sc, t in reversed(proctor.events[-8:]):
        st.sidebar.write(f"{t} — {name} ({sc})")


# =======================================================================
# STAGE 3 — REPORT
# =======================================================================
def report_stage():
    proctor = st.session_state.proctor
    results = st.session_state.results
    st.subheader("Interview report")

    avg = sum(r["score"] for r in results) / len(results) if results else 0
    risk = ("LOW" if proctor.score < 20 else
            "MEDIUM" if proctor.score < 50 else "HIGH")

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
        st.dataframe(pd.DataFrame(proctor.events,
                                  columns=["Event", "Score", "Time"]),
                     use_container_width=True)
    else:
        st.caption("No integrity events flagged.")

    lines = ["AI INTERVIEW REPORT",
             f"Role: {st.session_state.get('role','')}",
             f"Average answer score: {avg:.1f}/10",
             f"Integrity risk: {risk} (suspicion {proctor.score})", ""]
    for i, r in enumerate(results, 1):
        lines += [f"Q{i}: {r['question']}", f"  Answer: {r['answer']}",
                  f"  Score: {r['score']}/10 ({r['verdict']})",
                  f"  Feedback: {r['feedback']}", ""]
    st.download_button("Download report (.txt)", "\n".join(lines).encode(),
                       "interview_report.txt", "text/plain")

    if st.button("Start a new interview"):
        for k in ("questions", "idx", "results", "role"):
            st.session_state.pop(k, None)
        st.session_state.update(stage="setup", qstate="ask", played_idx=-1)
        st.rerun()


# =======================================================================
stage = st.session_state.stage
if stage == "setup":
    setup_stage()
elif stage == "interview":
    interview_stage()
elif stage == "report":
    report_stage()