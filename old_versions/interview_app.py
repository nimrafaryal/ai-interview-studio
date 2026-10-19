"""
interview_app.py  —  Full AI interview + proctoring (Phase C)
Run with:  streamlit run interview_app.py

Flow:
  Setup  -> enter role/JD, generate questions (Groq)
  Interview (per question):
      - AI speaks the question (local TTS -> browser plays it)
      - Candidate clicks "Answering": a live loop records the mic AND runs the
        proctor on the webcam at the same time (continuous monitoring)
      - Candidate unticks "Answering" -> answer is transcribed (Groq Whisper)
        and scored (Groq LLM)
  Report -> per-question scores + proctoring risk + downloadable summary

Needs:  pip install sounddevice   (interviewer.py, voice.py, engine.py, and
your GROQ_API_KEY must already be set up from the earlier steps.)
"""

import os
import wave
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from engine import Proctor
from interviewer import Interviewer
from voice import synth_to_file, Transcriber

try:
    import sounddevice as sd
    _HAS_AUDIO = True
except Exception:
    _HAS_AUDIO = False

SR = 16000  # mic sample rate
os.makedirs("answers", exist_ok=True)

st.set_page_config(page_title="AI Interview + Proctoring", layout="wide")
st.title("AI Interview + Proctoring")


# --- one-time, lightweight clients -------------------------------------
def _init_clients():
    if "interviewer" not in st.session_state:
        try:
            st.session_state.interviewer = Interviewer()
            st.session_state.transcriber = Transcriber()
        except Exception as e:
            st.error(f"Could not connect to Groq. Is GROQ_API_KEY set?  ({e})")
            st.stop()


for k, v in {"stage": "setup", "questions": [], "idx": 0, "results": []}.items():
    st.session_state.setdefault(k, v)


# =======================================================================
# STAGE 1 — SETUP
# =======================================================================
def setup_stage():
    _init_clients()
    st.subheader("1. Set up the interview")
    role = st.text_input("Role", "Junior Python Developer")
    jd = st.text_area("Job description / focus (optional)", "")
    n = st.slider("Number of questions", 3, 8, 4)
    diff = st.select_slider("Difficulty", ["easy", "mixed", "hard"], "mixed")

    if st.button("Generate questions & start"):
        with st.spinner("Generating questions..."):
            qs = st.session_state.interviewer.generate_questions(
                role, jd, n=n, difficulty=diff)
        if not qs:
            st.error("No questions came back. Try again.")
            return
        st.session_state.role = role
        st.session_state.questions = qs
        st.session_state.idx = 0
        st.session_state.results = []
        # load the heavy vision models only now, and start a fresh score
        if "proctor" not in st.session_state:
            with st.spinner("Loading proctoring models..."):
                st.session_state.proctor = Proctor()
        st.session_state.proctor.reset()
        st.session_state.stage = "interview"
        st.rerun()


# =======================================================================
# STAGE 2 — INTERVIEW (one question at a time)
# =======================================================================
def record_loop(question_text):
    """Blocking loop: record mic + run proctor on webcam until 'Answering' is
    unticked. Saves the answer WAV. Returns nothing (writes to disk)."""
    proctor = st.session_state.proctor
    idx = st.session_state.idx
    wav_path = f"answers/answer_{idx}.wav"

    video_slot = st.empty()
    score_slot = st.empty()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    chunks = []

    def audio_cb(indata, frames, t, status):
        chunks.append(indata.copy())

    stream = None
    if _HAS_AUDIO:
        stream = sd.InputStream(samplerate=SR, channels=1,
                                dtype="int16", callback=audio_cb)
        stream.start()

    try:
        while True:  # broken when the checkbox is unticked (Streamlit rerun)
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            annotated, score, _ = proctor.process_frame(frame)
            video_slot.image(annotated, channels="BGR", use_container_width=True)
            score_slot.metric("Live suspicion score", score)
    finally:
        cap.release()
        if stream is not None:
            stream.stop()
            stream.close()
        if chunks:
            data = np.concatenate(chunks, axis=0)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SR)
                wf.writeframes(data.tobytes())


def interview_stage():
    proctor = st.session_state.proctor
    qs = st.session_state.questions
    idx = st.session_state.idx
    total = len(qs)

    if idx >= total:
        st.session_state.stage = "report"
        st.rerun()

    q = qs[idx]
    qtext = q.get("question", "")
    st.subheader(f"Question {idx + 1} of {total}")
    st.markdown(f"**{qtext}**")
    st.caption(f"Topic: {q.get('topic','?')}  |  Difficulty: {q.get('difficulty','?')}")

    # Speak the question
    if st.button("Hear the question"):
        path = synth_to_file(qtext, f"answers/q_{idx}.wav")
        st.audio(open(path, "rb").read(), format="audio/wav", autoplay=True)

    if not _HAS_AUDIO:
        st.warning("sounddevice not installed — run: pip install sounddevice")

    wav_path = f"answers/answer_{idx}.wav"
    answering = st.checkbox("Answering (records mic + monitors camera)")

    if answering:
        st.info("Answer out loud. Untick the box above when you're done.")
        record_loop(qtext)
    else:
        if os.path.exists(wav_path):
            st.audio(open(wav_path, "rb").read(), format="audio/wav")
            if st.button("Submit & score this answer"):
                with st.spinner("Transcribing..."):
                    answer = st.session_state.transcriber.transcribe(wav_path)
                with st.spinner("Scoring..."):
                    ev = st.session_state.interviewer.evaluate_answer(
                        qtext, answer, st.session_state.role)
                st.session_state.results.append({
                    "question": qtext,
                    "answer": answer,
                    "score": ev.get("score", 0),
                    "verdict": ev.get("verdict", ""),
                    "feedback": ev.get("feedback", ""),
                    "followup": ev.get("followup", ""),
                    "suspicion_at_this_point": proctor.score,
                })
                st.session_state.idx += 1
                st.rerun()
        else:
            st.caption("Tick 'Answering' to record your spoken answer.")

    # live proctoring risk in the sidebar
    _sidebar_risk(proctor)


def _sidebar_risk(proctor):
    st.sidebar.header("Live integrity")
    s = proctor.score
    if s < 20:
        st.sidebar.success(f"LOW RISK — {s}")
    elif s < 50:
        st.sidebar.warning(f"MEDIUM RISK — {s}")
    else:
        st.sidebar.error(f"HIGH RISK — {s}")
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

    if results:
        avg = sum(r["score"] for r in results) / len(results)
    else:
        avg = 0
    risk = ("LOW" if proctor.score < 20 else
            "MEDIUM" if proctor.score < 50 else "HIGH")

    c1, c2, c3 = st.columns(3)
    c1.metric("Average answer score", f"{avg:.1f} / 10")
    c2.metric("Final suspicion score", proctor.score)
    c3.metric("Integrity risk", risk)

    st.divider()
    for i, r in enumerate(results, 1):
        st.markdown(f"**Q{i}. {r['question']}**")
        st.write(f"*Answer:* {r['answer'] or '(no answer captured)'}")
        st.write(f"*Score:* {r['score']}/10 — {r['verdict']}")
        st.write(f"*Feedback:* {r['feedback']}")
        st.divider()

    st.subheader("Proctoring event log")
    if proctor.events:
        ev_df = pd.DataFrame(proctor.events, columns=["Event", "Score", "Time"])
        st.dataframe(ev_df, use_container_width=True)
    else:
        st.caption("No integrity events flagged.")

    # downloadable combined report
    lines = [f"AI INTERVIEW REPORT",
             f"Role: {st.session_state.get('role','')}",
             f"Average answer score: {avg:.1f}/10",
             f"Integrity risk: {risk} (suspicion score {proctor.score})", ""]
    for i, r in enumerate(results, 1):
        lines += [f"Q{i}: {r['question']}",
                  f"  Answer: {r['answer']}",
                  f"  Score: {r['score']}/10 ({r['verdict']})",
                  f"  Feedback: {r['feedback']}", ""]
    report_txt = "\n".join(lines)
    st.download_button("Download report (.txt)", report_txt.encode(),
                       "interview_report.txt", "text/plain")

    if st.button("Start a new interview"):
        for k in ("questions", "idx", "results", "role"):
            st.session_state.pop(k, None)
        st.session_state.stage = "setup"
        st.rerun()


# =======================================================================
STAGE = st.session_state.stage
if STAGE == "setup":
    setup_stage()
elif STAGE == "interview":
    interview_stage()
elif STAGE == "report":
    report_stage()