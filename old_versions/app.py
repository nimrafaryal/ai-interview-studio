"""
app.py  —  AI Online Interview Proctoring System
Run with:  streamlit run app.py

On start, look straight at your screen for ~2 seconds while it CALIBRATES,
then move around to test. Use "Recalibrate" any time your seating changes.
"""

import cv2
import pandas as pd
import streamlit as st

from engine import Proctor

st.set_page_config(page_title="AI Proctoring System", layout="wide")
st.title("AI Online Interview Proctoring System")

if "proctor" not in st.session_state:
    with st.spinner("Loading models..."):
        st.session_state.proctor = Proctor()
proctor = st.session_state.proctor

# --- controls ----------------------------------------------------------
c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
run = c1.checkbox("Start Camera")
if c2.button("Reset"):
    proctor.reset()
    st.rerun()
if c3.button("Recalibrate"):
    proctor.recalibrate()
    st.rerun()
cam_index = c4.number_input("Camera index", 0, 5, 0, step=1)

# --- detection settings (applied to the persistent proctor each run) ---
with st.sidebar.expander("Detection settings", expanded=False):
    sens = st.select_slider("Gaze sensitivity",
                            ["Tight", "Normal", "Loose"], value="Normal")
    delay = st.slider("Look-away delay (sec)", 0.3, 3.0, 0.7, 0.1)
    obj_conf = st.slider("Object confidence", 0.25, 0.70, 0.40, 0.05)
    person_conf = st.slider("Person confidence", 0.40, 0.80, 0.60, 0.05)

proctor.set_sensitivity(sens)
proctor.look_away_seconds = delay
proctor.object_conf = obj_conf
proctor.person_conf = person_conf

video_slot = st.empty()
metric_slot = st.empty()

st.sidebar.header("Dashboard")
timeline_slot = st.sidebar.empty()
risk_slot = st.sidebar.empty()


def render_dashboard():
    with timeline_slot.container():
        st.subheader("Event Timeline")
        if not proctor.events:
            st.caption("No events yet.")
        else:
            for name, sc, t in reversed(proctor.events[-10:]):
                st.write(f"{t} — {name}  (score {sc})")
    with risk_slot.container():
        st.subheader("Risk Level")
        s = proctor.score
        if s < 20:
            st.success(f"LOW RISK — {s}")
        elif s < 50:
            st.warning(f"MEDIUM RISK — {s}")
        else:
            st.error(f"HIGH RISK — {s}")


if run:
    cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(int(cam_index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        st.error("Could not open the camera. Try a different camera index.")
    else:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    st.warning("Lost the camera feed.")
                    break
                frame = cv2.flip(frame, 1)
                annotated, score, _ = proctor.process_frame(frame)
                video_slot.image(annotated, channels="BGR", use_container_width=True)
                metric_slot.metric("Suspicion Score", score)
                render_dashboard()
        finally:
            cap.release()
else:
    video_slot.info("Camera is off. Tick 'Start Camera' to begin.")
    render_dashboard()

st.sidebar.divider()
if proctor.events:
    df = pd.DataFrame(proctor.events, columns=["Event", "Score", "Time"])
    st.sidebar.download_button(
        "Download Report (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="cheating_report.csv",
        mime="text/csv",
    )
else:
    st.sidebar.caption("No events to report yet.")