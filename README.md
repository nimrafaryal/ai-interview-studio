# AI Interview Studio

An AI interview system with live proctoring that runs on one laptop. Everything it uses is free.

* **AI interviewer**: Groq's free API (Llama 3.3 70B) writes the questions and scores the answers.
* **Speech-to-text**: Groq's free Whisper turns your spoken answer into text.
* **Voice**: your browser reads each question aloud (offline, no cost).
* **Proctoring**: MediaPipe tracks your face, eyes and head, and YOLOv8n spots phones and books. Both run offline on your CPU.

---

## First-time setup (about 10 minutes, once)

1. **Install Python 3.11** from https://www.python.org/downloads/release/python-3119/.
   On the first installer screen, tick **"Add python.exe to PATH"**.
2. **Get a free Groq key** at https://console.groq.com/keys.
   Sign in with Google, click *Create API Key*, and copy the key (it starts with `gsk_`).
3. Double-click **`setup.bat`**. It installs everything into a private `.venv` folder.

## Every time you use it

Double-click **`run.bat`**. The app opens in your browser at http://localhost:8501.

The first time, paste your Groq key into the box on the right of the setup screen and tick
*Remember on this laptop*. The app saves it to `.streamlit/secrets.toml`, so you only do this once.

## How an interview goes

| Step | What happens |
|---|---|
| **1 · Setup** | Enter the role, an optional job description and the candidate's skills or CV. Pick the number of questions, difficulty and time per answer. The AI writes the questions. |
| **2 · System check** | The camera turns on. Look at the screen for 2 seconds so it can calibrate your gaze. Click **Test mic**, which records 4 seconds and shows what Whisper heard. All checks must be green to start. |
| **3 · Interview** | Each question is shown and read aloud, and then recording starts by itself. Click **Next** when you finish, or the timer moves on. While you answer the next question, the AI transcribes and scores the previous one in the background. |
| **4 · Report** | You get an answer-quality score, a communication score, integrity (0–100), the share of time you looked at the screen, and an AI hiring recommendation. It also has feedback and an example strong answer for each question, an integrity chart, snapshots of flagged moments, and downloads (HTML report, CSV, JSON). Every report is saved automatically to `reports/`. |

## What the proctor flags

| Event | Penalty |
|---|---|
| Looking away for longer than the grace time (default 1.5 s) | −3 |
| Face not visible for more than 2 s | −6 |
| Book or notes | −8 |
| Phone | −15 |
| A second person or face | −20 |

Integrity starts at 100. **80 or above = low risk, 50–79 = medium, below 50 = high.**
Each type of event has a cooldown, and an object has to be seen several times in a row before it counts.
That stops the old problem where one phone was logged 30 times (see the old `cheating_report.csv`).
A water bottle no longer counts as cheating.

Under **System check → Proctoring settings** you can change the gaze sensitivity, the grace time and the alarm sound.

## Files

```
app.py          the app (the only file you run)
engine.py       proctoring: face, gaze, phone and person detection
devices.py      camera and microphone, each in a background thread
ai_brain.py     Groq: questions, scoring, Whisper, summary
report.py       builds the downloadable HTML report
models/         yolov8n.pt, face_landmarker.task, alarm.wav
old_versions/   your previous scripts, kept only for reference
```

## Troubleshooting

* **"Could not open camera"**: close Zoom, Teams, the Windows Camera app or any other app using the webcam. If you use an external webcam, try camera number 1.
* **The mic test says "silence"**: open Windows Settings → Privacy & security → Microphone and turn on *Let desktop apps access your microphone*. Then pick another microphone in the dropdown.
* **No voice reading the question**: use Chrome or Edge and click anywhere on the page once. Browsers only allow sound after you interact with the page.
* **"Key not working"**: paste the key again and check that no spaces were copied with it. The free tier has a daily limit, so if you hit it, wait and try again later.
* **Too many "Looking away" flags**: click *Recalibrate gaze*, or set the sensitivity to *Relaxed*.
* **Slow on an old laptop**: this is normal. YOLO only runs on every 5th frame, and the video preview runs at about 6–10 fps.
