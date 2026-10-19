"""
voice.py  —  The speech layer (Phase B)

  * Text-to-speech: local, offline, free  (pyttsx3 -> Windows voices)
  * Speech-to-text:  Groq's free Whisper  (audio never loads into your RAM)

TTS needs no API key. STT reuses your GROQ_API_KEY from Step 1.

SETUP:
  pip install pyttsx3
  (groq is already installed from Step 1)

TEST:
  python voice.py
  It writes a spoken sentence to a WAV, sends that WAV to Groq Whisper, and
  prints the transcription back. If the text round-trips, both halves work.

Note for later: in the Streamlit app we won't play audio on the server. We'll
save the question to a WAV and let the browser play it via st.audio, and we'll
capture the candidate's answer with st.audio_input. This file gives us the
building blocks for that.
"""

import os
import pyttsx3
from groq import Groq

# Groq speech models. Swap if one stops being available (check console.groq.com):
#   whisper-large-v3-turbo -> fast, great for short answers  (default)
#   whisper-large-v3       -> a touch more accurate, slower
STT_MODEL = "whisper-large-v3-turbo"


# ---- Text to speech (local, offline) ----------------------------------
def _new_engine(rate=170, voice_index=None):
    # A fresh engine per call avoids pyttsx3's known re-use quirks on Windows.
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)          # words per minute
    if voice_index is not None:
        voices = engine.getProperty("voices")
        if 0 <= voice_index < len(voices):
            engine.setProperty("voice", voices[voice_index].id)
    return engine


def synth_to_file(text, path="question.wav", rate=170, voice_index=None):
    """Write spoken audio to a WAV file. This is what the Streamlit app will use."""
    engine = _new_engine(rate, voice_index)
    engine.save_to_file(text, path)
    engine.runAndWait()
    engine.stop()
    return path


def speak(text, rate=170, voice_index=None):
    """Speak aloud immediately (blocking). Handy for CLI testing."""
    engine = _new_engine(rate, voice_index)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def list_voices():
    """See which installed Windows voices you can pick with voice_index."""
    engine = pyttsx3.init()
    return [(i, v.name) for i, v in enumerate(engine.getProperty("voices"))]


# ---- Speech to text (Groq Whisper, cloud, free) -----------------------
class Transcriber:
    def __init__(self, model=STT_MODEL, api_key=None):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Groq API key found. Set GROQ_API_KEY (same key as Step 1)."
            )
        self.client = Groq(api_key=api_key)
        self.model = model

    def transcribe(self, audio_path, language="en"):
        """Send an audio file to Groq Whisper and return the transcribed text."""
        with open(audio_path, "rb") as f:
            result = self.client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f.read()),
                model=self.model,
                language=language,
                response_format="text",
            )
        return str(result).strip()


# -----------------------------------------------------------------------
if __name__ == "__main__":
    sentence = ("Hello, this is a test of the AI interview system. "
                "Please describe your experience with Python.")

    print("Installed voices:")
    for i, name in list_voices():
        print(f"  [{i}] {name}")

    print("\n1) Synthesizing speech to test_tts.wav ...")
    synth_to_file(sentence, "test_tts.wav")

    print("2) Transcribing that WAV with Groq Whisper ...")
    text = Transcriber().transcribe("test_tts.wav")
    print("   Transcription:", text)

    print("\n3) Speaking a line out loud (you should hear this) ...")
    speak("Text to speech is working.")

    print("\nDone. If the transcription matches the sentence and you heard "
          "the voice, the speech layer works.")