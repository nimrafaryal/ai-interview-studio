"""
ai_brain.py  -  The AI interviewer (Groq free tier: LLM + Whisper speech-to-text)

Free key: https://console.groq.com/keys  (no credit card)

Improvements over interviewer.py / voice.py:
  * Model fallback: if one Groq model is retired or rate-limited, the next is tried.
  * Questions can be tailored to the candidate's CV / skills.
  * Answers are transcribed + scored IN THE BACKGROUND while the candidate is
    already on the next question, so the report is ready almost instantly.
  * Built-in question bank, so the interview still runs if the API is down.
  * Final hiring summary written by the AI.
"""

import os
import json
import time
import random

from groq import Groq

CHAT_MODELS = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b",
               "openai/gpt-oss-20b", "llama-3.1-8b-instant"]
STT_MODELS = ["whisper-large-v3-turbo", "whisper-large-v3"]

FALLBACK_QUESTIONS = [
    {"question": "Tell me about yourself and why you are interested in this role.", "topic": "Introduction", "difficulty": "easy"},
    {"question": "Describe a project you are proud of. What was your exact contribution?", "topic": "Experience", "difficulty": "easy"},
    {"question": "Tell me about a time you faced a difficult problem. How did you solve it?", "topic": "Problem solving", "difficulty": "medium"},
    {"question": "How do you learn a new technology or skill quickly? Give an example.", "topic": "Learning", "difficulty": "easy"},
    {"question": "Describe a disagreement with a teammate and how you handled it.", "topic": "Teamwork", "difficulty": "medium"},
    {"question": "How do you make sure your work is correct and high quality?", "topic": "Quality", "difficulty": "medium"},
    {"question": "Where do you see yourself in three years, and how does this role fit?", "topic": "Goals", "difficulty": "easy"},
    {"question": "What is one mistake you made at work or university, and what did you learn?", "topic": "Self-awareness", "difficulty": "medium"},
]


def _safe_json(text):
    if not text:
        return {}
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(t[a:b + 1])
        except Exception:
            pass
    return {}


class Brain:
    def __init__(self, api_key):
        if not api_key:
            raise RuntimeError("No Groq API key")
        self.client = Groq(api_key=api_key, timeout=60, max_retries=1)
        self.last_model = None

    # ------------------------------------------------------------- plumbing
    def _chat(self, messages, json_mode=True, temperature=0.4, max_tokens=1500):
        last = None
        for model in CHAT_MODELS:
            kw = dict(model=model, messages=messages, temperature=temperature,
                      max_tokens=max_tokens)
            if json_mode:
                kw["response_format"] = {"type": "json_object"}
            for attempt in range(2):
                try:
                    r = self.client.chat.completions.create(**kw)
                    self.last_model = model
                    return r.choices[0].message.content
                except Exception as e:
                    last = e
                    msg = str(e).lower()
                    if "model" in msg and ("not found" in msg or "decommission" in msg
                                           or "does not exist" in msg):
                        break                       # try next model
                    if "401" in msg or "invalid api key" in msg:
                        raise
                    time.sleep(1.5 * (attempt + 1))  # rate limit / network
        raise last

    def check_key(self):
        """Cheap call to confirm the key works. Returns (ok, message)."""
        try:
            self.client.models.list()
            return True, "Connected to Groq"
        except Exception as e:
            return False, str(e)[:200]

    # ------------------------------------------------------------ questions
    def generate_questions(self, role, jd="", cv="", n=5, difficulty="mixed"):
        system = ("You are a senior interviewer. Write clear, fair interview questions "
                  "that can be answered OUT LOUD in about a minute each. Mix technical "
                  "and behavioural questions. One question per item, no multi-part "
                  "questions, no code writing. Respond ONLY with JSON.")
        user = (f"Role: {role}\n"
                f"Job description / focus: {jd or 'general fit for the role'}\n"
                f"Candidate background / CV (may be empty): {cv[:3000] or 'not provided'}\n"
                f"Difficulty: {difficulty}\nNumber of questions: {n}\n"
                'Return: {"questions": [{"question": "...", "topic": "...", '
                '"difficulty": "easy|medium|hard", "ideal_points": ["key point", "..."]}]}')
        qs = []
        try:
            for _ in range(3):
                data = _safe_json(self._chat(
                    [{"role": "system", "content": system},
                     {"role": "user", "content": user}], temperature=0.8))
                qs += [q for q in data.get("questions", []) if q.get("question")]
                if len(qs) >= n:
                    break
        except Exception:
            pass
        if len(qs) < n:                       # top up from the offline bank
            pool = [q for q in FALLBACK_QUESTIONS if q not in qs]
            random.shuffle(pool)
            qs += pool[:n - len(qs)]
        return qs[:n]

    # ------------------------------------------------------------- scoring
    def evaluate(self, question, answer, role="", ideal_points=None):
        if not answer or len(answer.split()) < 3:
            return {"score": 0, "verdict": "No answer",
                    "feedback": "No meaningful spoken answer was captured.",
                    "strengths": [], "improvements": ["Answer the question out loud."],
                    "communication": 0}
        system = ("You are a strict but fair interviewer grading a SPOKEN answer that "
                  "was transcribed by speech-to-text (ignore small transcription errors "
                  "and filler words). Judge relevance, correctness, depth, structure and "
                  "clarity. Respond ONLY with JSON.")
        user = (f"Role: {role or 'general'}\nQuestion: {question}\n"
                f"Key points a strong answer covers: {ideal_points or 'use your judgement'}\n"
                f"Candidate answer: {answer}\n\n"
                'Return JSON: {"score": integer 0-10, "communication": integer 0-10, '
                '"verdict": "Excellent|Good|Average|Weak|Off-topic", '
                '"feedback": "2-3 sentences, speak to the candidate as \'you\'", '
                '"strengths": ["..."], "improvements": ["..."], '
                '"model_answer": "a 2-3 sentence example of a strong answer", '
                '"clean_answer": "the candidate\'s answer, lightly cleaned for reading: remove filler words '
                'and repetitions, fix obvious speech-to-text mistakes and punctuation. Keep their meaning and '
                'words; do NOT add anything they did not say."}')
        data = _safe_json(self._chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}], temperature=0.2))
        try:
            data["score"] = max(0, min(10, int(round(float(data.get("score", 0))))))
        except Exception:
            data["score"] = 0
        try:
            data["communication"] = max(0, min(10, int(round(float(data.get("communication", 0))))))
        except Exception:
            data["communication"] = 0
        for k in ("verdict", "feedback", "model_answer", "clean_answer"):
            data.setdefault(k, "")
        for k in ("strengths", "improvements"):
            if not isinstance(data.get(k), list):
                data[k] = []
        return data

    def summary(self, role, results, integrity, attention):
        brief = [{"q": r["question"], "score": r["score"], "verdict": r.get("verdict", "")}
                 for r in results]
        user = (f"Role: {role}\nPer-question results: {json.dumps(brief)}\n"
                f"Proctoring integrity: {integrity}/100, attention on screen: {attention:.0f}%\n"
                'Return JSON: {"recommendation": "Strong hire|Hire|Maybe|No hire", '
                '"summary": "3-4 sentence overall assessment", '
                '"top_strengths": ["..."], "focus_areas": ["..."]}')
        try:
            return _safe_json(self._chat(
                [{"role": "system", "content": "You are a hiring manager writing a short, "
                  "honest interview summary. Respond ONLY with JSON."},
                 {"role": "user", "content": user}], temperature=0.3))
        except Exception:
            return {}

    # --------------------------------------------------------- speech -> text
    def transcribe(self, path, context=""):
        """Speech -> text with Groq Whisper.

        Whisper sometimes INVENTS sentences when audio is silent or noisy
        ("Thanks for watching", "For the first time of the day..."). We ask for
        per-segment confidence and drop segments that are probably not speech."""
        last = None
        with open(path, "rb") as f:
            data = f.read()
        for model in STT_MODELS:
            try:
                r = self.client.audio.transcriptions.create(
                    file=(os.path.basename(path), data), model=model, language="en",
                    response_format="verbose_json", temperature=0.0,
                    prompt="A candidate answering a job interview question in English.")
                d = r.model_dump() if hasattr(r, "model_dump") else dict(r)
                segs = d.get("segments") or []
                if not segs:
                    return (d.get("text") or "").strip()
                kept = [sg.get("text", "") for sg in segs
                        if not (sg.get("no_speech_prob", 0) > 0.6 and sg.get("avg_logprob", 0) < -0.7)
                        and sg.get("compression_ratio", 1) < 2.6]
                return " ".join(t.strip() for t in kept).strip()
            except Exception as e:
                last = e
                time.sleep(1)
        raise last

    # ---------------------------------------------- full pipeline per answer
    def process_answer(self, item, role, silence_level):
        """Run in a background thread. item: dict with question, wav, peak."""
        out = dict(item)
        answer, err = "", ""
        if item.get("wav") and item.get("peak", 0) >= silence_level:
            try:
                answer = self.transcribe(item["wav"], item["question"])
            except Exception as e:
                err = f"Transcription failed: {e}"
        elif item.get("wav"):
            err = "Microphone captured only silence."
        else:
            err = "No audio recorded."
        try:
            ev = self.evaluate(item["question"], answer, role, item.get("ideal_points"))
        except Exception as e:
            ev = {"score": 0, "verdict": "Not scored", "communication": 0,
                  "feedback": f"AI scoring failed: {e}", "strengths": [], "improvements": []}
        out.update(ev)
        out.update(answer=answer, error=err)
        return out