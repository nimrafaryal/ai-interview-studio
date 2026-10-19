"""
interviewer.py  —  The AI interviewer brain (Phase A)

Talks to Groq's free API to:
  * generate interview questions from a role / job description
  * score a candidate's answer and return structured JSON

Groq is OpenAI-compatible and free (no credit card). This module has NO voice
and NO camera yet on purpose — get this working with typed input first, then the
speech and proctoring layers plug on top of it.

SETUP (do this once):
  1) pip install groq
  2) Get a free key at https://console.groq.com  (sign up, create API key)
  3) In PowerShell, set it for this session:
         $env:GROQ_API_KEY = "gsk_your_key_here"
     ...or permanently (reopen the terminal afterwards):
         setx GROQ_API_KEY "gsk_your_key_here"
  4) Test it:
         python interviewer.py
"""

import os
import json
import time

from groq import Groq

# Model IDs change over time on Groq. If you get a "model not found" error, run
# Interviewer().list_models() (or check console.groq.com) and swap this.
#   llama-3.3-70b-versatile -> best quality, lower daily limit  (good for interviews)
#   llama-3.1-8b-instant    -> faster + higher limits, a bit shallower
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _safe_json(text):
    """Parse JSON even if the model wrapped it in ``` fences."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        t = text.strip()
        for fence in ("```json", "```"):
            if t.startswith(fence):
                t = t[len(fence):]
        if t.endswith("```"):
            t = t[:-3]
        try:
            return json.loads(t.strip())
        except Exception:
            return {}


class Interviewer:
    def __init__(self, model=DEFAULT_MODEL, api_key=None):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Groq API key found. Set the GROQ_API_KEY environment "
                "variable (see the setup notes at the top of this file)."
            )
        self.client = Groq(api_key=api_key)
        self.model = model

    # -------------------------------------------------------------------
    def list_models(self):
        """Handy when a model id stops working — prints what's available to you."""
        return [m.id for m in self.client.models.list().data]

    def _chat(self, messages, json_mode=False, temperature=0.7,
              max_tokens=1024, retries=3):
        kwargs = dict(model=self.model, messages=messages,
                      temperature=temperature, max_tokens=max_tokens)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_err = None
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            except Exception as e:      # includes rate-limit (429) errors
                last_err = e
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)   # 1s, 2s, 4s backoff
        raise last_err

    # -------------------------------------------------------------------
    def generate_questions(self, role, job_description="", n=5,
                           difficulty="mixed"):
        """Return a list of dicts: {question, topic, difficulty}."""
        system = (
            "You are an experienced technical interviewer. You write clear, "
            "fair questions that can be answered out loud in 1-3 minutes. "
            "Avoid trick questions. Respond ONLY with JSON."
        )
        user = (
            f"Create {n} interview questions for the role: {role}.\n"
            f"Job description / focus: {job_description or 'general fit for the role'}.\n"
            f"Difficulty: {difficulty}.\n"
            'Return JSON of the form: '
            '{"questions": [{"question": "...", "topic": "...", '
            '"difficulty": "easy|medium|hard"}]}'
        )
        content = self._chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            json_mode=True, temperature=0.8,
        )
        return _safe_json(content).get("questions", [])

    def evaluate_answer(self, question, answer, role=""):
        """Return a dict: {score, verdict, feedback, strengths, concerns, followup}."""
        if not answer or not answer.strip():
            return {"score": 0, "verdict": "No answer",
                    "feedback": "The candidate did not provide an answer.",
                    "strengths": [], "concerns": ["No response given."],
                    "followup": ""}
        system = (
            "You are a strict but fair interviewer grading a spoken answer. "
            "Judge relevance, correctness, depth, and clarity. Be concise and "
            "constructive. Respond ONLY with JSON."
        )
        user = (
            f"Role: {role or 'general'}\n"
            f"Question: {question}\n"
            f"Candidate's answer (transcribed speech): {answer}\n\n"
            'Return JSON: {"score": 0-10 integer, "verdict": "short label", '
            '"feedback": "2-3 sentences", "strengths": ["..."], '
            '"concerns": ["..."], "followup": "one optional follow-up question or empty string"}'
        )
        content = self._chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            json_mode=True, temperature=0.2,
        )
        data = _safe_json(content)
        data.setdefault("score", 0)
        data.setdefault("feedback", "")
        return data


# -----------------------------------------------------------------------
# Smoke test:  python interviewer.py
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("Connecting to Groq...")
    iv = Interviewer()

    role = "Junior Python Developer"
    print(f"\nGenerating questions for: {role}\n" + "-" * 50)
    questions = iv.generate_questions(role, n=3, difficulty="easy")
    for i, q in enumerate(questions, 1):
        print(f"{i}. [{q.get('topic','?')}/{q.get('difficulty','?')}] {q.get('question')}")

    if questions:
        q1 = questions[0]["question"]
        sample = ("A list is mutable and uses square brackets, a tuple is "
                  "immutable and uses parentheses. I use tuples for fixed data.")
        print("\nScoring a sample answer to Q1\n" + "-" * 50)
        print(f"Q: {q1}\nA: {sample}\n")
        result = iv.evaluate_answer(q1, sample, role)
        print(json.dumps(result, indent=2))
    print("\nDone. If you see questions and a score above, your setup works.")