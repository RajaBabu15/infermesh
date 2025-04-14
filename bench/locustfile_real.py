"""Locust user classes for the InferMesh gateway load test.

Drives real HTTP traffic to http://localhost:8000. The gateway must be
running there; the run_locust.py orchestrator handles startup. Each request
becomes a real billable upstream call after the gateway forwards it.

User classes (selected via BENCH_LOCUST_CLASS env var):
  SharedPromptGroqUser   single-turn, identical system prompt
  DiverseGroqUser        unique uuid prefix per request
  ConversationalGroqUser multi-turn with session header propagation

Headless example:
  locust -f bench/locustfile_real.py --headless \\
         --users 25 --spawn-rate 5 --run-time 60s \\
         --host http://localhost:8000 \\
         --csv bench/results/run/cell
"""
from __future__ import annotations

import os
import random
import uuid

from locust import HttpUser, between, task

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

QUESTIONS = [
    "What is the capital of France?",
    "What does ACID stand for in databases?",
    "Briefly: what is recursion in programming?",
    "Explain TCP vs UDP in two sentences.",
    "What is 17 multiplied by 23?",
    "Name three common sorting algorithms.",
    "What is REST API in one sentence?",
    "Briefly: what is async/await in Python?",
    "What is HTTP/2?",
    "Explain the difference between SQL and NoSQL.",
    "What is a hash table?",
    "Define Big-O notation in one sentence.",
    "What is the CAP theorem?",
    "Briefly explain SSL vs TLS.",
    "What is dependency injection?",
]

SHARED_SYSTEM_PROMPT = (
    "You are a concise technical assistant. Answer in one or two short sentences. "
    "Be accurate and direct. Avoid filler words and disclaimers."
)

MODEL_GROQ = "llama-3.1-8b-instant"
MAX_TOKENS = int(os.getenv("BENCH_LOCUST_MAX_TOKENS", "40"))
SESSION_HEADER = "X-InferMesh-Session-ID"

ACTIVE_CLASS = os.getenv("BENCH_LOCUST_CLASS", "SharedPromptGroqUser")


# ---------------------------------------------------------------------------
# User classes. Only one is active per locust run, selected via env var.
# ---------------------------------------------------------------------------

if ACTIVE_CLASS == "SharedPromptGroqUser":

    class SharedPromptGroqUser(HttpUser):
        """All users share an identical 200-char system prompt."""
        wait_time = between(0.3, 1.5)

        @task
        def chat(self) -> None:
            self.client.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL_GROQ,
                    "messages": [
                        {"role": "system", "content": SHARED_SYSTEM_PROMPT},
                        {"role": "user", "content": random.choice(QUESTIONS)},
                    ],
                    "max_tokens": MAX_TOKENS,
                },
                name="POST /v1/chat/completions [shared_prefix]",
            )

elif ACTIVE_CLASS == "DiverseGroqUser":

    class DiverseGroqUser(HttpUser):
        """Each request gets a unique uuid prefix; no trie matches possible."""
        wait_time = between(0.3, 1.5)

        @task
        def chat(self) -> None:
            self.client.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL_GROQ,
                    "messages": [
                        {"role": "system",
                         "content": f"Session {uuid.uuid4().hex[:12]}. " + SHARED_SYSTEM_PROMPT},
                        {"role": "user", "content": random.choice(QUESTIONS)},
                    ],
                    "max_tokens": MAX_TOKENS,
                },
                name="POST /v1/chat/completions [diverse]",
            )

elif ACTIVE_CLASS == "ConversationalGroqUser":

    class ConversationalGroqUser(HttpUser):
        """
        Multi-turn conversation with session header propagation.
        Exercises sticky routing when disaggregation is enabled.
        """
        wait_time = between(0.5, 2.0)

        def on_start(self) -> None:
            self.session_id: str | None = None
            self.history: list[dict] = [
                {"role": "system", "content": SHARED_SYSTEM_PROMPT}
            ]
            self.turn = 0

        @task
        def turn_request(self) -> None:
            self.history.append({"role": "user", "content": random.choice(QUESTIONS)})
            headers = {SESSION_HEADER: self.session_id} if self.session_id else {}

            with self.client.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL_GROQ,
                    "messages": self.history,
                    "max_tokens": MAX_TOKENS,
                },
                headers=headers,
                name="POST /v1/chat/completions [conversational]",
                catch_response=True,
            ) as resp:
                if not resp.ok:
                    resp.failure(f"status={resp.status_code}")
                    return
                new_sid = resp.headers.get(SESSION_HEADER)
                if new_sid:
                    self.session_id = new_sid
                try:
                    content = resp.json()["choices"][0]["message"]["content"]
                    self.history.append({"role": "assistant", "content": content})
                except Exception:
                    pass
            self.turn += 1
            if self.turn >= 10:
                self.on_start()

else:
    raise ValueError(
        f"unknown BENCH_LOCUST_CLASS={ACTIVE_CLASS!r}; "
        "expected SharedPromptGroqUser, DiverseGroqUser, or ConversationalGroqUser"
    )
