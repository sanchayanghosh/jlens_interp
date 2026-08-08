"""Small Together.ai text-generation wrapper with bounded retries."""

from __future__ import annotations

import random
import time
from typing import Any

from together import Together


def generate_response(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    retries: int = 4,
) -> str:
    client = Together(api_key=api_key)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response: Any = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an AI assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("Together.ai returned an empty response.")
            return str(content).rstrip("\n ").lstrip("\n")
        except Exception as exc:  # Together exposes several transport/API exception classes.
            last_error = exc
            if attempt == retries - 1:
                break
            delay = min(2**attempt, 8) + random.random()
            time.sleep(delay)
    raise RuntimeError(f"Together.ai failed after {retries} attempts: {last_error}") from last_error
