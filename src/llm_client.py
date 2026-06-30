"""Shared LLM client — OpenAI (gpt-4o-mini) hoặc AntcoAI Gateway."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL


def is_llm_available() -> bool:
    return bool(LLM_API_KEY)


def chat(system: str, user: str, max_tokens: int = 512) -> str:
    """Gọi LLM qua OpenAI-compatible API."""
    if not LLM_API_KEY:
        raise RuntimeError("OPENAI_API_KEY chưa được cấu hình trong .env")

    from openai import OpenAI

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()
