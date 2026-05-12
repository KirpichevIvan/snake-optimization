from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from llm_as_optimizer.prompts import SYSTEM_PROMPT

load_dotenv()

POLZA_BASE_URL = "https://polza.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3.6-35b-a3b"
THETA_DIM = 4
DEFAULT_NUM_CANDIDATES = 3
# Таймаут HTTP к API (сек.); иначе зависание без строки в логе
DEFAULT_LLM_TIMEOUT_SEC = 240.0


def _response_format_json_schema(*, num_candidates: int, theta_dim: int = THETA_DIM) -> dict[str, Any]:
    """Формат ответа API: structured outputs (json_schema)."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "snake_optimization_step",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "hypothesis_note": {
                        "type": "string",
                        "description": "One sentence: task understanding + optimization hypothesis; why candidates fit.",
                        "minLength": 16,
                        "maxLength": 500,
                    },
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": theta_dim,
                            "maxItems": theta_dim,
                        },
                        "minItems": num_candidates,
                        "maxItems": num_candidates,
                    },
                },
                "required": ["candidates", "hypothesis_note"],
                "additionalProperties": False,
            },
        },
    }


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def ask_llm(
    user_content: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
    timeout_sec: float | None = None,
) -> dict[str, Any] | list[Any]:
    key = os.environ.get("POLZA_AI_API_KEY")
    if not key:
        msg = "Задайте POLZA_AI_API_KEY в окружении или в .env"
        raise RuntimeError(msg)

    to = timeout_sec if timeout_sec is not None else float(os.environ.get("LLM_TIMEOUT_SEC", str(DEFAULT_LLM_TIMEOUT_SEC)))
    client = OpenAI(base_url=POLZA_BASE_URL, api_key=key, timeout=to)
    response_format = _response_format_json_schema(num_candidates=num_candidates)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        response_format=response_format,  # type: ignore[arg-type]
    )
    raw = response.choices[0].message.content
    if raw is None or not raw.strip():
        msg = "Пустой ответ LLM"
        raise RuntimeError(msg)
    cleaned = _strip_json_fence(raw)
    out: object = json.loads(cleaned)
    if isinstance(out, dict | list):
        return out
    msg = f"Ожидался JSON-объект или массив, получено: {type(out).__name__}"
    raise TypeError(msg)
