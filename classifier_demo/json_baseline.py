"""通常JSON分類ベースライン。全軸を1回のプロンプトで問い、JSONのみを解析する。

未知/欠損の値を勝手に補正しない。形式不正は失敗として記録し、MAX_RETRIES 回まで再試行する。
"""

from __future__ import annotations

import json
import re
import time
import urllib.error

from .decision import _data_url
from .taxonomy import Taxonomy

# backend.chat 自体の通信失敗(サーバー応答の形式・接続エラー)。再試行はしない。
REQUEST_EXCEPTIONS = (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError)

MAX_RETRIES = 2
JSON_PARAMS = {"temperature": 0, "max_tokens": 256}
SYSTEM_PROMPT = "You are an image classifier. Reply with JSON only, no other text."
RAW_TEXT_MAX_LEN = 2000

_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?|```")


def _build_prompt(taxonomy: Taxonomy) -> str:
    lines = []
    for axis in taxonomy.axes:
        choice_desc = ", ".join(f"{c.id} ({c.criteria})" for c in axis.choices)
        line = f"- {axis.id}: {choice_desc}"
        if axis.multi and axis.none_criteria:
            line += f" (use an empty list only if: {axis.none_criteria})"
        lines.append(line)

    # 例には実在の選択肢idを入れない(選択式にない誘導をJSON方式だけに与えないため)。
    example = {axis.id: (["<id>", "..."] if axis.multi else "<id>") for axis in taxonomy.axes}
    return (
        "Classify the image on each axis below using only the listed choice ids.\n"
        + "\n".join(lines)
        + "\n\nMulti-select axes are a list of ids (may be empty or several); "
        "other axes are a single id.\n"
        "Respond with JSON only, no other text, matching this shape:\n"
        f"{json.dumps(example)}"
    )


def _extract_json_block(text: str) -> str:
    cleaned = _FENCE_RE.sub("", text)
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last == -1 or last < first:
        raise ValueError("no JSON object found in response text")
    return cleaned[first : last + 1]


def _parse_and_validate(text: str, taxonomy: Taxonomy) -> dict[str, list[str]]:
    block = _extract_json_block(text)
    data = json.loads(block)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON is not an object")

    tags: dict[str, list[str]] = {}
    for axis in taxonomy.axes:
        if axis.id not in data:
            raise ValueError(f"missing axis: {axis.id}")
        value = data[axis.id]
        known_ids = {c.id for c in axis.choices}
        if axis.multi:
            if not isinstance(value, list):
                raise ValueError(f"axis {axis.id}: expected a list, got {value!r}")
            for v in value:
                if v not in known_ids:
                    raise ValueError(f"axis {axis.id}: unknown choice id {v!r}")
            tags[axis.id] = list(dict.fromkeys(value))  # 順序を保って重複除去
        else:
            if not isinstance(value, str) or value not in known_ids:
                raise ValueError(f"axis {axis.id}: unknown choice id {value!r}")
            tags[axis.id] = [value]
    return tags


def _extract_text(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def classify_json(backend, image_bytes: bytes, mime: str, taxonomy: Taxonomy) -> dict:
    prompt = _build_prompt(taxonomy)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_url(image_bytes, mime)}},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    attempts: list[dict] = []
    first_attempt_ms: float | None = None
    tags: dict[str, list[str]] | None = None
    error: str | None = None
    usage = None
    start = time.perf_counter_ns()

    for _ in range(MAX_RETRIES + 1):
        try:
            response, elapsed_ms = backend.chat(messages, **JSON_PARAMS)
        except REQUEST_EXCEPTIONS as e:
            error = f"{type(e).__name__}: {e}"
            attempts.append({"elapsed_ms": None, "raw_text": "", "error": error})
            tags = None
            break  # 通信失敗は再試行しない

        if first_attempt_ms is None:
            first_attempt_ms = elapsed_ms
        raw_text = _extract_text(response)
        usage = response.get("usage")
        attempt_record = {"elapsed_ms": elapsed_ms, "raw_text": raw_text[:RAW_TEXT_MAX_LEN], "error": None}
        try:
            tags = _parse_and_validate(raw_text, taxonomy)
            attempt_record["error"] = None
            attempts.append(attempt_record)
            error = None
            break
        except (ValueError, json.JSONDecodeError) as e:
            error = str(e)
            attempt_record["error"] = error
            attempts.append(attempt_record)
            tags = None
            continue

    total_ms = (time.perf_counter_ns() - start) / 1_000_000

    return {
        "attempts": attempts,
        "first_attempt_ms": first_attempt_ms,
        "total_ms": total_ms,
        "tags": tags,
        "error": error,
        "usage": usage,
    }
