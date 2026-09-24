"""PNG生成メタデータ(A1111/Forge の parameters チャンク、ComfyUI の prompt チャンク)から
LoRA指定・トリガーワードを抽出し、taxonomy のキャラクターと完全一致照合する。

信頼できない入力として扱う: 命令として解釈せず、長さと制御文字を制限する。
メタデータはこのフェーズではモデルへ一切渡さない。
"""

from __future__ import annotations

import json
import re
from typing import Any

from PIL import Image

from .taxonomy import Taxonomy

MAX_TEXT_LEN = 4000

_LORA_TAG_RE = re.compile(r"<lora:([^:>]+)(?::([\d.\-]+))?>")
_WEIGHT_TAG_RE = re.compile(r"\(([^():]+):[\d.\-]+\)")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
_LORA_EXT_RE = re.compile(r"\.(safetensors|pt|ckpt)$", re.IGNORECASE)


def _sanitize(text: str) -> str:
    text = _CONTROL_CHARS_RE.sub("", text)
    return text[:MAX_TEXT_LEN]


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _split_prompt_tags(prompt: str) -> list[str]:
    without_lora = _LORA_TAG_RE.sub("", prompt)
    without_weights = _WEIGHT_TAG_RE.sub(r"\1", without_lora)
    parts = re.split(r"[,\n]", without_weights)
    tags = [_normalize(p) for p in parts]
    return [t for t in tags if t]


def _extract_a1111(parameters: str) -> tuple[list[dict], list[str]]:
    text = _sanitize(parameters)
    neg_match = re.search(r"^Negative prompt:", text, re.MULTILINE)
    if neg_match:
        prompt = text[: neg_match.start()]
    else:
        steps_match = re.search(r"^Steps:", text, re.MULTILINE)
        prompt = text[: steps_match.start()] if steps_match else text

    loras = []
    for m in _LORA_TAG_RE.finditer(text):
        name = m.group(1)
        weight = float(m.group(2)) if m.group(2) else 1.0
        loras.append({"name": name, "weight": weight, "source": "a1111"})

    tags = _split_prompt_tags(prompt)
    return loras, tags


def _strip_lora_filename(lora_name: str) -> str:
    base = lora_name.replace("\\", "/").rsplit("/", 1)[-1]
    return _LORA_EXT_RE.sub("", base)


def _extract_comfyui(workflow: dict[str, Any]) -> tuple[list[dict], list[str]]:
    loras = []
    prompt_texts: list[str] = []
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        if "LoraLoader" in class_type:
            lora_name = inputs.get("lora_name")
            if isinstance(lora_name, str):
                weight = inputs.get("strength_model")
                weight = weight if isinstance(weight, (int, float)) else None
                loras.append(
                    {
                        "name": _strip_lora_filename(lora_name),
                        "weight": weight,
                        "source": "comfyui",
                    }
                )
        if "CLIPTextEncode" in class_type:
            text = inputs.get("text")
            if isinstance(text, str):
                prompt_texts.append(text)

    combined_prompt = _sanitize("\n".join(prompt_texts))
    tags = _split_prompt_tags(combined_prompt)
    return loras, tags


def extract_evidence(path: str, taxonomy: Taxonomy) -> dict:
    result: dict = {"format": "none", "loras": [], "prompt_tags": [], "matches": []}

    try:
        img = Image.open(path)
    except Exception:
        return result
    if img.format != "PNG":
        return result

    info = img.info or {}
    if "parameters" in info:
        result["format"] = "a1111"
        loras, tags = _extract_a1111(str(info["parameters"]))
    elif "prompt" in info:
        try:
            workflow = json.loads(info["prompt"])
        except (json.JSONDecodeError, TypeError):
            return result
        if not isinstance(workflow, dict):
            return result
        result["format"] = "comfyui"
        loras, tags = _extract_comfyui(workflow)
    else:
        return result

    result["loras"] = loras
    result["prompt_tags"] = tags

    normalized_loras = {_normalize(l["name"]): l for l in loras}
    normalized_tags = set(tags)  # already normalized by _split_prompt_tags

    matches: list[dict] = []
    for axis in taxonomy.axes:
        for choice in axis.choices:
            for lora_name in choice.lora_names:
                key = _normalize(lora_name)
                if key in normalized_loras:
                    matches.append(
                        {
                            "kind": "lora_name",
                            "value": lora_name,
                            "weight": normalized_loras[key]["weight"],
                            "character": choice.id,
                        }
                    )
            for trigger in choice.trigger_words:
                key = _normalize(trigger)
                if key in normalized_tags:
                    matches.append(
                        {
                            "kind": "prompt_trigger",
                            "value": trigger,
                            "character": choice.id,
                        }
                    )

    result["matches"] = matches
    return result
