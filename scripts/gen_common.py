"""dataset/generation スクリプト共通ヘルパー(軽量MVP: 必要最小限のみ)。

- prompts.yaml の読み込み
- 枠(slot)からプロンプト文字列・seed を組み立てる
- 生成記録(log.csv)・パラメータJSONの書き出し
- 同一 slot+attempt の上書き拒否(後選び防止)
"""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = REPO_ROOT / "dataset" / "generation" / "prompts.yaml"
GENERATION_DIR = REPO_ROOT / "dataset" / "generation"
STAGING_DIR = REPO_ROOT / "dataset" / "staging"
LOG_CSV_PATH = REPO_ROOT / "dataset" / "generation" / "log.csv"

LOG_FIELDS = [
    "slot_id",
    "attempt",
    "seed",
    "tool",
    "file",
    "sha256",
    "generated_at",
    "result",
    "reason",
]

TRIGGER_WORD = "fet_alisa_uniform, "


def load_prompts(path: Path | None = None) -> dict:
    """prompts.yaml を読み込んで dict を返す。"""
    import yaml  # 遅延importでスクリプト全体のimport負荷を軽く保つ

    p = path or PROMPTS_PATH
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_slot(cfg: dict, slot_id: str) -> dict:
    """cfg['slots'] から slot_id 一致の枠を返す。無ければ KeyError。"""
    for slot in cfg["slots"]:
        if slot["id"] == slot_id:
            return slot
    raise KeyError(f"slot not found in prompts.yaml: {slot_id}")


def slots_for_tool(cfg: dict, tool: str) -> list[dict]:
    """指定 tool の枠をファイル順で返す(--slots 未指定時のデフォルト用)。"""
    return [s for s in cfg["slots"] if s["tool"] == tool]


def build_prompt(slot: dict, cfg: dict, tool: str) -> str:
    """positive_prefix + (forgeのみ: LoRAタグ) + (trigger時: トリガーワード) + 置換後のslotプロンプト。

    ComfyUI では LoRA はテキストではなく LoraLoaderModelOnly ノードで指定するため、
    ここでは LoRA タグを一切テキストに入れない。
    """
    common = cfg["common"]
    parts = [common["positive_prefix"]]

    weight = slot.get("lora_weight")
    if weight is not None and tool == "forge":
        parts.append(f"<lora:{cfg['lora_forge']}:{weight}> ")

    if slot.get("trigger"):
        parts.append(TRIGGER_WORD)

    prompt_text = slot["prompt"].format(
        alisa=cfg["fragments"]["alisa"],
        second=cfg["fragments"]["second"],
    )
    parts.append(prompt_text)

    return "".join(parts)


def seed_for_attempt(slot: dict, attempt: int) -> int:
    """attempt k(1始まり)の seed = slot.seed + (k - 1)。"""
    return slot["seed"] + (attempt - 1)


def staging_image_path(slot_id: str, attempt: int) -> Path:
    return STAGING_DIR / f"{slot_id}_a{attempt}.png"


def params_json_path(slot_id: str, attempt: int) -> Path:
    return GENERATION_DIR / f"{slot_id}_a{attempt}.json"


def check_not_generated(slot_id: str, attempt: int) -> None:
    """同じ slot+attempt の画像/paramsが既にあれば拒否する(後選び防止)。"""
    img = staging_image_path(slot_id, attempt)
    params = params_json_path(slot_id, attempt)
    if img.exists():
        raise FileExistsError(f"既に画像が存在します(再生成しません): {img}")
    if params.exists():
        raise FileExistsError(f"既にparamsが存在します(再生成しません): {params}")


def write_params(slot_id: str, attempt: int, data: dict) -> Path:
    """params JSON を書く。既存なら上書き拒否。"""
    p = params_json_path(slot_id, attempt)
    if p.exists():
        raise FileExistsError(f"既にparamsが存在します(上書きしません): {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def append_log(row: dict[str, Any]) -> None:
    """log.csv に1行追記する。ファイルが無ければヘッダ付きで作成する。"""
    LOG_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_CSV_PATH.exists()
    with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- HTTP(urllib のみ。requests等の追加依存を増やさない) ----


def http_get_json(url: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_json(url: str, payload: dict, timeout: float = 600) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_bytes(url: str, timeout: float = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()
