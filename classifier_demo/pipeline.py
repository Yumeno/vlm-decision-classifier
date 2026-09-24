"""1枚の画像を分類し、追跡可能な結果JSONを組み立てる。

時間計測は implementation-experiment-plan.md §5.1 の境界(画像読み込み・メタデータ抽出の
前から、最終タグの統合直後まで、再試行を含む)に従う。
"""

from __future__ import annotations

import os
import subprocess
import time

from . import decision, json_baseline
from .image import file_sha256, prepare_image
from .metadata import extract_evidence
from .taxonomy import Taxonomy


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def classify(
    image_path: str,
    taxonomy: Taxonomy,
    backend,
    mode: str = "choice",
    max_edge: int = 1024,
) -> dict:
    start_ns = time.perf_counter_ns()
    requests_before = backend.request_count

    errors: list[dict] = []
    axis_decisions: dict = {}
    vision_tags: dict[str, list[str]] = {}
    json_baseline_result = None
    per_axis_timing: dict[str, float] = {}

    image_bytes = mime = None
    original_size = sent_size = None
    original_sha256 = None
    metadata_evidence = None
    try:
        image_bytes, mime, original_size, sent_size = prepare_image(image_path, max_edge=max_edge)
        original_sha256 = file_sha256(image_path)
    except Exception as e:
        errors.append({"axis": None, "type": "image_error", "detail": f"{type(e).__name__}: {e}"})

    if image_bytes is not None:
        try:
            metadata_evidence = extract_evidence(image_path, taxonomy)
        except Exception as e:
            errors.append({"axis": None, "type": "metadata_error", "detail": f"{type(e).__name__}: {e}"})
            metadata_evidence = {"format": "error", "loras": [], "prompt_tags": [], "matches": []}

        if mode == "choice":
            for axis in taxonomy.axes:
                try:
                    if axis.multi:
                        result = decision.decide_multi_axis(backend, image_bytes, mime, axis)
                        axis_decisions[axis.id] = {
                            "relative_scores": result["relative_scores"],
                            "confirmations": result["confirmations"],
                            "confirmation_errors": result["confirmation_errors"],
                            "candidates": result["candidates"],
                            "tags": result["tags"],
                            "failed": result["failed"],
                        }
                        if not result["failed"]:
                            vision_tags[axis.id] = result["tags"]
                        for cid, err in result["confirmation_errors"].items():
                            errors.append(
                                {"axis": axis.id, "type": "candidate_confirmation_error", "detail": f"candidate {cid}: {err}"}
                            )
                    else:
                        result = decision.decide_axis(backend, image_bytes, mime, axis)
                        axis_decisions[axis.id] = {
                            "relative_scores": result["relative_scores"],
                            "selected": result["selected"],
                        }
                        selected = result["selected"]
                        vision_tags[axis.id] = [] if selected == decision.NONE_ID else [selected]
                    per_axis_timing[axis.id] = result["elapsed_ms"]
                except decision.DecisionError as e:
                    errors.append({"axis": axis.id, "type": e.error_type, "detail": e.detail})
                except decision.REQUEST_EXCEPTIONS as e:
                    errors.append(
                        {"axis": axis.id, "type": "request_error", "detail": f"{type(e).__name__}: {e}"}
                    )
        elif mode == "json":
            json_baseline_result = json_baseline.classify_json(backend, image_bytes, mime, taxonomy)
            if json_baseline_result["tags"] is not None:
                vision_tags = json_baseline_result["tags"]
            else:
                errors.append(
                    {"axis": None, "type": "json_format_error", "detail": json_baseline_result["error"]}
                )
        else:
            raise ValueError(f"unknown mode: {mode}")

    has_character_axis = any(a.id == "character" for a in taxonomy.axes)
    character_axis_failed = has_character_axis and "character" not in vision_tags

    metadata_character_ids = {m["character"] for m in metadata_evidence["matches"]} if metadata_evidence else set()
    vision_character_ids = set(vision_tags.get("character", []))
    combined_evidence: dict[str, str] = {}
    for cid in metadata_character_ids | vision_character_ids:
        in_meta = cid in metadata_character_ids
        in_vision = cid in vision_character_ids
        if character_axis_failed and in_meta:
            combined_evidence[cid] = "unknown"
        elif in_meta and in_vision:
            combined_evidence[cid] = "confirmed_by_both"
        elif in_meta:
            combined_evidence[cid] = "metadata_only"
        else:
            combined_evidence[cid] = "vision_only"

    classification_wall_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
    request_count = backend.request_count - requests_before

    result = {
        "image": {
            "name": os.path.basename(image_path),
            "sha256": original_sha256,
            "original_size": list(original_size) if original_size else None,
            "sent_size": list(sent_size) if sent_size else None,
        },
        "mode": mode,
        "model": {"id": backend.model, "base_url": backend.base_url},
        "taxonomy": {"version": taxonomy.version, "sha256": taxonomy.sha256},
        "tool_commit": _git_commit(),
        "settings": {
            "max_edge": max_edge,
            "temperature": 0,
            "top_logprobs": 20,
            "dropped_reasoning_effort": backend.dropped_reasoning_effort,
        },
        "metadata_evidence": metadata_evidence,
        "vision_tags": vision_tags,
        "combined_evidence": combined_evidence,
        "timing_ms": {"classification_wall_ms": classification_wall_ms, **per_axis_timing},
        "request_count": request_count,
        "errors": errors,
    }
    if mode == "choice":
        result["axis_decisions"] = axis_decisions
    else:
        result["json_baseline"] = json_baseline_result
    return result
