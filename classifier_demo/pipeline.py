"""1枚の画像を分類し、追跡可能な結果JSONを組み立てる。

時間計測は implementation-experiment-plan.md §5.1 の境界(画像読み込み・メタデータ抽出の
前から、最終タグの統合直後まで、再試行を含む)に従う。
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import decision, json_baseline
from .image import JPEG_QUALITY, file_sha256, prepare_image
from .metadata import extract_evidence
from .taxonomy import Axis, Taxonomy


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


def _process_axis(
    axis: Axis, backend, image_bytes: bytes, mime: str, on_progress, progress_lock, confirm: bool, rank_threshold: float
) -> dict:
    """1軸分の判定を実行し、classify() へマージするための結果をまとめて返す。

    axis_concurrency>=2 のときは複数スレッドから並行して呼ばれるため、on_progress の
    呼び出しだけロックで直列化する(backend.request_count 自体は ChatBackend 側のロックで
    保護済み)。この関数自体は元の逐次ループの1回分と完全に同じ処理・同じイベント形を返す。
    """
    event: dict = {"type": "axis", "axis_id": axis.id, "multi": axis.multi, "error": None}
    axis_errors: list[dict] = []
    axis_decision: dict | None = None
    vision_tags_value: list[str] | None = None
    elapsed_ms: float | None = None

    try:
        if axis.multi:
            result = decision.decide_multi_axis(
                backend, image_bytes, mime, axis, confirm=confirm, rank_threshold=rank_threshold
            )
            axis_decision = {
                "relative_scores": result["relative_scores"],
                "confirmations": result["confirmations"],
                "confirmation_errors": result["confirmation_errors"],
                "candidates": result["candidates"],
                "tags": result["tags"],
                "failed": result["failed"],
            }
            if not result["failed"]:
                vision_tags_value = result["tags"]
            for cid, err in result["confirmation_errors"].items():
                axis_errors.append(
                    {"axis": axis.id, "type": "candidate_confirmation_error", "detail": f"candidate {cid}: {err}"}
                )
            event.update(
                {
                    "relative_scores": result["relative_scores"],
                    "candidates": result["candidates"],
                    "confirmations": result["confirmations"],
                    "confirmation_errors": result["confirmation_errors"],
                    "tags": result["tags"],
                    "failed": result["failed"],
                    "elapsed_ms": result["elapsed_ms"],
                    "confirm": confirm,
                    "rank_threshold": rank_threshold,
                }
            )
        else:
            result = decision.decide_axis(backend, image_bytes, mime, axis)
            axis_decision = {
                "relative_scores": result["relative_scores"],
                "selected": result["selected"],
            }
            selected = result["selected"]
            vision_tags_value = [] if selected == decision.NONE_ID else [selected]
            event.update(
                {
                    "relative_scores": result["relative_scores"],
                    "selected": result["selected"],
                    "elapsed_ms": result["elapsed_ms"],
                }
            )
        elapsed_ms = result["elapsed_ms"]
    except decision.DecisionError as e:
        axis_errors.append({"axis": axis.id, "type": e.error_type, "detail": e.detail})
        event["error"] = {"type": e.error_type, "detail": e.detail}
    except decision.REQUEST_EXCEPTIONS as e:
        axis_errors.append({"axis": axis.id, "type": "request_error", "detail": f"{type(e).__name__}: {e}"})
        event["error"] = {"type": "request_error", "detail": f"{type(e).__name__}: {e}"}

    if on_progress is not None:
        with progress_lock:
            on_progress(event)

    return {
        "axis_id": axis.id,
        "axis_decision": axis_decision,
        "vision_tags": vision_tags_value,
        "errors": axis_errors,
        "elapsed_ms": elapsed_ms,
    }


def _classify_bundled(
    taxonomy: Taxonomy, backend, image_bytes: bytes, mime: str, rank_threshold: float, on_progress,
    multi_mode: str = "rank",
):
    """E7 束ね質問: リクエスト1回で全軸を判定し、choice と同じ形の結果・イベントにする。
    on_progress は1リクエストの完了後に軸ごとのイベントをまとめて出す。"""
    axis_decisions: dict = {}
    vision_tags: dict[str, list[str]] = {}
    errors: list[dict] = []
    events: list[dict] = []
    bundled_info: dict = {"raw_text": None, "positions": {}, "request_ms": None}

    try:
        res = decision.decide_bundled(backend, image_bytes, mime, taxonomy, rank_threshold, multi_mode)
        axis_results = res["axes"]
        bundled_info = {
            "raw_text": res["raw_text"],
            "positions": {aid: r["position"] for aid, r in axis_results.items() if r.get("error") is None},
            "request_ms": res["elapsed_ms"],
        }
        request_error = None
    except decision.DecisionError as e:
        axis_results, request_error = {}, {"type": e.error_type, "detail": e.detail}
    except decision.REQUEST_EXCEPTIONS as e:
        axis_results, request_error = {}, {"type": "request_error", "detail": f"{type(e).__name__}: {e}"}

    for axis in taxonomy.axes:
        event: dict = {
            "type": "axis", "axis_id": axis.id, "multi": axis.multi, "error": None, "bundled": True,
            "elapsed_ms": bundled_info["request_ms"],
        }
        r = axis_results.get(axis.id)
        err = request_error if request_error is not None else (r or {}).get("error")
        if err is not None:
            errors.append({"axis": axis.id, "type": err["type"], "detail": err["detail"]})
            event["error"] = err
        elif axis.multi:
            # yn(E7e)のときは候補ごとの P(yes) を confirmations に入れ、UI の yes/no 表示に載せる。
            yn = multi_mode == "yn"
            confirmations = r["confirmations"] if yn else {}
            axis_decisions[axis.id] = {
                "relative_scores": r["relative_scores"],
                "confirmations": confirmations,
                "confirmation_errors": {},
                "candidates": r["candidates"],
                "tags": r["tags"],
                "failed": False,
            }
            vision_tags[axis.id] = r["tags"]
            event.update(
                {
                    "relative_scores": r["relative_scores"], "candidates": r["candidates"],
                    "confirmations": confirmations, "confirmation_errors": {}, "tags": r["tags"], "failed": False,
                    "confirm": yn, "rank_threshold": rank_threshold, "multi_mode": multi_mode,
                }
            )
        else:
            axis_decisions[axis.id] = {"relative_scores": r["relative_scores"], "selected": r["selected"]}
            vision_tags[axis.id] = [] if r["selected"] == decision.NONE_ID else [r["selected"]]
            event.update({"relative_scores": r["relative_scores"], "selected": r["selected"]})
        events.append(event)

    if on_progress is not None:
        for event in events:
            on_progress(event)
    return {"axis_decisions": axis_decisions, "vision_tags": vision_tags, "errors": errors, "bundled": bundled_info}


def classify(
    image_path: str,
    taxonomy: Taxonomy,
    backend,
    mode: str = "choice",
    max_edge: int = 1024,
    on_progress=None,
    axis_concurrency: int = 1,
    confirm: bool = False,
    rank_threshold: float = 0.5,
    image_format: str = "jpeg",
    bundled_multi: str = "rank",
) -> dict:
    """`on_progress` はデモUIサーバー用の任意コールバック(既定Noneなら未使用・
    既存の挙動と戻り値は変わらない)。呼ばれる順序: メタデータイベント1回 →
    (choice/bundledモードのみ)軸ごとに1回(bundledは1リクエスト完了後にまとめて)。`axis_concurrency<=1` なら taxonomy.axes の順で逐次、
    2以上なら ThreadPoolExecutor で軸を並列実行する(on_progress は完了順に呼ばれうるが、
    戻り値の axis_decisions/vision_tags/errors/per_axis_timing は常に taxonomy.axes の順で
    組み立てるため、内容・順序は axis_concurrency の値によらず同じになる)。
    `confirm`/`rank_threshold` は複数選択軸(character・outfit等)の採用方法を決める
    (decision.decide_multi_axis 参照)。既定は confirm=False, rank_threshold=0.5。
    呼び出し側で例外を出さないこと。"""
    start_ns = time.perf_counter_ns()
    requests_before = backend.request_count

    errors: list[dict] = []
    axis_decisions: dict = {}
    vision_tags: dict[str, list[str]] = {}
    json_baseline_result = None
    bundled_info = None
    per_axis_timing: dict[str, float] = {}

    image_bytes = mime = None
    original_size = sent_size = None
    original_sha256 = None
    metadata_evidence = None
    try:
        image_bytes, mime, original_size, sent_size = prepare_image(
            image_path, max_edge=max_edge, image_format=image_format
        )
        original_sha256 = file_sha256(image_path)
    except Exception as e:
        errors.append({"axis": None, "type": "image_error", "detail": f"{type(e).__name__}: {e}"})

    if image_bytes is not None:
        try:
            metadata_evidence = extract_evidence(image_path, taxonomy)
        except Exception as e:
            errors.append({"axis": None, "type": "metadata_error", "detail": f"{type(e).__name__}: {e}"})
            metadata_evidence = {"format": "error", "loras": [], "prompt_tags": [], "matches": []}

        if on_progress is not None:
            on_progress({"type": "metadata", "metadata_evidence": metadata_evidence})

        if mode == "choice":
            progress_lock = threading.Lock()
            if axis_concurrency <= 1:
                axis_outcomes = [
                    _process_axis(axis, backend, image_bytes, mime, on_progress, progress_lock, confirm, rank_threshold)
                    for axis in taxonomy.axes
                ]
            else:
                with ThreadPoolExecutor(max_workers=axis_concurrency) as executor:
                    futures = [
                        executor.submit(
                            _process_axis,
                            axis,
                            backend,
                            image_bytes,
                            mime,
                            on_progress,
                            progress_lock,
                            confirm,
                            rank_threshold,
                        )
                        for axis in taxonomy.axes
                    ]
                    outcomes_by_axis_id = {}
                    for future in as_completed(futures):
                        outcome = future.result()
                        outcomes_by_axis_id[outcome["axis_id"]] = outcome
                axis_outcomes = [outcomes_by_axis_id[axis.id] for axis in taxonomy.axes]

            # axis_concurrency の値によらず、taxonomy.axes の順でマージする
            # (on_progress は完了順に呼ばれうるが、戻り値の順序・内容は逐次実行と同じにする)。
            for outcome in axis_outcomes:
                axis_id = outcome["axis_id"]
                if outcome["axis_decision"] is not None:
                    axis_decisions[axis_id] = outcome["axis_decision"]
                if outcome["vision_tags"] is not None:
                    vision_tags[axis_id] = outcome["vision_tags"]
                errors.extend(outcome["errors"])
                if outcome["elapsed_ms"] is not None:
                    per_axis_timing[axis_id] = outcome["elapsed_ms"]
        elif mode == "bundled":
            out = _classify_bundled(taxonomy, backend, image_bytes, mime, rank_threshold, on_progress, bundled_multi)
            axis_decisions = out["axis_decisions"]
            vision_tags = out["vision_tags"]
            errors.extend(out["errors"])
            bundled_info = out["bundled"]
            if bundled_info["request_ms"] is not None:
                per_axis_timing["bundled_request_ms"] = bundled_info["request_ms"]
        elif mode == "json":
            json_baseline_result = json_baseline.classify_json(backend, image_bytes, mime, taxonomy)
            if json_baseline_result["tags"] is not None:
                vision_tags = json_baseline_result["tags"]
            else:
                errors.append(
                    {"axis": None, "type": "json_format_error", "detail": json_baseline_result["error"]}
                )
            if on_progress is not None:
                on_progress({"type": "json_result", **json_baseline_result})
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
            "image_format": image_format,
            "jpeg_quality": JPEG_QUALITY if image_format == "jpeg" else None,
            "temperature": 0,
            "top_logprobs": 20,
            "dropped_reasoning_effort": backend.dropped_reasoning_effort,
            "confirm": confirm,
            "rank_threshold": rank_threshold,
            "bundled_multi": bundled_multi,
        },
        "metadata_evidence": metadata_evidence,
        "vision_tags": vision_tags,
        "combined_evidence": combined_evidence,
        "timing_ms": {"classification_wall_ms": classification_wall_ms, **per_axis_timing},
        "request_count": request_count,
        "errors": errors,
    }
    if mode in ("choice", "bundled"):
        result["axis_decisions"] = axis_decisions
        if mode == "bundled":
            result["bundled"] = bundled_info
    else:
        result["json_baseline"] = json_baseline_result
    return result
