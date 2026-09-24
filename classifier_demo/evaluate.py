"""manifest検証と評価実行。

`check_manifest` は doc/dataset-plan.md §4 の manifest仕様を検証する。
`run_evaluate` は同じ検証をまず行い(エラーがあれば中断)、
doc/implementation-experiment-plan.md §5・§5.1 の評価規則・時間測定プロトコルに沿って
`classifier_demo.pipeline.classify` を1ケースずつ逐次実行し、
ケース別結果JSON・run.json・cases.csv・summary.md を出力する。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import pipeline, report
from .decision import decide_axis
from .image import prepare_image
from .pipeline import _git_commit
from .taxonomy import load as load_taxonomy

REQUIRED_CASE_FIELDS = [
    "case_id",
    "source_image_id",
    "derived_from",
    "image_path",
    "image_sha256",
    "split",
    "scenario",
    "expected",
    "expected_metadata",
    "generation",
    "rights",
    "review",
]
REQUIRED_EXPECTED_FIELDS = ["image_type", "art_style", "subject", "character"]
REQUIRED_EXPECTED_METADATA_FIELDS = ["format", "loras", "trigger_words", "characters", "artificial"]


@dataclass
class ManifestCheck:
    cases: list[dict]
    errors: list[str]
    scenario_counts: dict[str, int] = field(default_factory=dict)
    source_count: int = 0
    derived_count: int = 0
    rights_false_count: int = 0


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest_lines(manifest_path: str) -> tuple[list[dict], list[str]]:
    cases: list[dict] = []
    errors: list[str] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append(f"line {lineno}: invalid JSON ({e})")
    return cases, errors


def check_manifest(manifest_path: str) -> ManifestCheck:
    """manifest を検証する。パスは実行時のカレントディレクトリ(リポジトリルート想定)から解決する。"""
    cases, errors = _load_manifest_lines(manifest_path)

    seen_ids: set[str] = set()
    scenario_counts: dict[str, int] = {}
    source_count = 0
    derived_count = 0
    rights_false_count = 0

    for idx, case in enumerate(cases):
        label = case.get("case_id", f"<line {idx + 1}>")

        missing = [f for f in REQUIRED_CASE_FIELDS if f not in case]
        if missing:
            errors.append(f"case {label}: missing field(s) {missing}")
            continue  # 必須項目が無いと以降のチェックができない

        case_id = case["case_id"]
        if case_id in seen_ids:
            errors.append(f"case {case_id}: duplicate case_id")
        else:
            seen_ids.add(case_id)

        missing_exp = [f for f in REQUIRED_EXPECTED_FIELDS if f not in case["expected"]]
        if missing_exp:
            errors.append(f"case {case_id}: expected missing field(s) {missing_exp}")

        missing_meta = [
            f for f in REQUIRED_EXPECTED_METADATA_FIELDS if f not in case["expected_metadata"]
        ]
        if missing_meta:
            errors.append(f"case {case_id}: expected_metadata missing field(s) {missing_meta}")

        rights = case["rights"]
        if "rights_confirmed" not in rights:
            errors.append(f"case {case_id}: rights missing rights_confirmed")
        elif rights["rights_confirmed"] is not True:
            rights_false_count += 1

        image_path = case["image_path"]
        if not os.path.exists(image_path):
            errors.append(f"case {case_id}: image_path does not exist: {image_path}")
        else:
            actual_sha = sha256_file(image_path)
            if actual_sha != case["image_sha256"]:
                errors.append(
                    f"case {case_id}: image_sha256 mismatch "
                    f"(manifest {case['image_sha256']}, actual {actual_sha})"
                )

        scenario = case.get("scenario", "<missing>")
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        if case.get("derived_from") is None:
            source_count += 1
        else:
            derived_count += 1

    by_id = {c["case_id"]: c for c in cases if "case_id" in c}
    for case in cases:
        case_id = case.get("case_id")
        if case_id is None or "derived_from" not in case:
            continue
        derived_from = case["derived_from"]
        if derived_from is None:
            continue
        origin = by_id.get(derived_from)
        if origin is None:
            errors.append(f"case {case_id}: derived_from refers to unknown case_id {derived_from!r}")
        elif origin.get("source_image_id") != case.get("source_image_id"):
            errors.append(
                f"case {case_id}: derived_from case {derived_from} has a different source_image_id"
            )

    return ManifestCheck(
        cases=cases,
        errors=errors,
        scenario_counts=scenario_counts,
        source_count=source_count,
        derived_count=derived_count,
        rights_false_count=rights_false_count,
    )


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def _relpath(path: str) -> str:
    try:
        return os.path.relpath(path, os.getcwd())
    except ValueError:
        # Windows でドライブが異なる場合など relpath が計算できないときはファイル名のみ残す。
        return os.path.basename(path)


def _classify_safe(image_path: str, taxonomy, backend, mode: str, max_edge: int) -> dict:
    """pipeline.classify を呼ぶ。1ケースの予期しない例外で全体を止めないための保険。

    pipeline.classify 自体は画像・メタデータ・軸ごとのエラーを内部で捕捉して結果に記録するため、
    通常はここで例外を捕まえることはない。
    """
    start_ns = time.perf_counter_ns()
    try:
        return pipeline.classify(image_path, taxonomy, backend, mode=mode, max_edge=max_edge)
    except Exception as e:
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        return {
            "image": {"name": os.path.basename(image_path)},
            "mode": mode,
            "vision_tags": {},
            "metadata_evidence": None,
            "combined_evidence": {},
            "timing_ms": {"classification_wall_ms": elapsed_ms},
            "request_count": 0,
            "errors": [{"axis": None, "type": "unhandled_error", "detail": f"{type(e).__name__}: {e}"}],
        }


def _run_warmup(cases: list[dict], taxonomy, backend, warmup: int, max_edge: int) -> dict:
    info = {"count": warmup, "elapsed_ms": 0.0, "errors": 0}
    if warmup <= 0 or not cases:
        return info

    axis = taxonomy.axes[0]
    try:
        image_bytes, mime, _, _ = prepare_image(cases[0]["image_path"], max_edge=max_edge)
    except Exception:
        info["errors"] += warmup
        return info

    for _ in range(warmup):
        try:
            result = decide_axis(backend, image_bytes, mime, axis)
            info["elapsed_ms"] += result["elapsed_ms"]
        except Exception:
            info["errors"] += 1
    return info


def run_evaluate(
    manifest_path: str,
    taxonomy_path: str,
    backend,
    modes: list[str],
    max_edge: int,
    warmup: int,
    runtime_label: str | None,
    note: str | None,
    output_dir: str,
    runtime_info_path: str | None = None,
) -> int:
    check = check_manifest(manifest_path)
    if check.errors:
        for e in check.errors:
            print(f"check-manifest error: {e}", file=sys.stderr)
        return 1

    # runtime-info は評価を始める前に読む。読めない/不正なら評価そのものを始めない。
    runtime_info = None
    runtime_info_file = None
    if runtime_info_path:
        try:
            with open(runtime_info_path, "r", encoding="utf-8") as f:
                runtime_info = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"runtime-info error: failed to read {runtime_info_path}: {e}", file=sys.stderr)
            return 1
        runtime_info_file = {
            "name": os.path.basename(runtime_info_path),
            "sha256": sha256_file(runtime_info_path),
        }

    taxonomy = load_taxonomy(taxonomy_path)

    excluded_cases: list[dict] = []
    evaluated_cases: list[dict] = []
    evaluated_by_scenario: dict[str, int] = {}
    for case in check.cases:
        if case["rights"].get("rights_confirmed") is not True:
            excluded_cases.append({"case_id": case["case_id"], "reason": "rights_not_confirmed"})
        else:
            evaluated_cases.append(case)
            scenario = case.get("scenario", "<missing>")
            evaluated_by_scenario[scenario] = evaluated_by_scenario.get(scenario, 0) + 1

    cases_dir = os.path.join(output_dir, "cases")
    os.makedirs(cases_dir, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()

    warmup_info = _run_warmup(evaluated_cases, taxonomy, backend, warmup, max_edge)

    records: list[dict] = []
    for i, case in enumerate(evaluated_cases):
        order = list(modes) if i % 2 == 0 else list(reversed(modes))
        for order_index, mode in enumerate(order):
            result = _classify_safe(case["image_path"], taxonomy, backend, mode, max_edge)
            records.append({"case": case, "mode": mode, "order_index": order_index, "result": result})
            out_path = os.path.join(cases_dir, f"{case['case_id']}.{mode}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    finished = datetime.now(timezone.utc).isoformat()

    run_info = {
        "started": started,
        "finished": finished,
        "tool_commit": {"commit": _git_commit(), "dirty": _git_dirty()},
        "manifest": {"path": _relpath(manifest_path), "sha256": sha256_file(manifest_path)},
        "taxonomy": {"version": taxonomy.version, "sha256": taxonomy.sha256},
        "model": backend.model,
        "base_url": backend.base_url,
        "modes": modes,
        "max_edge": max_edge,
        "warmup": warmup_info,
        "runtime_label": runtime_label,
        "runtime_info": runtime_info,
        "runtime_info_file": runtime_info_file,
        "note": note,
        "platform": {"platform": platform.platform(), "python_version": sys.version},
        "case_counts": {
            "total": len(check.cases),
            "evaluated": len(evaluated_cases),
            "excluded": len(excluded_cases),
            "manifest_by_scenario": check.scenario_counts,
            "evaluated_by_scenario": evaluated_by_scenario,
        },
        "excluded_cases": excluded_cases,
    }
    with open(os.path.join(output_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    csv_rows = [
        report.build_case_row(r["case"], r["mode"], r["order_index"], r["result"]) for r in records
    ]
    report.write_cases_csv(os.path.join(output_dir, "cases.csv"), csv_rows)

    summary_text = report.build_summary(evaluated_cases, records, modes)
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    return 0
