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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import decision, json_baseline, pipeline, report
from .image import JPEG_QUALITY, prepare_image
from .pipeline import _git_commit
from .taxonomy import Taxonomy
from .taxonomy import load as load_taxonomy

VALID_MODES = {"choice", "json", "bundled", "json_schema", "dgemma_choice", "dgemma_json"}


def case_mode_order(modes: list[str], i: int) -> list[str]:
    """ケースiのモード実行順(順序効果の抑制)。2モードは交互(偶数=順、奇数=逆順)、
    3モード以上は i%n だけ回転する。"""
    if len(modes) >= 3:
        k = i % len(modes)
        return list(modes[k:]) + list(modes[:k])
    return list(modes) if i % 2 == 0 else list(reversed(modes))


def validate_modes(modes: list[str]) -> None:
    """--modes の妥当性を検証する。不正なら ValueError を送出する。"""
    if not modes:
        raise ValueError("--modes must not be empty")
    invalid = [m for m in modes if m not in VALID_MODES]
    if invalid:
        raise ValueError(f"invalid mode(s): {invalid} (choose from {sorted(VALID_MODES)})")
    if len(modes) != len(set(modes)):
        raise ValueError(f"duplicate modes are not allowed: {modes}")


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


def check_manifest(manifest_path: str, taxonomy: Taxonomy) -> ManifestCheck:
    """manifest を検証する。パスは実行時のカレントディレクトリ(リポジトリルート想定)から解決する。

    `expected` の各フィールドは taxonomy の軸一覧から検証する(ハードコードしない):
    軸ごとにキーがあること、単一選択軸は文字列で choice id に含まれること、
    複数選択軸は list で各要素が choice id に含まれること。
    """
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

        expected = case["expected"]
        for axis in taxonomy.axes:
            if axis.id not in expected:
                errors.append(f"case {case_id}: expected missing field(s) ['{axis.id}']")
                continue
            value = expected[axis.id]
            known_ids = {c.id for c in axis.choices}
            if axis.multi:
                if not isinstance(value, list):
                    errors.append(
                        f"case {case_id}: expected.{axis.id} must be a list (multi axis), got {value!r}"
                    )
                    continue
                non_str = [v for v in value if not isinstance(v, str)]
                if non_str:
                    errors.append(
                        f"case {case_id}: expected.{axis.id} has non-string element(s) {non_str}"
                    )
                unknown = [v for v in value if isinstance(v, str) and v not in known_ids]
                if unknown:
                    errors.append(f"case {case_id}: expected.{axis.id} has unknown choice id(s) {unknown}")
            else:
                if not isinstance(value, str):
                    errors.append(
                        f"case {case_id}: expected.{axis.id} must be a string (single axis), got {value!r}"
                    )
                elif value not in known_ids:
                    errors.append(f"case {case_id}: expected.{axis.id} has unknown choice id {value!r}")

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
        elif origin.get("derived_from") is not None:
            errors.append(
                f"case {case_id}: derived_from case {derived_from} is itself derived "
                "(nested derivation is not allowed)"
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


def _classify_safe(
    image_path: str,
    taxonomy,
    backend,
    mode: str,
    max_edge: int,
    axis_concurrency: int = 1,
    confirm: bool = False,
    rank_threshold: float = 0.5,
    image_format: str = "jpeg",
    bundled_multi: str = "rank",
) -> dict:
    """pipeline.classify を呼ぶ。1ケースの予期しない例外で全体を止めないための保険。

    pipeline.classify 自体は画像・メタデータ・軸ごとのエラーを内部で捕捉して結果に記録するため、
    通常はここで例外を捕まえることはない。
    """
    start_ns = time.perf_counter_ns()
    try:
        return pipeline.classify(
            image_path,
            taxonomy,
            backend,
            mode=mode,
            max_edge=max_edge,
            axis_concurrency=axis_concurrency,
            confirm=confirm,
            rank_threshold=rank_threshold,
            image_format=image_format,
            bundled_multi=bundled_multi,
        )
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


def _run_warmup(cases: list[dict], taxonomy, backend, warmup: int, max_edge: int, image_format: str = "jpeg") -> dict:
    """先頭ケースの画像・先頭軸の選択式質問で `warmup` 回のウォームアップを送る。

    使ったケースID・軸ID・試行回数・各試行のelapsed_msをそのまま記録する
    (結果自体は破棄し、分類時間の計測には含めない)。
    """
    info: dict = {"count": warmup, "case_id": None, "axis_id": None, "elapsed_ms": [], "errors": 0}
    if warmup <= 0 or not cases:
        return info

    case = cases[0]
    axis = taxonomy.axes[0]
    info["case_id"] = case["case_id"]
    info["axis_id"] = axis.id

    try:
        image_bytes, mime, _, _ = prepare_image(case["image_path"], max_edge=max_edge, image_format=image_format)
    except Exception:
        info["errors"] += warmup
        return info

    for _ in range(warmup):
        try:
            result = decision.decide_axis(backend, image_bytes, mime, axis)
            info["elapsed_ms"].append(result["elapsed_ms"])
        except Exception:
            info["errors"] += 1
    return info


def _run_prime(
    case: dict, backend, max_edge: int, parallel: int = 1, image_format: str = "jpeg", mode: str = "choice"
) -> dict:
    """E9 ホットロード用: 1ケースにつき、画像をサーバーのキャッシュに載せる準備リクエストを
    parallel 本同時に送る(並列送信時に全スロットへ画像を載せるため。parallel=1 なら1本)。
    mode が json のときは JSON方式と同じ system文の準備(json_baseline.prime)を1本だけ送る
    (キャッシュは先頭一致なので、方式ごとに自分の先頭を載せる必要がある)。

    elapsed_ms は開始から全完了までの壁時計時間。prime自体の失敗(画像読み込み・通信の
    いずれも)はエラーとしてケースの記録に残し、以降の判定は通常どおり続ける。
    """
    parallel = max(1, parallel)
    prime_fn = decision.prime
    if mode in ("json", "json_schema", "dgemma_json"):
        prime_fn, parallel = json_baseline.prime, 1
    start = time.perf_counter_ns()
    try:
        image_bytes, mime, _, _ = prepare_image(case["image_path"], max_edge=max_edge, image_format=image_format)
        if parallel == 1:
            # N=1 は従来どおり(prime自身の計測値をそのまま使う)
            result = prime_fn(backend, image_bytes, mime)
            return {"elapsed_ms": result["elapsed_ms"], "error": None, "parallel": 1}
        else:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = [executor.submit(prime_fn, backend, image_bytes, mime) for _ in range(parallel)]
                errors = []
                for f in futures:
                    try:
                        f.result()
                    except Exception as e:
                        errors.append(f"{type(e).__name__}: {e}")
            if errors:
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                return {"elapsed_ms": elapsed_ms, "error": "; ".join(errors), "parallel": parallel}
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        return {"elapsed_ms": elapsed_ms, "error": None, "parallel": parallel}
    except Exception as e:
        return {"elapsed_ms": None, "error": f"{type(e).__name__}: {e}", "parallel": parallel}


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
    dataset_version: str | None = None,
    prime: bool = False,
    axis_concurrency: int = 1,
    confirm: bool = False,
    rank_threshold: float = 0.5,
    image_format: str = "jpeg",
    bundled_multi: str = "rank",
) -> int:
    try:
        validate_modes(modes)
    except ValueError as e:
        print(f"invalid --modes: {e}", file=sys.stderr)
        return 1

    taxonomy = load_taxonomy(taxonomy_path)

    check = check_manifest(manifest_path, taxonomy)
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

    excluded_cases: list[dict] = []
    rights_excluded_ids: set[str] = set()
    provisionally_evaluated: list[dict] = []
    for case in check.cases:
        if case["rights"].get("rights_confirmed") is not True:
            excluded_cases.append({"case_id": case["case_id"], "reason": "rights_not_confirmed"})
            rights_excluded_ids.add(case["case_id"])
        else:
            provisionally_evaluated.append(case)

    # 元ケースがrights未確認で除外された派生ケースも合わせて除外する。派生の派生は
    # check_manifest で禁止しているため、1段だけ見れば連鎖は起きない。
    evaluated_cases: list[dict] = []
    for case in provisionally_evaluated:
        derived_from = case.get("derived_from")
        if derived_from and derived_from in rights_excluded_ids:
            excluded_cases.append({"case_id": case["case_id"], "reason": "source excluded"})
        else:
            evaluated_cases.append(case)

    evaluated_by_scenario: dict[str, int] = {}
    for case in evaluated_cases:
        scenario = case.get("scenario", "<missing>")
        evaluated_by_scenario[scenario] = evaluated_by_scenario.get(scenario, 0) + 1

    cases_dir = os.path.join(output_dir, "cases")
    os.makedirs(cases_dir, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()

    warmup_info = _run_warmup(evaluated_cases, taxonomy, backend, warmup, max_edge, image_format)

    records: list[dict] = []
    for i, case in enumerate(evaluated_cases):
        order = case_mode_order(modes, i)
        for order_index, mode in enumerate(order):
            # prime は各モードの計測の直前に、そのモード自身の先頭(system文+画像)で送る
            # (キャッシュは先頭一致のため。選択式の準備ではJSONに効かない)。
            # classification_wall_ms には含めない(別記録)。
            # dgemma_choice は自前の dgemma-server が読み出しを自前で行い、先頭だけを載せる手段が無いので準備しない。
            prime_info = (
                _run_prime(case, backend, max_edge, axis_concurrency if axis_concurrency >= 2 else 1, image_format, mode)
                if prime and mode != "dgemma_choice"
                else None
            )
            result = _classify_safe(
                case["image_path"], taxonomy, backend, mode, max_edge, axis_concurrency, confirm, rank_threshold, image_format,
                bundled_multi,
            )
            if prime_info is not None:
                result["prime_ms"] = prime_info["elapsed_ms"]
                result["prime_error"] = prime_info["error"]
                result["prime_parallel"] = prime_info["parallel"]
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
        "dataset_version": dataset_version,
        "model": backend.model,
        "base_url": backend.base_url,
        # DiffusionGemma(issue #4)の run のみ。vLLM の commit 等は --runtime-info / --note に書く。
        "dgemma": (
            {
                "structured_url": backend.structured_url,
                "samples": backend.samples,
                "seed": backend.seed,
                "template": backend.template,
                "instruction": backend.instruction,
                "yn_style": backend.yn_style,
                "order": backend.order,
                "steps": backend.steps,
                "adaptive_threshold": backend.adaptive_threshold,
                "adaptive_max": backend.adaptive_max if backend.adaptive_threshold is not None else None,
                "max_per_read": backend.max_per_read,
                "max_soft_tokens": backend.max_soft_tokens,
                "extra_body": backend.extra_body,
                "dropped_params": backend.dropped_params,
                "prime_skipped_modes": ["dgemma_choice"] if prime and "dgemma_choice" in modes else [],
            }
            if hasattr(backend, "structured_url")
            else None
        ),
        "modes": modes,
        "max_edge": max_edge,
        "image_format": image_format,
        "jpeg_quality": JPEG_QUALITY if image_format == "jpeg" else None,
        "warmup": warmup_info,
        "prime": prime,
        "prime_scope": "per_mode" if prime else None,  # prime=true で prime_scope が欠落している旧run.jsonは、選択式の準備を1回だけ送っていた(JSONに効かない)
        "prime_parallel": (axis_concurrency if axis_concurrency >= 2 else 1) if prime else None,
        "axis_concurrency": axis_concurrency,
        "confirm": confirm,
        "rank_threshold": rank_threshold,
        "bundled_multi": bundled_multi,
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
        report.build_case_row(r["case"], r["mode"], r["order_index"], r["result"], taxonomy) for r in records
    ]
    report.write_cases_csv(os.path.join(output_dir, "cases.csv"), csv_rows, taxonomy)

    summary_text = report.build_summary(
        evaluated_cases, records, modes, taxonomy, confirm, rank_threshold, max_edge, image_format, bundled_multi
    )
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    return 0
