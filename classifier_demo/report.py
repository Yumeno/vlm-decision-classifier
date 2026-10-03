"""評価結果の採点・集計・レポート生成。

採点規則は doc/dataset-plan.md §2.5-2.7、集計規則は
doc/implementation-experiment-plan.md §5 に従う。

軸の一覧(単一選択/複数選択)は taxonomy から取る(ハードコードしない)。
複数選択軸(character・outfit など)はすべて同じ集合採点(完全一致・TP/FP/FN・
micro/macro)を受ける。character 固有なのはメタデータ照合(LoRA・トリガーワード)
と、派生ケース比較・シナリオ表の「キャラ完全一致」列だけ。
"""

from __future__ import annotations

import csv
import itertools
import math
import re
import statistics
from collections import Counter

from .image import JPEG_QUALITY
from .taxonomy import Taxonomy

_WS_RE = re.compile(r"\s+")

STATIC_CSV_FIELDS = ["case_id", "source_image_id", "derived_from", "scenario", "mode", "order_index"]

METADATA_CSV_FIELDS = [
    "expected_meta_format",
    "detected_meta_format",
    "expected_meta_chars",
    "detected_meta_chars",
    "meta_chars_ok",
    "expected_meta_loras",
    "detected_meta_loras",
    "meta_loras_ok",
    "expected_meta_triggers",
    "detected_meta_triggers",
    "meta_triggers_ok",
    "expected_meta_artificial",
]

TIMING_CSV_FIELDS = [
    "classification_wall_ms",
    "prime_ms",
    "request_count",
    "error_types",
    "json_attempts",
    "json_first_attempt_ms",
    "json_total_ms",
    "completion_tokens",
]


def single_axis_ids(taxonomy: Taxonomy) -> list[str]:
    return [a.id for a in taxonomy.axes if not a.multi]


def multi_axis_ids(taxonomy: Taxonomy) -> list[str]:
    return [a.id for a in taxonomy.axes if a.multi]


def axis_choice_ids(taxonomy: Taxonomy, axis_id: str) -> list[str]:
    return [c.id for c in taxonomy.axis(axis_id).choices]


def csv_fieldnames(taxonomy: Taxonomy) -> list[str]:
    fields = list(STATIC_CSV_FIELDS)
    for axis_id in single_axis_ids(taxonomy):
        fields += [f"expected_{axis_id}", f"pred_{axis_id}", f"ok_{axis_id}"]
    for axis_id in multi_axis_ids(taxonomy):
        fields += [
            f"expected_{axis_id}",
            f"pred_{axis_id}",
            f"exact_{axis_id}",
            f"tp_{axis_id}",
            f"fp_{axis_id}",
            f"fn_{axis_id}",
            f"{axis_id}_axis_failed",
        ]
    fields += METADATA_CSV_FIELDS + TIMING_CSV_FIELDS
    return fields


def nearest_rank_percentile(values: list[float], pct: float) -> float:
    """nearest-rank法での百分位点。

    昇順ソート後、rank = ceil(pct/100 * N)(1始まり、最小1・最大N)番目の値を返す。
    例: N=5, pct=90 -> rank=ceil(4.5)=5 -> 5番目(最大値)。
    """
    if not values:
        raise ValueError("empty values")
    ordered = sorted(values)
    n = len(ordered)
    rank = max(1, min(n, math.ceil(pct / 100 * n)))
    return ordered[rank - 1]


def score_single_axis(case: dict, result: dict, axis_id: str) -> tuple[str, str, bool]:
    """単一選択軸1つの (expected, pred, ok) を返す。

    軸が `vision_tags` に無い、または空(欠落・失敗)の場合は pred を空文字にし、
    期待値が何であっても ok を明示的に False にする(複数選択軸の失敗規則と同じ扱い)。
    """
    tags = result.get("vision_tags") or {}
    axis_tags = tags.get(axis_id)
    failed = not axis_tags
    pred = axis_tags[0] if axis_tags else ""
    expected = case["expected"][axis_id]
    ok = (not failed) and pred == expected
    return expected, pred, ok


def score_multi_axis(case: dict, result: dict, axis_id: str) -> dict:
    """複数選択軸1つの採点。dataset-plan §2.6 の規則(character・outfit など共通):
    - 空集合どうしは完全一致とみなす。
    - 軸が失敗した場合は、予測集合を空として TP/FP/FN を数えるが、
      exact_match は(期待が空集合でも)常に不正解とする。
    """
    tags = result.get("vision_tags") or {}
    failed = axis_id not in tags
    predicted = set(tags.get(axis_id, []))
    expected = set(case["expected"][axis_id])
    exact_match = (not failed) and predicted == expected
    return {
        "expected": expected,
        "predicted": predicted,
        "exact_match": exact_match,
        "tp": predicted & expected,
        "fp": predicted - expected,
        "fn": expected - predicted,
        "failed": failed,
    }


def _all_axes_ok(case: dict, result: dict, taxonomy: Taxonomy) -> bool:
    """単一選択軸すべて正解、かつ複数選択軸すべて完全一致(シナリオ表・ペア比較の「全軸正解」)。"""
    if not all(score_single_axis(case, result, a)[2] for a in single_axis_ids(taxonomy)):
        return False
    return all(score_multi_axis(case, result, a)["exact_match"] for a in multi_axis_ids(taxonomy))


def _normalize_lora_name(name: str) -> str:
    return _WS_RE.sub(" ", name.strip().lower())


def _lora_multiset(loras: list[dict]) -> Counter:
    """正規化名+weightの組を多重集合として数える(重複LoRA指定を潰さないため`set`にしない)。"""
    return Counter((_normalize_lora_name(l["name"]), l.get("weight")) for l in loras)


def format_loras(loras: list[dict]) -> str:
    return ";".join(sorted(f"{l['name']}:{l.get('weight')}" for l in loras))


def score_metadata(case: dict, result: dict) -> dict:
    """メタデータの採点(format・character・LoRA・トリガーワード検出)。

    抽出自体が失敗した場合は常に不正解とする(`artificial` はメタデータから検出できないため採点しない)。
    メタデータの証拠は character 軸(LoRA名・トリガーワード照合)専用で、outfit 等の他の
    複数選択軸には対応する証拠が無いため一般化しない。
    """
    evidence = result.get("metadata_evidence")
    failed = evidence is None or evidence.get("format") == "error"
    detected_format = evidence["format"] if evidence else None
    detected_loras = evidence["loras"] if evidence else []
    detected_chars = {m["character"] for m in evidence["matches"]} if evidence else set()
    detected_triggers = (
        {m["value"] for m in evidence["matches"] if m["kind"] == "prompt_trigger"} if evidence else set()
    )

    expected_meta = case["expected_metadata"]
    expected_format = expected_meta["format"]
    expected_loras = expected_meta.get("loras", [])
    expected_chars = set(expected_meta["characters"])
    expected_triggers = set(expected_meta.get("trigger_words", []))

    return {
        "expected_format": expected_format,
        "detected_format": detected_format,
        "format_ok": (not failed) and detected_format == expected_format,
        "expected_chars": expected_chars,
        "detected_chars": detected_chars,
        "chars_ok": (not failed) and detected_chars == expected_chars,
        "expected_loras": expected_loras,
        "detected_loras": detected_loras,
        "loras_ok": (not failed) and _lora_multiset(detected_loras) == _lora_multiset(expected_loras),
        "expected_triggers": expected_triggers,
        "detected_triggers": detected_triggers,
        "triggers_ok": (not failed) and detected_triggers == expected_triggers,
        "expected_artificial": expected_meta.get("artificial"),
        "failed": failed,
    }


def build_case_row(case: dict, mode: str, order_index: int, result: dict, taxonomy: Taxonomy) -> dict:
    row = {
        "case_id": case["case_id"],
        "source_image_id": case["source_image_id"],
        "derived_from": case.get("derived_from") or "",
        "scenario": case.get("scenario", ""),
        "mode": mode,
        "order_index": order_index,
    }

    for axis_id in single_axis_ids(taxonomy):
        expected, pred, ok = score_single_axis(case, result, axis_id)
        row[f"expected_{axis_id}"] = expected
        row[f"pred_{axis_id}"] = pred
        row[f"ok_{axis_id}"] = ok

    for axis_id in multi_axis_ids(taxonomy):
        scored = score_multi_axis(case, result, axis_id)
        row[f"expected_{axis_id}"] = ";".join(sorted(scored["expected"]))
        row[f"pred_{axis_id}"] = ";".join(sorted(scored["predicted"]))
        row[f"exact_{axis_id}"] = scored["exact_match"]
        row[f"tp_{axis_id}"] = len(scored["tp"])
        row[f"fp_{axis_id}"] = len(scored["fp"])
        row[f"fn_{axis_id}"] = len(scored["fn"])
        row[f"{axis_id}_axis_failed"] = scored["failed"]

    meta = score_metadata(case, result)
    row["expected_meta_format"] = meta["expected_format"]
    row["detected_meta_format"] = meta["detected_format"] if meta["detected_format"] is not None else ""
    row["expected_meta_chars"] = ";".join(sorted(meta["expected_chars"]))
    row["detected_meta_chars"] = ";".join(sorted(meta["detected_chars"]))
    row["meta_chars_ok"] = meta["chars_ok"]
    row["expected_meta_loras"] = format_loras(meta["expected_loras"])
    row["detected_meta_loras"] = format_loras(meta["detected_loras"])
    row["meta_loras_ok"] = meta["loras_ok"]
    row["expected_meta_triggers"] = ";".join(sorted(meta["expected_triggers"]))
    row["detected_meta_triggers"] = ";".join(sorted(meta["detected_triggers"]))
    row["meta_triggers_ok"] = meta["triggers_ok"]
    row["expected_meta_artificial"] = meta["expected_artificial"]

    timing = result.get("timing_ms") or {}
    row["classification_wall_ms"] = timing.get("classification_wall_ms", "")
    # E9: --prime 有効時のみ result に付く(評価器側で付加。pipeline.classify自体は関知しない)。
    row["prime_ms"] = result.get("prime_ms", "")
    row["request_count"] = result.get("request_count", "")
    error_types = [e.get("type", "") for e in result.get("errors") or []]
    row["error_types"] = ";".join(dict.fromkeys(error_types))

    if mode in ("json", "json_schema", "dgemma_json"):
        jb = result.get("json_baseline") or {}
        row["json_attempts"] = len(jb.get("attempts") or [])
        row["json_first_attempt_ms"] = jb.get("first_attempt_ms", "")
        row["json_total_ms"] = jb.get("total_ms", "")
        usage = jb.get("usage") or {}
        row["completion_tokens"] = usage.get("completion_tokens", "")
    else:
        row["json_attempts"] = ""
        row["json_first_attempt_ms"] = ""
        row["json_total_ms"] = ""
        row["completion_tokens"] = ""

    return row


def write_cases_csv(path: str, rows: list[dict], taxonomy: Taxonomy) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames(taxonomy))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pct_str(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "N=0"
    return f"{numerator / denominator:.1%} ({numerator}/{denominator})"


def _dual_pct_str(numerator_src: int, denominator_src: int, numerator_all: int, denominator_all: int) -> str:
    """画像判定の主集計(元画像分母)と、水増し確認用の参考値(全ケース分母)を並記する。"""
    return f"元画像 {_pct_str(numerator_src, denominator_src)} ／ 全ケース {_pct_str(numerator_all, denominator_all)}"


def _is_success(result: dict) -> bool:
    return not (result.get("errors") or [])


def _multi_axis_stats(scores: list[dict], choice_ids: list[str]) -> dict[str, dict]:
    """候補ごとの tp/fp/fn/precision/recall/f1 と micro集計を返す(複数選択軸すべてに共通)。"""
    stats: dict[str, dict] = {}
    micro_tp = micro_fp = micro_fn = 0
    for cid in choice_ids:
        tp = sum(1 for s in scores if cid in s["tp"])
        fp = sum(1 for s in scores if cid in s["fp"])
        fn = sum(1 for s in scores if cid in s["fn"])
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        stats[cid] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    mp = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    mr = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    mf1 = 2 * mp * mr / (mp + mr) if (mp + mr) else 0.0
    stats["micro"] = {"tp": micro_tp, "fp": micro_fp, "fn": micro_fn, "precision": mp, "recall": mr, "f1": mf1}
    return stats


def _macro_prf1(stats: dict[str, dict], choice_ids: list[str]) -> dict:
    """choice_ids の候補別値を単純平均したmacro Precision/Recall/F1。

    分母(Precisionは TP+FP、Recallは TP+FN)が0で値を定義できない候補は、
    その候補だけ平均から除く(0として扱うと過小評価になるため)。F1はPrecision・
    Recallの両方が定義できる候補のみで平均する。1件も無ければ None(n/a)。
    """
    precisions = [stats[cid]["precision"] for cid in choice_ids if (stats[cid]["tp"] + stats[cid]["fp"]) > 0]
    recalls = [stats[cid]["recall"] for cid in choice_ids if (stats[cid]["tp"] + stats[cid]["fn"]) > 0]
    f1s = [
        stats[cid]["f1"]
        for cid in choice_ids
        if (stats[cid]["tp"] + stats[cid]["fp"]) > 0 and (stats[cid]["tp"] + stats[cid]["fn"]) > 0
    ]
    return {
        "precision": statistics.mean(precisions) if precisions else None,
        "recall": statistics.mean(recalls) if recalls else None,
        "f1": statistics.mean(f1s) if f1s else None,
    }


def _macro_val(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def _pred_label(axis_score: dict) -> str:
    if axis_score["failed"]:
        return "FAILED"
    predicted = axis_score["predicted"]
    return ";".join(sorted(predicted)) if predicted else "(空)"


def _same_pred_label(o_score: dict, d_score: dict) -> str:
    """予測が同じかどうかのラベル。両方失敗している場合は偶然の一致と区別する。"""
    if o_score["failed"] and d_score["failed"]:
        return "同一(失敗)"
    if o_score["failed"] != d_score["failed"]:
        return "不一致"
    return "一致" if o_score["predicted"] == d_score["predicted"] else "不一致"


def _latency_block(recs: list[dict], mode: str) -> list[str]:
    lines: list[str] = []
    all_ms = [
        r["result"]["timing_ms"]["classification_wall_ms"]
        for r in recs
        if (r["result"].get("timing_ms") or {}).get("classification_wall_ms") is not None
    ]
    success_ms = [
        r["result"]["timing_ms"]["classification_wall_ms"]
        for r in recs
        if _is_success(r["result"]) and (r["result"].get("timing_ms") or {}).get("classification_wall_ms") is not None
    ]
    if all_ms:
        lines.append(
            f"  - 全件(N={len(all_ms)}): mean={statistics.mean(all_ms):.1f}ms "
            f"p50={nearest_rank_percentile(all_ms, 50):.1f}ms p90={nearest_rank_percentile(all_ms, 90):.1f}ms"
        )
    else:
        lines.append("  - 全件: N=0")
    if success_ms:
        lines.append(
            f"  - 成功のみ(N={len(success_ms)}): mean={statistics.mean(success_ms):.1f}ms "
            f"p50={nearest_rank_percentile(success_ms, 50):.1f}ms p90={nearest_rank_percentile(success_ms, 90):.1f}ms"
        )
    req_counts = [r["result"].get("request_count", 0) for r in recs if r["result"].get("request_count") is not None]
    if req_counts:
        lines.append(f"  - 平均リクエスト数: {statistics.mean(req_counts):.2f}")

    # E9 --prime: prime_ms が result に付いているケースのみ(prime有効時のみ)。
    prime_ms_values = [
        r["result"]["prime_ms"] for r in recs if isinstance(r["result"].get("prime_ms"), (int, float))
    ]
    if prime_ms_values:
        pp = max((r["result"].get("prime_parallel") or 1) for r in recs)
        par = f"、{pp}本同時" if pp >= 2 else ""
        lines.append(
            f"  - prime(画像の読み込み{par}、N={len(prime_ms_values)}): "
            f"mean={statistics.mean(prime_ms_values):.1f}ms "
            f"p50={nearest_rank_percentile(prime_ms_values, 50):.1f}ms "
            f"p90={nearest_rank_percentile(prime_ms_values, 90):.1f}ms"
        )
        combined_ms = []
        for r in recs:
            p = r["result"].get("prime_ms")
            c = (r["result"].get("timing_ms") or {}).get("classification_wall_ms")
            if isinstance(p, (int, float)) and isinstance(c, (int, float)):
                combined_ms.append(p + c)
        if combined_ms:
            lines.append(f"  - prime + 判定(N={len(combined_ms)}): mean={statistics.mean(combined_ms):.1f}ms")

    if mode in ("json", "json_schema", "dgemma_json") and recs:
        fmt_errors = sum(1 for r in recs if (r["result"].get("json_baseline") or {}).get("tags") is None)
        attempts = [len((r["result"].get("json_baseline") or {}).get("attempts") or []) for r in recs]
        lines.append(f"  - 形式不正率: {_pct_str(fmt_errors, len(recs))}")
        if attempts:
            lines.append(f"  - 平均試行回数: {statistics.mean(attempts):.2f}")
    return lines


def _pair_compare(
    base_by_case: dict[str, dict], other_by_case: dict[str, dict], taxonomy: Taxonomy
) -> tuple[int, int, int, int, int]:
    """base/other 間のペア比較。「キャラ完全一致」は character 軸専用(仕様どおり)、
    「全軸正解」は単一軸すべて+複数選択軸すべての完全一致(taxonomy 由来)。
    """
    common_ids = sorted(set(base_by_case) & set(other_by_case))
    char_base_wins = char_other_wins = 0
    all_base_wins = all_other_wins = 0
    for cid in common_ids:
        rb, ro = base_by_case[cid], other_by_case[cid]
        cb = score_multi_axis(rb["case"], rb["result"], "character")["exact_match"]
        co = score_multi_axis(ro["case"], ro["result"], "character")["exact_match"]
        if cb and not co:
            char_base_wins += 1
        elif co and not cb:
            char_other_wins += 1
        ab = _all_axes_ok(rb["case"], rb["result"], taxonomy)
        ao = _all_axes_ok(ro["case"], ro["result"], taxonomy)
        if ab and not ao:
            all_base_wins += 1
        elif ao and not ab:
            all_other_wins += 1
    return char_base_wins, char_other_wins, all_base_wins, all_other_wins, len(common_ids)


def build_summary(
    evaluated_cases: list[dict],
    records: list[dict],
    modes: list[str],
    taxonomy: Taxonomy,
    confirm: bool = False,
    rank_threshold: float = 0.5,
    max_edge: int | None = None,
    image_format: str | None = None,
    bundled_multi: str = "rank",
) -> str:
    """summary.md の本文を組み立てる。records は
    {"case", "mode", "order_index", "result"} の dict のリスト(実行順)。

    画像判定(単一軸・複数選択軸・シナリオ別・latency・モード間比較)は
    派生ケース(メタデータ除去コピー)を含めると同じ画像が二重に効いてしまうため、
    元画像ケース(`derived_from` が null)だけを分母にする(dataset-plan §1 の水増し防止)。
    メタデータの採点は派生ケースが本題(画素同一・メタデータのみ相違)なので全ケースで出す。
    """
    single_axes = single_axis_ids(taxonomy)
    multi_axes = multi_axis_ids(taxonomy)

    source_cases = [c for c in evaluated_cases if not c.get("derived_from")]
    derived_cases = [c for c in evaluated_cases if c.get("derived_from")]
    n_source = len(source_cases)

    records_source = [r for r in records if not r["case"].get("derived_from")]
    by_mode_source: dict[str, list[dict]] = {m: [] for m in modes}
    for r in records_source:
        by_mode_source[r["mode"]].append(r)

    by_mode_all: dict[str, list[dict]] = {m: [] for m in modes}
    for r in records:
        by_mode_all[r["mode"]].append(r)

    lines: list[str] = []
    lines.append("# 評価サマリー")
    lines.append("")
    condition = (
        f"実行条件: confirm={'オン' if confirm else 'オフ'}(複数選択軸の候補ごとのyes/no確認)"
        f", rank_threshold={rank_threshold}(confirmオフ時の採用閾値)"
    )
    if "bundled" in modes:
        condition += f", 束ね質問の複数選択: {bundled_multi}(rank=相対スコアと閾値 / yn=候補ごとのYes/No欄)"
    if image_format is not None:
        condition += f", 送信画像: {image_format}(quality={JPEG_QUALITY})" if image_format == "jpeg" else f", 送信画像: {image_format}"
        condition += f"・長辺{max_edge}"
    lines.append(condition)
    lines.append("")
    lines.append(
        f"評価ケース数(全体): N={len(evaluated_cases)}"
        f"(元画像 {n_source} 件 / 派生(メタデータ除去コピー) {len(derived_cases)} 件)"
    )
    lines.append("")

    lines.append(f"## 単一軸の正答率(元画像 N={n_source} 件。全ケース分母も併記)")
    lines.append("")
    lines.append("| 軸 | " + " | ".join(modes) + " |")
    lines.append("|---|" + "---|" * len(modes))
    for axis_id in single_axes:
        cells = []
        for mode in modes:
            recs_src = by_mode_source[mode]
            recs_all = by_mode_all[mode]
            correct_src = sum(1 for r in recs_src if score_single_axis(r["case"], r["result"], axis_id)[2])
            correct_all = sum(1 for r in recs_all if score_single_axis(r["case"], r["result"], axis_id)[2])
            cells.append(_dual_pct_str(correct_src, len(recs_src), correct_all, len(recs_all)))
        lines.append(f"| {axis_id} | " + " | ".join(cells) + " |")
    lines.append("")

    for axis_id in multi_axes:
        choice_ids = axis_choice_ids(taxonomy, axis_id)
        lines.append(f"## {axis_id}(集合、元画像 N={n_source} 件。全ケース分母も併記)")
        lines.append("")
        lines.append("### 完全一致率(集合の完全一致。空集合どうしは一致)")
        lines.append("")
        scores_by_mode_source = {
            mode: [score_multi_axis(r["case"], r["result"], axis_id) for r in by_mode_source[mode]] for mode in modes
        }
        scores_by_mode_all = {
            mode: [score_multi_axis(r["case"], r["result"], axis_id) for r in by_mode_all[mode]] for mode in modes
        }
        lines.append("| " + " | ".join(modes) + " |")
        lines.append("|" + "---|" * len(modes))
        cells = []
        for mode in modes:
            scores_src = scores_by_mode_source[mode]
            scores_all = scores_by_mode_all[mode]
            exact_src = sum(1 for s in scores_src if s["exact_match"])
            exact_all = sum(1 for s in scores_all if s["exact_match"])
            cells.append(_dual_pct_str(exact_src, len(scores_src), exact_all, len(scores_all)))
        lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        lines.append("### 候補別 TP/FP/FN・Precision/Recall/F1(各列は 元画像 / 全ケース)")
        lines.append("")
        lines.append(
            "micro は全候補合算のTP/FP/FNから計算。macro は候補別値(TP/FP/FNの列は対象外)の"
            "単純平均で、分母(Precisionは TP+FP、Recallは TP+FN)が0で値を定義できない候補は"
            "平均から除く(該当時は `n/a`)。"
        )
        lines.append("")
        for mode in modes:
            lines.append(f"#### {mode}")
            lines.append("")
            lines.append(
                "| 候補 | TP(元) | TP(全) | FP(元) | FP(全) | FN(元) | FN(全) | "
                "Precision(元) | Precision(全) | Recall(元) | Recall(全) | F1(元) | F1(全) |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            stats_src = _multi_axis_stats(scores_by_mode_source[mode], choice_ids)
            stats_all = _multi_axis_stats(scores_by_mode_all[mode], choice_ids)
            for cid in choice_ids + ["micro"]:
                s, a = stats_src[cid], stats_all[cid]
                lines.append(
                    f"| {cid} | {s['tp']} | {a['tp']} | {s['fp']} | {a['fp']} | {s['fn']} | {a['fn']} | "
                    f"{s['precision']:.2f} | {a['precision']:.2f} | {s['recall']:.2f} | {a['recall']:.2f} | "
                    f"{s['f1']:.2f} | {a['f1']:.2f} |"
                )
            macro_src = _macro_prf1(stats_src, choice_ids)
            macro_all = _macro_prf1(stats_all, choice_ids)
            lines.append(
                f"| macro | - | - | - | - | - | - | "
                f"{_macro_val(macro_src['precision'])} | {_macro_val(macro_all['precision'])} | "
                f"{_macro_val(macro_src['recall'])} | {_macro_val(macro_all['recall'])} | "
                f"{_macro_val(macro_src['f1'])} | {_macro_val(macro_all['f1'])} |"
            )
            lines.append("")

    lines.append(
        f"## シナリオ別(キャラ完全一致・全軸正解、元画像 N={n_source} 件。全ケース分母も併記)"
    )
    lines.append("")
    # シナリオ一覧は元画像・派生を合わせた全評価ケースから作る(派生ケースだけにしか
    # 現れないシナリオがあっても取りこぼさないため)。元画像側に件数が無いシナリオは
    # 「元画像欄」を `-` にする(0/0 の割合表示にしない)。
    # 「キャラ完全一致」列は character 軸専用(仕様どおり)。「全軸正解」は単一軸+
    # 複数選択軸すべて(outfit 等を含む)の完全一致。
    scenarios = sorted({c.get("scenario", "") for c in evaluated_cases})
    for mode in modes:
        lines.append(f"### {mode}")
        lines.append("")
        lines.append("| シナリオ | N(元画像/全ケース) | キャラ完全一致 | 全軸正解 |")
        lines.append("|---|---|---|---|")
        recs_src = by_mode_source[mode]
        recs_all = by_mode_all[mode]
        for scenario in scenarios:
            sc_recs_src = [r for r in recs_src if r["case"].get("scenario", "") == scenario]
            sc_recs_all = [r for r in recs_all if r["case"].get("scenario", "") == scenario]
            n_src = len(sc_recs_src)
            n_all = len(sc_recs_all)
            if n_src == 0 and n_all == 0:
                continue

            def _all_axes_ok_count(recs: list[dict]) -> int:
                return sum(1 for r in recs if _all_axes_ok(r["case"], r["result"], taxonomy))

            char_exact_all = sum(
                1 for r in sc_recs_all if score_multi_axis(r["case"], r["result"], "character")["exact_match"]
            )
            all_ok_all = _all_axes_ok_count(sc_recs_all)
            if n_src == 0:
                lines.append(
                    f"| {scenario} | - ／ {n_all} | "
                    f"元画像 - ／ 全ケース {_pct_str(char_exact_all, n_all)} | "
                    f"元画像 - ／ 全ケース {_pct_str(all_ok_all, n_all)} |"
                )
                continue
            char_exact_src = sum(
                1 for r in sc_recs_src if score_multi_axis(r["case"], r["result"], "character")["exact_match"]
            )
            all_ok_src = _all_axes_ok_count(sc_recs_src)
            lines.append(
                f"| {scenario} | {n_src} ／ {n_all} | "
                f"{_dual_pct_str(char_exact_src, n_src, char_exact_all, n_all)} | "
                f"{_dual_pct_str(all_ok_src, n_src, all_ok_all, n_all)} |"
            )
        lines.append("")

    lines.append("## メタデータ(全ケース、モード非依存)")
    lines.append("")
    meta_mode = "choice" if "choice" in modes else modes[0]
    meta_recs = by_mode_all[meta_mode]
    n_meta = len(meta_recs)
    meta_scores = [score_metadata(r["case"], r["result"]) for r in meta_recs]
    format_ok = sum(1 for s in meta_scores if s["format_ok"])
    chars_ok = sum(1 for s in meta_scores if s["chars_ok"])
    loras_ok = sum(1 for s in meta_scores if s["loras_ok"])
    triggers_ok = sum(1 for s in meta_scores if s["triggers_ok"])
    lines.append(f"(集計元モード: {meta_mode}、元画像・派生を含む全ケース)")
    lines.append("")
    lines.append(f"- format 正答率: {_pct_str(format_ok, n_meta)}")
    lines.append(f"- characters 完全一致率: {_pct_str(chars_ok, n_meta)}")
    lines.append(f"- loras 完全一致率: {_pct_str(loras_ok, n_meta)}")
    lines.append(f"- trigger_words 完全一致率: {_pct_str(triggers_ok, n_meta)}")
    lines.append(
        "- `artificial` は画像から検出できない記録上の事実なので採点対象外"
        "(ケース別CSVに期待値のみ出力)。"
    )
    lines.append("")

    lines.append("## 派生ケース(メタデータ除去コピー)")
    lines.append("")
    lines.append(
        "画素は元ケースと同一で生成メタデータのみ除去したコピー。character の予測・正誤が"
        "元ケースと同じかどうかを確認する(画像判定の集計には含めない)。"
    )
    lines.append("")
    if derived_cases:
        lines.append(
            "| 派生ケース | 元ケース | mode | 元predicted | 派生predicted | 予測同一 | 元正誤 | 派生正誤 | 正誤同一 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        records_by_case_mode = {(r["case"]["case_id"], r["mode"]): r for r in records}
        for d in derived_cases:
            for mode in modes:
                d_rec = records_by_case_mode.get((d["case_id"], mode))
                origin_rec = records_by_case_mode.get((d["derived_from"], mode))
                if d_rec is None or origin_rec is None:
                    continue
                d_char = score_multi_axis(d, d_rec["result"], "character")
                o_char = score_multi_axis(origin_rec["case"], origin_rec["result"], "character")
                same_correct = d_char["exact_match"] == o_char["exact_match"]
                lines.append(
                    f"| {d['case_id']} | {d['derived_from']} | {mode} | "
                    f"{_pred_label(o_char)} | {_pred_label(d_char)} | {_same_pred_label(o_char, d_char)} | "
                    f"{'正' if o_char['exact_match'] else '誤'} | {'正' if d_char['exact_match'] else '誤'} | "
                    f"{'一致' if same_correct else '不一致'} |"
                )
    else:
        lines.append("派生ケースなし。")
    lines.append("")

    lines.append(f"## Latency(元画像 N={n_source} 件。全ケース分母(N={len(evaluated_cases)} 件)も併記)")
    lines.append("")
    for mode in modes:
        lines.append(f"### {mode}")
        lines.append("")
        lines.append("- 元画像:")
        lines.extend(_latency_block(by_mode_source[mode], mode))
        lines.append("- 全ケース:")
        lines.extend(_latency_block(by_mode_all[mode], mode))
        lines.append("")

    lines.append("## 失敗件数(エラー種別、全ケース)")
    lines.append("")
    for mode in modes:
        lines.append(f"### {mode}")
        lines.append("")
        counts: dict[str, int] = {}
        for r in by_mode_all[mode]:
            for e in r["result"].get("errors") or []:
                t = e.get("type", "unknown")
                counts[t] = counts.get(t, 0) + 1
        if counts:
            for t in sorted(counts):
                lines.append(f"- {t}: {counts[t]}")
        else:
            lines.append("- なし")
        lines.append("")

    if len(modes) >= 2:
        lines.append(f"## モード間のペア比較(元画像 N={n_source} 件。全ケース分母も併記)")
        lines.append("")
        for base, other in itertools.combinations(modes, 2):
            base_by_case_src = {r["case"]["case_id"]: r for r in by_mode_source[base]}
            other_by_case_src = {r["case"]["case_id"]: r for r in by_mode_source[other]}
            cb_wins_src, co_wins_src, ab_wins_src, ao_wins_src, n_common_src = _pair_compare(
                base_by_case_src, other_by_case_src, taxonomy
            )
            lines.append(f"### 元画像({base} vs {other}、N={n_common_src})")
            lines.append("")
            lines.append(f"- キャラ完全一致: {base}のみ正解 {cb_wins_src} 件 / {other}のみ正解 {co_wins_src} 件")
            lines.append(f"- 全軸正解: {base}のみ正解 {ab_wins_src} 件 / {other}のみ正解 {ao_wins_src} 件")
            lines.append("")

            base_by_case_all = {r["case"]["case_id"]: r for r in by_mode_all[base]}
            other_by_case_all = {r["case"]["case_id"]: r for r in by_mode_all[other]}
            cb_wins_all, co_wins_all, ab_wins_all, ao_wins_all, n_common_all = _pair_compare(
                base_by_case_all, other_by_case_all, taxonomy
            )
            lines.append(f"### 全ケース({base} vs {other}、N={n_common_all})")
            lines.append("")
            lines.append(f"- キャラ完全一致: {base}のみ正解 {cb_wins_all} 件 / {other}のみ正解 {co_wins_all} 件")
            lines.append(f"- 全軸正解: {base}のみ正解 {ab_wins_all} 件 / {other}のみ正解 {ao_wins_all} 件")
            lines.append("")

    lines.append("## 注記")
    lines.append("")
    lines.append("N が小さいため点推定を強い結論として扱わないこと。ケース別CSV(`cases.csv`)が一次資料。")
    lines.append("")

    return "\n".join(lines)
