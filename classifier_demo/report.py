"""評価結果の採点・集計・レポート生成。

採点規則は doc/dataset-plan.md §2.5-2.7、集計規則は
doc/implementation-experiment-plan.md §5 に従う。
"""

from __future__ import annotations

import csv
import math
import statistics

# manifest の `expected` に含まれる単一軸(character は別扱い)。
SINGLE_AXES = ["image_type", "art_style", "subject"]

# character 軸の候補id(taxonomy/default.yaml と対応)。
CHARACTERS = ["alisa", "second_original", "other_original"]

CSV_FIELDNAMES = [
    "case_id",
    "source_image_id",
    "derived_from",
    "scenario",
    "mode",
    "order_index",
    "expected_image_type",
    "pred_image_type",
    "ok_image_type",
    "expected_art_style",
    "pred_art_style",
    "ok_art_style",
    "expected_subject",
    "pred_subject",
    "ok_subject",
    "expected_character",
    "pred_character",
    "character_exact_match",
    "character_tp",
    "character_fp",
    "character_fn",
    "char_axis_failed",
    "expected_meta_format",
    "detected_meta_format",
    "expected_meta_chars",
    "detected_meta_chars",
    "meta_chars_ok",
    "classification_wall_ms",
    "request_count",
    "error_types",
    "json_attempts",
    "json_first_attempt_ms",
    "json_total_ms",
    "completion_tokens",
]


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
    """単一軸1つの (expected, pred, ok) を返す。軸が失敗(欠落)していれば pred は空文字。"""
    tags = result.get("vision_tags") or {}
    axis_tags = tags.get(axis_id)
    pred = axis_tags[0] if axis_tags else ""
    expected = case["expected"][axis_id]
    return expected, pred, pred == expected


def score_character(case: dict, result: dict) -> dict:
    """character軸の採点。dataset-plan §2.6 の規則:
    - 空集合どうしは完全一致とみなす。
    - 軸が失敗した場合は、予測集合を空として TP/FP/FN を数えるが、
      exact_match は(期待が空集合でも)常に不正解とする。
    """
    tags = result.get("vision_tags") or {}
    failed = "character" not in tags
    predicted = set(tags.get("character", []))
    expected = set(case["expected"]["character"])
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


def score_metadata(case: dict, result: dict) -> dict:
    """メタデータの採点(format・character検出)。抽出自体が失敗した場合は常に不正解とする。"""
    evidence = result.get("metadata_evidence")
    failed = evidence is None or evidence.get("format") == "error"
    detected_format = evidence["format"] if evidence else None
    detected_chars = {m["character"] for m in evidence["matches"]} if evidence else set()
    expected_meta = case["expected_metadata"]
    expected_format = expected_meta["format"]
    expected_chars = set(expected_meta["characters"])
    return {
        "expected_format": expected_format,
        "detected_format": detected_format,
        "format_ok": (not failed) and detected_format == expected_format,
        "expected_chars": expected_chars,
        "detected_chars": detected_chars,
        "chars_ok": (not failed) and detected_chars == expected_chars,
        "failed": failed,
    }


def build_case_row(case: dict, mode: str, order_index: int, result: dict) -> dict:
    row = {
        "case_id": case["case_id"],
        "source_image_id": case["source_image_id"],
        "derived_from": case.get("derived_from") or "",
        "scenario": case.get("scenario", ""),
        "mode": mode,
        "order_index": order_index,
    }

    for axis_id in SINGLE_AXES:
        expected, pred, ok = score_single_axis(case, result, axis_id)
        row[f"expected_{axis_id}"] = expected
        row[f"pred_{axis_id}"] = pred
        row[f"ok_{axis_id}"] = ok

    char = score_character(case, result)
    row["expected_character"] = ";".join(sorted(char["expected"]))
    row["pred_character"] = ";".join(sorted(char["predicted"]))
    row["character_exact_match"] = char["exact_match"]
    row["character_tp"] = len(char["tp"])
    row["character_fp"] = len(char["fp"])
    row["character_fn"] = len(char["fn"])
    row["char_axis_failed"] = char["failed"]

    meta = score_metadata(case, result)
    row["expected_meta_format"] = meta["expected_format"]
    row["detected_meta_format"] = meta["detected_format"] if meta["detected_format"] is not None else ""
    row["expected_meta_chars"] = ";".join(sorted(meta["expected_chars"]))
    row["detected_meta_chars"] = ";".join(sorted(meta["detected_chars"]))
    row["meta_chars_ok"] = meta["chars_ok"]

    timing = result.get("timing_ms") or {}
    row["classification_wall_ms"] = timing.get("classification_wall_ms", "")
    row["request_count"] = result.get("request_count", "")
    error_types = [e.get("type", "") for e in result.get("errors") or []]
    row["error_types"] = ";".join(dict.fromkeys(error_types))

    if mode == "json":
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


def write_cases_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pct_str(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "N=0"
    return f"{numerator / denominator:.1%} ({numerator}/{denominator})"


def _is_success(result: dict) -> bool:
    return not (result.get("errors") or [])


def build_summary(evaluated_cases: list[dict], records: list[dict], modes: list[str]) -> str:
    """summary.md の本文を組み立てる。records は
    {"case", "mode", "order_index", "result"} の dict のリスト(実行順)。
    """
    by_mode: dict[str, list[dict]] = {m: [] for m in modes}
    for r in records:
        by_mode[r["mode"]].append(r)

    lines: list[str] = []
    lines.append("# 評価サマリー")
    lines.append("")
    lines.append(f"評価ケース数(全体): N={len(evaluated_cases)}")
    lines.append("")

    lines.append("## 単一軸の正答率")
    lines.append("")
    lines.append("| 軸 | " + " | ".join(modes) + " |")
    lines.append("|---|" + "---|" * len(modes))
    for axis_id in SINGLE_AXES:
        cells = []
        for mode in modes:
            recs = by_mode[mode]
            n = len(recs)
            correct = sum(1 for r in recs if score_single_axis(r["case"], r["result"], axis_id)[2])
            cells.append(_pct_str(correct, n))
        lines.append(f"| {axis_id} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## キャラクター")
    lines.append("")
    lines.append("### 完全一致率(集合の完全一致。空集合どうしは一致)")
    lines.append("")
    char_scores_by_mode = {
        mode: [score_character(r["case"], r["result"]) for r in by_mode[mode]] for mode in modes
    }
    lines.append("| " + " | ".join(modes) + " |")
    lines.append("|" + "---|" * len(modes))
    cells = []
    for mode in modes:
        scores = char_scores_by_mode[mode]
        exact = sum(1 for s in scores if s["exact_match"])
        cells.append(_pct_str(exact, len(scores)))
    lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### キャラ別 TP/FP/FN・Precision/Recall/F1")
    lines.append("")
    for mode in modes:
        lines.append(f"#### {mode}")
        lines.append("")
        lines.append("| キャラ | TP | FP | FN | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|---|---|---|")
        scores = char_scores_by_mode[mode]
        micro_tp = micro_fp = micro_fn = 0
        for ch in CHARACTERS:
            tp = sum(1 for s in scores if ch in s["tp"])
            fp = sum(1 for s in scores if ch in s["fp"])
            fn = sum(1 for s in scores if ch in s["fn"])
            micro_tp += tp
            micro_fp += fp
            micro_fn += fn
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            lines.append(f"| {ch} | {tp} | {fp} | {fn} | {precision:.2f} | {recall:.2f} | {f1:.2f} |")
        mp = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
        mr = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
        mf1 = 2 * mp * mr / (mp + mr) if (mp + mr) else 0.0
        lines.append(f"| micro | {micro_tp} | {micro_fp} | {micro_fn} | {mp:.2f} | {mr:.2f} | {mf1:.2f} |")
        lines.append("")

    lines.append("## シナリオ別(キャラ完全一致・全軸正解)")
    lines.append("")
    scenarios = sorted({c.get("scenario", "") for c in evaluated_cases})
    for mode in modes:
        lines.append(f"### {mode}")
        lines.append("")
        lines.append("| シナリオ | N | キャラ完全一致 | 全軸正解 |")
        lines.append("|---|---|---|---|")
        recs = by_mode[mode]
        for scenario in scenarios:
            sc_recs = [r for r in recs if r["case"].get("scenario", "") == scenario]
            n = len(sc_recs)
            if n == 0:
                continue
            char_exact = sum(1 for r in sc_recs if score_character(r["case"], r["result"])["exact_match"])
            all_ok = 0
            for r in sc_recs:
                axes_ok = all(score_single_axis(r["case"], r["result"], a)[2] for a in SINGLE_AXES)
                if axes_ok and score_character(r["case"], r["result"])["exact_match"]:
                    all_ok += 1
            lines.append(f"| {scenario} | {n} | {_pct_str(char_exact, n)} | {_pct_str(all_ok, n)} |")
        lines.append("")

    lines.append("## 元画像単位の集計")
    lines.append("")
    source_ids = sorted({c["source_image_id"] for c in evaluated_cases})
    derived_count = sum(1 for c in evaluated_cases if c.get("derived_from"))
    lines.append(f"元画像数: {len(source_ids)}  派生ケース数: {derived_count}")
    lines.append("")
    for mode in modes:
        recs = [r for r in by_mode[mode] if not r["case"].get("derived_from")]
        exact = sum(1 for r in recs if score_character(r["case"], r["result"])["exact_match"])
        lines.append(f"- {mode}: 元画像のみのキャラ完全一致率 {_pct_str(exact, len(recs))}")
    lines.append("")

    lines.append("## メタデータ(モード非依存)")
    lines.append("")
    meta_mode = "choice" if "choice" in modes else modes[0]
    meta_recs = by_mode[meta_mode]
    n_meta = len(meta_recs)
    format_ok = sum(1 for r in meta_recs if score_metadata(r["case"], r["result"])["format_ok"])
    chars_ok = sum(1 for r in meta_recs if score_metadata(r["case"], r["result"])["chars_ok"])
    lines.append(f"(集計元モード: {meta_mode})")
    lines.append("")
    lines.append(f"- format 正答率: {_pct_str(format_ok, n_meta)}")
    lines.append(f"- characters 完全一致率: {_pct_str(chars_ok, n_meta)}")
    lines.append("")

    lines.append("## Latency")
    lines.append("")
    for mode in modes:
        recs = by_mode[mode]
        lines.append(f"### {mode}")
        lines.append("")
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
                f"- 全件(N={len(all_ms)}): mean={statistics.mean(all_ms):.1f}ms "
                f"p50={nearest_rank_percentile(all_ms, 50):.1f}ms p90={nearest_rank_percentile(all_ms, 90):.1f}ms"
            )
        if success_ms:
            lines.append(
                f"- 成功のみ(N={len(success_ms)}): mean={statistics.mean(success_ms):.1f}ms "
                f"p50={nearest_rank_percentile(success_ms, 50):.1f}ms p90={nearest_rank_percentile(success_ms, 90):.1f}ms"
            )
        req_counts = [r["result"].get("request_count", 0) for r in recs if r["result"].get("request_count") is not None]
        if req_counts:
            lines.append(f"- 平均リクエスト数: {statistics.mean(req_counts):.2f}")
        if mode == "json" and recs:
            fmt_errors = sum(1 for r in recs if (r["result"].get("json_baseline") or {}).get("tags") is None)
            attempts = [len((r["result"].get("json_baseline") or {}).get("attempts") or []) for r in recs]
            lines.append(f"- 形式不正率: {_pct_str(fmt_errors, len(recs))}")
            if attempts:
                lines.append(f"- 平均試行回数: {statistics.mean(attempts):.2f}")
        lines.append("")

    lines.append("## 失敗件数(エラー種別)")
    lines.append("")
    for mode in modes:
        lines.append(f"### {mode}")
        lines.append("")
        counts: dict[str, int] = {}
        for r in by_mode[mode]:
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
        lines.append("## モード間のペア比較")
        lines.append("")
        base, other = modes[0], modes[1]
        base_by_case = {r["case"]["case_id"]: r for r in by_mode[base]}
        other_by_case = {r["case"]["case_id"]: r for r in by_mode[other]}
        common_ids = sorted(set(base_by_case) & set(other_by_case))
        char_base_wins = char_other_wins = 0
        all_base_wins = all_other_wins = 0
        for cid in common_ids:
            rb, ro = base_by_case[cid], other_by_case[cid]
            cb = score_character(rb["case"], rb["result"])["exact_match"]
            co = score_character(ro["case"], ro["result"])["exact_match"]
            if cb and not co:
                char_base_wins += 1
            elif co and not cb:
                char_other_wins += 1
            ab = cb and all(score_single_axis(rb["case"], rb["result"], a)[2] for a in SINGLE_AXES)
            ao = co and all(score_single_axis(ro["case"], ro["result"], a)[2] for a in SINGLE_AXES)
            if ab and not ao:
                all_base_wins += 1
            elif ao and not ab:
                all_other_wins += 1
        lines.append(f"({base} vs {other}、N={len(common_ids)})")
        lines.append("")
        lines.append(f"- キャラ完全一致: {base}のみ正解 {char_base_wins} 件 / {other}のみ正解 {char_other_wins} 件")
        lines.append(f"- 全軸正解: {base}のみ正解 {all_base_wins} 件 / {other}のみ正解 {all_other_wins} 件")
        lines.append("")

    lines.append("## 注記")
    lines.append("")
    lines.append("N が小さいため点推定を強い結論として扱わないこと。ケース別CSV(`cases.csv`)が一次資料。")
    lines.append("")

    return "\n".join(lines)
