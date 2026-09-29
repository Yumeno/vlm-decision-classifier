"""最終の取り直し(F1〜F3)の集計。

`results/final/<F1|F2|F3>_e<長辺>_<a|b>_r<繰り返し>/` の run.json と cases.csv を読み、
条件・長辺・方式ごとに繰り返しのあいだの平均と標準偏差を `summary.md` にまとめる。
標準ライブラリのみ。summary.md のテキストは解析せず、cases.csv の列から計算する。

    python scripts/aggregate_final.py --input results/final --output doc/experiments/final
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

_DIR_RE = re.compile(r"^(F\d)_e(\d+)_([ab])_r(\d+)$")
COND_LABEL = {"F1": "F1", "F2": "F2", "F3": "F3"}
MAX_EXAMPLES = 5


def load_runs(input_dir: Path) -> list[dict]:
    """完了した実行(run.json と cases.csv が両方あるもの)だけを読む。"""
    runs = []
    for d in sorted(input_dir.iterdir()):
        m = _DIR_RE.match(d.name)
        if not m or not d.is_dir():
            continue
        run_json, cases_csv = d / "run.json", d / "cases.csv"
        if not (run_json.is_file() and cases_csv.is_file()):
            continue  # 実行中(未完了)
        meta = json.loads(run_json.read_text(encoding="utf-8"))
        with open(cases_csv, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        runs.append(
            {
                "name": d.name,
                "cond": m.group(1),
                "edge": int(m.group(2)),
                "kind": m.group(3),
                "rep": int(m.group(4)),
                "meta": meta,
                "rows": rows,
            }
        )
    return runs


def method_label(meta: dict, mode: str) -> str:
    if mode == "choice":
        return "choice(確認オン)" if meta.get("confirm") else "choice(確認オフ)"
    if mode == "bundled":
        return f"bundled({meta.get('bundled_multi', '?')})"
    return mode


def axes_of(rows: list[dict]) -> tuple[list[str], list[str]]:
    """cases.csv の列名から軸を取る。ok_<軸>=単一軸、exact_<軸>=複数選択軸。"""
    cols = list(rows[0].keys()) if rows else []
    singles = [c[3:] for c in cols if c.startswith("ok_")]
    multis = [c[6:] for c in cols if c.startswith("exact_")]
    return singles, multis


def is_true(v: str) -> bool:
    return str(v).strip().lower() == "true"


def is_source(row: dict) -> bool:
    return not row.get("derived_from")


def mean_sd(values: list[float]) -> tuple[float, float | None]:
    m = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) >= 2 else None
    return m, sd


def fmt_ms(values: list[float]) -> str:
    if not values:
        return "-"
    m, sd = mean_sd(values)
    sd_s = f"{sd:.0f}" if sd is not None else "n/a"
    return f"{m:.0f} ± {sd_s} ({len(values)})"


def fmt_pct(values: list[float]) -> str:
    if not values:
        return "-"
    m, sd = mean_sd(values)
    sd_s = f"{sd:.1f}" if sd is not None else "n/a"
    return f"{m:.1f} ± {sd_s} ({len(values)})"


def fmt_num(values: list[float], digits: int = 2) -> str:
    if not values:
        return "-"
    return f"{statistics.mean(values):.{digits}f}"


def _floats(rows: list[dict], col: str) -> list[float]:
    out = []
    for r in rows:
        v = r.get(col, "")
        if v not in ("", None):
            out.append(float(v))
    return out


def group_units(runs: list[dict]) -> dict[tuple, list[tuple[dict, list[dict]]]]:
    """(cond, edge, 方式) -> [(run, その方式の行)] 。繰り返し順。"""
    units: dict[tuple, list] = defaultdict(list)
    for run in runs:
        by_mode: dict[str, list[dict]] = defaultdict(list)
        for r in run["rows"]:
            by_mode[r["mode"]].append(r)
        for mode, rows in by_mode.items():
            key = (run["cond"], run["edge"], method_label(run["meta"], mode))
            units[key].append((run, rows))
    for v in units.values():
        v.sort(key=lambda t: t[0]["rep"])
    return units


def _sort_key(key: tuple) -> tuple:
    cond, edge, label = key
    return (cond, -edge, label)


def build_summary(runs: list[dict]) -> str:
    lines: list[str] = []
    if not runs:
        return "# 最終の取り直し(F1〜F3)の集計\n\n完了した実行がありません。\n"

    reps_all = sorted({r["rep"] for r in runs})
    singles, multis = axes_of(runs[0]["rows"])
    axes = singles + multis
    units = group_units(runs)

    lines.append("# 最終の取り直し(F1〜F3)の集計")
    lines.append("")
    lines.append(
        f"集計した実行: {len(runs)} 回分(繰り返し {', '.join(str(r) for r in reps_all)}。"
        f"計画は 3 回 × 12 = 36 回)。存在する繰り返しだけで平均し、表の括弧内は平均に使った回数。"
    )
    lines.append("")

    # 1. 条件
    lines.append("## 1. 条件")
    lines.append("")
    commits = {(r["meta"].get("tool_commit") or {}).get("commit") for r in runs}
    dirties = {(r["meta"].get("tool_commit") or {}).get("dirty") for r in runs}
    if len(commits) == 1 and len(dirties) == 1:
        lines.append(f"tool_commit: 全実行で同一 `{next(iter(commits))}`、dirty={next(iter(dirties))}。")
    else:
        lines.append("tool_commit: **実行間で不一致**。")
        for r in runs:
            tc = r["meta"].get("tool_commit") or {}
            lines.append(f"- {r['name']}: `{tc.get('commit')}` dirty={tc.get('dirty')}")
    lines.append("")
    lines.append("| 条件 | モデル | サーバー | 長辺 | 回 | 方式 | 繰り返し数 | 繰り返し |")
    lines.append("|---|---|---|---|---|---|---|---|")
    cond_keys = sorted({(r["cond"], r["edge"], r["kind"]) for r in runs}, key=lambda k: (k[0], -k[1], k[2]))
    for cond, edge, kind in cond_keys:
        rs = sorted([r for r in runs if (r["cond"], r["edge"], r["kind"]) == (cond, edge, kind)], key=lambda r: r["rep"])
        meta = rs[0]["meta"]
        server = (meta.get("runtime_info") or {}).get("server") or {}
        patched = server.get("image_cache_patch")
        server_s = "改造" if patched else "未改造" if patched is not None else "?"
        methods = ", ".join(method_label(meta, m) for m in meta.get("modes", []))
        lines.append(
            f"| {cond} | {meta.get('model')} | {server_s} | {edge} | {kind} | {methods} | {len(rs)} | "
            f"{', '.join('r' + str(r['rep']) for r in rs)} |"
        )
    lines.append("")

    # 2. 時間
    lines.append("## 2. 時間(元画像 N=31、派生の -strip を除く)")
    lines.append("")
    lines.append(
        "各繰り返しで元画像の値を平均し、その繰り返しごとの値の「平均 ± 標準偏差(回数)」を出す。単位は ms。"
    )
    lines.append("")
    lines.append("### 2.1 classification_wall_ms とリクエスト数")
    lines.append("")
    lines.append("| 条件 | 長辺 | 方式 | N(元画像) | classification_wall_ms | 平均リクエスト数 |")
    lines.append("|---|---|---|---|---|---|")
    for key in sorted(units, key=_sort_key):
        cond, edge, label = key
        wall, req, ns = [], [], set()
        for _, rows in units[key]:
            src = [r for r in rows if is_source(r)]
            ns.add(len(src))
            w = _floats(src, "classification_wall_ms")
            if w:
                wall.append(statistics.mean(w))
            q = _floats(src, "request_count")
            if q:
                req.append(statistics.mean(q))
        n_s = "/".join(str(n) for n in sorted(ns))
        lines.append(f"| {cond} | {edge} | {label} | {n_s} | {fmt_ms(wall)} | {fmt_num(req)} |")
    lines.append("")

    lines.append("### 2.2 prime_ms(条件 × 長辺、a・b 回をあわせた実行ごとの値)")
    lines.append("")
    lines.append("prime_ms は 1 回の実行内で方式によらず同じケースごとの値なので、実行ごとに 1 つの方式の行だけを使う。")
    lines.append("")
    lines.append("| 条件 | 長辺 | prime_ms | 実行数 |")
    lines.append("|---|---|---|---|")
    for cond, edge in sorted({(r["cond"], r["edge"]) for r in runs}, key=lambda k: (k[0], -k[1])):
        vals = []
        for r in runs:
            if (r["cond"], r["edge"]) != (cond, edge):
                continue
            first_mode = r["rows"][0]["mode"] if r["rows"] else None
            src = [x for x in r["rows"] if x["mode"] == first_mode and is_source(x)]
            p = _floats(src, "prime_ms")
            if p:
                vals.append(statistics.mean(p))
        lines.append(f"| {cond} | {edge} | {fmt_ms(vals)} | {len(vals)} |")
    lines.append("")

    # 3. 精度
    lines.append("## 3. 精度(元画像 N=31、正答率 %)")
    lines.append("")
    lines.append(
        "単一軸は正答率(ok_<軸>)、複数選択軸(" + "・".join(multis) + ")は集合の完全一致率(exact_<軸>)。"
        "失敗したケースは分母に残り不正解として数える(cases.csv の採点どおり)。各繰り返しの率の「平均 ± 標準偏差(回数)」。"
    )
    lines.append("")
    lines.append("| 条件 | 長辺 | 方式 | " + " | ".join(axes) + " |")
    lines.append("|---|---|---|" + "---|" * len(axes))
    for key in sorted(units, key=_sort_key):
        cond, edge, label = key
        cells = []
        for ax in axes:
            col = f"ok_{ax}" if ax in singles else f"exact_{ax}"
            per_rep = []
            for _, rows in units[key]:
                src = [r for r in rows if is_source(r)]
                if src:
                    per_rep.append(100 * sum(is_true(r[col]) for r in src) / len(src))
            cells.append(fmt_pct(per_rep))
        lines.append(f"| {cond} | {edge} | {label} | " + " | ".join(cells) + " |")
    lines.append("")

    # 4. 失敗
    lines.append("## 4. 失敗(全ケース、派生を含む)")
    lines.append("")
    lines.append("cases.csv の error_types 列(行ごとに種別を重複なく列挙)を種別ごとに数えた行数。")
    lines.append("")
    lines.append("| 条件 | 長辺 | 方式 | 合計 | 種別ごとの合計 | 回ごとの内訳 |")
    lines.append("|---|---|---|---|---|---|")
    for key in sorted(units, key=_sort_key):
        cond, edge, label = key
        total: Counter = Counter()
        per_rep_s = []
        for run, rows in units[key]:
            c: Counter = Counter()
            for r in rows:
                for t in filter(None, (r.get("error_types") or "").split(";")):
                    c[t] += 1
            total.update(c)
            per_rep_s.append(f"r{run['rep']}({run['kind']}): {sum(c.values())}")
        kinds = ", ".join(f"{t}={n}" for t, n in sorted(total.items())) or "なし"
        lines.append(
            f"| {cond} | {edge} | {label} | {sum(total.values())} | {kinds} | {'; '.join(per_rep_s)} |"
        )
    lines.append("")

    # 5. 揺れ
    lines.append("## 5. 揺れ(繰り返しのあいだで予測が変わったケース × 軸、元画像)")
    lines.append("")
    lines.append(
        "同じ条件・長辺・方式で、繰り返しのあいだに pred_<軸> が 1 つでも違った (ケース, 軸) の数。"
        f"例は最大 {MAX_EXAMPLES} 件(`ケース/軸: 各回の予測`)。繰り返しが 1 回だけの単位は比較できない(-)。"
    )
    lines.append("")
    lines.append("| 条件 | 長辺 | 方式 | 繰り返し数 | 揺れた (ケース, 軸) / 全 (ケース, 軸) | 例 |")
    lines.append("|---|---|---|---|---|---|")
    for key in sorted(units, key=_sort_key):
        cond, edge, label = key
        reps = units[key]
        if len(reps) < 2:
            lines.append(f"| {cond} | {edge} | {label} | {len(reps)} | - | - |")
            continue
        preds: dict[tuple, list[str]] = defaultdict(list)
        for run, rows in reps:
            for r in rows:
                if not is_source(r):
                    continue
                for ax in axes:
                    preds[(r["case_id"], ax)].append(r.get(f"pred_{ax}", ""))
        changed = [(k, v) for k, v in sorted(preds.items()) if len(set(v)) > 1]
        examples = "; ".join(f"{c}/{a}: {' | '.join(v)}" for (c, a), v in changed[:MAX_EXAMPLES]) or "なし"
        lines.append(f"| {cond} | {edge} | {label} | {len(reps)} | {len(changed)} / {len(preds)} | {examples} |")
    lines.append("")

    # 6. 注記
    lines.append("## 6. 注記")
    lines.append("")
    lines.append("- 元画像は 31 件と小標本。少数の差から強い結論を書かないこと。")
    lines.append("- 繰り返しは最大 3 回。標準偏差は回のあいだのばらつき(標本標準偏差、回数 1 のときは n/a)で、画像の選び方の不確かさは含まない。")
    lines.append("- スコアは列挙候補内で再正規化した相対スコアであり、正答確率ではない。")
    lines.append("- 一次資料は各実行フォルダの run.json・cases.csv。この表はそこから計算したもの。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default="results/final")
    ap.add_argument("--output", default="doc/experiments/final")
    args = ap.parse_args()
    runs = load_runs(Path(args.input))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(build_summary(runs), encoding="utf-8")
    print(f"{len(runs)} 回分を集計 -> {out / 'summary.md'}")


if __name__ == "__main__":
    main()
