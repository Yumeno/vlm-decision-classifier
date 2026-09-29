"""E10: 出力形式の頑健性ストレステスト。

温度0では同じ画像の出力は決定的なので、形式不正の確率は「多数の異なる画像」で見る。
ここでは白色ノイズ画像(seed 0..N-1 から決定的に生成)を使う。正解ラベルは不要で、
「出力形式が有効か」と時間だけを方式ごとに記録する(実画像の失敗率ではなく、方式間の頑健性の比較)。

方式(--modes): choice / json / json_schema / bundled / bundled_yn(bundled + bundled_multi=yn)。
準備(prime)は方式ごとに自分の先頭で送り(evaluate と同じ)、classification_wall_ms には含めない。
実行ロジックは classifier_demo.evaluate の関数を再利用する。サーバーの起動・終了はしない。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone

from classifier_demo.backend import ChatBackend
from classifier_demo.evaluate import (
    _classify_safe,
    _git_dirty,
    _run_prime,
    _run_warmup,
    case_mode_order,
    sha256_file,
)
from classifier_demo.image import JPEG_QUALITY
from classifier_demo.pipeline import _git_commit
from classifier_demo.taxonomy import load as load_taxonomy

SIZE = 1024
MAX_EDGE = 1024
RANK_THRESHOLD = 0.5
WARMUP_SEED = 10**6  # 計測対象(0..N-1)と重ならないウォームアップ用画像
# 方式名 -> (pipeline のモード, bundled_multi)
MODE_TABLE = {
    "choice": ("choice", "rank"),
    "json": ("json", "rank"),
    "json_schema": ("json_schema", "rank"),
    "bundled": ("bundled", "rank"),
    "bundled_yn": ("bundled", "yn"),
}
CSV_FIELDS = [
    "image_seed", "mode", "order_index", "ok", "error_types", "error_detail",
    "classification_wall_ms", "prime_ms", "prime_error", "request_count", "json_attempts", "completion_tokens",
]


def make_noise_image(seed: int, path: str) -> None:
    """seed から決定的な 1024x1024 RGB 一様ノイズを PNG で保存する(Pillow と random のみ)。"""
    from PIL import Image

    raw = random.Random(seed).randbytes(SIZE * SIZE * 3)
    Image.frombytes("RGB", (SIZE, SIZE), raw).save(path, format="PNG")


def build_row(seed: int, mode: str, order_index: int, result: dict) -> dict:
    errors = result.get("errors") or []
    jb = result.get("json_baseline") or {}
    return {
        "image_seed": seed,
        "mode": mode,
        "order_index": order_index,
        "ok": not errors,
        "error_types": ";".join(dict.fromkeys(e.get("type", "") for e in errors)),
        "error_detail": " | ".join(str(e.get("detail", ""))[:200] for e in errors),
        "classification_wall_ms": (result.get("timing_ms") or {}).get("classification_wall_ms", ""),
        "prime_ms": result.get("prime_ms", ""),
        "prime_error": result.get("prime_error") or "",
        "request_count": result.get("request_count", ""),
        "json_attempts": len(jb.get("attempts") or []) if jb else "",
        "completion_tokens": (jb.get("usage") or {}).get("completion_tokens", ""),
    }


def _mean(rows: list[dict], key: str) -> str:
    vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
    return f"{statistics.mean(vals):.1f}" if vals else "-"


def build_summary(rows: list[dict], modes: list[str], n: int) -> str:
    lines = [
        "# E10 出力形式ストレステスト",
        "",
        f"白色ノイズ画像 N={n}(実画像の失敗率ではなく、方式間の頑健性の比較)。失敗 = 分類結果の errors が1件以上。",
        "",
        "| 方式 | 失敗/N | 失敗種別 | 平均判定時間(ms) | 平均prime(ms) | 平均リクエスト数 | 平均JSON試行数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in modes:
        rs = [r for r in rows if r["mode"] == m]
        fails = [r for r in rs if not r["ok"]]
        types = Counter(t for r in fails for t in r["error_types"].split(";") if t)
        types_str = ", ".join(f"{k}={v}" for k, v in types.most_common()) or "-"
        lines.append(
            f"| {m} | {len(fails)}/{len(rs)} | {types_str} | {_mean(rs, 'classification_wall_ms')} | "
            f"{_mean(rs, 'prime_ms')} | {_mean(rs, 'request_count')} | {_mean(rs, 'json_attempts')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="E10 format robustness stress test (white-noise images)")
    p.add_argument("--base-url", default="http://127.0.0.1:1235/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--modes", default="choice,json,json_schema,bundled,bundled_yn")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--base-seed", type=int, default=0, help="画像seedは base-seed .. base-seed+N-1")
    p.add_argument("--taxonomy", default="taxonomy/default.yaml")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--prime", action="store_true", help="方式ごとに自分の先頭の準備リクエストを送る(evaluate と同じ)")
    p.add_argument("--runtime-info", default=None)
    p.add_argument("--runtime-label", default=None)
    p.add_argument("--note", default=None)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODE_TABLE] + (["<duplicate>"] if len(modes) != len(set(modes)) else [])
    if not modes or bad:
        print(f"invalid --modes: {bad} (choose from {sorted(MODE_TABLE)})", file=sys.stderr)
        return 1

    runtime_info = runtime_info_file = None
    if args.runtime_info:
        try:
            with open(args.runtime_info, "r", encoding="utf-8") as f:
                runtime_info = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"runtime-info error: {e}", file=sys.stderr)
            return 1
        runtime_info_file = {"name": os.path.basename(args.runtime_info), "sha256": sha256_file(args.runtime_info)}

    taxonomy = load_taxonomy(args.taxonomy)
    backend = ChatBackend(base_url=args.base_url, model=args.model)
    img_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    seeds = list(range(args.base_seed, args.base_seed + args.n))
    paths = {}
    for s in seeds + [WARMUP_SEED]:
        paths[s] = os.path.join(img_dir, f"noise_{s}.png")
        make_noise_image(s, paths[s])

    started = datetime.now(timezone.utc).isoformat()
    warmup_info = _run_warmup(
        [{"case_id": f"noise_{WARMUP_SEED}", "image_path": paths[WARMUP_SEED]}],
        taxonomy, backend, args.warmup, MAX_EDGE, "jpeg",
    )

    rows: list[dict] = []
    for i, seed in enumerate(seeds):
        case = {"case_id": f"noise_{seed}", "image_path": paths[seed]}
        for order_index, m in enumerate(case_mode_order(modes, i)):
            pipe_mode, bmulti = MODE_TABLE[m]
            prime_info = _run_prime(case, backend, MAX_EDGE, 1, "jpeg", pipe_mode) if args.prime else None
            result = _classify_safe(
                paths[seed], taxonomy, backend, pipe_mode, MAX_EDGE, 1, False, RANK_THRESHOLD, "jpeg", bmulti
            )
            if prime_info is not None:
                result["prime_ms"] = prime_info["elapsed_ms"]
                result["prime_error"] = prime_info["error"]
            rows.append(build_row(seed, m, order_index, result))

    run_info = {
        "experiment": "E10",
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "tool_commit": {"commit": _git_commit(), "dirty": _git_dirty()},
        "taxonomy": {"version": taxonomy.version, "sha256": taxonomy.sha256},
        "model": backend.model,
        "base_url": backend.base_url,
        "modes": modes,
        "n": args.n,
        "base_seed": args.base_seed,
        "seeds": [seeds[0], seeds[-1]] if seeds else [],
        "image": {"kind": "white_noise_uniform_rgb", "size": SIZE, "generator": "random.Random(seed).randbytes"},
        "max_edge": MAX_EDGE,
        "image_format": "jpeg",
        "jpeg_quality": JPEG_QUALITY,
        "temperature": 0,
        "confirm": False,
        "rank_threshold": RANK_THRESHOLD,
        "prime": args.prime,
        "prime_scope": "per_mode" if args.prime else None,
        "warmup": warmup_info,
        "runtime_label": args.runtime_label,
        "runtime_info": runtime_info,
        "runtime_info_file": runtime_info_file,
        "note": args.note,
        "platform": {"platform": platform.platform(), "python_version": sys.version},
    }
    with open(os.path.join(args.output_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "cases.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(build_summary(rows, modes, args.n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
