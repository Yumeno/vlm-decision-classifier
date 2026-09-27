"""Phase 4 / E5用の小さなベンチマークスクリプト。

llama.cppの画像キャッシュ改造(doc/patches/llamacpp-mtmd-checkpoint.patch)の有無を、
未改造版/改造版のllama-serverを別々に起動した状態でこのスクリプトをそれぞれ実行して比較する
(サーバーの起動・切り替えはユーザーが行う。本スクリプトはサーバーを起動・終了しない)。

計測する2つの小テスト(doc/implementation-experiment-plan.md Phase 4 手順3):

- テスト1(同じ画像・質問だけ変更): 各繰り返しの先頭で画像Bの1軸目への順位付けリクエストを1回送り
  (直前の繰り返しの最終リクエストが画像Aのままキャッシュに残っている状態を崩し、
  次の画像Aへの1軸目リクエストが必ず再エンコードを要する状態にする)、続けて画像Aについて
  taxonomyの全軸を1軸ずつ逐次送る(選択式パイプラインが軸ごとに送る最初の順位付けリクエストと
  同じもの)。これをrepeats回繰り返し、各繰り返しの1軸目(再エンコードが必要)と2軸目以降
  (画像は同じでキャッシュが効くはずの箇所)の経過時間を比べる。画像Bへのリセットリクエストは
  `reset`として別に記録し、経過時間の要約(中央値)には含めない。リセット自体が失敗した場合、
  キャッシュが画像Bに切り替わった保証がないため、その繰り返しの画像Aの記録には
  `reset_failed: true`を付けて記録は残しつつ(送信自体は止めない)、要約の件数・中央値からは除外し、
  除外した繰り返し数を`excluded_repeats`に記録する。
- テスト2(画像切替 A→B→A): 各軸についてA→B→Aの順で同じ順位付けリクエストを送り、
  1回目のAと2回目のAで最上位ラベル・相対スコアが変わっていないか(画像混線がないか)を確認する。
  途中のBリクエストが失敗した場合、画像が実際にBへ切り替わった保証がないため、その組には
  `b_failed: true`を付け、`top_label_match`・`max_abs_diff`はNone(比較無効)として要約から除外し、
  無効にした組数を`invalid_groups`に記録する。

プロンプト文・リクエストパラメータ・画像前処理はclassifier_demoの既存関数
(decision.choose / image.prepare_image / taxonomy.load)をそのまま使う。
yes/no確認リクエスト(decide_multi_axisの候補確認)は送らない。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone

from classifier_demo import decision
from classifier_demo.backend import ChatBackend
from classifier_demo.evaluate import _git_dirty
from classifier_demo.image import file_sha256, prepare_image
from classifier_demo.pipeline import _git_commit
from classifier_demo.taxonomy import Axis, Taxonomy, load as load_taxonomy


def _send_ranking(backend, image_bytes: bytes, mime: str, axis: Axis) -> dict:
    """axisの最初の順位付けリクエスト(decision.choose)を1回送り、結果を記録用に正規化する。
    失敗しても例外を投げず、エラーとして記録する。"""
    try:
        result = decision.choose(
            backend, image_bytes, mime, axis.question, axis.choices, axis.allow_none, axis.none_criteria
        )
    except decision.DecisionError as e:
        return {
            "elapsed_ms": None,
            "top_label": None,
            "relative_scores": None,
            "error": f"{e.error_type}: {e.detail}",
        }
    except decision.REQUEST_EXCEPTIONS as e:
        return {
            "elapsed_ms": None,
            "top_label": None,
            "relative_scores": None,
            "error": f"{type(e).__name__}: {e}",
        }
    return {
        "elapsed_ms": result["elapsed_ms"],
        "top_label": result["selected"],
        "relative_scores": result["relative_scores"],
        "error": None,
    }


def _max_abs_diff(a: dict | None, b: dict | None) -> float | None:
    if a is None or b is None:
        return None
    keys = set(a) | set(b)
    if not keys:
        return None
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def run_test1(
    backend,
    image_a_bytes: bytes,
    mime_a: str,
    image_b_bytes: bytes,
    mime_b: str,
    taxonomy: Taxonomy,
    repeats: int,
) -> dict:
    """テスト1: 各繰り返しの先頭で画像Bの1軸目への順位付けリクエストを1回送って
    キャッシュ上の画像をBに切り替えてから(`reset`として記録)、画像Aについてtaxonomyの
    全軸を1軸ずつ逐次送る。repeats回繰り返す。`reset`は要約(中央値)の計算に含めない。

    reset自体が失敗した場合はキャッシュが画像Bへ切り替わった保証がないため、Aのリクエストは
    送る(止めない)が、その繰り返しの全記録に`reset_failed: true`を付けて要約の件数・中央値から
    除外し、除外した繰り返し数を`excluded_repeats`に数える。"""
    first_axis = taxonomy.axes[0]
    requests: list[dict] = []
    resets: list[dict] = []
    excluded_repeats = 0
    for r in range(1, repeats + 1):
        reset_record = _send_ranking(backend, image_b_bytes, mime_b, first_axis)
        reset_record.update({"repeat": r, "axis": first_axis.id})
        resets.append(reset_record)

        reset_failed = reset_record["error"] is not None
        if reset_failed:
            excluded_repeats += 1

        for axis_index, axis in enumerate(taxonomy.axes, start=1):
            record = _send_ranking(backend, image_a_bytes, mime_a, axis)
            record.update(
                {"repeat": r, "axis_index": axis_index, "axis": axis.id, "reset_failed": reset_failed}
            )
            requests.append(record)

    eligible = [req for req in requests if not req["reset_failed"]]
    first_axis_times = [req["elapsed_ms"] for req in eligible if req["axis_index"] == 1 and req["error"] is None]
    later_axes_times = [req["elapsed_ms"] for req in eligible if req["axis_index"] > 1 and req["error"] is None]
    failures = sum(1 for req in requests if req["error"] is not None)

    summary = {
        "first_axis_elapsed_ms_median": _median(first_axis_times),
        "first_axis_count": len(first_axis_times),
        "later_axes_elapsed_ms_median": _median(later_axes_times),
        "later_axes_count": len(later_axes_times),
        "excluded_repeats": excluded_repeats,
        "failures": failures,
    }
    return {"requests": requests, "resets": resets, "summary": summary}


def run_test2(
    backend,
    image_a_bytes: bytes,
    mime_a: str,
    image_b_bytes: bytes,
    mime_b: str,
    taxonomy: Taxonomy,
    repeats: int,
) -> dict:
    """テスト2: 各軸についてA→B→Aの順で送る。repeats回繰り返す。

    Bリクエストが失敗した場合、画像が実際にBへ切り替わった保証がないため、その組は
    `b_failed: true`を付けて`top_label_match`・`max_abs_diff`をNone(比較無効)にし、
    要約の`label_mismatch_count`・`max_abs_diff_overall`から除外する。無効にした組数は
    `invalid_groups`に数える。"""
    groups: list[dict] = []
    for r in range(1, repeats + 1):
        for axis in taxonomy.axes:
            a1 = _send_ranking(backend, image_a_bytes, mime_a, axis)
            b = _send_ranking(backend, image_b_bytes, mime_b, axis)
            a2 = _send_ranking(backend, image_a_bytes, mime_a, axis)

            b_failed = b["error"] is not None
            if b_failed:
                top_label_match = None
                max_abs_diff = None
            else:
                if a1["error"] is None and a2["error"] is None:
                    top_label_match = a1["top_label"] == a2["top_label"]
                else:
                    top_label_match = None
                max_abs_diff = _max_abs_diff(a1["relative_scores"], a2["relative_scores"])

            groups.append(
                {
                    "repeat": r,
                    "axis": axis.id,
                    "requests": {"a1": a1, "b": b, "a2": a2},
                    "b_failed": b_failed,
                    "top_label_match": top_label_match,
                    "max_abs_diff": max_abs_diff,
                }
            )

    invalid_groups = sum(1 for g in groups if g["b_failed"])
    label_mismatch_count = sum(1 for g in groups if g["top_label_match"] is False)
    diffs = [g["max_abs_diff"] for g in groups if g["max_abs_diff"] is not None]
    failures = sum(
        1 for g in groups for req in g["requests"].values() if req["error"] is not None
    )

    summary = {
        "label_mismatch_count": label_mismatch_count,
        "max_abs_diff_overall": max(diffs) if diffs else None,
        "invalid_groups": invalid_groups,
        "failures": failures,
    }
    return {"groups": groups, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_cache",
        description="Phase 4 / E5: llama.cpp画像キャッシュ改造の再現用の小さなベンチマーク",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--images",
        nargs=2,
        default=["dataset/images/M01.png", "dataset/images/G05.png"],
        metavar=("IMAGE_A", "IMAGE_B"),
        help="1枚目がA、2枚目がB(既定: dataset/images/M01.png dataset/images/G05.png)",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--taxonomy", default="taxonomy/default.yaml")
    parser.add_argument("--max-edge", type=int, default=1024)
    parser.add_argument("--label", default=None, help="この実行を識別する任意のラベル(例: vanilla/patched)")
    parser.add_argument("--output", required=True, help="出力JSONのパス(親フォルダがなければ作る)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    started = datetime.now(timezone.utc).isoformat()

    taxonomy = load_taxonomy(args.taxonomy)
    backend = ChatBackend(base_url=args.base_url, model=args.model)

    image_a_path, image_b_path = args.images
    image_a_bytes, mime_a, _, _ = prepare_image(image_a_path, max_edge=args.max_edge)
    image_b_bytes, mime_b, _, _ = prepare_image(image_b_path, max_edge=args.max_edge)

    warmup_axis = taxonomy.axes[0]
    warmup_record = _send_ranking(backend, image_b_bytes, mime_b, warmup_axis)
    warmup_record["axis"] = warmup_axis.id
    warmup_record["image"] = "b"

    test1 = run_test1(backend, image_a_bytes, mime_a, image_b_bytes, mime_b, taxonomy, args.repeats)
    test2 = run_test2(backend, image_a_bytes, mime_a, image_b_bytes, mime_b, taxonomy, args.repeats)

    finished = datetime.now(timezone.utc).isoformat()

    output = {
        "started": started,
        "finished": finished,
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "images": {
            "a": {"path": image_a_path, "sha256": file_sha256(image_a_path)},
            "b": {"path": image_b_path, "sha256": file_sha256(image_b_path)},
        },
        "taxonomy": {"path": args.taxonomy, "version": taxonomy.version, "sha256": taxonomy.sha256},
        "max_edge": args.max_edge,
        "repeats": args.repeats,
        "tool_commit": {"commit": _git_commit(), "dirty": _git_dirty()},
        "warmup": warmup_record,
        "test1_same_image_change_question": test1,
        "test2_image_switch": test2,
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    t1s = test1["summary"]
    t2s = test2["summary"]
    print(f"written: {args.output}")
    print(
        "test1: first-axis median="
        f"{t1s['first_axis_elapsed_ms_median']}ms (n={t1s['first_axis_count']}), "
        "later-axes median="
        f"{t1s['later_axes_elapsed_ms_median']}ms (n={t1s['later_axes_count']}), "
        f"excluded_repeats={t1s['excluded_repeats']}, failures={t1s['failures']}"
    )
    print(
        "test2: label_mismatch_count="
        f"{t2s['label_mismatch_count']}, max_abs_diff_overall={t2s['max_abs_diff_overall']}, "
        f"invalid_groups={t2s['invalid_groups']}, failures={t2s['failures']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
