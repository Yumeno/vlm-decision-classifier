"""CLI: probe / classify。"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

from PIL import Image

from . import evaluate, pipeline, server, systemone
from .backend import ChatBackend
from .decision import Choice, DecisionError, choose
from .taxonomy import load as load_taxonomy


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _make_probe_image() -> bytes:
    img = Image.new("RGB", (64, 64), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def cmd_probe(args: argparse.Namespace) -> int:
    backend = ChatBackend(base_url=args.base_url, model=args.model)

    try:
        models_response = backend.list_models()
        model_ids = [m.get("id") for m in models_response.get("data", [])]
        listed = args.model in model_ids
        print(f"GET /models: ok, model '{args.model}' listed = {listed}")
    except Exception as e:
        print(f"GET /models: failed ({e})")
        return 1

    image_bytes = _make_probe_image()
    choices = [
        Choice(id="red", name="red", criteria="the image is mostly red"),
        Choice(id="blue", name="blue", criteria="the image is mostly blue"),
        Choice(id="green", name="green", criteria="the image is mostly green"),
    ]

    try:
        result = choose(
            backend,
            image_bytes,
            "image/png",
            "What is the main color of this image?",
            choices,
            allow_none=False,
        )
    except DecisionError as e:
        thinking = e.error_type == "thinking_before_answer"
        print(f"choose: failed ({e.error_type}: {e.detail})")
        print(f"logprobs present: {'no' if e.error_type != 'no_label_tokens' else 'yes'}")
        print(f"thinking detected: {'yes' if thinking else 'no'}")
        print(f"reasoning_effort dropped: {backend.dropped_reasoning_effort}")
        return 1
    except Exception as e:
        print(f"choose: failed ({e})")
        return 1

    print(f"answer: {result['selected']}")
    print(f"relative_scores: {result['relative_scores']}")
    print(f"elapsed_ms: {result['elapsed_ms']:.1f}")
    print("logprobs present: yes")
    print("thinking detected: no")
    print(f"reasoning_effort dropped: {backend.dropped_reasoning_effort}")
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    taxonomy = load_taxonomy(args.taxonomy)
    backend = ChatBackend(base_url=args.base_url, model=args.model)
    result = pipeline.classify(
        args.image,
        taxonomy,
        backend,
        mode=args.mode,
        max_edge=args.max_edge,
        image_format=args.image_format,
        confirm=args.confirm,
        rank_threshold=args.rank_threshold,
        bundled_multi=args.bundled_multi,
    )

    print(f"image: {result['image']['name']}  mode: {result['mode']}")
    for axis_id, tags in result["vision_tags"].items():
        print(f"  {axis_id}: {tags}")
    if result["errors"]:
        print(f"errors: {result['errors']}")
    print(f"classification_wall_ms: {result['timing_ms']['classification_wall_ms']:.1f}")
    print(f"request_count: {result['request_count']}")

    output_json = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"written: {args.output}")
    else:
        print(output_json)
    return 0


def cmd_check_manifest(args: argparse.Namespace) -> int:
    taxonomy = load_taxonomy(args.taxonomy)
    result = evaluate.check_manifest(args.manifest, taxonomy)
    print(f"cases: {len(result.cases)}")
    for scenario in sorted(result.scenario_counts):
        print(f"  scenario {scenario}: {result.scenario_counts[scenario]}")
    print(f"source images: {result.source_count}  derived: {result.derived_count}")
    print(f"rights_confirmed=false: {result.rights_false_count}")
    if result.errors:
        print(f"errors: {len(result.errors)}")
        for e in result.errors:
            print(f"  - {e}")
        return 1
    print("OK")
    return 0


def _modes_type(value: str) -> list[str]:
    """--modes の argparse type。'choice,json' のようなCSVを検証済みのリストへ変換する。"""
    modes = [m.strip() for m in value.split(",") if m.strip()]
    try:
        evaluate.validate_modes(modes)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    return modes


def cmd_evaluate(args: argparse.Namespace) -> int:
    backend = ChatBackend(base_url=args.base_url, model=args.model)
    return evaluate.run_evaluate(
        manifest_path=args.manifest,
        taxonomy_path=args.taxonomy,
        backend=backend,
        modes=args.modes,
        max_edge=args.max_edge,
        image_format=args.image_format,
        warmup=args.warmup,
        runtime_label=args.runtime_label,
        note=args.note,
        output_dir=args.output_dir,
        runtime_info_path=args.runtime_info,
        dataset_version=args.dataset_version,
        prime=args.prime,
        axis_concurrency=args.axis_concurrency,
        confirm=args.confirm,
        rank_threshold=args.rank_threshold,
        bundled_multi=args.bundled_multi,
    )


def cmd_serve(args: argparse.Namespace) -> int:
    taxonomy = load_taxonomy(args.taxonomy)
    server.serve(args.port, taxonomy)
    return 0


def cmd_systemone(args: argparse.Namespace) -> int:
    import dataclasses

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    if args.state_file:
        with open(args.state_file, encoding="utf-8") as f:
            state = f.read()
    else:
        state = args.state or ""
    client = systemone.SystemOneClient(
        base_url=args.base_url, model=args.model, max_edge=args.max_edge, image_format=args.image_format
    )
    try:
        response = client.system_one(state, questions, images=args.image, prime=args.prime)
    except systemone.SystemOneError as e:
        print(f"SystemOneError: {e}", file=sys.stderr)
        return 1
    print(json.dumps(dataclasses.asdict(response), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="classifier_demo")
    sub = parser.add_subparsers(dest="command", required=True)

    probe_parser = sub.add_parser(
        "probe", help="check server connectivity, vision and logprobs support"
    )
    probe_parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    probe_parser.add_argument("--model", required=True)
    probe_parser.set_defaults(func=cmd_probe)

    classify_parser = sub.add_parser("classify", help="classify one image")
    classify_parser.add_argument("image")
    classify_parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    classify_parser.add_argument("--model", required=True)
    classify_parser.add_argument("--mode", choices=["choice", "json", "bundled", "json_schema"], default="choice")
    classify_parser.add_argument("--taxonomy", default="taxonomy/default.yaml")
    classify_parser.add_argument("--max-edge", type=int, default=1024)
    classify_parser.add_argument(
        "--image-format",
        choices=["jpeg", "png"],
        default="jpeg",
        help="モデルへ送る画像の形式(既定jpeg、quality 90)。E1〜E9の再現には png を指定する",
    )
    classify_parser.add_argument(
        "--confirm",
        action="store_true",
        help="複数選択軸(character・outfit等)で上位候補ごとにyes/noを確認する(既定オフ)。"
        "指定するとE1〜E4等これまでの実験と同じ判定になる",
    )
    classify_parser.add_argument(
        "--rank-threshold",
        type=float,
        default=0.5,
        help="--confirm を指定しないとき、複数選択軸で採用する相対スコアの閾値(既定0.5)",
    )
    classify_parser.add_argument(
        "--bundled-multi",
        choices=["rank", "yn"],
        default="rank",
        help="束ね質問(bundled)での複数選択軸の扱い。rank=相対スコアと閾値(既定)、yn=候補ごとのYes/No欄(E7e)",
    )
    classify_parser.add_argument("--output")
    classify_parser.set_defaults(func=cmd_classify)

    check_manifest_parser = sub.add_parser(
        "check-manifest", help="validate dataset/manifest.jsonl (run from repo root)"
    )
    check_manifest_parser.add_argument("--manifest", default="dataset/manifest.jsonl")
    check_manifest_parser.add_argument("--taxonomy", default="taxonomy/default.yaml")
    check_manifest_parser.set_defaults(func=cmd_check_manifest)

    evaluate_parser = sub.add_parser(
        "evaluate", help="run check-manifest then classify every evaluated case (run from repo root)"
    )
    evaluate_parser.add_argument("--manifest", default="dataset/manifest.jsonl")
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    evaluate_parser.add_argument(
        "--modes",
        default="choice,json",
        type=_modes_type,
        help="comma-separated choice/json/bundled/json_schema, no duplicates (default: choice,json). 3モード指定時はケースごとに実行順を回転する",
    )
    evaluate_parser.add_argument("--taxonomy", default="taxonomy/default.yaml")
    evaluate_parser.add_argument("--max-edge", type=int, default=1024)
    evaluate_parser.add_argument(
        "--image-format",
        choices=["jpeg", "png"],
        default="jpeg",
        help="モデルへ送る画像の形式(既定jpeg、quality 90)。E1〜E9の再現には png を指定する",
    )
    evaluate_parser.add_argument("--warmup", type=int, default=1)
    evaluate_parser.add_argument("--runtime-label", default=None)
    evaluate_parser.add_argument(
        "--runtime-info",
        default=None,
        help="JSON file describing the run environment (model/mmproj SHA256, server commit, "
        "patch, launch args, GPU offload, etc.); stored as-is in run.json",
    )
    evaluate_parser.add_argument(
        "--dataset-version",
        default=None,
        help="dataset version tag (e.g. dataset-v1.0.0) to cross-check against DATASET_CARD.md; "
        "stored as-is in run.json",
    )
    evaluate_parser.add_argument("--note", default=None)
    evaluate_parser.add_argument("--output-dir", required=True)
    evaluate_parser.add_argument(
        "--prime",
        action="store_true",
        help="E9: 各ケースの判定前に画像だけの準備リクエストを1回送り(ホットロード)、"
        "prime_ms を別記録する(classification_wall_msには含めない)",
    )
    evaluate_parser.add_argument(
        "--axis-concurrency",
        type=int,
        default=1,
        help="E9: choiceモードで軸ごとの質問を同時に送る数(既定1=逐次)",
    )
    evaluate_parser.add_argument(
        "--confirm",
        action="store_true",
        help="複数選択軸(character・outfit等)で上位候補ごとにyes/noを確認する(既定オフ)。"
        "指定するとE1〜E4等これまでの実験と同じ判定になる(再現には本フラグが必要)",
    )
    evaluate_parser.add_argument(
        "--rank-threshold",
        type=float,
        default=0.5,
        help="--confirm を指定しないとき、複数選択軸で採用する相対スコアの閾値(既定0.5)",
    )
    evaluate_parser.add_argument(
        "--bundled-multi",
        choices=["rank", "yn"],
        default="rank",
        help="束ね質問(bundled)での複数選択軸の扱い。rank=相対スコアと閾値(既定)、yn=候補ごとのYes/No欄(E7e)",
    )
    evaluate_parser.set_defaults(func=cmd_evaluate)

    serve_parser = sub.add_parser(
        "serve", help="run the recording demo UI server (loopback only, no auto model load)"
    )
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--taxonomy", default="taxonomy/default.yaml")
    serve_parser.set_defaults(func=cmd_serve)

    so_parser = sub.add_parser(
        "systemone", help="Jev-style questions (choice/noul/score) against a local VLM (experimental)"
    )
    so_parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    so_parser.add_argument("--model", required=True)
    so_parser.add_argument("--questions", required=True, help="JSON file: {question_id: {type, instructions, criteria}}")
    so_parser.add_argument("--state", help="state text")
    so_parser.add_argument("--state-file", help="read state text from a file")
    so_parser.add_argument("--image", action="append", default=[], help="image path or data URL (repeatable)")
    so_parser.add_argument("--max-edge", type=int, default=1024)
    so_parser.add_argument("--image-format", choices=["jpeg", "png"], default="jpeg")
    so_parser.add_argument("--prime", action="store_true", help="send one prefix-only request first")
    so_parser.set_defaults(func=cmd_systemone)

    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
