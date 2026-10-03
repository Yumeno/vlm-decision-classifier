"""CLI: probe / classify。"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

from PIL import Image

from . import dgemma_server, evaluate, pipeline, server, systemone
from .backend import ChatBackend
from .decision import Choice, DecisionError, choose
from .dgemma import DgemmaBackend
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


def _int_min(minimum: int):
    def parse(value: str) -> int:
        try:
            n = int(value)
        except ValueError:
            n = minimum - 1
        if n < minimum:
            raise argparse.ArgumentTypeError(f"must be an integer >= {minimum}")
        return n

    return parse


def _add_dgemma_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dgemma-url",
        default=None,
        help="DiffusionGemma(issue #4): 自前の dgemma-server のURL(例 http://127.0.0.1:8012。/v1/systemone を使う)。"
        "dgemma_choice で必須。--base-url は vLLM 本体(例 http://127.0.0.1:8000/v1)",
    )
    p.add_argument("--dgemma-samples", type=_int_min(1), default=1, help="dgemma_choice のノイズ draw の回数(整数、既定1)")
    p.add_argument("--dgemma-seed", type=int, default=0, help="dgemma_choice のノイズの seed(既定0)")
    p.add_argument(
        "--dgemma-template", choices=["keyed", "numbered"], default="keyed",
        help="回答テンプレート。keyed=`<質問id>: <ラベル>`(既定)、numbered=`Q<n>: <ラベル>`",
    )
    p.add_argument(
        "--dgemma-instruction", choices=["default", "strict", "strict_sys"], default="default",
        help="プロンプト文面。strict=system と各質問末尾の指示、strict_sys=system のみ強い文面(テンプレートは同じ)。default 以外のときだけ送る",
    )
    p.add_argument(
        "--dgemma-yn-style", choices=["slash", "lines", "letters", "yn"], default="slash",
        help="複数選択の yes/no 質問の描き方。slash=`yes / no`(既定)、lines=`yes: present...`/`no: not present...`、letters=A/B、yn=`Y: yes`/`N: no`(質問文はそのまま)。slash 以外のときだけ送る",
    )
    p.add_argument(
        "--dgemma-catchall-style", choices=["default", "list"], default="default",
        help="「その他」系(catch_all)の選択肢の質問文。list=同じ軸の他の選択肢を挙げて「それらを除いて他にあるか」と聞く(クライアント側。taxonomy は変えない)",
    )
    p.add_argument(
        "--dgemma-steps", type=_int_min(1), default=1,
        help="デノイズのステップ数(既定1=1ステップ読み)。2以上ではスロット以外の位置を固定して複数ステップ回し、最終ステップの分布を読む。2以上のときだけ送る",
    )
    p.add_argument(
        "--dgemma-adaptive-threshold", type=float, default=None,
        help="適応的な再読み出し(issue #12)。1回目で正規化エントロピーがこの値以上の質問がある(または失敗した)ときだけ追加で読む。指定時のみ送る(samples は1のまま)",
    )
    p.add_argument(
        "--dgemma-adaptive-max", type=_int_min(2), default=3,
        help="適応的再読み出しの総読み出し回数の上限(既定3)",
    )
    p.add_argument(
        "--dgemma-order", choices=["taxonomy", "character_first"], default="taxonomy",
        help="質問の並び。character_first=character 軸の質問を最初の複数選択軸の前へ(クライアント側)",
    )
    p.add_argument(
        "--dgemma-max-per-read", type=_int_min(0), default=0,
        help="1回の読み出しに入れる質問数の上限(診断用。0=全質問を1回で)",
    )
    p.add_argument(
        "--dgemma-max-soft-tokens", type=_int_min(1), default=None,
        help="画像トークン予算(vLLM の mm_processor_kwargs.max_soft_tokens。例 70/140/280)。未指定なら送らない",
    )


def _make_backend(args: argparse.Namespace, modes: list[str]) -> ChatBackend | None:
    """dgemma_* のモードを含むときは DgemmaBackend(thinking 無効を全リクエストに足す)、
    それ以外は従来の ChatBackend。dgemma_choice に --dgemma-url が無ければ None。"""
    if any(m.startswith("dgemma_") for m in modes):
        if "dgemma_choice" in modes and not args.dgemma_url:
            print("--dgemma-url is required for dgemma_choice", file=sys.stderr)
            return None
        return DgemmaBackend(
            base_url=args.base_url,
            model=args.model,
            structured_url=args.dgemma_url,
            samples=args.dgemma_samples,
            seed=args.dgemma_seed,
            template=args.dgemma_template,
            instruction=args.dgemma_instruction,
            yn_style=args.dgemma_yn_style,
            order=args.dgemma_order,
            adaptive_threshold=args.dgemma_adaptive_threshold,
            adaptive_max=args.dgemma_adaptive_max,
            steps=args.dgemma_steps,
            catchall_style=args.dgemma_catchall_style,
            max_per_read=args.dgemma_max_per_read,
            max_soft_tokens=args.dgemma_max_soft_tokens,
        )
    return ChatBackend(base_url=args.base_url, model=args.model)


def cmd_classify(args: argparse.Namespace) -> int:
    taxonomy = load_taxonomy(args.taxonomy)
    backend = _make_backend(args, [args.mode])
    if backend is None:
        return 1
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
    backend = _make_backend(args, args.modes)
    if backend is None:
        return 1
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
    classify_parser.add_argument("--mode", choices=sorted(evaluate.VALID_MODES), default="choice")
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
    _add_dgemma_args(classify_parser)
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
        help="comma-separated choice/json/bundled/json_schema/dgemma_choice/dgemma_json, no duplicates (default: choice,json). 3モード指定時はケースごとに実行順を回転する",
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
    _add_dgemma_args(evaluate_parser)
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

    ds_parser = sub.add_parser(
        "dgemma-server", help="DiffusionGemma(vLLM)用の判定サーバー。POST /v1/systemone(issue #4)"
    )
    dgemma_server.add_arguments(ds_parser)
    ds_parser.set_defaults(func=dgemma_server.serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
