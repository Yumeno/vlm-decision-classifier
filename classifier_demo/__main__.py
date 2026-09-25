"""CLI: probe / classify。"""

from __future__ import annotations

import argparse
import io
import json
import sys

from PIL import Image

from . import pipeline
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
        args.image, taxonomy, backend, mode=args.mode, max_edge=args.max_edge
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
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"written: {args.output}")
    else:
        print(output_json)
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
    classify_parser.add_argument("--mode", choices=["choice", "json"], default="choice")
    classify_parser.add_argument("--taxonomy", default="taxonomy/default.yaml")
    classify_parser.add_argument("--max-edge", type=int, default=1024)
    classify_parser.add_argument("--output")
    classify_parser.set_defaults(func=cmd_classify)

    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
