"""Forge Neo (Automatic1111互換API) で dataset/generation/prompts.yaml の枠を生成する。

使い方:
    python scripts/generate_forge.py --url http://127.0.0.1:7870 --attempt 1
    python scripts/generate_forge.py --url http://127.0.0.1:7870 --attempt 1 --slots A01,A02

- モデルの切り替えはしない(Forge Neo は override_settings でのcheckpoint変更を無視するため、
  作者が手動でロードしておく)。起動チェックポイントが anima-base-v1.0 でなければ即中断する。
- 1枠ずつ順番に生成する(並列化しない)。
- 同じ slot+attempt の再生成は拒否する(後選び防止。dataset-plan.md §2.7)。
"""

from __future__ import annotations

import argparse
import base64
import re
import sys
import time
from datetime import datetime, timezone

import gen_common as gc


def check_checkpoint(url: str) -> str:
    """起動中チェックポイントが anima-base-v1.0 であることを確認する。違えば中断。"""
    options = gc.http_get_json(f"{url}/sdapi/v1/options")
    checkpoint = options.get("sd_model_checkpoint", "")
    if "anima-base-v1.0" not in checkpoint:
        print(
            f"[中断] Forge Neo のロード中チェックポイントが anima-base-v1.0 ではありません: "
            f"{checkpoint!r}\n"
            "作者がForge Neo側で手動でモデルを切り替えてから再実行してください。",
            file=sys.stderr,
        )
        sys.exit(1)
    return checkpoint


def parse_version(parameters_text: str) -> str:
    m = re.search(r"Version:\s*([^\n,]+)", parameters_text or "")
    return m.group(1).strip() if m else ""


def generate_one(url: str, cfg: dict, slot_id: str, attempt: int) -> None:
    slot = gc.get_slot(cfg, slot_id)
    if slot["tool"] != "forge":
        raise ValueError(f"{slot_id} は tool={slot['tool']} であり forge 用ではありません")

    gc.check_not_generated(slot_id, attempt)

    seed = gc.seed_for_attempt(slot, attempt)
    prompt = gc.build_prompt(slot, cfg, "forge")
    common = cfg["common"]
    width, height = slot["size"]

    payload = {
        "prompt": prompt,
        "negative_prompt": common["negative"],
        "seed": seed,
        "steps": common["steps"],
        "cfg_scale": common["cfg"],
        "sampler_name": common["sampler_forge"],
        "scheduler": common["scheduler_forge"],
        "width": width,
        "height": height,
    }

    start = time.perf_counter()
    response = gc.http_post_json(f"{url}/sdapi/v1/txt2img", payload, timeout=600)
    elapsed = time.perf_counter() - start

    images = response.get("images") or []
    if not images:
        raise RuntimeError(f"txt2img応答にimagesがありません: keys={list(response.keys())}")

    png_bytes = base64.b64decode(images[0])
    sha256 = gc.sha256_bytes(png_bytes)
    generated_at = datetime.now(timezone.utc).isoformat()

    # 埋め込まれた parameters テキストを読む(再エンコードせず、返ってきたPNGバイト列そのまま保存する)
    import io

    from PIL import Image

    with Image.open(io.BytesIO(png_bytes)) as im:
        parameters_text = im.info.get("parameters", "")

    img_path = gc.staging_image_path(slot_id, attempt)
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(png_bytes)

    version = parse_version(parameters_text)

    params_data = {
        "tool": "forge_neo",
        "request": payload,
        "returned_parameters": parameters_text,
        "version": version,
        "elapsed_seconds": elapsed,
        "png_sha256": sha256,
        "generated_at": generated_at,
    }
    gc.write_params(slot_id, attempt, params_data)

    if "Model hash: bd43b7cffe" not in parameters_text:
        gc.append_log(
            {
                "slot_id": slot_id,
                "attempt": attempt,
                "seed": seed,
                "tool": "forge_neo",
                "file": str(img_path.relative_to(gc.REPO_ROOT)),
                "sha256": sha256,
                "generated_at": generated_at,
                "result": "setup_error",
                "reason": "returned parameters does not contain Model hash: bd43b7cffe",
            }
        )
        print(
            f"[中断] {slot_id} attempt={attempt}: 返ってきた画像のModel hashがbd43b7cffeでは"
            "ありません。ファイルは保存済みですがサーバのモデル設定を確認してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    gc.append_log(
        {
            "slot_id": slot_id,
            "attempt": attempt,
            "seed": seed,
            "tool": "forge_neo",
            "file": str(img_path.relative_to(gc.REPO_ROOT)),
            "sha256": sha256,
            "generated_at": generated_at,
            "result": "generated",
            "reason": "",
        }
    )
    print(f"  -> {img_path.name} (sha256={sha256[:12]}..., {elapsed:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forge Neo でdataset生成枠(forge)を生成する")
    parser.add_argument("--url", default="http://127.0.0.1:7870")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--slots", default=None, help="カンマ区切りのslot id。省略時はtool=forgeの全枠")
    args = parser.parse_args()

    cfg = gc.load_prompts()
    check_checkpoint(args.url)

    if args.slots:
        slot_ids = [s.strip() for s in args.slots.split(",") if s.strip()]
    else:
        slot_ids = [s["id"] for s in gc.slots_for_tool(cfg, "forge")]

    failures = []
    for i, slot_id in enumerate(slot_ids, start=1):
        print(f"[{i}/{len(slot_ids)}] {slot_id} attempt={args.attempt} 生成中...")
        try:
            generate_one(args.url, cfg, slot_id, args.attempt)
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 - 1枠の失敗で全体を止めない
            print(f"  [エラー] {slot_id}: {e}", file=sys.stderr)
            failures.append(slot_id)

    if failures:
        print(f"失敗した枠: {failures}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
