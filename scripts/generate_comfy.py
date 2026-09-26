"""ComfyUI で dataset/generation/prompts.yaml の枠(tool: comfyui)を生成する。

使い方:
    python scripts/generate_comfy.py --url http://127.0.0.1:8188 --attempt 1
    python scripts/generate_comfy.py --url http://127.0.0.1:8188 --attempt 1 --slots A03,A04

ベースのワークフローは HuggingFace の Anima 公式サンプル画像
(circlestone-labs/Anima の example.png)に埋め込まれた API形式グラフを
dataset/generation/comfy_anima_workflow.json として保存したもの。

ノードは class_type で特定する(HFのサンプルが将来更新されても
KSampler/EmptyLatentImage/UNETLoader/VAEDecode の関係から辿れるようにするため)。
現時点のサンプルグラフの構造は以下の通り(コミット時点で確認済み):
  - "44" UNETLoader        (unet_name を anima-base-v1.0.safetensors に上書き)
  - "11"/"12" CLIPTextEncode (KSamplerのpositive/negative参照から特定。text を上書き)
  - "19" KSampler           (seed/steps/cfg/sampler_name/scheduler を上書き)
  - "28" EmptyLatentImage   (width/height を上書き)
  - "8"  VAEDecode          (そのまま)
  - "1"  PreviewImage       (SaveImageへ差し替え。PreviewImageは type=temp 保存のため
                              /view?type=output で拾えない。filename_prefixを追加)
LoRAを使う枠は、UNETLoaderの直後にLoraLoaderModelOnlyノードを挿入し、
KSamplerのmodel入力をそちらに繋ぎ替える。
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlencode

import gen_common as gc

WORKFLOW_PATH = gc.GENERATION_DIR / "comfy_anima_workflow.json"
UNET_FILENAME = "anima-base-v1.0.safetensors"
POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 600


def load_workflow() -> dict:
    import json

    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def find_by_class(graph: dict, class_type: str) -> list[str]:
    return [nid for nid, node in graph.items() if node.get("class_type") == class_type]


def locate_nodes(graph: dict) -> dict:
    """class_type からノードIDを特定する。想定と違えば明示的にエラーにする。"""
    ksampler_ids = find_by_class(graph, "KSampler")
    unet_ids = find_by_class(graph, "UNETLoader")
    latent_ids = find_by_class(graph, "EmptyLatentImage")
    vaedecode_ids = find_by_class(graph, "VAEDecode")
    output_ids = find_by_class(graph, "PreviewImage") + find_by_class(graph, "SaveImage")

    if len(ksampler_ids) != 1:
        raise RuntimeError(f"KSamplerノードが1個ではありません: {ksampler_ids}")
    if len(unet_ids) != 1:
        raise RuntimeError(f"UNETLoaderノードが1個ではありません: {unet_ids}")
    if len(latent_ids) != 1:
        raise RuntimeError(f"EmptyLatentImageノードが1個ではありません: {latent_ids}")
    if len(vaedecode_ids) != 1:
        raise RuntimeError(f"VAEDecodeノードが1個ではありません: {vaedecode_ids}")
    if len(output_ids) != 1:
        raise RuntimeError(f"PreviewImage/SaveImageノードが1個ではありません: {output_ids}")

    ksampler_id = ksampler_ids[0]
    ksampler_inputs = graph[ksampler_id]["inputs"]
    positive_id = ksampler_inputs["positive"][0]
    negative_id = ksampler_inputs["negative"][0]

    return {
        "ksampler_id": ksampler_id,
        "unet_id": unet_ids[0],
        "latent_id": latent_ids[0],
        "vaedecode_id": vaedecode_ids[0],
        "output_id": output_ids[0],
        "positive_id": positive_id,
        "negative_id": negative_id,
    }


def build_graph(slot: dict, cfg: dict, seed: int, template: dict, output_prefix: str) -> dict:
    graph = copy.deepcopy(template)
    nodes = locate_nodes(graph)
    common = cfg["common"]

    graph[nodes["unet_id"]]["inputs"]["unet_name"] = UNET_FILENAME

    positive_text = gc.build_prompt(slot, cfg, "comfyui")
    graph[nodes["positive_id"]]["inputs"]["text"] = positive_text
    graph[nodes["negative_id"]]["inputs"]["text"] = common["negative"]

    ks_inputs = graph[nodes["ksampler_id"]]["inputs"]
    ks_inputs["seed"] = seed
    ks_inputs["steps"] = common["steps"]
    ks_inputs["cfg"] = common["cfg"]
    ks_inputs["sampler_name"] = common["sampler_comfy"]
    ks_inputs["scheduler"] = common["scheduler_comfy"]

    width, height = slot["size"]
    graph[nodes["latent_id"]]["inputs"]["width"] = width
    graph[nodes["latent_id"]]["inputs"]["height"] = height

    weight = slot.get("lora_weight")
    if weight is not None:
        lora_node_id = "lora_1"
        while lora_node_id in graph:
            lora_node_id += "_"
        graph[lora_node_id] = {
            "inputs": {
                "model": [nodes["unet_id"], 0],
                "lora_name": "Anima\\fet_alisa_uniform\\" + cfg["lora_comfy"] + ".safetensors",
                "strength_model": weight,
            },
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": "Load LoRA Model Only"},
        }
        ks_inputs["model"] = [lora_node_id, 0]

    # PreviewImage は type=temp にしか保存しないため SaveImage に差し替える(type=output で拾うため)
    out_node = graph[nodes["output_id"]]
    images_input = out_node["inputs"]["images"]
    graph[nodes["output_id"]] = {
        "inputs": {"images": images_input, "filename_prefix": output_prefix},
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    }

    return graph


def check_lora(url: str, cfg: dict) -> None:
    """LoraLoaderModelOnlyの入力仕様と、使うlora_nameが選択肢にあることを確認する。違えば中断。"""
    info = gc.http_get_json(f"{url}/object_info/LoraLoaderModelOnly")
    node_info = info.get("LoraLoaderModelOnly")
    if node_info is None:
        print("[中断] ComfyUIにLoraLoaderModelOnlyノードが見つかりません。", file=sys.stderr)
        sys.exit(1)

    required = node_info.get("input", {}).get("required", {})
    print(f"LoraLoaderModelOnly required inputs: {list(required.keys())}")

    expected_keys = {"model", "lora_name", "strength_model"}
    actual_keys = set(required.keys())
    if actual_keys != expected_keys:
        print(
            f"[中断] LoraLoaderModelOnlyのrequired入力が想定と違います。"
            f"想定: {sorted(expected_keys)} / 実際: {sorted(actual_keys)}",
            file=sys.stderr,
        )
        sys.exit(1)

    lora_name_spec = required.get("lora_name")
    choices: list[str] = []
    if isinstance(lora_name_spec, list) and lora_name_spec and isinstance(lora_name_spec[0], list):
        choices = lora_name_spec[0]

    target = "Anima\\fet_alisa_uniform\\" + cfg["lora_comfy"] + ".safetensors"
    if target not in choices:
        candidates = [c for c in choices if cfg["lora_comfy"] in c]
        print(
            f"[中断] ComfyUIのlora_name選択肢に {target!r} が見つかりません。\n"
            f"{cfg['lora_comfy']!r} を含む候補: {candidates}",
            file=sys.stderr,
        )
        sys.exit(1)


def submit(url: str, graph: dict, client_id: str) -> str:
    response = gc.http_post_json(
        f"{url}/prompt", {"prompt": graph, "client_id": client_id}, timeout=60
    )
    node_errors = response.get("node_errors") or {}
    if node_errors:
        raise RuntimeError(f"/prompt がnode_errorsを返しました: {node_errors}")
    prompt_id = response.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"/prompt応答にprompt_idがありません: {response}")
    return prompt_id


def wait_for_history(url: str, prompt_id: str) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        history = gc.http_get_json(f"{url}/history/{prompt_id}")
        entry = history.get(prompt_id)
        if entry and entry.get("outputs"):
            return entry
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"生成待ちがタイムアウトしました(prompt_id={prompt_id})")


def fetch_image(url: str, output_node: dict) -> bytes:
    images = output_node.get("images") or []
    if not images:
        raise RuntimeError(f"出力ノードにimagesがありません: {output_node}")
    image_ref = images[0]
    qs = urlencode(
        {
            "filename": image_ref["filename"],
            "subfolder": image_ref.get("subfolder", ""),
            "type": image_ref.get("type", "output"),
        }
    )
    return gc.http_get_bytes(f"{url}/view?{qs}")


def generate_one(url: str, cfg: dict, template: dict, slot_id: str, attempt: int) -> None:
    slot = gc.get_slot(cfg, slot_id)
    if slot["tool"] != "comfyui":
        raise ValueError(f"{slot_id} は tool={slot['tool']} であり comfyui 用ではありません")

    gc.check_not_generated(slot_id, attempt)

    seed = gc.seed_for_attempt(slot, attempt)
    output_prefix = f"{slot_id}_a{attempt}"
    graph = build_graph(slot, cfg, seed, template, output_prefix)
    client_id = str(uuid.uuid4())

    # 送信直前にマーカーを書く。以降、結果にかかわらずこのslot+attemptは二度と送らない。
    gc.write_submitted_marker(
        slot_id,
        attempt,
        {
            "slot_id": slot_id,
            "attempt": attempt,
            "seed": seed,
            "tool": "comfyui",
            "graph": graph,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    sha256 = ""
    img_path = None
    image_saved = False
    try:
        start = time.perf_counter()
        prompt_id = submit(url, graph, client_id)
        print(f"  prompt_id={prompt_id}")
        gc.write_prompt_id(slot_id, attempt, prompt_id)
        history_entry = wait_for_history(url, prompt_id)
        elapsed = time.perf_counter() - start

        nodes = locate_nodes(graph)
        output_node = history_entry["outputs"].get(nodes["output_id"])
        if output_node is None:
            raise RuntimeError(f"historyの出力に{nodes['output_id']}がありません: {history_entry}")

        png_bytes = fetch_image(url, output_node)
        sha256 = gc.sha256_bytes(png_bytes)
        generated_at = datetime.now(timezone.utc).isoformat()

        img_path = gc.staging_image_path(slot_id, attempt)
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(png_bytes)
        image_saved = True

        params_data = {
            "tool": "comfyui",
            "prompt_id": prompt_id,
            "graph": graph,
            "elapsed_seconds": elapsed,
            "png_sha256": sha256,
            "generated_at": generated_at,
        }
        gc.write_params(slot_id, attempt, params_data)
    except Exception as e:
        # 送信後の失敗(タイムアウト・取得失敗・保存失敗など)は必ずログしてから次の枠へ進めるようにする。
        # 画像保存が済んでいれば file/sha256 は分かっているのでログに残す(保存前の失敗は空のまま)。
        gc.append_log(
            {
                "slot_id": slot_id,
                "attempt": attempt,
                "seed": seed,
                "tool": "comfyui",
                "file": str(img_path.relative_to(gc.REPO_ROOT)) if image_saved else "",
                "sha256": sha256 if image_saved else "",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "result": "error",
                "reason": str(e)[:300],
            }
        )
        raise

    gc.append_log(
        {
            "slot_id": slot_id,
            "attempt": attempt,
            "seed": seed,
            "tool": "comfyui",
            "file": str(img_path.relative_to(gc.REPO_ROOT)),
            "sha256": sha256,
            "generated_at": generated_at,
            "result": "generated",
            "reason": "",
        }
    )
    print(f"  -> {img_path.name} (sha256={sha256[:12]}..., {elapsed:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="ComfyUI でdataset生成枠(comfyui)を生成する")
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--attempt", type=int, default=1, choices=[1, 2])
    parser.add_argument("--slots", default=None, help="カンマ区切りのslot id。省略時はtool=comfyuiの全枠")
    args = parser.parse_args()

    cfg = gc.load_prompts()
    template = load_workflow()
    check_lora(args.url, cfg)

    if args.slots:
        slot_ids = [s.strip() for s in args.slots.split(",") if s.strip()]
    else:
        slot_ids = [s["id"] for s in gc.slots_for_tool(cfg, "comfyui")]

    failures = []
    for i, slot_id in enumerate(slot_ids, start=1):
        print(f"[{i}/{len(slot_ids)}] {slot_id} attempt={args.attempt} 生成中...")
        try:
            generate_one(args.url, cfg, template, slot_id, attempt=args.attempt)
        except Exception as e:  # noqa: BLE001 - 1枠の失敗で全体を止めない
            print(f"  [エラー] {slot_id}: {e}", file=sys.stderr)
            failures.append(slot_id)

    if failures:
        print(f"失敗した枠: {failures}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
