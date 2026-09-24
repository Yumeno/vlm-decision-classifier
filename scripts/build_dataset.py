"""dataset v1.0.0 を組み立てる(doc/dataset-plan.md §4 manifest仕様)。

入力:
- 作者の最終ラベル: dataset/staging/_labels/final/labels/*.json (31枠)
- 1回目の読み戻し: dataset/staging/_labels/labels/*.json (規則整合の5件修正前)
- 検収画像: dataset/staging/*.png / *.jpg (枠ごとに最新attemptを採用)
- 生成記録: dataset/generation/{id}_a{attempt}.json

出力:
- dataset/labels/labels_final.json, dataset/labels/labels_initial_readback.json
- dataset/images/{id}.{ext}, dataset/images/{A01..A04}-strip.png
- dataset/manifest.jsonl (31ソース + 4派生 = 35ケース)

classifier_demo は import しない(メタデータ読み取りは本スクリプト内の独立実装)。
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from pathlib import Path

from PIL import Image

from strip_metadata import strip_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGING = REPO_ROOT / "dataset" / "staging"
GENERATION = REPO_ROOT / "dataset" / "generation"
LOG_CSV = GENERATION / "log.csv"
LABELS_FINAL_DIR = STAGING / "_labels" / "final" / "labels"
LABELS_INITIAL_DIR = STAGING / "_labels" / "labels"
IMAGES_OUT = REPO_ROOT / "dataset" / "images"
LABELS_OUT = REPO_ROOT / "dataset" / "labels"
MANIFEST_OUT = REPO_ROOT / "dataset" / "manifest.jsonl"

SCENARIO_PREFIX = {
    "G": "general",
    "A": "alisa_lora",
    "N": "alisa_no_lora",
    "S": "similar",
    "M": "multi",
    "C": "conflict",
    "O": "outfit_scene",
}

ALISA_LORA_NAMES = {"fet-alisa-uniform-anima-v4u", "fet-alisa-uniform-anima-v4u_comfy"}
ALISA_TRIGGER_WORDS = ("fet_alisa_uniform",)

BASE_MODEL = "anima-base-v1.0"
BASE_MODEL_SHA256 = "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e"
EXTERNAL_TOOL_VERSION = {
    "external:codex_cli": "codex-cli 0.155.1",
    "external:antigravity_cli": "agy 1.2.7",
}
RIGHTS_TERMS = {
    "forge_neo": "CircleStone Labs Non-Commercial License v1.2 §2.e (Outputs)",
    "comfyui": "CircleStone Labs Non-Commercial License v1.2 §2.e (Outputs)",
    "external:codex_cli": "OpenAI terms: outputs owned by the user (checked 2026-09-25)",
    "external:antigravity_cli": "Google terms: outputs usable by the user (checked 2026-09-25)",
}
EDIT_SLOTS = {"O02", "O04", "O06", "O07"}  # 外部ツールで attempt1 を参照編集した枠
OWN_REFERENCE_SLOTS = {"N01", "N02"}  # 参照画像が作者自身のキャラの枠
TOKEN_TO_ATTEMPT = {"a1": "1", "a2": "2", "a2b": "2b"}
DERIVED_SOURCE_SLOTS = ["A01", "A02", "A03", "A04"]
RIGHTS_CONFIRMATION_PATH = REPO_ROOT / "dataset" / "labels" / "rights_confirmation.json"
RIGHTS_CONFIRMATION_REF = "dataset/labels/rights_confirmation.json"
C02_INJECT_SCRIPT = GENERATION / "C02_inject_metadata.py"

LORA_TAG_RE = re.compile(r"<lora:([^:>]+)(?::([0-9.]+))?>")
EMPHASIS_TAG_RE = re.compile(r"^\(+(.*?)(?::[0-9.]+)?\)+$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_label_file(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return d.get("data", d)


def load_labels_dir(dir_path: Path) -> dict[str, dict]:
    return {p.stem: load_label_file(p) for p in sorted(dir_path.glob("*.json"))}


def load_log_rows() -> list[dict]:
    with LOG_CSV.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pick_attempt(slot_id: str) -> tuple[str, Path]:
    """検収画像の最新attemptを選ぶ。C02 は人工メタデータ付きの a1 を使う例外。"""
    if slot_id == "C02":
        path = STAGING / "C02_a1_injected.png"
        if not path.exists():
            raise SystemExit(f"[エラー] {path} がありません")
        return "a1", path

    for token in ("a2b", "a2", "a1"):
        matches = sorted(
            p for p in STAGING.glob(f"{slot_id}_{token}.*") if p.suffix.lower() in (".png", ".jpg")
        )
        if len(matches) > 1:
            raise SystemExit(f"[エラー] {slot_id}_{token} の画像が複数見つかりました: {matches}")
        if matches:
            return token, matches[0]
    raise SystemExit(f"[エラー] {slot_id} の検収画像が見つかりません")


def find_loras_in_text(text: str) -> list[tuple[str, float]]:
    return [(name, float(weight) if weight else 1.0) for name, weight in LORA_TAG_RE.findall(text)]


def find_loras_in_graph(graph: dict) -> list[tuple[str, float]]:
    loras = []
    for node in graph.values():
        if "LoraLoader" in node.get("class_type", ""):
            lora_name = node["inputs"].get("lora_name", "")
            name = Path(lora_name.replace("\\", "/")).stem
            weight = float(node["inputs"].get("strength_model", 1.0))
            loras.append((name, weight))
    return loras


def find_comfy_positive_text(graph: dict) -> str:
    for node in graph.values():
        if node.get("class_type") == "KSampler":
            pos_ref = node["inputs"].get("positive")
            if pos_ref:
                pos_node = graph.get(str(pos_ref[0]))
                if pos_node and pos_node.get("class_type") == "CLIPTextEncode":
                    return pos_node["inputs"].get("text", "")
    return ""


def normalize_tag(tag: str) -> str:
    """`(tag:1.2)` のような強調記法を素のタグ文字列に戻す。"""
    tag = tag.strip()
    m = EMPHASIS_TAG_RE.match(tag)
    return m.group(1).strip() if m else tag


def find_trigger_words_in_text(text: str) -> list[str]:
    """カンマ区切りタグを走査してトリガーワードの完全一致を探す。

    `<lora:...:weight>` タグはスペース区切りで別のタグ(トリガーワード等)とカンマなしで
    連結されることがあるため、先に除去してから分割する。
    """
    cleaned = LORA_TAG_RE.sub("", text)
    tags = [normalize_tag(t) for t in re.split(r"[,\n]", cleaned)]
    tags = [t for t in tags if t]
    return [w for w in ALISA_TRIGGER_WORDS if w in tags]


def characters_from_loras_and_triggers(loras: list[tuple[str, float]], trigger_words: list[str]) -> list[str]:
    has_alisa = any(name.lower().strip() in ALISA_LORA_NAMES for name, _ in loras) or bool(trigger_words)
    return ["alisa"] if has_alisa else []


def read_embedded_metadata(image_path: Path) -> dict:
    """完成画像に実際に埋め込まれたメタデータだけから expected_metadata を作る(独立実装)。"""
    with Image.open(image_path) as im:
        info = dict(im.info)

    if "parameters" in info:
        text = info["parameters"]
        positive = text.split("Negative prompt:", 1)[0]
        loras = find_loras_in_text(positive)
        trigger_words = find_trigger_words_in_text(positive)
        return {
            "format": "a1111",
            "loras": [{"name": n, "weight": w} for n, w in loras],
            "trigger_words": trigger_words,
            "characters": characters_from_loras_and_triggers(loras, trigger_words),
            "artificial": False,
        }

    if "prompt" in info:
        graph = json.loads(info["prompt"])
        loras = find_loras_in_graph(graph)
        positive_text = find_comfy_positive_text(graph)
        trigger_words = find_trigger_words_in_text(positive_text)
        return {
            "format": "comfyui",
            "loras": [{"name": n, "weight": w} for n, w in loras],
            "trigger_words": trigger_words,
            "characters": characters_from_loras_and_triggers(loras, trigger_words),
            "artificial": False,
        }

    return {"format": "none", "loras": [], "trigger_words": [], "characters": [], "artificial": False}


def cross_check_metadata(slot_id: str, embedded: dict, record: dict) -> None:
    """独立実装(画像から読んだ値)と生成記録(依頼した内容)が食い違っていないか確認する。"""
    tool = record.get("tool", "")
    if tool == "forge_neo":
        rec_loras = find_loras_in_text(record["request"]["prompt"])
    elif tool == "comfyui":
        rec_loras = find_loras_in_graph(record["graph"])
    elif tool.startswith("external:"):
        # 外部生成ツールは生成記録にメタデータ埋め込みの仕組みがないので、
        # 画像側にも何も埋め込まれていない(format=none)ことだけ確認する。
        if embedded["format"] != "none":
            raise SystemExit(
                f"[エラー] {slot_id}: 外部生成画像のはずですが埋め込みメタデータがあります "
                f"(format={embedded['format']})"
            )
        return
    else:
        return

    emb_set = {(l["name"].lower().strip(), round(l["weight"], 4)) for l in embedded["loras"]}
    rec_set = {(n.lower().strip(), round(w, 4)) for n, w in rec_loras}
    if emb_set != rec_set:
        raise SystemExit(
            f"[エラー] {slot_id}: 画像に埋め込まれたLoRA情報が生成記録と一致しません "
            f"(画像={emb_set}, 記録={rec_set})"
        )


def load_c02_injected_parameters() -> str:
    """C02_inject_metadata.py を import・実行せず、AST から PARAMETERS 定数の文字列だけを取り出す。"""
    tree = ast.parse(C02_INJECT_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PARAMETERS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise SystemExit(f"[エラー] {C02_INJECT_SCRIPT} に PARAMETERS 定数が見つかりません")


def verify_c02_injected_metadata(image_path: Path) -> None:
    """C02 の画像に書き込まれた parameters チャンクが、注入スクリプトの定数と完全一致するか確認する。"""
    with Image.open(image_path) as im:
        actual = im.info.get("parameters")
    expected = load_c02_injected_parameters()
    if actual != expected:
        raise SystemExit(
            "[エラー] C02: 画像の parameters チャンクが "
            f"{C02_INJECT_SCRIPT} の PARAMETERS 定数と一致しません"
        )


def tool_version_of(tool: str, record: dict) -> str | None:
    if tool == "forge_neo":
        m = re.search(r"Version:\s*(\S+)", record.get("returned_parameters", ""))
        return m.group(1) if m else None
    if tool == "comfyui":
        return None
    return EXTERNAL_TOOL_VERSION.get(tool)


def seed_of(tool: str, record: dict) -> int | None:
    if tool == "forge_neo":
        return record["request"]["seed"]
    if tool == "comfyui":
        for node in record["graph"].values():
            if node.get("class_type") == "KSampler":
                return node["inputs"]["seed"]
    return None


def edit_of_source(slot_id: str, record: dict) -> str | None:
    if slot_id not in EDIT_SLOTS:
        return None
    marker = f"参照画像: dataset/staging/{slot_id}_a1."
    if marker in record.get("response_note", ""):
        return f"{slot_id}_a1"
    return None


def load_rights_confirmation() -> dict:
    if not RIGHTS_CONFIRMATION_PATH.exists():
        raise SystemExit(f"[エラー] {RIGHTS_CONFIRMATION_PATH} がありません")
    return json.loads(RIGHTS_CONFIRMATION_PATH.read_text(encoding="utf-8"))


def rights_of(slot_id: str, tool: str, edit_of: str | None, confirmed_slots: set[str]) -> dict:
    terms = RIGHTS_TERMS[tool]
    if edit_of is not None:
        terms += " + source image under CircleStone §2.e"
    if slot_id in OWN_REFERENCE_SLOTS:
        terms += " + reference image is the author's own character"
    return {
        "terms": terms,
        "rights_confirmed": slot_id in confirmed_slots,
        "confirmation_record": RIGHTS_CONFIRMATION_REF,
    }


def verify_recorded_sha256(slot_id: str, token: str, actual_sha: str, log_rows: list[dict]) -> None:
    if slot_id == "C02":
        rows = [r for r in log_rows if r["tool"] == "artificial_metadata"]
        if not rows:
            raise SystemExit("[エラー] log.csv に artificial_metadata 行がありません")
        expected_sha = rows[0]["sha256"]
    else:
        record_path = GENERATION / f"{slot_id}_{token}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        expected_sha = record.get("png_sha256") or record.get("sha256")
    if actual_sha != expected_sha:
        raise SystemExit(
            f"[エラー] {slot_id} ({token}): sha256が生成記録と一致しません "
            f"(画像={actual_sha}, 記録={expected_sha})"
        )


def build_source_case(
    slot_id: str, labels_final: dict, log_rows: list[dict], confirmed_slots: set[str]
) -> dict:
    token, src_path = pick_attempt(slot_id)
    data = src_path.read_bytes()
    actual_sha = sha256_bytes(data)
    verify_recorded_sha256(slot_id, token, actual_sha, log_rows)

    ext = src_path.suffix.lstrip(".").lower()
    dest = IMAGES_OUT / f"{slot_id}.{ext}"
    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)

    record_token = "a1" if slot_id == "C02" else token
    record_path = GENERATION / f"{slot_id}_{record_token}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    tool = record["tool"]

    embedded = read_embedded_metadata(dest)
    if slot_id == "C02":
        verify_c02_injected_metadata(dest)
        embedded["artificial"] = True
    else:
        cross_check_metadata(slot_id, embedded, record)

    edit_of = edit_of_source(slot_id, record)
    generation = {
        "tool": tool,
        "tool_version": tool_version_of(tool, record),
        "base_model": BASE_MODEL if tool in ("forge_neo", "comfyui") else None,
        "base_model_sha256": BASE_MODEL_SHA256 if tool in ("forge_neo", "comfyui") else None,
        "seed": seed_of(tool, record),
        "attempt": TOKEN_TO_ATTEMPT[record_token],
        "edit_of": edit_of,
        "params_file": f"dataset/generation/{slot_id}_{record_token}.json",
    }

    label_entry = labels_final[slot_id]
    note = label_entry.get("note") or None

    return {
        "case_id": slot_id,
        "source_image_id": slot_id,
        "derived_from": None,
        "image_path": f"dataset/images/{slot_id}.{ext}",
        "image_sha256": actual_sha,
        "split": "test",
        "scenario": SCENARIO_PREFIX[slot_id[0]],
        "expected": label_entry["labels"],
        "expected_metadata": embedded,
        "generation": generation,
        "rights": rights_of(slot_id, tool, edit_of, confirmed_slots),
        "review": {"status": "accepted", "note": note},
    }


def verify_pixel_match(src_path: Path, dst_path: Path) -> None:
    """派生ファイルを書き出し後に読み直し、元画像と画素(mode・size・tobytes)が一致するか再確認する。"""
    with Image.open(src_path) as a, Image.open(dst_path) as b:
        a.load()
        b.load()
        if a.mode != b.mode or a.size != b.size or a.tobytes() != b.tobytes():
            raise SystemExit(f"[エラー] 画素が一致しません: {src_path} -> {dst_path}")


def build_derived_case(slot_id: str, source_case: dict) -> dict:
    src_path = IMAGES_OUT / Path(source_case["image_path"]).name
    dst_path = IMAGES_OUT / f"{slot_id}-strip.png"
    strip_metadata(src_path, dst_path)  # strip_metadata 内でも画素一致を検証済み
    verify_pixel_match(src_path, dst_path)  # 書き出し後に読み直しての独立再確認
    actual_sha = sha256_bytes(dst_path.read_bytes())

    return {
        "case_id": f"{slot_id}-strip",
        "source_image_id": slot_id,
        "derived_from": slot_id,
        "image_path": f"dataset/images/{slot_id}-strip.png",
        "image_sha256": actual_sha,
        "split": "test",
        "scenario": source_case["scenario"],
        "expected": source_case["expected"],
        "expected_metadata": {
            "format": "none",
            "loras": [],
            "trigger_words": [],
            "characters": [],
            "artificial": False,
        },
        "generation": source_case["generation"],
        "rights": source_case["rights"],
        "review": {
            "status": "accepted",
            "note": f"{slot_id} のメタデータ除去コピー (scripts/strip_metadata.py)",
        },
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def print_summary(cases: list[dict]) -> None:
    scenario_counts: dict[str, int] = {}
    source_count = 0
    derived_count = 0
    for case in cases:
        scenario_counts[case["scenario"]] = scenario_counts.get(case["scenario"], 0) + 1
        if case["derived_from"] is None:
            source_count += 1
        else:
            derived_count += 1

    print(f"cases: {len(cases)} (source={source_count}, derived={derived_count})")
    for scenario in sorted(scenario_counts):
        print(f"  scenario {scenario}: {scenario_counts[scenario]}")

    print("\nexpected_metadata:")
    for case in cases:
        m = case["expected_metadata"]
        print(
            f"  {case['case_id']:>10}: format={m['format']:<8} "
            f"characters={m['characters']} artificial={m['artificial']}"
        )

    # 素朴な異常検知: LoRA/トリガーが記録されている想定の枠以外で characters が付いていないか、
    # 逆に想定の枠で付いていないか(狙いとの比較。判定基準はあくまで§2.5の実測)
    expect_alisa_meta = {"A01", "A02", "A03", "A04", "C01", "C02", "M01", "M02"}
    print("\nanomalies (expected_metadata.characters vs plan expectation):")
    found = False
    for case in cases:
        slot = case["source_image_id"]
        if case["derived_from"] is not None:
            continue
        has_alisa = "alisa" in case["expected_metadata"]["characters"]
        should_have = slot in expect_alisa_meta
        if has_alisa != should_have:
            found = True
            print(f"  {slot}: characters={case['expected_metadata']['characters']} (plan expects alisa={should_have})")
    if not found:
        print("  none")


def main() -> None:
    labels_final = load_labels_dir(LABELS_FINAL_DIR)
    labels_initial = load_labels_dir(LABELS_INITIAL_DIR)

    if len(labels_final) != 31:
        raise SystemExit(f"[エラー] labels_final の件数が31ではありません: {len(labels_final)}")

    write_json(LABELS_OUT / "labels_final.json", labels_final)
    write_json(LABELS_OUT / "labels_initial_readback.json", labels_initial)

    log_rows = load_log_rows()
    confirmed_slots = set(load_rights_confirmation().get("slots", []))

    slot_ids = sorted(labels_final.keys())
    cases = [
        build_source_case(slot_id, labels_final, log_rows, confirmed_slots) for slot_id in slot_ids
    ]
    by_id = {c["case_id"]: c for c in cases}
    for slot_id in DERIVED_SOURCE_SLOTS:
        cases.append(build_derived_case(slot_id, by_id[slot_id]))

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_OUT.open("w", encoding="utf-8", newline="\n") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print_summary(cases)
    print(f"\n-> {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
