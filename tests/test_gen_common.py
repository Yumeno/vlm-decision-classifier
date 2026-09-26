import csv
import json

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import gen_common as gc
import strip_metadata


@pytest.fixture
def cfg():
    return {
        "lora_forge": "fet-alisa-uniform-anima-v4u",
        "lora_comfy": "fet-alisa-uniform-anima-v4u_comfy",
        "common": {
            "steps": 36,
            "cfg": 4.5,
            "sampler_forge": "ER SDE",
            "scheduler_forge": "Simple",
            "sampler_comfy": "er_sde",
            "scheduler_comfy": "simple",
            "positive_prefix": "masterpiece, best quality, score_7, safe, ",
            "negative": "worst quality, low quality",
        },
        "fragments": {
            "alisa": "brown hair, pink vest, red neck ribbon",
            "second": "brown hair, navy blazer, green necktie",
        },
    }


def make_slot(**overrides):
    slot = {
        "id": "A01",
        "tool": "forge",
        "seed": 20260911,
        "size": [1024, 1024],
        "lora_weight": 1.0,
        "trigger": True,
        "prompt": "1girl, {alisa}, office, upper body",
    }
    slot.update(overrides)
    return slot


# ---- build_prompt ----


def test_build_prompt_forge_with_lora_and_trigger(cfg):
    slot = make_slot()
    prompt = gc.build_prompt(slot, cfg, "forge")
    assert prompt.startswith("masterpiece, best quality, score_7, safe, ")
    assert "<lora:fet-alisa-uniform-anima-v4u:1.0>" in prompt
    assert "fet_alisa_uniform, " in prompt
    assert "brown hair, pink vest, red neck ribbon" in prompt
    # 順序: prefix -> lora -> trigger -> prompt本文
    prefix_idx = prompt.index("masterpiece")
    lora_idx = prompt.index("<lora:")
    trigger_idx = prompt.index("fet_alisa_uniform,")
    body_idx = prompt.index("office")
    assert prefix_idx < lora_idx < trigger_idx < body_idx


def test_build_prompt_forge_without_lora_no_trigger(cfg):
    slot = make_slot(lora_weight=None, trigger=False, prompt="no humans, scenery")
    prompt = gc.build_prompt(slot, cfg, "forge")
    assert "<lora:" not in prompt
    assert "fet_alisa_uniform" not in prompt
    assert "no humans, scenery" in prompt


def test_build_prompt_forge_weight_zero_still_emits_lora_tag(cfg):
    # C01: 重み0でも記録上はLoRA指定が「ある」ことがdataset-plan.mdの要件
    slot = make_slot(id="C01", lora_weight=0.0, trigger=False, prompt="1girl, {second}, corridor")
    prompt = gc.build_prompt(slot, cfg, "forge")
    assert "<lora:fet-alisa-uniform-anima-v4u:0.0>" in prompt
    assert "fet_alisa_uniform" not in prompt


def test_build_prompt_comfyui_no_lora_tag_in_text_but_trigger_present(cfg):
    slot = make_slot(id="A03", tool="comfyui")
    prompt = gc.build_prompt(slot, cfg, "comfyui")
    # ComfyUIではLoRAはテキストではなくノードで指定するため、テキストに<lora:...>は出ない
    assert "<lora:" not in prompt
    assert "fet_alisa_uniform, " in prompt


def test_build_prompt_comfyui_second_fragment_substitution(cfg):
    slot = make_slot(id="S01", tool="comfyui", lora_weight=None, trigger=False,
                      prompt="1girl, {second}, classroom")
    prompt = gc.build_prompt(slot, cfg, "comfyui")
    assert "navy blazer, green necktie" in prompt
    assert "{second}" not in prompt


# ---- seed_for_attempt ----


def test_seed_for_attempt_1_is_slot_seed():
    slot = make_slot(seed=20260911)
    assert gc.seed_for_attempt(slot, 1) == 20260911


def test_seed_for_attempt_2_is_seed_plus_one():
    slot = make_slot(seed=20260911)
    assert gc.seed_for_attempt(slot, 2) == 20260912


# ---- write_params / check_not_generated: 上書き拒否 ----


def test_write_params_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "GENERATION_DIR", tmp_path)

    gc.write_params("A01", 1, {"tool": "forge_neo"})
    with pytest.raises(FileExistsError):
        gc.write_params("A01", 1, {"tool": "forge_neo"})

    data = json.loads((tmp_path / "A01_a1.json").read_text(encoding="utf-8"))
    assert data == {"tool": "forge_neo"}


def test_check_not_generated_raises_if_image_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "GENERATION_DIR", tmp_path)
    monkeypatch.setattr(gc, "STAGING_DIR", tmp_path)

    img = tmp_path / "A01_a1.png"
    img.write_bytes(b"fake png bytes")

    with pytest.raises(FileExistsError):
        gc.check_not_generated("A01", 1)


def test_check_not_generated_ok_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "GENERATION_DIR", tmp_path)
    monkeypatch.setattr(gc, "STAGING_DIR", tmp_path)

    gc.check_not_generated("A01", 1)  # 例外が出なければOK


def test_write_submitted_marker_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "GENERATION_DIR", tmp_path)

    gc.write_submitted_marker("A01", 1, {"slot_id": "A01", "attempt": 1, "seed": 1})
    with pytest.raises(FileExistsError):
        gc.write_submitted_marker("A01", 1, {"slot_id": "A01", "attempt": 1, "seed": 1})

    data = json.loads((tmp_path / "A01_a1.submitted.json").read_text(encoding="utf-8"))
    assert data["seed"] == 1


def test_write_submitted_marker_uses_exclusive_create(tmp_path, monkeypatch):
    # write_submitted_marker自身が"x"モードの新規作成のみを試みる。
    # write_submitted_markerを経由せず外部から同名ファイルが先に置かれていても、
    # 「存在チェック→書き込み」の2段階を踏まずに作成そのもので失敗し、内容を壊さない。
    monkeypatch.setattr(gc, "GENERATION_DIR", tmp_path)
    marker = tmp_path / "A01_a1.submitted.json"
    marker.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        gc.write_submitted_marker("A01", 1, {"slot_id": "A01", "attempt": 1, "seed": 1})

    assert marker.read_text(encoding="utf-8") == "existing"


def test_check_not_generated_raises_if_marker_exists_even_without_result(tmp_path, monkeypatch):
    # 送信済みマーカーだけがある(=送信したが結果がまだ/失敗した)状態でも再送信を拒否する
    monkeypatch.setattr(gc, "GENERATION_DIR", tmp_path)
    monkeypatch.setattr(gc, "STAGING_DIR", tmp_path)

    gc.write_submitted_marker("A01", 1, {"slot_id": "A01", "attempt": 1, "seed": 1})

    assert not gc.staging_image_path("A01", 1).exists()
    assert not gc.params_json_path("A01", 1).exists()
    with pytest.raises(FileExistsError):
        gc.check_not_generated("A01", 1)


def test_check_not_generated_does_not_block_other_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "GENERATION_DIR", tmp_path)
    monkeypatch.setattr(gc, "STAGING_DIR", tmp_path)

    gc.write_submitted_marker("A01", 1, {"slot_id": "A01", "attempt": 1, "seed": 1})
    gc.check_not_generated("A01", 2)  # attempt違いは拒否されない


# ---- append_log ----


def test_append_log_creates_header_and_appends(tmp_path, monkeypatch):
    log_path = tmp_path / "log.csv"
    monkeypatch.setattr(gc, "LOG_CSV_PATH", log_path)

    gc.append_log(
        {
            "slot_id": "A01",
            "attempt": 1,
            "seed": 20260911,
            "tool": "forge_neo",
            "file": "dataset/staging/A01_a1.png",
            "sha256": "abc123",
            "generated_at": "2026-09-25T00:00:00+00:00",
            "result": "generated",
            "reason": "",
        }
    )
    gc.append_log(
        {
            "slot_id": "A01",
            "attempt": 2,
            "seed": 20260912,
            "tool": "forge_neo",
            "file": "dataset/staging/A01_a2.png",
            "sha256": "def456",
            "generated_at": "2026-09-25T00:01:00+00:00",
            "result": "generated",
            "reason": "retry after reject",
        }
    )

    with open(log_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["slot_id"] == "A01"
    assert rows[0]["attempt"] == "1"
    assert rows[1]["reason"] == "retry after reject"
    assert list(rows[0].keys()) == gc.LOG_FIELDS


# ---- load_prompts: 実ファイルの読み込み確認 ----


def test_load_prompts_real_file_has_expected_shape():
    cfg = gc.load_prompts()
    assert cfg["lora_forge"] == "fet-alisa-uniform-anima-v4u"
    slot_ids = {s["id"] for s in cfg["slots"]}
    assert "A01" in slot_ids
    assert "C01" in slot_ids


# ---- strip_metadata ----


def test_strip_metadata_removes_parameters_keeps_pixels(tmp_path):
    src = tmp_path / "src.png"
    dst = tmp_path / "dst.png"

    im = Image.new("RGB", (8, 6), color=(10, 20, 30))
    im.putpixel((0, 0), (255, 0, 0))
    info = PngInfo()
    info.add_text("parameters", "masterpiece, best quality\nSteps: 36, Seed: 1")
    im.save(src, pnginfo=info)

    with Image.open(src) as check_src:
        assert check_src.info.get("parameters")

    strip_metadata.strip_metadata(src, dst)

    with Image.open(src) as before, Image.open(dst) as after:
        before.load()
        after.load()
        assert "parameters" not in after.info
        assert before.tobytes() == after.tobytes()
        assert before.size == after.size
        assert before.mode == after.mode


def test_strip_metadata_errors_on_missing_src(tmp_path):
    with pytest.raises(FileNotFoundError):
        strip_metadata.strip_metadata(tmp_path / "does_not_exist.png", tmp_path / "dst.png")
