import json
from pathlib import Path

from PIL import Image, PngImagePlugin

from classifier_demo import metadata
from classifier_demo.taxonomy import Axis, Choice, Taxonomy


def _character_taxonomy() -> Taxonomy:
    axis = Axis(
        id="character",
        question="Which character appears in this image?",
        multi=True,
        allow_none=True,
        choices=[
            Choice(
                id="alisa",
                name="Alisa",
                criteria="criteria",
                lora_names=["fet-alisa-uniform-anima-v4u"],
                trigger_words=["fet_alisa_uniform"],
            ),
            Choice(id="second_original", name="second", criteria="criteria"),
            Choice(id="other_original", name="other", criteria="criteria", catch_all=True),
        ],
    )
    return Taxonomy(version="test", axes=[axis], sha256="deadbeef")


def _save_png(path: Path, text_chunks: dict[str, str]) -> None:
    img = Image.new("RGB", (8, 8), (255, 0, 0))
    info = PngImagePlugin.PngInfo()
    for key, value in text_chunks.items():
        info.add_text(key, value)
    img.save(path, pnginfo=info)


def test_a1111_lora_exact_match(tmp_path):
    path = tmp_path / "a.png"
    params = (
        "masterpiece, <lora:fet-alisa-uniform-anima-v4u:0.8>, 1girl\n"
        "Negative prompt: bad quality\n"
        "Steps: 20, Sampler: Euler"
    )
    _save_png(path, {"parameters": params})
    tax = _character_taxonomy()
    evidence = metadata.extract_evidence(str(path), tax)
    assert evidence["format"] == "a1111"
    assert evidence["loras"] == [
        {"name": "fet-alisa-uniform-anima-v4u", "weight": 0.8, "source": "a1111"}
    ]
    matches = [m for m in evidence["matches"] if m["kind"] == "lora_name"]
    assert matches == [
        {"kind": "lora_name", "value": "fet-alisa-uniform-anima-v4u", "weight": 0.8, "character": "alisa"}
    ]


def test_a1111_lora_similar_name_no_match(tmp_path):
    path = tmp_path / "a.png"
    params = "masterpiece, <lora:fet_alisa_uniform_ilpen2:0.8>, 1girl\nSteps: 20"
    _save_png(path, {"parameters": params})
    tax = _character_taxonomy()
    evidence = metadata.extract_evidence(str(path), tax)
    lora_matches = [m for m in evidence["matches"] if m["kind"] == "lora_name"]
    assert lora_matches == []


def test_a1111_lora_uppercase_variant_matches(tmp_path):
    path = tmp_path / "a.png"
    params = "masterpiece, <lora:FET-ALISA-UNIFORM-ANIMA-V4U:0.8>, 1girl\nSteps: 20"
    _save_png(path, {"parameters": params})
    tax = _character_taxonomy()
    evidence = metadata.extract_evidence(str(path), tax)
    lora_matches = [m for m in evidence["matches"] if m["kind"] == "lora_name"]
    assert len(lora_matches) == 1
    assert lora_matches[0]["character"] == "alisa"


def test_comfyui_loraloader_match(tmp_path):
    path = tmp_path / "c.png"
    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": "Anima/fet_alisa_uniform/fet-alisa-uniform-anima-v4u.safetensors",
                "strength_model": 0.8,
            },
        },
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "masterpiece, 1girl"}},
    }
    _save_png(path, {"prompt": json.dumps(workflow)})
    tax = _character_taxonomy()
    evidence = metadata.extract_evidence(str(path), tax)
    assert evidence["format"] == "comfyui"
    assert evidence["loras"] == [
        {"name": "fet-alisa-uniform-anima-v4u", "weight": 0.8, "source": "comfyui"}
    ]
    lora_matches = [m for m in evidence["matches"] if m["kind"] == "lora_name"]
    assert lora_matches[0]["character"] == "alisa"


def test_trigger_word_exact_vs_suffixed(tmp_path):
    path_exact = tmp_path / "exact.png"
    _save_png(path_exact, {"parameters": "1girl, fet_alisa_uniform, outdoors\nSteps: 20"})
    path_suffixed = tmp_path / "suffixed.png"
    _save_png(path_suffixed, {"parameters": "1girl, fet_alisa_uniform_x, outdoors\nSteps: 20"})

    tax = _character_taxonomy()

    evidence_exact = metadata.extract_evidence(str(path_exact), tax)
    trigger_matches = [m for m in evidence_exact["matches"] if m["kind"] == "prompt_trigger"]
    assert len(trigger_matches) == 1
    assert trigger_matches[0]["character"] == "alisa"

    evidence_suffixed = metadata.extract_evidence(str(path_suffixed), tax)
    trigger_matches_suffixed = [m for m in evidence_suffixed["matches"] if m["kind"] == "prompt_trigger"]
    assert trigger_matches_suffixed == []


def test_no_metadata_format_none(tmp_path):
    path = tmp_path / "plain.png"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(path)
    tax = _character_taxonomy()
    evidence = metadata.extract_evidence(str(path), tax)
    assert evidence["format"] == "none"
    assert evidence["matches"] == []
