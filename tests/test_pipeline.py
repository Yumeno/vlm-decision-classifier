import urllib.error
from pathlib import Path

from PIL import Image, PngImagePlugin

from classifier_demo import pipeline, taxonomy
from tests.fakes import FakeBackend, make_logprobs_response, make_no_logprobs_response


def _make_image_with_alisa_lora(path: Path) -> None:
    img = Image.new("RGB", (16, 16), (200, 100, 100))
    info = PngImagePlugin.PngInfo()
    info.add_text(
        "parameters",
        "masterpiece, <lora:fet-alisa-uniform-anima-v4u:0.8>, 1girl\nSteps: 20",
    )
    img.save(path, pnginfo=info)


def test_classify_choice_mode_records_axis_failure_and_succeeds_others(tmp_path):
    image_path = tmp_path / "img.png"
    _make_image_with_alisa_lora(image_path)

    tax = taxonomy.load("taxonomy/default.yaml")

    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),  # image_type -> illustration
            make_logprobs_response({"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}),  # art_style -> anime_2d
            make_logprobs_response(
                {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.05, "E": 0.03, "F": 0.02}
            ),  # subject -> person
            make_no_logprobs_response(),  # character axis fails
        ]
    )

    result = pipeline.classify(str(image_path), tax, backend, mode="choice", max_edge=64)

    assert result["mode"] == "choice"
    assert result["vision_tags"]["image_type"] == ["illustration"]
    assert result["vision_tags"]["art_style"] == ["anime_2d"]
    assert result["vision_tags"]["subject"] == ["person"]
    assert "character" not in result["vision_tags"]

    assert len(result["errors"]) == 1
    assert result["errors"][0]["axis"] == "character"
    assert result["errors"][0]["type"] == "no_logprobs"

    # メタデータで alisa の LoRA が一致しているが、character軸が失敗しているので unknown
    assert result["combined_evidence"] == {"alisa": "unknown"}
    assert result["metadata_evidence"]["format"] == "a1111"

    # 画像の絶対パスを結果に含めない
    assert result["image"]["name"] == "img.png"
    assert "\\" not in result["image"]["name"] and "/" not in result["image"]["name"]

    assert result["timing_ms"]["classification_wall_ms"] > 0
    assert result["request_count"] == 4
    assert "axis_decisions" in result
    assert "image_type" in result["axis_decisions"]


def test_classify_records_request_error_and_continues_other_axes(tmp_path):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (16, 16), (0, 0, 0)).save(image_path)
    tax = taxonomy.load("taxonomy/default.yaml")

    backend = FakeBackend(
        [
            urllib.error.URLError("connection refused"),  # image_type -> 通信失敗
            make_logprobs_response({"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}),  # art_style -> anime_2d
            make_logprobs_response(
                {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.05, "E": 0.03, "F": 0.02}
            ),  # subject -> person
            make_logprobs_response(
                {"A": 0.6, "B": 0.0001, "C": 0.1999, "D": 0.2}
            ),  # character ranking -> alisa top, second_original below floor
            make_logprobs_response({"A": 0.9, "B": 0.1}),  # alisa yes/no -> yes
        ]
    )

    result = pipeline.classify(str(image_path), tax, backend, mode="choice", max_edge=64)

    assert "image_type" not in result["vision_tags"]
    assert result["vision_tags"]["art_style"] == ["anime_2d"]
    assert any(e["axis"] == "image_type" and e["type"] == "request_error" for e in result["errors"])


def test_classify_image_error_is_recorded_without_crashing(tmp_path):
    missing_path = tmp_path / "does_not_exist.png"
    tax = taxonomy.load("taxonomy/default.yaml")
    backend = FakeBackend([])

    result = pipeline.classify(str(missing_path), tax, backend, mode="choice", max_edge=64)

    assert result["vision_tags"] == {}
    assert result["metadata_evidence"] is None
    assert any(e["type"] == "image_error" for e in result["errors"])
    assert result["timing_ms"]["classification_wall_ms"] > 0
    assert result["request_count"] == 0


def test_classify_character_confirmation_failure_excludes_from_vision_tags(tmp_path):
    # character軸の確認(yes/no)が1件失敗した場合、その軸は失敗扱いとなり、
    # vision_tags に character を入れない(catch_all へもフォールバックしない)。
    image_path = tmp_path / "img.png"
    _make_image_with_alisa_lora(image_path)
    tax = taxonomy.load("taxonomy/default.yaml")

    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),  # image_type
            make_logprobs_response({"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}),  # art_style
            make_logprobs_response(
                {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.05, "E": 0.03, "F": 0.02}
            ),  # subject
            make_logprobs_response(
                {"A": 0.6, "B": 0.0001, "C": 0.1999, "D": 0.2}
            ),  # character ranking -> alisa candidate only
            urllib.error.URLError("connection refused"),  # alisa yes/no -> 通信失敗
        ]
    )

    result = pipeline.classify(str(image_path), tax, backend, mode="choice", max_edge=64)

    assert "character" not in result["vision_tags"]
    assert result["axis_decisions"]["character"]["failed"] is True
    assert result["axis_decisions"]["character"]["tags"] == []
    # メタデータで alisa の LoRA が一致しているが、character軸が失敗しているので unknown
    assert result["combined_evidence"] == {"alisa": "unknown"}
    assert any(
        e["axis"] == "character" and e["type"] == "candidate_confirmation_error" for e in result["errors"]
    )


def test_classify_metadata_error_is_recorded_and_classification_continues(tmp_path, monkeypatch):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (16, 16), (0, 0, 0)).save(image_path)
    tax = taxonomy.load("taxonomy/default.yaml")

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "extract_evidence", _raise)

    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),  # image_type
            make_logprobs_response({"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}),  # art_style
            make_logprobs_response(
                {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.05, "E": 0.03, "F": 0.02}
            ),  # subject
            make_logprobs_response(
                {"A": 0.6, "B": 0.0001, "C": 0.1999, "D": 0.2}
            ),  # character ranking -> alisa candidate only
            make_logprobs_response({"A": 0.1, "B": 0.9}),  # alisa yes/no -> no
        ]
    )

    result = pipeline.classify(str(image_path), tax, backend, mode="choice", max_edge=64)

    assert result["metadata_evidence"] == {"format": "error", "loras": [], "prompt_tags": [], "matches": []}
    assert any(e["type"] == "metadata_error" for e in result["errors"])
    # メタデータ抽出が失敗しても分類自体は続行される
    assert result["vision_tags"]["image_type"] == ["illustration"]
