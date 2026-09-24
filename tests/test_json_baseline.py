from classifier_demo import json_baseline
from classifier_demo.taxonomy import Axis, Choice, Taxonomy
from tests.fakes import FakeBackend, make_text_response


def _small_taxonomy() -> Taxonomy:
    image_type = Axis(
        id="image_type",
        question="q",
        multi=False,
        allow_none=False,
        choices=[
            Choice(id="illustration", name="illustration", criteria="c"),
            Choice(id="comic", name="comic", criteria="c"),
        ],
    )
    character = Axis(
        id="character",
        question="q",
        multi=True,
        allow_none=True,
        choices=[
            Choice(id="alisa", name="Alisa", criteria="c"),
            Choice(id="other_original", name="other", criteria="c", catch_all=True),
        ],
    )
    return Taxonomy(version="test", axes=[image_type, character], sha256="deadbeef")


def test_classify_json_strips_fences_and_succeeds():
    text = '```json\n{"image_type": "illustration", "character": ["alisa"]}\n```'
    backend = FakeBackend([make_text_response(text)])
    tax = _small_taxonomy()
    result = json_baseline.classify_json(backend, b"img", "image/png", tax)
    assert result["tags"] == {"image_type": ["illustration"], "character": ["alisa"]}
    assert len(result["attempts"]) == 1
    assert result["error"] is None


def test_classify_json_unknown_value_is_format_error():
    text = '{"image_type": "not_a_real_choice", "character": []}'
    backend = FakeBackend([make_text_response(text)] * 3)
    tax = _small_taxonomy()
    result = json_baseline.classify_json(backend, b"img", "image/png", tax)
    assert result["tags"] is None
    assert result["error"] is not None
    assert len(result["attempts"]) == 3  # MAX_RETRIES=2 -> 3回とも失敗


def test_classify_json_missing_axis_is_format_error():
    text = '{"image_type": "illustration"}'
    backend = FakeBackend([make_text_response(text)] * 3)
    tax = _small_taxonomy()
    result = json_baseline.classify_json(backend, b"img", "image/png", tax)
    assert result["tags"] is None
    assert "character" in result["error"]


def test_classify_json_retry_then_success():
    bad_text = "not json at all"
    good_text = '{"image_type": "comic", "character": []}'
    backend = FakeBackend([make_text_response(bad_text), make_text_response(good_text)])
    tax = _small_taxonomy()
    result = json_baseline.classify_json(backend, b"img", "image/png", tax)
    assert result["tags"] == {"image_type": ["comic"], "character": []}
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["error"] is not None
    assert result["attempts"][1]["error"] is None


def test_classify_json_retries_exhausted_records_failure():
    backend = FakeBackend([make_text_response("garbage")] * 3)
    tax = _small_taxonomy()
    result = json_baseline.classify_json(backend, b"img", "image/png", tax)
    assert result["tags"] is None
    assert result["error"] is not None
    assert len(result["attempts"]) == 3
    assert all(a["error"] is not None for a in result["attempts"])
