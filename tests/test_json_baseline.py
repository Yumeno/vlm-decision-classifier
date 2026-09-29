import urllib.error

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


def test_classify_json_dedupes_multi_select_ids():
    text = '{"image_type": "illustration", "character": ["alisa", "alisa", "other_original"]}'
    backend = FakeBackend([make_text_response(text)])
    tax = _small_taxonomy()
    result = json_baseline.classify_json(backend, b"img", "image/png", tax)
    assert result["tags"]["character"] == ["alisa", "other_original"]


def test_classify_json_request_error_stops_without_retry():
    backend = FakeBackend([urllib.error.URLError("connection refused"), make_text_response("unused")])
    tax = _small_taxonomy()
    result = json_baseline.classify_json(backend, b"img", "image/png", tax)
    assert result["tags"] is None
    assert "URLError" in result["error"]
    assert len(result["attempts"]) == 1  # 通信失敗は再試行しない


def test_build_prompt_example_uses_placeholder_not_real_choice_id():
    tax = _small_taxonomy()
    prompt = json_baseline._build_prompt(tax)
    assert "<id>" in prompt
    # 先頭選択肢(illustration/alisa)を回答例として誘導しない
    assert '"illustration"' not in prompt.split("shape:")[-1]
    assert '"alisa"' not in prompt.split("shape:")[-1]


def test_build_prompt_notes_none_criteria_for_multi_axis_with_none_criteria():
    image_type = Axis(
        id="image_type",
        question="q",
        multi=False,
        allow_none=False,
        choices=[Choice(id="illustration", name="illustration", criteria="c")],
    )
    character = Axis(
        id="character",
        question="q",
        multi=True,
        allow_none=True,
        none_criteria="no character appears in the image",
        choices=[
            Choice(id="alisa", name="Alisa", criteria="c"),
            Choice(id="other_original", name="other", criteria="c", catch_all=True),
        ],
    )
    tax = Taxonomy(version="test", axes=[image_type, character], sha256="deadbeef")
    prompt = json_baseline._build_prompt(tax)
    assert "(use an empty list only if: no character appears in the image)" in prompt
    # none_criteria が無い軸には注記を付けない
    assert prompt.count("use an empty list only if") == 1


class _CapturingBackend:
    def __init__(self, response):
        self.payloads = []
        self._response = response

    def chat(self, messages, **params):
        self.payloads.append((messages, params))
        return self._response, 1.0


def test_prime_shares_prefix_with_classify_json():
    # キャッシュは先頭一致。system文と画像部分(text より前)が classify_json と同一であること。
    text = '{"image_type": "illustration", "character": []}'
    backend = _CapturingBackend(make_text_response(text))
    json_baseline.classify_json(backend, b"img", "image/png", _small_taxonomy())
    json_baseline.prime(backend, b"img", "image/png")
    (cls_msgs, _), (prime_msgs, prime_params) = backend.payloads
    assert prime_msgs[0] == cls_msgs[0]  # system
    assert prime_msgs[1]["content"][0] == cls_msgs[1]["content"][0]  # 画像
    assert prime_params["max_tokens"] == 1


def test_build_json_schema_has_enum_for_all_axes():
    schema = json_baseline.build_json_schema(_small_taxonomy())
    assert schema["required"] == ["image_type", "character"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["image_type"]["enum"] == ["illustration", "comic"]
    char = schema["properties"]["character"]
    assert char["type"] == "array" and char["items"]["enum"] == ["alisa", "other_original"]


def test_classify_json_constrained_sends_response_format():
    text = '{"image_type": "comic", "character": []}'
    tax = _small_taxonomy()
    backend = FakeBackend([make_text_response(text)])
    result = json_baseline.classify_json(backend, b"img", "image/png", tax, constrained=True)
    assert result["tags"] == {"image_type": ["comic"], "character": []}
    rf = backend.calls[0]["params"]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"]["required"] == ["image_type", "character"]


def test_classify_json_unconstrained_has_no_response_format():
    backend = FakeBackend([make_text_response('{"image_type": "comic", "character": []}')])
    json_baseline.classify_json(backend, b"img", "image/png", _small_taxonomy())
    assert "response_format" not in backend.calls[0]["params"]
