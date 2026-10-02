"""DiffusionGemma 方式(dgemma_choice / dgemma_json)の純粋ロジック。偽バックエンドのみ、実モデルの出力は使わない。"""

import urllib.error

from PIL import Image

from classifier_demo import decision, dgemma, evaluate, pipeline
from classifier_demo.taxonomy import Axis, Choice, Taxonomy
from tests.fakes import FakeBackend, make_text_response


def _tax() -> Taxonomy:
    single = Axis(
        id="style", question="What style?", multi=False, allow_none=False,
        choices=[Choice(id="anime", name="anime style", criteria="cel"), Choice(id="other", name="other", criteria="rest")],
    )
    single_none = Axis(
        id="subj", question="What subject?", multi=False, allow_none=True, none_criteria="nothing",
        choices=[Choice(id="p", name="person", criteria="a person"), Choice(id="o", name="object", criteria="an item")],
    )
    multi = Axis(
        id="outfit", question="Which outfits?", multi=True, allow_none=True, none_criteria="no character",
        choices=[
            Choice(id="maid", name="maid outfit", criteria="apron"),
            Choice(id="swim", name="swimsuit", criteria="swimsuit"),
            Choice(id="rest", name="other clothing", criteria="other", catch_all=True),
        ],
    )
    return Taxonomy(version="t", axes=[single, single_none, multi], sha256="s")


class FakeDgemmaBackend(FakeBackend):
    structured_url = "http://fake:8011"
    samples = "auto"
    extra_body: dict = {}

    def systemone(self, body):
        self.request_count += 1
        self.calls.append({"body": body})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, 5.0


def _ok_response(**override) -> dict:
    answers = {
        "q0": {"type": "choice", "choice": "anime style", "probabilities": {"anime style": 0.8, "other": 0.2}, "confidence": 0.9},
        "q1": {
            "type": "choice", "choice": "none of the above", "confidence": 0.5,
            "probabilities": {"person": 0.1, "object": 0.2, "none of the above": 0.7},
        },
        "q2": {"type": "noul", "noul": 0.9},
        "q3": {"type": "noul", "noul": 0.5},
        "q4": {"type": "noul", "noul": 0.49},
    }
    answers.update(override)
    return {
        "model": "dgemma", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 3},
        "diagnostics": {"samples": {"n": 4}, "timing": {"total_ms": 123.0, "reads": 4}},
    }


def test_build_request_maps_axes_to_choice_and_noul_questions():
    body, plan = dgemma.build_request(_tax(), b"img", "image/jpeg", "dgemma", 1)
    qs = body["questions"]
    assert list(qs) == ["q0", "q1", "q2", "q3", "q4"]  # 短い連番ID。軸・選択肢idへは plan で戻す
    assert qs["q0"] == {
        "type": "choice", "instructions": "What style?", "criteria": {"anime style": "cel", "other": "rest"},
    }
    assert list(qs["q1"]["criteria"]) == ["person", "object", dgemma.NONE_NAME]  # allow_none は選択肢に足す
    assert qs["q1"]["criteria"][dgemma.NONE_NAME] == "nothing"
    assert qs["q2"] == {"type": "noul", "instructions": "Is maid outfit (apron) present in the image?"}
    assert body["samples"] == 1 and body["model"] == "dgemma"
    assert body["images"][0].startswith("data:image/jpeg;base64,")
    assert plan["outfit"]["qids"] == {"maid": "q2", "swim": "q3", "rest": "q4"}
    assert plan["subj"]["names"][dgemma.NONE_NAME] == decision.NONE_ID
    # 正解(メタデータ・LoRA名など)をモデルに渡す欄が無い
    assert set(body) == {"model", "state", "questions", "images", "samples"}


def test_build_request_rejects_duplicate_option_names():
    ax = Axis(id="a", question="q", multi=False, allow_none=False,
              choices=[Choice(id="x", name="same", criteria="1"), Choice(id="y", name="same", criteria="2")])
    try:
        dgemma.build_request(Taxonomy(version="t", axes=[ax], sha256="s"), b"i", "image/jpeg", "m")
    except decision.DecisionError as e:
        assert e.error_type == "request_error"
    else:
        raise AssertionError("expected DecisionError")


def test_parse_response_maps_back_to_choice_ids_and_thresholds_multi():
    tax = _tax()
    _, plan = dgemma.build_request(tax, b"i", "image/jpeg", "dgemma")
    res = dgemma.parse_response(tax, plan, _ok_response())
    assert res["style"]["selected"] == "anime"
    assert res["style"]["relative_scores"] == {"anime": 0.8, "other": 0.2}
    assert res["subj"]["selected"] == decision.NONE_ID
    m = res["outfit"]
    assert m["tags"] == ["maid", "swim"]  # >=0.5 を P(yes) 降順。0.49 は不採用(catch_all も通常候補)
    assert m["confirmations"]["rest"] == 0.49 and m["failed"] is False


def test_parse_response_marks_only_the_broken_axis_as_failed():
    tax = _tax()
    _, plan = dgemma.build_request(tax, b"i", "image/jpeg", "dgemma")
    resp = _ok_response(q3=None)
    del resp["answers"]["q0"]
    res = dgemma.parse_response(tax, plan, resp)
    assert res["style"]["error"]["type"] == "dgemma_format_error"
    assert res["outfit"]["error"]["type"] == "dgemma_format_error"
    assert res["subj"]["selected"] == decision.NONE_ID  # 他の軸は判定される


def test_parse_response_rejects_unknown_choice():
    tax = _tax()
    _, plan = dgemma.build_request(tax, b"i", "image/jpeg", "dgemma")
    bad = {"type": "choice", "choice": "zzz", "probabilities": {"anime style": 0.5, "other": 0.5}}
    res = dgemma.parse_response(tax, plan, _ok_response(q0=bad))
    assert "error" in res["style"]


def _image(tmp_path):
    p = tmp_path / "img.png"
    Image.new("RGB", (16, 16), (1, 2, 3)).save(p)
    return str(p)


def test_classify_dgemma_choice_records_tags_scores_and_server_diagnostics(tmp_path):
    backend = FakeDgemmaBackend([_ok_response()])
    result = pipeline.classify(_image(tmp_path), _tax(), backend, mode="dgemma_choice", max_edge=64)
    assert result["errors"] == []
    assert result["vision_tags"] == {"style": ["anime"], "subj": [], "outfit": ["maid", "swim"]}
    assert result["dgemma"]["samples_n"] == 4 and result["dgemma"]["reads"] == 4
    assert result["request_count"] == 1
    assert result["timing_ms"]["classification_wall_ms"] > 0 and result["timing_ms"]["dgemma_request_ms"] == 5.0


def test_classify_dgemma_choice_http_error_fails_all_axes_without_raising(tmp_path):
    err = decision.DecisionError("http_error", "HTTP 422: bad")
    backend = FakeDgemmaBackend([err])
    result = pipeline.classify(_image(tmp_path), _tax(), backend, mode="dgemma_choice", max_edge=64)
    assert result["vision_tags"] == {}  # 強制しない。失敗は分母に残る
    assert [e["axis"] for e in result["errors"]] == ["style", "subj", "outfit"]
    assert all(e["type"] == "http_error" for e in result["errors"])


def test_classify_dgemma_choice_connection_error_is_request_error(tmp_path):
    backend = FakeDgemmaBackend([urllib.error.URLError("refused")])
    result = pipeline.classify(_image(tmp_path), _tax(), backend, mode="dgemma_choice", max_edge=64)
    assert {e["type"] for e in result["errors"]} == {"request_error"}


def test_classify_dgemma_json_uses_plain_json_without_response_format(tmp_path):
    text = '{"style": "anime", "subj": "p", "outfit": ["maid"]}'
    backend = FakeDgemmaBackend([make_text_response(text)])
    result = pipeline.classify(_image(tmp_path), _tax(), backend, mode="dgemma_json", max_edge=64)
    assert result["errors"] == []
    assert result["vision_tags"]["outfit"] == ["maid"]
    assert "response_format" not in backend.calls[0]["params"]


def test_dgemma_modes_are_valid_and_backend_adds_thinking_off():
    evaluate.validate_modes(["dgemma_choice", "dgemma_json"])
    b = dgemma.DgemmaBackend(base_url="http://x/v1", model="dgemma", structured_url="http://y:8011/")
    assert b.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert b.structured_url == "http://y:8011"


def test_backend_chat_drops_temperature_but_keeps_max_tokens(monkeypatch):
    sent = {}

    def fake_post(self, path, payload):
        sent.update(payload)
        return {}

    monkeypatch.setattr(dgemma.ChatBackend, "_post", fake_post)
    b = dgemma.DgemmaBackend(base_url="http://x/v1", model="dgemma")
    b.chat([], temperature=0, max_tokens=256)
    assert "temperature" not in sent and sent["max_tokens"] == 256
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert b.dropped_params == ["temperature"]


def test_systemone_without_url_is_a_decision_error():
    b = dgemma.DgemmaBackend(base_url="http://x/v1", model="dgemma")
    try:
        b.systemone({})
    except decision.DecisionError as e:
        assert e.error_type == "request_error"
    else:
        raise AssertionError("expected DecisionError")
