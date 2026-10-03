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
    structured_url = "http://fake:8012"
    samples = 1
    seed = 0
    template = "keyed"
    max_per_read = 0
    max_soft_tokens = None
    instruction = "default"
    yn_style = "slash"
    order = "taxonomy"
    adaptive_threshold = None
    adaptive_max = 3
    steps = 1
    catchall_style = "default"
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
        "style": {"type": "choice", "choice": "anime style", "probabilities": {"anime style": 0.8, "other": 0.2}, "confidence": 0.9},
        "subj": {
            "type": "choice", "choice": "none of the above", "confidence": 0.5,
            "probabilities": {"person": 0.1, "object": 0.2, "none of the above": 0.7},
        },
        "outfit_maid": {"type": "noul", "noul": 0.9},
        "outfit_swim": {"type": "noul", "noul": 0.5},
        "outfit_rest": {"type": "noul", "noul": 0.49},
    }
    answers.update(override)
    return {
        "model": "dgemma", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 3},
        "errors": {},
        "diagnostics": {
            "samples": 1, "total_ms": 123.0, "tokenize_ms": 0.0,
            "reads": [{"questions": [], "canvas_width": 128, "ms_per_sample": [100.0], "prompt_tokens": 50, "cached_tokens": [None]}],
        },
    }


def test_build_request_maps_axes_to_choice_and_noul_questions():
    body, plan = dgemma.build_request(_tax(), b"img", "image/jpeg", "dgemma", 2, 7, "numbered", 3, 140)
    qs = body["questions"]
    # 読みやすいID: 単一選択=軸id、複数選択の選択肢=<軸id>_<選択肢id>
    assert list(qs) == ["style", "subj", "outfit_maid", "outfit_swim", "outfit_rest"]
    assert qs["style"] == {
        "type": "choice", "instructions": "What style?", "criteria": {"anime style": "cel", "other": "rest"},
    }
    assert list(qs["subj"]["criteria"]) == ["person", "object", dgemma.NONE_NAME]  # allow_none は選択肢に足す
    assert qs["subj"]["criteria"][dgemma.NONE_NAME] == "nothing"
    assert qs["outfit_maid"] == {"type": "noul", "instructions": "Is maid outfit (apron) present in the image?", "subject": "maid outfit (apron)"}
    assert body["samples"] == 2 and body["seed"] == 7 and body["template"] == "numbered" and body["max_per_read"] == 3
    assert body["model"] == "dgemma" and body["mm_processor_kwargs"] == {"max_soft_tokens": 140}
    assert body["images"][0].startswith("data:image/jpeg;base64,")
    assert plan["outfit"]["qids"] == {"maid": "outfit_maid", "swim": "outfit_swim", "rest": "outfit_rest"}
    assert plan["subj"]["names"][dgemma.NONE_NAME] == decision.NONE_ID
    # 正解(メタデータ・LoRA名など)をモデルに渡す欄が無い。mm_processor_kwargs は指定時のみ
    body2, _ = dgemma.build_request(_tax(), b"img", "image/jpeg", "dgemma")
    assert set(body2) == {"model", "state", "questions", "images", "samples", "seed", "template", "max_per_read"}
    assert body2["samples"] == 1 and body2["seed"] == 0 and body2["template"] == "keyed" and body2["max_per_read"] == 0


def test_build_request_rejects_ids_that_violate_the_question_id_pattern():
    ax = Axis(id="Bad.Axis", question="q", multi=False, allow_none=False,
              choices=[Choice(id="x", name="a", criteria="1"), Choice(id="y", name="b", criteria="2")])
    multi = Axis(id="outfit", question="q", multi=True, allow_none=False,
                 choices=[Choice(id="swim.suit", name="a", criteria="1")])
    for axis in (ax, multi):
        try:
            dgemma.build_request(Taxonomy(version="t", axes=[axis], sha256="s"), b"i", "image/jpeg", "m")
        except decision.DecisionError as e:
            assert e.error_type == "request_error" and "must match" in e.detail
        else:
            raise AssertionError("expected DecisionError")


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
    resp = _ok_response(outfit_swim=None)
    del resp["answers"]["style"]
    res = dgemma.parse_response(tax, plan, resp)
    assert res["style"]["error"]["type"] == "dgemma_format_error"
    assert res["outfit"]["error"]["type"] == "dgemma_format_error"
    assert res["subj"]["selected"] == decision.NONE_ID  # 他の軸は判定される


def test_parse_response_rejects_unknown_choice():
    tax = _tax()
    _, plan = dgemma.build_request(tax, b"i", "image/jpeg", "dgemma")
    bad = {"type": "choice", "choice": "zzz", "probabilities": {"anime style": 0.5, "other": 0.5}}
    res = dgemma.parse_response(tax, plan, _ok_response(style=bad))
    assert "error" in res["style"]


def test_parse_response_server_errors_fail_the_axis_and_any_failed_multi_option_fails_the_axis():
    tax = _tax()
    _, plan = dgemma.build_request(tax, b"i", "image/jpeg", "dgemma")
    resp = _ok_response()
    resp["errors"] = {"outfit_swim": {"type": "label_mass_low", "detail": "x"}, "subj": {"type": "missing_label_logprob", "detail": "y"}}
    del resp["answers"]["outfit_swim"]
    res = dgemma.parse_response(tax, plan, resp)
    assert res["style"]["selected"] == "anime"
    assert "label_mass_low" in res["outfit"]["error"]["detail"]  # 1選択肢の失敗で軸ごと失敗
    assert "missing_label_logprob" in res["subj"]["error"]["detail"]


def _image(tmp_path):
    p = tmp_path / "img.png"
    Image.new("RGB", (16, 16), (1, 2, 3)).save(p)
    return str(p)


def test_classify_dgemma_choice_records_tags_scores_and_server_diagnostics(tmp_path):
    backend = FakeDgemmaBackend([_ok_response()])
    result = pipeline.classify(_image(tmp_path), _tax(), backend, mode="dgemma_choice", max_edge=64)
    assert result["errors"] == []
    assert result["vision_tags"] == {"style": ["anime"], "subj": [], "outfit": ["maid", "swim"]}
    d = result["dgemma"]
    assert d["samples"] == 1 and d["seed"] == 0 and d["template"] == "keyed" and d["max_per_read"] == 0
    assert d["reads_n"] == 1 and d["read_ms"] == [[100.0]] and d["cached_tokens"] == [[None]] and d["server_total_ms"] == 123.0
    body = backend.calls[0]["body"]
    assert body["seed"] == 0 and body["template"] == "keyed" and "mm_processor_kwargs" not in body
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
    b = dgemma.DgemmaBackend(base_url="http://x/v1", model="dgemma", structured_url="http://y:8012/")
    assert b.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert b.structured_url == "http://y:8012"


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


def test_build_request_sends_instruction_only_when_not_default():
    body, _ = dgemma.build_request(_tax(), b"i", "image/jpeg", "dgemma", instruction="strict")
    assert body["instruction"] == "strict"
    body2, _ = dgemma.build_request(_tax(), b"i", "image/jpeg", "dgemma")
    assert "instruction" not in body2


def test_build_request_subject_dedupe_yn_style_and_order():
    ax_single = Axis(id="s", question="q", multi=False, allow_none=False,
                     choices=[Choice(id="x", name="a", criteria="1"), Choice(id="y", name="b", criteria="2")])
    outfit = Axis(id="outfit", question="q", multi=True, allow_none=False,
                  choices=[Choice(id="swim", name="Swimsuit", criteria="swimsuit"), Choice(id="maid", name="maid", criteria="apron")])
    char = Axis(id="character", question="q", multi=True, allow_none=False, choices=[Choice(id="alisa", name="Alisa", criteria="a girl")])
    tax = Taxonomy(version="t", axes=[ax_single, outfit, char], sha256="s")
    body, _ = dgemma.build_request(tax, b"i", "image/jpeg", "m", yn_style="letters", order="character_first")
    assert list(body["questions"]) == ["s", "character_alisa", "outfit_swim", "outfit_maid"]
    assert body["yn_style"] == "letters"
    qs = body["questions"]
    assert qs["outfit_swim"]["subject"] == "Swimsuit"  # criteria が name と同じ(大小無視)なら name だけ
    assert qs["outfit_maid"]["subject"] == "maid (apron)"
    assert qs["outfit_maid"]["instructions"] == "Is maid (apron) present in the image?"  # 従来の文面のまま
    body2, plan = dgemma.build_request(tax, b"i", "image/jpeg", "m")
    assert list(body2["questions"]) == ["s", "outfit_swim", "outfit_maid", "character_alisa"] and "yn_style" not in body2
    assert set(plan) == {"s", "outfit", "character"}


def test_build_request_adaptive_fields_only_when_threshold_given():
    body, _ = dgemma.build_request(_tax(), b"i", "image/jpeg", "m", adaptive_threshold=0.6, adaptive_max=4)
    assert body["adaptive_threshold"] == 0.6 and body["adaptive_max"] == 4 and body["samples"] == 1
    body2, _ = dgemma.build_request(_tax(), b"i", "image/jpeg", "m")
    assert "adaptive_threshold" not in body2 and "adaptive_max" not in body2


def test_build_request_steps_only_when_gt_1():
    body, _ = dgemma.build_request(_tax(), b"i", "image/jpeg", "m", steps=4)
    assert body["steps"] == 4
    assert "steps" not in dgemma.build_request(_tax(), b"i", "image/jpeg", "m")[0]
    assert "steps" not in dgemma.build_request(_tax(), b"i", "image/jpeg", "m", steps=1)[0]


def test_parse_response_carries_uncertain_flags_without_changing_selection():
    tax = _tax()
    _, plan = dgemma.build_request(tax, b"i", "image/jpeg", "dgemma")
    plain = dgemma.parse_response(tax, plan, _ok_response())
    assert "uncertain" not in plain["style"] and "uncertain_options" not in plain["outfit"]
    flagged = _ok_response()
    flagged["answers"]["style"].update({"uncertain": True, "entropy": 0.9})
    flagged["answers"]["outfit_maid"].update({"uncertain": False, "entropy": 0.1})
    flagged["answers"]["outfit_swim"].update({"uncertain": True, "entropy": 0.99})
    res = dgemma.parse_response(tax, plan, flagged)
    assert res["style"]["uncertain"] is True and res["style"]["entropy"] == 0.9
    assert res["style"]["selected"] == plain["style"]["selected"]
    assert res["outfit"]["uncertain_options"] == ["swim"] and res["outfit"]["tags"] == plain["outfit"]["tags"]


def test_catchall_style_list_rewrites_only_catch_all_questions():
    import pathlib

    from classifier_demo.taxonomy import load

    tax = load(str(pathlib.Path(__file__).resolve().parent.parent / "taxonomy" / "default.yaml"))
    default, _ = dgemma.build_request(tax, b"i", "image/jpeg", "m")
    listed, _ = dgemma.build_request(tax, b"i", "image/jpeg", "m", catchall_style="list")
    assert dgemma.build_request(tax, b"i", "image/jpeg", "m", catchall_style="default")[0] == default  # 既定は従来どおり
    diff = {k for k in default["questions"] if default["questions"][k] != listed["questions"][k]}
    assert diff == {"outfit_other", "character_other_original"}  # catch_all だけが変わる
    assert listed["questions"]["outfit_other"]["instructions"] == (
        "Apart from sailor school uniform, school uniform, office wear, swimsuit, maid outfit, "
        "shrine maiden outfit and nun's habit, is there at least one other clothing "
        "(clothing whose type is recognizable but fits none of the above) in the image?"
    )
    assert listed["questions"]["character_other_original"]["instructions"] == (
        "Apart from Alisa and Second original character, is there at least one other character "
        "(any person or humanoid character who is neither Alisa nor the second original character) in the image?"
    )
    assert listed["questions"]["outfit_other"]["subject"].endswith(", apart from sailor school uniform, school uniform, office wear, swimsuit, maid outfit, shrine maiden outfit and nun's habit")


def test_catchall_style_list_joining_and_edge_cases():
    def axis(choices):
        return Taxonomy(version="t", axes=[Axis(id="ax", question="q", multi=True, allow_none=False, choices=choices)], sha256="s")

    def q(choices):
        body, _ = dgemma.build_request(axis(choices), b"i", "image/jpeg", "m", catchall_style="list")
        return body["questions"]["ax_z"]["instructions"]

    rest = Choice(id="z", name="Rest", criteria="rest", catch_all=True)
    one = [Choice(id="a", name="aa", criteria="1"), rest]
    two = [Choice(id="a", name="aa", criteria="1"), Choice(id="b", name="bb", criteria="2"), rest]
    three = two + [Choice(id="c", name="cc", criteria="3")]
    assert q(one) == "Apart from aa, is there at least one Rest in the image?"
    assert q(two).startswith("Apart from aa and bb, is there")
    assert q(three).startswith("Apart from aa, bb and cc, is there")
    # catch_all 同士は列挙しない。兄弟が無ければ従来の文面
    only = [Choice(id="a", name="aa", criteria="1", catch_all=True), Choice(id="z", name="Rest2", criteria="r", catch_all=True)]
    body, _ = dgemma.build_request(axis(only), b"i", "image/jpeg", "m", catchall_style="list")
    assert body["questions"]["ax_a"]["instructions"] == "Is aa (1) present in the image?"
    plain = [Choice(id="a", name="aa", criteria="1"), Choice(id="z", name="bb", criteria="2")]
    assert dgemma.build_request(axis(plain), b"i", "image/jpeg", "m", catchall_style="list")[0] == dgemma.build_request(axis(plain), b"i", "image/jpeg", "m")[0]
