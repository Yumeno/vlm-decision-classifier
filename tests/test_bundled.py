"""E7 束ね質問(decide_bundled / pipeline mode="bundled" / 3モードの順序回転)。偽バックエンドのみ。"""

import math

from classifier_demo import decision, evaluate, pipeline
from classifier_demo.taxonomy import Axis, Choice, Taxonomy
from tests.fakes import FakeBackend


def _tax() -> Taxonomy:
    a = Axis(
        id="a1", question="qa", multi=False, allow_none=False,
        choices=[Choice(id="x", name="X", criteria="cx"), Choice(id="y", name="Y", criteria="cy")],
    )
    b = Axis(
        id="a2", question="qb", multi=False, allow_none=True,
        choices=[Choice(id="p", name="P", criteria="cp"), Choice(id="q", name="Q", criteria="cq")],
    )
    c = Axis(
        id="a3", question="qc", multi=True, allow_none=True,
        choices=[
            Choice(id="m", name="M", criteria="cm"),
            Choice(id="n", name="N", criteria="cn"),
            Choice(id="rest", name="rest", criteria="cr", catch_all=True),
        ],
    )
    return Taxonomy(version="t", axes=[a, b, c], sha256="s")


def _entry(token: str, probs: dict[str, float]) -> dict:
    return {
        "token": token,
        "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in probs.items()],
    }


def _resp(entries: list[dict]) -> dict:
    return {"choices": [{"message": {"content": ""}, "logprobs": {"content": entries}}], "usage": {}}


def _ans(k, token: str, probs: dict[str, float]) -> list[dict]:
    """1軸ぶんの出力 "<k>: <記号>\n"(番号・コロン・空白・記号・改行の5トークン)。答えは index 3。"""
    return [
        _entry(str(k), {str(k): 1.0}),
        _entry(":", {":": 1.0}),
        _entry(" ", {" ": 1.0}),
        _entry(token, probs),
        _entry("\n", {"\n": 1.0}),
    ]


def test_numbered_positions_and_scores_per_axis():
    entries = (
        _ans(1, "A", {"A": 0.8, "B": 0.2})
        + _ans(2, " B", {" B": 0.6, "A": 0.1, "C": 0.3})  # 空白付きラベル
        + _ans(3, "D", {"C": 0.7, "D": 0.2, "A": 0.1})
    )
    backend = FakeBackend([_resp(entries)])
    res = decision.decide_bundled(backend, b"img", "image/png", _tax(), rank_threshold=0.5)

    assert backend.request_count == 1
    axes = res["axes"]
    assert [axes[k]["position"] for k in ("a1", "a2", "a3")] == [3, 8, 13]
    assert axes["a1"]["selected"] == "x"
    assert math.isclose(axes["a1"]["relative_scores"]["x"], 0.8)
    # 位置8の top: " B"=0.6, A=0.1, C=0.3 -> A:p, B:q, C:none(空白付きは合算)
    assert axes["a2"]["selected"] == "q"
    assert math.isclose(axes["a2"]["relative_scores"]["q"], 0.6)
    assert math.isclose(axes["a2"]["relative_scores"]["p"], 0.1)
    assert math.isclose(axes["a2"]["relative_scores"]["__none__"], 0.3)


def test_second_axis_selected_by_argmax_of_its_own_position():
    entries = (
        _ans(1, "A", {"A": 1.0})
        + _ans(2, "B", {"B": 0.6, "A": 0.4})
        + _ans(3, "C", {"A": 0.6, "B": 0.3, "C": 0.1})
    )
    res = decision.decide_bundled(FakeBackend([_resp(entries)]), b"i", "image/png", _tax(), 0.5)
    axes = res["axes"]
    assert axes["a2"]["selected"] == "q"
    # multi 軸: m が閾値0.5以上 -> 採用
    assert axes["a3"]["candidates"] == ["m"]
    assert axes["a3"]["tags"] == ["m"]


def test_multi_axis_falls_back_like_confirm_off():
    # 閾値以上の非catch_allが無く、全体argmaxがcatch_all(C)ならcatch_allをタグにする
    entries = (
        _ans(1, "A", {"A": 1.0})
        + _ans(2, "A", {"A": 1.0})
        + _ans(3, "C", {"A": 0.2, "B": 0.2, "C": 0.6})
    )
    res = decision.decide_bundled(FakeBackend([_resp(entries)]), b"i", "image/png", _tax(), 0.7)
    assert res["axes"]["a3"]["candidates"] == []
    assert res["axes"]["a3"]["tags"] == ["rest"]


def test_skipped_middle_axis_does_not_shift_others():
    # 軸2が抜けても、軸3は番号で軸3に割り当てられる(E7aのずれの再発防止)
    entries = _ans(1, "A", {"A": 0.9, "B": 0.1}) + _ans(3, "A", {"A": 0.9, "B": 0.1})
    res = decision.decide_bundled(FakeBackend([_resp(entries)]), b"i", "image/png", _tax(), 0.5)
    axes = res["axes"]
    assert axes["a1"]["error"] is None and axes["a1"]["position"] == 3
    assert axes["a2"]["error"]["type"] == "bundled_format_error"
    assert axes["a3"]["error"] is None and axes["a3"]["position"] == 8


def test_out_of_order_numbers_assigned_by_number():
    entries = _ans(2, "B", {"B": 1.0}) + _ans(1, "A", {"A": 1.0}) + _ans(3, "A", {"A": 1.0})
    res = decision.decide_bundled(FakeBackend([_resp(entries)]), b"i", "image/png", _tax(), 0.5)
    axes = res["axes"]
    assert (axes["a1"]["position"], axes["a2"]["position"], axes["a3"]["position"]) == (8, 3, 13)
    assert axes["a1"]["selected"] == "x" and axes["a2"]["selected"] == "q"


def test_first_answer_wins_for_duplicate_number():
    entries = _ans(1, "A", {"A": 1.0}) + _ans(1, "B", {"B": 1.0})
    res = decision.decide_bundled(FakeBackend([_resp(entries)]), b"i", "image/png", _tax(), 0.5)
    assert res["axes"]["a1"]["position"] == 3 and res["axes"]["a1"]["selected"] == "x"


def test_colon_glued_to_label_is_format_error():
    # ":A" は1トークンにくっついているのでラベルと一致せず、黙って読み替えない
    entries = [
        _entry("1", {"1": 1.0}),
        _entry(":A", {":A": 1.0}),
        _entry("\n", {"\n": 1.0}),
    ]
    res = decision.decide_bundled(FakeBackend([_resp(entries)]), b"i", "image/png", _tax(), 0.5)
    for k in ("a1", "a2", "a3"):
        assert res["axes"][k]["error"]["type"] == "bundled_format_error"


def test_missing_labels_recorded_per_axis_without_forcing():
    entries = _ans(1, "A", {"A": 0.9, "B": 0.1}) + [_entry("hello", {"hello": 1.0})]
    res = decision.decide_bundled(FakeBackend([_resp(entries)]), b"i", "image/png", _tax(), 0.5)
    axes = res["axes"]
    assert axes["a1"]["error"] is None and axes["a1"]["selected"] == "x"
    assert axes["a2"]["error"]["type"] == "bundled_format_error"
    assert axes["a3"]["error"]["type"] == "bundled_format_error"
    assert "selected" not in axes["a2"]


def test_pipeline_bundled_one_request_and_choice_shaped_result(tmp_path):
    from PIL import Image

    img = tmp_path / "i.png"
    Image.new("RGB", (16, 16), (1, 2, 3)).save(img)
    entries = (
        _ans(1, "A", {"A": 0.8, "B": 0.2})
        + _ans(2, "B", {"A": 0.3, "B": 0.7})
        + _ans(3, "A", {"A": 0.9, "B": 0.05, "C": 0.05})
    )
    backend = FakeBackend([_resp(entries)])
    events = []
    result = pipeline.classify(str(img), _tax(), backend, mode="bundled", max_edge=64, on_progress=events.append)

    assert result["mode"] == "bundled"
    assert result["request_count"] == 1
    assert result["errors"] == []
    assert result["vision_tags"] == {"a1": ["x"], "a2": ["q"], "a3": ["m"]}
    assert set(result["axis_decisions"]) == {"a1", "a2", "a3"}
    assert [e["axis_id"] for e in events if e["type"] == "axis"] == ["a1", "a2", "a3"]


def test_pipeline_bundled_axis_errors_recorded(tmp_path):
    from PIL import Image

    img = tmp_path / "i.png"
    Image.new("RGB", (16, 16)).save(img)
    backend = FakeBackend([_resp(_ans(1, "A", {"A": 1.0}))])
    result = pipeline.classify(str(img), _tax(), backend, mode="bundled", max_edge=64)
    assert result["vision_tags"] == {"a1": ["x"]}
    assert [(e["axis"], e["type"]) for e in result["errors"]] == [
        ("a2", "bundled_format_error"),
        ("a3", "bundled_format_error"),
    ]


def test_three_mode_order_rotates_and_two_mode_alternates():
    modes = ["choice", "json", "bundled"]
    orders = [evaluate.case_mode_order(modes, i) for i in range(4)]
    assert orders[0] == ["choice", "json", "bundled"]
    assert orders[1] == ["json", "bundled", "choice"]
    assert orders[2] == ["bundled", "choice", "json"]
    assert orders[3] == orders[0]
    assert evaluate.case_mode_order(["choice", "json"], 0) == ["choice", "json"]
    assert evaluate.case_mode_order(["choice", "json"], 1) == ["json", "choice"]
    evaluate.validate_modes(modes)
