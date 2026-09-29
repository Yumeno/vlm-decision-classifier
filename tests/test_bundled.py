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


# ---- E7e: 複数選択軸を候補ごとの Y/N 欄にする(multi_mode="yn") ----


def _yn(k, letter_probs: dict[str, float]) -> list[dict]:
    """サブ質問 "<k>: <記号>\n"。k は "3a" のような文字列。"""
    return _ans(k, "A", letter_probs)


def test_yn_prompt_lists_sub_questions_for_multi_axis_only():
    prompt = decision._build_bundled_prompt(_tax().axes, "yn")
    assert "Question 1:\nqa\n" in prompt and "Question 2:\nqb\n" in prompt  # 単一選択は従来どおり
    for key, name, crit in (("3a", "M", "cm"), ("3b", "N", "cn"), ("3c", "rest", "cr")):  # catch_all も含む
        assert f"Question {key}:\nIs {name} ({crit}) present in the image?\n\nA. yes\nB. no" in prompt
    assert "Question 3:" not in prompt and "qc" not in prompt
    # 回答の雛形が欄の番号順に全行入る
    template = decision.BUNDLED_TEMPLATE_INSTRUCTION + "\n" + "\n".join(
        f"{k}: ?" for k in ("1", "2", "3a", "3b", "3c")
    )
    assert prompt.endswith(template)
    # rank もサブ質問なし・軸数ぶんの雛形(1: ? 〜 3: ?)で終わる
    rank_prompt = decision._build_bundled_prompt(_tax().axes, "rank")
    rank_template = decision.BUNDLED_TEMPLATE_INSTRUCTION + "\n" + "\n".join(f"{k}: ?" for k in ("1", "2", "3"))
    assert "3a" not in rank_prompt and rank_prompt.endswith(rank_template)


def test_yn_reads_p_yes_and_adopts_multiple():
    entries = (
        _ans(1, "A", {"A": 0.8, "B": 0.2})
        + _ans(2, "B", {"A": 0.3, "B": 0.7})
        + _yn("3a", {"A": 0.99, "B": 0.01})
        + _yn("3b", {" A": 0.6, "B": 0.4})  # 空白付きラベルも合算
        + _yn("3c", {"A": 0.1, "B": 0.9})
    )
    backend = FakeBackend([_resp(entries)])
    res = decision.decide_bundled(backend, b"i", "image/png", _tax(), multi_mode="yn")
    assert backend.request_count == 1
    a3 = res["axes"]["a3"]
    assert a3["error"] is None
    assert math.isclose(a3["confirmations"]["m"], 0.99)
    assert math.isclose(a3["confirmations"]["n"], 0.6)
    assert math.isclose(a3["confirmations"]["rest"], 0.1)
    assert a3["tags"] == ["m", "n"]  # 2つ同時に採用(P(yes)降順)
    assert a3["candidates"] == ["m", "n", "rest"]
    assert res["axes"]["a1"]["selected"] == "x" and res["axes"]["a2"]["selected"] == "q"


def test_yn_catch_all_adopted_as_normal_candidate_and_empty_when_none():
    entries = (
        _ans(1, "A", {"A": 1.0}) + _ans(2, "A", {"A": 1.0})
        + _yn("3a", {"A": 0.2, "B": 0.8}) + _yn("3b", {"A": 0.2, "B": 0.8}) + _yn("3c", {"A": 0.7, "B": 0.3})
    )
    res = decision.decide_bundled(FakeBackend([_resp(entries)]), b"i", "image/png", _tax(), multi_mode="yn")
    assert res["axes"]["a3"]["tags"] == ["rest"]
    entries[-5:] = _yn("3c", {"A": 0.3, "B": 0.7})
    res = decision.decide_bundled(FakeBackend([_resp(entries)]), b"i", "image/png", _tax(), multi_mode="yn")
    assert res["axes"]["a3"]["tags"] == []  # 全部 no なら空(catch_all へのフォールバックもしない)


def test_yn_missing_sub_question_is_axis_format_error():
    entries = (
        _ans(1, "A", {"A": 1.0}) + _ans(2, "A", {"A": 1.0})
        + _yn("3a", {"A": 0.9, "B": 0.1}) + _yn("3c", {"A": 0.9, "B": 0.1})  # 3b が抜けた
    )
    res = decision.decide_bundled(FakeBackend([_resp(entries)]), b"i", "image/png", _tax(), multi_mode="yn")
    err = res["axes"]["a3"]["error"]
    assert err["type"] == "bundled_format_error" and "3b=n" in err["detail"]
    assert res["axes"]["a1"]["error"] is None  # 他の軸には影響しない


def test_yn_max_tokens_scales_with_fields():
    entries = _ans(1, "A", {"A": 1.0})
    class Recording(FakeBackend):
        max_tokens: list = []

        def chat(self, messages, **params):
            self.max_tokens.append(params["max_tokens"])
            return super().chat(messages, **params)

    backend = Recording([_resp(entries), _resp(entries)])
    backend.max_tokens = []
    decision.decide_bundled(backend, b"i", "image/png", _tax(), multi_mode="yn")
    decision.decide_bundled(backend, b"i", "image/png", _tax(), multi_mode="rank")
    assert backend.max_tokens == [5 * (1 + 1 + 3), 5 * 3]


def test_rank_is_default_and_unchanged_by_sub_numbers():
    # rank では "3a:" のような番号は無視され、従来どおり "3:" だけが軸3の答え
    entries = _ans(1, "A", {"A": 1.0}) + _ans(2, "A", {"A": 1.0}) + _yn("3a", {"A": 1.0}) + _ans(3, "A", {"A": 0.9, "B": 0.1})
    res = decision.decide_bundled(FakeBackend([_resp(entries)]), b"i", "image/png", _tax(), 0.5)
    assert res["axes"]["a3"]["position"] == 18 and res["axes"]["a3"]["tags"] == ["m"]


def test_pipeline_bundled_yn_records_confirmations_and_setting(tmp_path):
    from PIL import Image

    img = tmp_path / "i.png"
    Image.new("RGB", (16, 16), (1, 2, 3)).save(img)
    entries = (
        _ans(1, "A", {"A": 0.8, "B": 0.2}) + _ans(2, "B", {"A": 0.3, "B": 0.7})
        + _yn("3a", {"A": 0.99, "B": 0.01}) + _yn("3b", {"A": 0.98, "B": 0.02}) + _yn("3c", {"A": 0.1, "B": 0.9})
    )
    events = []
    result = pipeline.classify(
        str(img), _tax(), FakeBackend([_resp(entries)]), mode="bundled", max_edge=64,
        on_progress=events.append, bundled_multi="yn",
    )
    assert result["request_count"] == 1 and result["errors"] == []
    assert result["settings"]["bundled_multi"] == "yn"
    assert result["vision_tags"]["a3"] == ["m", "n"]
    assert result["axis_decisions"]["a3"]["confirmations"]["m"] > 0.9
    ev = [e for e in events if e["type"] == "axis" and e["axis_id"] == "a3"][0]
    assert ev["confirm"] is True and ev["multi_mode"] == "yn" and ev["candidates"] == ["m", "n", "rest"]
