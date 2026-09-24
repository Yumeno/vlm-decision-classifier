import math

import pytest

from classifier_demo import decision
from classifier_demo.decision import (
    DecisionError,
    NONE_ID,
    choose,
    decide_multi_axis,
    extract_top_logprobs,
    pool_labels,
)
from classifier_demo.taxonomy import Axis, Choice
from tests.fakes import FakeBackend, make_logprobs_response
import urllib.error


def test_pool_labels_merges_token_variants():
    # "A" と " A" は同一ラベルとして合算される
    top_logprobs = [
        {"token": "A", "logprob": math.log(0.3)},
        {"token": " A", "logprob": math.log(0.3)},
        {"token": "B", "logprob": math.log(0.4)},
    ]
    scores = pool_labels(top_logprobs, ["A", "B"])
    # 合算後: A=0.6, B=0.4, 合計1.0なので再正規化しても同じ値
    assert scores["A"] == pytest.approx(0.6)
    assert scores["B"] == pytest.approx(0.4)


def test_pool_labels_renormalizes_over_listed_labels():
    top_logprobs = [
        {"token": "A", "logprob": math.log(0.6)},
        {"token": " A", "logprob": math.log(0.1)},
        {"token": "B", "logprob": math.log(0.2)},
        {"token": "C", "logprob": math.log(0.05)},
    ]
    scores = pool_labels(top_logprobs, ["A", "B", "C"])
    total = 0.7 + 0.2 + 0.05
    assert scores["A"] == pytest.approx(0.7 / total)
    assert scores["B"] == pytest.approx(0.2 / total)
    assert scores["C"] == pytest.approx(0.05 / total)
    assert sum(scores.values()) == pytest.approx(1.0)


def test_pool_labels_zero_mass_raises_with_observed_tokens():
    top_logprobs = [{"token": t, "logprob": math.log(0.1)} for t in ["X", "Y", "Z", "W", "V", "U"]]
    with pytest.raises(DecisionError) as exc_info:
        pool_labels(top_logprobs, ["A", "B"])
    assert exc_info.value.error_type == "no_label_tokens"
    assert "'X'" in exc_info.value.detail


def test_pool_labels_non_finite_mass_raises():
    # logprob が極端に大きいと exp() が inf になり得る -> 非有限値も no_label_tokens 扱い
    top_logprobs = [{"token": "A", "logprob": 1e6}]
    with pytest.raises(DecisionError) as exc_info:
        pool_labels(top_logprobs, ["A", "B"])
    assert exc_info.value.error_type == "no_label_tokens"


def test_extract_top_logprobs_thinking_detected():
    response = {
        "choices": [
            {
                "message": {"content": "", "reasoning_content": "let me think"},
                "logprobs": {"content": []},
            }
        ],
        "usage": {},
    }
    with pytest.raises(DecisionError) as exc_info:
        extract_top_logprobs(response)
    assert exc_info.value.error_type == "thinking_before_answer"


def test_extract_top_logprobs_thinking_via_usage_tokens():
    response = {
        "choices": [{"message": {"content": ""}, "logprobs": {"content": []}}],
        "usage": {"completion_tokens_details": {"reasoning_tokens": 5}},
    }
    with pytest.raises(DecisionError) as exc_info:
        extract_top_logprobs(response)
    assert exc_info.value.error_type == "thinking_before_answer"


def test_extract_top_logprobs_no_logprobs_without_thinking():
    response = {
        "choices": [{"message": {"content": ""}, "logprobs": {"content": []}}],
        "usage": {},
    }
    with pytest.raises(DecisionError) as exc_info:
        extract_top_logprobs(response)
    assert exc_info.value.error_type == "no_logprobs"


def _character_axis() -> Axis:
    return Axis(
        id="character",
        question="Which character appears in this image?",
        multi=True,
        allow_none=True,
        choices=[
            Choice(id="alisa", name="Alisa", criteria="alisa criteria"),
            Choice(id="second_original", name="Second original", criteria="second criteria"),
            Choice(id="other_original", name="other", criteria="other criteria", catch_all=True),
        ],
    )


def test_choose_selects_argmax():
    backend = FakeBackend([make_logprobs_response({"A": 0.7, "B": 0.2, "C": 0.1})])
    choices = [
        Choice(id="x", name="x", criteria="x"),
        Choice(id="y", name="y", criteria="y"),
        Choice(id="z", name="z", criteria="z"),
    ]
    result = choose(backend, b"img", "image/png", "q?", choices, allow_none=False)
    assert result["selected"] == "x"
    assert result["relative_scores"]["x"] == pytest.approx(0.7)


def test_decide_multi_axis_candidate_filter_and_threshold():
    axis = _character_axis()
    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.5, "B": 0.3, "C": 0.15, "D": 0.05}),  # ranking
            make_logprobs_response({"A": 0.9, "B": 0.1}),  # alisa yes/no -> yes
            make_logprobs_response({"A": 0.2, "B": 0.8}),  # second_original yes/no -> no
        ]
    )
    result = decide_multi_axis(backend, b"img", "image/png", axis)
    assert set(result["candidates"]) == {"alisa", "second_original"}
    assert result["tags"] == ["alisa"]


def test_decide_multi_axis_floor_excludes_low_scoring_candidate():
    axis = _character_axis()
    # second_original のスコアが alisa の0.001倍未満 -> 候補から除外され、yes/noを呼ばない
    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.5, "B": 0.0001, "C": 0.0999, "D": 0.4}),  # ranking
            make_logprobs_response({"A": 0.9, "B": 0.1}),  # alisa yes/no -> yes
        ]
    )
    result = decide_multi_axis(backend, b"img", "image/png", axis)
    assert result["candidates"] == ["alisa"]
    assert result["tags"] == ["alisa"]


def test_decide_multi_axis_catch_all_fallback():
    axis = _character_axis()
    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.2, "B": 0.1, "C": 0.6, "D": 0.1}),  # ranking: other_original top among non-none
            make_logprobs_response({"A": 0.1, "B": 0.9}),  # alisa yes/no -> no
            make_logprobs_response({"A": 0.1, "B": 0.9}),  # second_original yes/no -> no
        ]
    )
    result = decide_multi_axis(backend, b"img", "image/png", axis)
    assert result["tags"] == ["other_original"]


def test_decide_multi_axis_none_top_no_fallback():
    axis = _character_axis()
    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.1, "B": 0.05, "C": 0.05, "D": 0.8}),  # ranking: none top, alisa top among non-none
            make_logprobs_response({"A": 0.1, "B": 0.9}),  # alisa yes/no -> no
            make_logprobs_response({"A": 0.1, "B": 0.9}),  # second_original yes/no -> no
        ]
    )
    result = decide_multi_axis(backend, b"img", "image/png", axis)
    assert result["tags"] == []
    assert result["relative_scores"][NONE_ID] == pytest.approx(0.8)


def test_decide_multi_axis_none_top_overrides_catch_all_among_non_none():
    # catch_all(other_original) は __none__ を除く候補の中では最高スコアだが、
    # __none__ を含めた全体では __none__ が最高 -> catch_all フォールバックは適用しない。
    axis = _character_axis()
    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.05, "B": 0.05, "C": 0.3, "D": 0.6}),  # ranking
            make_logprobs_response({"A": 0.1, "B": 0.9}),  # alisa yes/no -> no
            make_logprobs_response({"A": 0.1, "B": 0.9}),  # second_original yes/no -> no
        ]
    )
    result = decide_multi_axis(backend, b"img", "image/png", axis)
    assert result["tags"] == []


def test_decide_multi_axis_candidate_confirmation_error_recorded():
    axis = _character_axis()
    backend = FakeBackend(
        [
            make_logprobs_response({"A": 0.5, "B": 0.3, "C": 0.15, "D": 0.05}),  # ranking
            urllib.error.URLError("connection refused"),  # alisa yes/no -> 通信失敗
            make_logprobs_response({"A": 0.9, "B": 0.1}),  # second_original yes/no -> yes
        ]
    )
    result = decide_multi_axis(backend, b"img", "image/png", axis)
    assert "alisa" not in result["confirmations"]
    assert "alisa" in result["confirmation_errors"]
    assert "URLError" in result["confirmation_errors"]["alisa"]
    assert result["tags"] == ["second_original"]
