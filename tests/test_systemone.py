import math

import pytest
from PIL import Image

from classifier_demo.systemone import SystemOneClient, SystemOneError
from tests.fakes import FakeBackend, make_logprobs_response, make_no_logprobs_response, make_thinking_response


def client(responses):
    backend = FakeBackend(responses)
    return SystemOneClient(backend=backend), backend


CHOICE = {"type": "choice", "instructions": "Dept?", "criteria": {"returns": "r", "billing": {"x": 1}, "other": None}}


def test_choice_labels_and_mapping_back():
    c, b = client([make_logprobs_response({"A": 0.2, "B": 0.6, "C": 0.2})])
    r = c.system_one("s", {"q": CHOICE})
    a = r.choices["q"]
    assert a.choice == "billing"
    assert list(a.probabilities) == ["returns", "billing", "other"]
    prompt = b.calls[0]["messages"][1]["content"][-1]["text"]
    assert "A. returns - r" in prompt and 'B. billing - {"x": 1}' in prompt and "C. other" in prompt
    assert b.calls[0]["params"]["max_tokens"] == 1 and b.calls[0]["params"]["top_logprobs"] == 20


def test_probabilities_renormalized_over_labels():
    # 列挙外トークン(Z)の質量は捨てて A/B 内で再正規化する
    c, _ = client([make_logprobs_response({"A": 0.3, "B": 0.1, "Z": 0.6})])
    p = c.system_one("s", {"q": {"type": "choice", "criteria": {"x": None, "y": None}}}).choices["q"].probabilities
    assert p["x"] == pytest.approx(0.75) and p["y"] == pytest.approx(0.25)
    assert sum(p.values()) == pytest.approx(1.0)


def test_noul_p_yes():
    c, _ = client([make_logprobs_response({"A": 0.9, "B": 0.1})])
    assert c.system_one("s", {"q": {"type": "noul", "instructions": "ok?"}}).nouls["q"].noul == pytest.approx(0.9)


def test_score_expected_value():
    c, _ = client([make_logprobs_response({"0": 0.1, "1": 0.1, "2": 0.8})])
    a = c.system_one("s", {"q": {"type": "score", "criteria": ["low", "mid", "high"]}}).scores["q"]
    assert a.score == pytest.approx(1.7)
    assert a.probabilities[2] == pytest.approx(0.8) and a.legend[0] == "low"


def test_too_many_labels_is_error():
    crit = {f"k{i}": None for i in range(53)}
    c, b = client([])
    with pytest.raises(SystemOneError, match="52"):
        c.system_one("s", {"q": {"type": "choice", "criteria": crit}})
    assert b.request_count == 0


def test_52_labels_ok():
    crit = {f"k{i}": None for i in range(52)}
    c, _ = client([make_logprobs_response({"a": 1.0})])
    assert c.system_one("s", {"q": {"type": "choice", "criteria": crit}}).choices["q"].choice == "k26"


def test_unknown_type_and_empty_questions():
    c, _ = client([])
    with pytest.raises(SystemOneError, match="unknown type"):
        c.system_one("s", {"q": {"type": "rank"}})
    with pytest.raises(SystemOneError):
        c.system_one("s", {})


@pytest.mark.parametrize("resp", [make_logprobs_response({"Z": 1.0}), make_thinking_response(), make_no_logprobs_response()])
def test_failures_are_explicit(resp):
    c, _ = client([resp])
    with pytest.raises(SystemOneError, match="'q'"):
        c.system_one("s", {"q": CHOICE})


def test_prefix_identical_across_questions(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(img)
    c, b = client([make_logprobs_response({"A": 1.0, "B": 1.0}), make_logprobs_response({"A": 1.0, "B": 1.0})])
    c.system_one({"k": "日本語"}, {"q1": {"type": "noul"}, "q2": CHOICE | {"criteria": {"a": None, "b": None}}}, images=[str(img)])
    m1, m2 = (call["messages"] for call in b.calls)
    assert m1[0] == m2[0] and m1[0]["role"] == "system"
    c1, c2 = m1[1]["content"], m2[1]["content"]
    assert c1[:-1] == c2[:-1] and c1[-1] != c2[-1]
    assert [p["type"] for p in c1[:-1]] == ["image_url", "text"]
    assert '{"k": "日本語"}' in c1[1]["text"]


def test_prime_recorded_separately():
    c, b = client([make_logprobs_response({"A": 1.0, "B": 1.0}), make_logprobs_response({"A": 0.5, "B": 0.5})])
    r = c.system_one("s", {"q": {"type": "noul"}}, prime=True)
    assert b.request_count == 2 and r.usage.request_count == 2
    assert r.usage.prime_elapsed_ms == 1.0 and r.usage.total_elapsed_ms == 1.0
    assert "logprobs" not in b.calls[0]["params"]


def test_control_chars_stripped_and_length_error():
    c, b = client([make_logprobs_response({"A": 1.0, "B": 1.0})])
    c.system_one("a\x00b", {"q": {"type": "noul"}})
    assert "State:\nab" in b.calls[0]["messages"][1]["content"][0]["text"]
    with pytest.raises(SystemOneError, match="too long"):
        c.system_one("x" * 200_000, {"q": {"type": "noul"}})
