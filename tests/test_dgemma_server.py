"""dgemma-server の純粋ロジック。/tokenize と vLLM の応答は偽物を注入する(実モデルの出力は使わない)。"""

import math
import random
import re

import pytest

from classifier_demo import dgemma_server as ds


class FakeTokenizer:
    """特殊トークン・単語・空白/記号1字ごとに1トークン。単語ごとに安定した id を振る。"""

    def __init__(self):
        self.ids: dict[str, int] = {}

    def __call__(self, text):
        out = []
        for piece in re.findall(r"<\|?\w+\|?>|\w+|\s|[^\w\s]", text):
            out.append(self.ids.setdefault(piece, len(self.ids) + 1))
        return out


def _questions():
    return [
        ds.Question("style", "choice", "What style?", [("anime", "cel"), ("photo", "real"), ("other", "")]),
        ds.Question("outfit_maid", "noul", "Maid?", []),
        ds.Question("outfit_swim", "noul", "Swim?", []),
    ]


def _body(**kw):
    body = {
        "model": "m",
        "state": "Classify.",
        "questions": {
            "style": {"type": "choice", "instructions": "What style?", "criteria": {"anime": "cel", "photo": "real", "other": ""}},
            "outfit_maid": {"type": "noul", "instructions": "Maid?"},
            "outfit_swim": {"type": "noul", "instructions": "Swim?"},
        },
        "images": ["data:image/jpeg;base64,AAAA"],
    }
    body.update(kw)
    return body


# ---- テンプレート


def test_build_template_keyed_and_numbered():
    tok = ds.Tokenizer(FakeTokenizer())
    qs = _questions()
    t = ds.build_template(tok, ds.read_keys("keyed", qs), qs)
    assert t.keys == ["style", "outfit_maid", "outfit_swim"]
    assert [s.qid for s in t.slots] == ["style", "outfit_maid", "outfit_swim"]
    assert set(t.slots[0].label_ids) == {"A", "B", "C"} and set(t.slots[1].label_ids) == {"yes", "no"}
    assert t.tokens[-1] == tok(ds.TURN_CLOSE_TEXT)[0]
    # 各スロット位置のトークンはベース(先頭ラベル)の id
    assert t.tokens[t.slots[0].pos] == t.slots[0].label_ids["A"]
    n = ds.build_template(tok, ds.read_keys("numbered", qs), qs)
    assert n.keys == ["Q1", "Q2", "Q3"]
    assert n.tokens != t.tokens


def test_build_template_detects_label_that_splits_into_two_tokens():
    base = FakeTokenizer()

    def tokenize(text):
        return base(text.replace("B\n", "B B\n"))  # ラベル B だけ2トークンになる

    qs = _questions()[:1]
    with pytest.raises(ds.RequestError) as e:
        ds.build_template(ds.Tokenizer(tokenize), ["style"], qs)
    assert e.value.status == 422 and "token count" in e.value.message


def test_build_template_detects_duplicate_label_ids_and_multiple_diff_positions():
    base = FakeTokenizer()

    def same_id(text):  # A も B も同じトークン列になる(区別できない)
        return base(text.replace("B", "A").replace("C", "A"))

    with pytest.raises(ds.RequestError):
        ds.build_template(ds.Tokenizer(same_id), ["style"], _questions()[:1])

    def two_diffs(text):  # ラベルによって行頭のキーも変わる
        ids = base(text)
        return [x + 1000 for x in ids[:3]] + ids[3:] if " C\n" in text else ids

    with pytest.raises(ds.RequestError) as e:
        ds.build_template(ds.Tokenizer(two_diffs), ["style"], _questions()[:1])
    assert "exactly one" in e.value.message or "distinct" in e.value.message


def test_build_template_detects_bpe_boundary_mismatch_with_whole_text():
    base = FakeTokenizer()

    def tokenize(text):
        ids = base(text)
        return ids + [999] if text.endswith(ds.TURN_CLOSE_TEXT) and len(text) > 40 else ids  # 全文だけ結果が違う

    with pytest.raises(ds.RequestError) as e:
        ds.build_template(ds.Tokenizer(tokenize), ["style"], _questions()[:1])
    assert e.value.status == 422 and "whole template" in e.value.message


def test_tokenizer_caches_and_counts_time_once():
    calls = []

    def fn(text):
        calls.append(text)
        return [1]

    tok = ds.Tokenizer(fn)
    tok("a")
    tok("a")
    assert calls == ["a"]


# ---- キャンバス幅・ノイズ


def test_canvas_width_rounds_up_to_32_and_errors_on_overflow():
    assert ds.canvas_width(10, 256) == 32
    assert ds.canvas_width(31, 256) == 32
    assert ds.canvas_width(32, 256) == 64  # +1 の PAD を残す
    assert ds.canvas_width(255, 256) == 256
    assert ds.canvas_width(50, 40 + 20) == 60  # 丸めが canvas を超えるなら canvas
    with pytest.raises(ds.RequestError) as e:
        ds.canvas_width(256, 256)
    assert e.value.status == 422


def test_build_canvas_is_deterministic_with_noise_only_in_slots():
    tok = ds.Tokenizer(FakeTokenizer())
    qs = _questions()
    t = ds.build_template(tok, ds.read_keys("keyed", qs), qs)
    c1 = ds.build_canvas(t, 64, random.Random("0:0:0"), 1000, 0)
    c2 = ds.build_canvas(t, 64, random.Random("0:0:0"), 1000, 0)
    c3 = ds.build_canvas(t, 64, random.Random("0:0:1"), 1000, 0)
    assert c1 == c2 and c1 != c3 and len(c1) == 64
    slot_pos = {s.pos for s in t.slots}
    for i, tid in enumerate(c1):
        if i not in slot_pos:
            assert tid == (t.tokens[i] if i < len(t.tokens) else 0)
        else:
            assert 0 <= tid < 1000
    assert all(x == 0 for x in c1[len(t.tokens):])


# ---- スロットの読み


def _content(slot_lps, n=40):
    """slot_lps: {pos: {token_id: logprob}} から logprobs.content を作る。"""
    content = [{"token": "token_id:5", "logprob": -0.1, "top_logprobs": []} for _ in range(n)]
    for pos, lps in slot_lps.items():
        content[pos] = {"token": "token_id:5", "logprob": -1, "top_logprobs": [{"token": f"token_id:{t}", "logprob": lp} for t, lp in lps.items()]}
    return content


SLOT = ds.Slot("q", 3, {"A": 11, "B": 12})


def test_read_slot_renormalizes_and_reports_mass():
    lp_a, lp_b = math.log(0.3), math.log(0.1)
    probs, mass, reason = ds.read_slot(_content({3: {11: lp_a, 12: lp_b, 99: -0.5}}), SLOT, ["A", "B"], 0.2)
    assert reason is None and mass == pytest.approx(0.4)
    assert probs["A"] == pytest.approx(0.75) and probs["B"] == pytest.approx(0.25)
    assert sum(probs.values()) == pytest.approx(1)


def test_read_slot_invalid_cases():
    assert ds.read_slot(_content({3: {11: -1.0}}), SLOT, ["A", "B"], 0.1)[2] == "missing_label_logprob"
    p, mass, reason = ds.read_slot(_content({3: {11: math.log(0.1), 12: math.log(0.1)}}), SLOT, ["A", "B"], 0.5)
    assert p is None and reason == "label_mass_low" and mass == pytest.approx(0.2)
    assert ds.read_slot(_content({}, n=2), SLOT, ["A", "B"], 0.5)[2] == "short_content"
    bad = [{"token": "x", "logprob": -1, "top_logprobs": [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -0.1}]}] * 5
    assert ds.read_slot(bad, SLOT, ["A", "B"], 0.5)[2] == "missing_label_logprob"  # token_id: 形式でないものは読まない


def test_aggregate_question_averages_valid_samples_only():
    q = ds.Question("style", "choice", "?", [("anime", ""), ("photo", "")])
    samples = [({"A": 0.8, "B": 0.2}, 0.9, None), (None, 0.1, "label_mass_low"), ({"A": 0.4, "B": 0.6}, 0.9, None)]
    ans, err = ds.aggregate_question(q, samples)
    assert err is None and ans["choice"] == "anime"
    assert ans["probabilities"] == {"anime": pytest.approx(0.6), "photo": pytest.approx(0.4)}
    n = ds.Question("o", "noul", "?", [])
    ans, _ = ds.aggregate_question(n, [({"yes": 0.2, "no": 0.8}, 1, None), ({"yes": 0.4, "no": 0.6}, 1, None)])
    assert ans == {"type": "noul", "noul": pytest.approx(0.3)}


def test_aggregate_question_all_invalid_is_an_error_not_an_answer():
    q = ds.Question("style", "choice", "?", [("a", ""), ("b", "")])
    ans, err = ds.aggregate_question(q, [(None, 0.1, "label_mass_low"), (None, None, "missing_label_logprob"), (None, 0.2, "label_mass_low")])
    assert ans is None and err["type"] == "label_mass_low" and "no valid sample" in err["detail"]


# ---- 分割・入力検証


def test_split_reads_keeps_request_order_and_zero_means_one_read():
    qs = [ds.Question(f"q{i}", "noul", "", []) for i in range(5)]
    assert [[q.qid for q in g] for g in ds.split_reads(qs, 0)] == [["q0", "q1", "q2", "q3", "q4"]]
    assert [[q.qid for q in g] for g in ds.split_reads(qs, 2)] == [["q0", "q1"], ["q2", "q3"], ["q4"]]


def test_parse_request_validation():
    r = ds.parse_request(_body(samples=3, seed=5, template="numbered", mm_processor_kwargs={"max_soft_tokens": 70}))
    assert [q.qid for q in r.questions] == ["style", "outfit_maid", "outfit_swim"]
    assert r.samples == 3 and r.seed == 5 and r.template == "numbered" and r.mm_processor_kwargs == {"max_soft_tokens": 70}
    assert r.questions[0].labels == ["A", "B", "C"] and r.questions[1].labels == ["yes", "no"]
    bad_bodies = [
        _body(template="free"),
        _body(samples=0),
        _body(samples=True),
        _body(max_per_read=-1),
        _body(images=[]),
        _body(images=["http://x"]),
        _body(mm_processor_kwargs=[1]),
        _body(questions={}),
        _body(questions={"Bad-Id": {"type": "noul", "instructions": ""}}),
        _body(questions={"1abc": {"type": "noul", "instructions": ""}}),
        _body(questions={"a" * 49: {"type": "noul", "instructions": ""}}),
        _body(questions={"q": {"type": "score", "instructions": ""}}),
        _body(questions={"q": {"type": "choice", "instructions": "", "criteria": {f"o{i}": "" for i in range(27)}}}),
        _body(questions={"q": {"type": "choice", "instructions": "", "criteria": {"only": ""}}}),
        "not a dict",
    ]
    for b in bad_bodies:
        with pytest.raises(ds.RequestError) as e:
            ds.parse_request(b)
        assert e.value.status == 400, b
    ok = ds.parse_request(_body(questions={"q": {"type": "choice", "instructions": "x", "criteria": {f"o{i}": "" for i in range(26)}}}))
    assert ok.questions[0].labels[-1] == "Z"


def test_parse_request_cleans_untrusted_text():
    q = {"q": {"type": "noul", "instructions": "a\x00b\nc‮" + "x" * 5000}}
    r = ds.parse_request(_body(questions=q))
    assert "\x00" not in r.questions[0].instructions and "\n" not in r.questions[0].instructions
    assert len(r.questions[0].instructions) <= 1000


# ---- エンジン全体(偽 vLLM)


class FakeVllm:
    """プロンプトのキャンバスを見て、各スロットに指定の分布を返す。"""

    def __init__(self, engine_holder, dist):
        self.holder, self.dist, self.payloads = engine_holder, dist, []

    def __call__(self, payload):
        self.payloads.append(payload)
        eng = self.holder["engine"]
        canvas = payload["vllm_xargs"]["diffusion_seed_canvas"]
        width = payload["vllm_xargs"]["diffusion_canvas_length"]
        assert len(canvas) == width
        block = payload["messages"][1]["content"][-1]["text"]
        slots = next(pl for pl in eng.last_plans if pl.block == block).template.slots
        content = [{"token": "token_id:5", "logprob": -0.1, "top_logprobs": []} for _ in range(width)]
        for slot in slots:
            probs = self.dist[slot.qid]
            content[slot.pos]["top_logprobs"] = [
                {"token": f"token_id:{slot.label_ids[lab]}", "logprob": math.log(p)} for lab, p in probs.items()
            ]
        return {
            "choices": [{"logprobs": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": width, "total_tokens": 100 + width, "prompt_tokens_details": {"cached_tokens": 64}},
        }


def _engine(dist, **cfg):
    holder = {}
    fake = FakeVllm(holder, dist)
    eng = ds.Engine(ds.Config(vocab_size=1000, **cfg), FakeTokenizer(), fake)
    orig = eng.plan_reads

    def plan_reads(req):
        eng.last_plans = orig(req)
        return eng.last_plans

    eng.plan_reads = plan_reads
    holder["engine"] = eng
    return eng, fake


DIST = {
    "style": {"A": 0.6, "B": 0.3, "C": 0.1},
    "outfit_maid": {"yes": 0.9, "no": 0.1},
    "outfit_swim": {"yes": 0.2, "no": 0.8},
}


def test_engine_one_read_all_questions_and_request_shape():
    eng, fake = _engine(DIST)
    out = eng.handle(_body(mm_processor_kwargs={"max_soft_tokens": 140}, seed=3))
    assert len(fake.payloads) == 1  # 全質問を1回で
    p = fake.payloads[0]
    assert p["logprobs"] is True and p["return_tokens_as_token_ids"] is True
    assert p["chat_template_kwargs"] == {"enable_thinking": False} and p["mm_processor_kwargs"] == {"max_soft_tokens": 140}
    assert p["vllm_xargs"]["diffusion_max_steps"] == 1 and p["vllm_xargs"]["diffusion_read_only"] is True
    for forbidden in ("temperature", "top_k", "top_p", "seed", "min_p"):
        assert forbidden not in p
    msgs = p["messages"]
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == ds.SYSTEM_PROMPT
    content = msgs[1]["content"]
    assert content[0]["type"] == "image_url" and content[1]["text"] == "Classify."  # 画像が先頭、次に state
    assert content[2]["text"].startswith("style: What style?\n  A: anime (cel)\n  B: photo (real)\n  C: other\n")
    assert content[2]["text"].endswith("outfit_swim: Swim?\n  yes / no")
    assert out["answers"]["style"]["choice"] == "anime"
    assert out["answers"]["style"]["probabilities"]["anime"] == pytest.approx(0.6)
    assert out["answers"]["outfit_maid"]["noul"] == pytest.approx(0.9)
    assert out["errors"] == {}
    d = out["diagnostics"]
    assert d["template"] == "keyed" and d["seed"] == 3 and d["samples"] == 1 and len(d["reads"]) == 1
    assert d["reads"][0]["canvas_width"] % 32 == 0 and d["reads"][0]["cached_tokens"] == [64]
    assert d["label_mass"]["style"] == [pytest.approx(1.0)]
    assert out["usage"]["prompt_tokens"] == 100


def test_engine_numbered_uses_q_keys_in_prompt_and_response_keeps_ids():
    eng, fake = _engine(DIST)
    out = eng.handle(_body(template="numbered"))
    text = fake.payloads[0]["messages"][1]["content"][2]["text"]
    assert text.startswith("Q1: What style?") and "Q2: Maid?" in text and "style" not in text.split("\n")[0][:2]
    assert set(out["answers"]) == {"style", "outfit_maid", "outfit_swim"}


def test_engine_max_per_read_splits_into_consecutive_reads_and_samples_vary_noise():
    eng, fake = _engine(DIST)
    out = eng.handle(_body(max_per_read=2, samples=2, seed=1))
    assert len(fake.payloads) == 4  # 2 reads x 2 samples
    assert [r["questions"] for r in out["diagnostics"]["reads"]] == [["style", "outfit_maid"], ["outfit_swim"]]
    canvases = {tuple(p["vllm_xargs"]["diffusion_seed_canvas"]) for p in fake.payloads}
    assert len(canvases) == 4  # read/sample ごとに別ノイズ
    assert set(out["answers"]) == {"style", "outfit_maid", "outfit_swim"}
    # 同じ seed なら再現
    eng2, fake2 = _engine(DIST)
    eng2.handle(_body(max_per_read=2, samples=2, seed=1))
    assert {tuple(p["vllm_xargs"]["diffusion_seed_canvas"]) for p in fake2.payloads} == canvases


def test_engine_label_mass_below_threshold_goes_to_errors_without_answer():
    dist = dict(DIST)
    dist["style"] = {"A": 0.2, "B": 0.1, "C": 0.05}  # 質量 0.35 < 0.5
    eng, _ = _engine(dist)
    out = eng.handle(_body())
    assert "style" not in out["answers"]
    assert out["errors"]["style"]["type"] == "label_mass_low"
    assert out["diagnostics"]["label_mass"]["style"] == [pytest.approx(0.35)]
    assert "outfit_maid" in out["answers"]  # 他の質問は答える


def test_engine_samples_average_and_per_sample_diagnostics():
    eng, fake = _engine(DIST)
    out = eng.handle(_body(samples=3))
    assert len(fake.payloads) == 3
    assert len(out["diagnostics"]["per_sample"]["outfit_maid"]) == 3
    assert out["diagnostics"]["reads"][0]["ms_per_sample"][0] is not None


def test_engine_overflow_is_an_explicit_error_not_a_silent_split():
    eng, fake = _engine(DIST, canvas=16)
    with pytest.raises(ds.RequestError) as e:
        eng.handle(_body())
    assert e.value.status == 422 and fake.payloads == []


def test_engine_upstream_failure_for_all_reads_is_502_and_partial_failure_is_per_question():
    def always_fail(payload):
        raise ds.UpstreamError("down")

    eng = ds.Engine(ds.Config(vocab_size=1000), FakeTokenizer(), always_fail)
    with pytest.raises(ds.UpstreamError):
        eng.handle(_body())

    eng2, fake = _engine(DIST)
    original = fake.__call__
    calls = {"n": 0}

    def flaky(payload):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ds.UpstreamError("boom")
        return original(payload)

    eng2.chat_fn = flaky
    out = eng2.handle(_body(max_per_read=2))  # read 0 は成功、read 1 は失敗
    assert set(out["answers"]) == {"style", "outfit_maid"}
    assert out["errors"]["outfit_swim"]["type"] == "upstream_error"


def test_malformed_upstream_shapes_degrade_to_per_sample_failures():
    assert ds.read_slot([{"top_logprobs": ["x", None, 3]}] * 5, SLOT, ["A", "B"], 0.5)[2] == "missing_label_logprob"
    assert ds.read_slot(["notadict"] * 5, SLOT, ["A", "B"], 0.5)[2] == "missing_label_logprob"
    eng, fake = _engine(DIST)
    eng.chat_fn = lambda payload: {"choices": [{"logprobs": {"content": 5}}], "usage": "bad"}
    with pytest.raises(ds.UpstreamError):  # 全読み出しが失敗なら 502 相当
        eng.handle(_body())
    eng.chat_fn = lambda payload: (_ for _ in ()).throw(KeyError("zzz"))  # 想定外の例外もクラッシュしない
    with pytest.raises(ds.UpstreamError) as e:
        eng.handle(_body())
    assert "KeyError" in str(e.value)


def test_non_string_template_is_400():
    with pytest.raises(ds.RequestError) as e:
        ds.parse_request(_body(template=["keyed"]))
    assert e.value.status == 400
