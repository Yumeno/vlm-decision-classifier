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
            probs = self.dist.get(slot.qid)
            if probs is None:
                continue
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


# ---- 名寄せ(別表記のトークンも数える)


def _alias_slot(options, tokenizer=None):
    tok = ds.Tokenizer(tokenizer or FakeTokenizer())
    q = ds.Question("art_style", "choice", "?", options)
    t = ds.build_template(tok, ["art_style"], [q])
    return tok, t.slots[0]


def test_aliases_include_letter_name_and_case_variants_and_yes_no_forms():
    tok, slot = _alias_slot([("anime", ""), ("photo", "")])
    ids = lambda *words: sorted(tok(f"art_style: {w}\n")[slot.pos - len(tok(ds.SCAFFOLD_TEXT))] for w in words)
    assert slot.aliases["A"] == ids("A", "anime", "Anime")
    assert slot.aliases["B"] == ids("B", "photo", "Photo")
    assert slot.label_ids["A"] in slot.aliases["A"] and slot.conflicts == []
    n = ds.Question("o", "noul", "?", [])
    t = ds.build_template(tok, ["o"], [n]).slots[0]
    assert len(t.aliases["yes"]) == 3 and len(t.aliases["no"]) == 3
    assert set(t.aliases["yes"]).isdisjoint(t.aliases["no"])


def test_alias_ids_claimed_by_several_options_are_dropped_and_recorded():
    tok, slot = _alias_slot([("photo", ""), ("Photo", ""), ("other", "")])
    photo = tok("x: photo\n")[3]
    assert photo not in slot.aliases["A"] and photo not in slot.aliases["B"] and photo in slot.conflicts
    assert slot.label_ids["A"] in slot.aliases["A"] and slot.label_ids["C"] in slot.aliases["C"]
    # 名前が別選択肢の記号と同じ(選択肢 B の名前が "A")なら、記号の持ち主 A だけが残す
    tok2, slot2 = _alias_slot([("first", ""), ("A", "")])
    assert slot2.label_ids["A"] in slot2.aliases["A"] and slot2.label_ids["A"] not in slot2.aliases["B"]
    assert slot2.label_ids["A"] in slot2.conflicts


def test_alias_with_different_prefix_tokens_is_skipped():
    base = FakeTokenizer()

    def tokenize(text):
        ids = base(text)
        return [999] + ids if "zzname" in text else ids  # スロットより前が変わる表記

    _, slot = _alias_slot([("zzname", ""), ("plain", "")], tokenize)
    assert base.ids["zzname"] not in slot.aliases["A"]  # 前が変わる表記は使わない(Zzname は前が同じなので残る)
    assert slot.label_ids["A"] in slot.aliases["A"]
    assert len(slot.aliases["B"]) == 3


def test_read_slot_sums_alias_mass_per_option():
    slot = ds.Slot("q", 3, {"A": 11, "B": 12}, {"A": [11, 21], "B": [12, 22]})
    lps = {11: math.log(0.05), 21: math.log(0.6), 12: math.log(0.1), 22: math.log(0.05)}
    probs, mass, reason = ds.read_slot(_content({3: lps}), slot, ["A", "B"], 0.5)
    assert reason is None and mass == pytest.approx(0.8)
    assert probs["A"] == pytest.approx(0.65 / 0.8) and probs["B"] == pytest.approx(0.15 / 0.8)
    # 別表記の欠けは 0。正規ラベルの欠けは無効
    probs, mass, _ = ds.read_slot(_content({3: {11: math.log(0.6), 12: math.log(0.2)}}), slot, ["A", "B"], 0.5)
    assert mass == pytest.approx(0.8)
    assert ds.read_slot(_content({3: {21: -0.1, 12: -0.1}}), slot, ["A", "B"], 0.5)[2] == "missing_label_logprob"


def test_engine_counts_aliases_requests_their_ids_and_records_them():
    eng, fake = _engine(DIST)
    out = eng.handle(_body())
    slots = {s.qid: s for s in eng.last_plans[0].template.slots}
    asked = set(fake.payloads[0]["logprob_token_ids"])
    assert set(slots["style"].all_ids()) <= asked and len(asked) > len(slots["style"].label_ids) + 2
    assert out["diagnostics"]["alias_ids"]["style"]["anime"] == slots["style"].alias_ids("A")
    assert out["diagnostics"]["alias_ids"]["outfit_maid"]["yes"] == slots["outfit_maid"].alias_ids("yes")
    assert out["diagnostics"]["alias_conflicts"] == {}


def test_instruction_variants_change_only_prompt_text():
    expected_default = (
        "style: What style?\n  A: anime (cel)\n  B: photo (real)\n  C: other\n"
        "outfit_maid: Maid?\n  yes / no\noutfit_swim: Swim?\n  yes / no"
    )
    eng, fake = _engine(DIST)
    eng.handle(_body())
    eng.handle(_body(instruction="strict"))
    d, s = fake.payloads
    assert d["messages"][0]["content"] == ds.SYSTEM_PROMPT and d["messages"][1]["content"][2]["text"] == expected_default
    assert s["messages"][0]["content"] == ds.STRICT_SYSTEM_PROMPT != ds.SYSTEM_PROMPT
    text = s["messages"][1]["content"][2]["text"]
    assert "  Answer with one letter: A, B or C" in text and text.count("  Answer with yes or no") == 2
    assert s["vllm_xargs"]["diffusion_canvas_length"] == d["vllm_xargs"]["diffusion_canvas_length"]  # テンプレートは不変
    with pytest.raises(ds.RequestError):
        ds.parse_request(_body(instruction="loud"))


def _yn_body(style, **kw):
    q = {
        "style": {"type": "choice", "instructions": "What style?", "criteria": {"anime": "cel", "photo": "real"}},
        "outfit_maid": {"type": "noul", "instructions": "Is maid present?", "subject": "maid outfit (apron)"},
        "outfit_swim": {"type": "noul", "instructions": "Is swim present?"},
    }
    return _body(questions=q, yn_style=style, **kw)


def _block(req):
    return ds.question_block(ds.read_keys("keyed", req.questions), req.questions, req.instruction)


def test_yn_style_rendering_and_labels():
    slash = ds.parse_request(_yn_body("slash"))
    assert _block(slash) == (
        "style: What style?\n  A: anime (cel)\n  B: photo (real)\n"
        "outfit_maid: Is maid present?\n  yes / no\noutfit_swim: Is swim present?\n  yes / no"
    )
    assert ds.parse_request(_yn_body("slash")).questions[1].labels == ["yes", "no"]
    lines = ds.parse_request(_yn_body("lines"))
    assert _block(lines).endswith(
        "outfit_maid: maid outfit (apron)\n  yes: present in the image\n  no: not present in the image\n"
        "outfit_swim: Is swim present?\n  yes: present in the image\n  no: not present in the image"
    )  # subject が無ければ instructions
    assert lines.questions[1].labels == ["yes", "no"]
    letters = ds.parse_request(_yn_body("letters"))
    assert "outfit_maid: maid outfit (apron)\n  A: present in the image\n  B: not present in the image" in _block(letters)
    assert letters.questions[1].labels == ["A", "B"] and letters.questions[1].yes_label == "A"
    assert ds.parse_request(_yn_body("slash")).yn_style == "slash"
    for bad in ("dots", 3, None):
        with pytest.raises(ds.RequestError) as e:
            ds.parse_request(_yn_body(bad))
        assert e.value.status == 400


def test_letters_yn_aliases_and_answer_maps_a_to_yes():
    eng, fake = _engine({"style": {"A": 0.6, "B": 0.3}, "outfit_maid": {"A": 0.7, "B": 0.1}, "outfit_swim": {"A": 0.1, "B": 0.7}})
    out = eng.handle(_yn_body("letters"))
    slot = {s.qid: s for s in eng.last_plans[0].template.slots}["outfit_maid"]
    tok = lambda w: eng.tok("outfit_maid: " + w + chr(10))[3]  # key, ":", " ", label
    assert set(slot.label_ids) == {"A", "B"}
    assert slot.alias_ids("A") == sorted({tok("A"), tok("yes"), tok("Yes"), tok("YES")})
    assert slot.alias_ids("B") == sorted({tok("B"), tok("no"), tok("No"), tok("NO")})
    assert out["answers"]["outfit_maid"]["noul"] == pytest.approx(0.7 / 0.8)
    assert out["answers"]["outfit_swim"]["noul"] == pytest.approx(0.1 / 0.8)


def test_strict_sys_changes_only_system_text_and_keeps_default_and_strict():
    eng, fake = _engine(DIST)
    for ins in ("default", "strict", "strict_sys"):
        eng.handle(_body(instruction=ins))
    d, s, ss = fake.payloads
    assert d["messages"][0]["content"] == ds.SYSTEM_PROMPT and s["messages"][0]["content"] == ds.STRICT_SYSTEM_PROMPT
    assert ss["messages"][0]["content"] == ds.STRICT_SYS_ONLY_PROMPT not in (ds.SYSTEM_PROMPT, ds.STRICT_SYSTEM_PROMPT)
    text = ss["messages"][1]["content"][2]["text"]
    assert "Answer with" not in text and text == d["messages"][1]["content"][2]["text"]
    assert "Answer with" in s["messages"][1]["content"][2]["text"]


def test_yn_style_yn_rendering_labels_and_mapping():
    req = ds.parse_request(_yn_body("yn"))
    assert _block(req).endswith(
        "outfit_maid: Is maid present?\n  Y: yes\n  N: no\noutfit_swim: Is swim present?\n  Y: yes\n  N: no"
    )  # 質問文は instructions(subject は使わない)
    q = req.questions[1]
    assert q.labels == ["Y", "N"] and q.yes_label == "Y"
    strict = ds.parse_request(_yn_body("yn", instruction="strict"))
    assert _block(strict).count("  Answer with one letter: Y or N") == 2
    eng, _ = _engine({"style": {"A": 0.6, "B": 0.3}, "outfit_maid": {"Y": 0.7, "N": 0.1}, "outfit_swim": {"Y": 0.1, "N": 0.7}})
    out = eng.handle(_yn_body("yn"))
    slot = {s.qid: s for s in eng.last_plans[0].template.slots}["outfit_maid"]
    tok = lambda w: eng.tok("outfit_maid: " + w + chr(10))[3]
    assert set(slot.label_ids) == {"Y", "N"}
    assert slot.alias_ids("Y") == sorted({tok("Y"), tok("yes"), tok("Yes"), tok("YES")})
    assert slot.alias_ids("N") == sorted({tok("N"), tok("no"), tok("No"), tok("NO")})
    assert out["answers"]["outfit_maid"]["noul"] == pytest.approx(0.7 / 0.8)
    assert out["answers"]["outfit_swim"]["noul"] == pytest.approx(0.1 / 0.8)


# ---- 適応的な再読み出し(issue #12)


def test_normalized_entropy():
    assert ds.normalized_entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert ds.normalized_entropy([1 / 3] * 3) == pytest.approx(1.0)
    assert ds.normalized_entropy([1.0, 0.0]) == pytest.approx(0.0)
    assert ds.normalized_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert 0 < ds.normalized_entropy([0.9, 0.1]) < 0.5  # 2値ではビット数のエントロピーに等しい
    assert ds.normalized_entropy([0.9, 0.1]) == pytest.approx(-(0.9 * math.log2(0.9) + 0.1 * math.log2(0.1)))


def _seq_engine(seq):
    """呼び出し n 回目に seq[min(n, 最後)] の分布を返す偽 vLLM。"""
    eng, fake = _engine(seq[0])
    orig = fake.__call__
    n = {"i": 0}

    def call(payload):
        fake.dist = seq[min(n["i"], len(seq) - 1)]
        n["i"] += 1
        return orig(payload)

    eng.chat_fn = call
    return eng, fake


def test_adaptive_validation():
    for bad in (0, -0.1, 1.5, "x", True):
        with pytest.raises(ds.RequestError) as e:
            ds.parse_request(_body(adaptive_threshold=bad))
        assert e.value.status == 400
    with pytest.raises(ds.RequestError):
        ds.parse_request(_body(adaptive_threshold=0.5, samples=2))
    with pytest.raises(ds.RequestError):
        ds.parse_request(_body(adaptive_threshold=0.5, adaptive_max=1))
    assert ds.parse_request(_body(adaptive_max=1)).adaptive_threshold is None  # 適応が無効なら adaptive_max は無視
    r = ds.parse_request(_body(adaptive_threshold=1))
    assert r.adaptive_threshold == 1.0 and r.adaptive_max == 3
    assert ds.parse_request(_body()).adaptive_threshold is None


def test_adaptive_not_triggered_reads_once():
    eng, fake = _engine(DIST)
    out = eng.handle(_body(adaptive_threshold=0.95))
    ad = out["diagnostics"]["adaptive"]
    assert len(fake.payloads) == 1 and ad["triggered"] is False and ad["reads_used"] == 1
    assert ad["trigger_questions"] == {} and ad["replaced"] == []
    assert set(ad["first_read_entropy"]) == {"style", "outfit_maid", "outfit_swim"}
    assert "adaptive" not in _engine(DIST)[0].handle(_body())["diagnostics"]


def test_adaptive_replaces_only_triggered_questions_with_average():
    first = {"style": {"A": 0.4, "B": 0.35, "C": 0.25}, "outfit_maid": {"yes": 0.9, "no": 0.1}, "outfit_swim": {"yes": 0.2, "no": 0.8}}
    second = {"style": {"A": 0.1, "B": 0.8, "C": 0.1}, "outfit_maid": {"yes": 0.5, "no": 0.5}, "outfit_swim": {"yes": 0.5, "no": 0.5}}
    eng, fake = _seq_engine([first, second])
    out = eng.handle(_body(adaptive_threshold=0.9, adaptive_max=2))
    ad = out["diagnostics"]["adaptive"]
    assert len(fake.payloads) == 2 and ad["triggered"] is True and ad["reads_used"] == 2
    assert set(ad["trigger_questions"]) == {"style"} and ad["replaced"] == ["style"]
    assert all(0 <= h <= 1 for h in ad["first_read_entropy"].values())
    probs = out["answers"]["style"]["probabilities"]
    assert probs["anime"] == pytest.approx((0.4 / 1.0 + 0.1 / 1.0) / 2)
    assert probs["photo"] == pytest.approx((0.35 + 0.8) / 2)  # 置き換え: 全サンプルの平均
    assert out["answers"]["outfit_maid"]["noul"] == pytest.approx(0.9)  # 非対象は1回目のまま(2回目が違っても)
    assert out["answers"]["outfit_swim"]["noul"] == pytest.approx(0.2)
    assert len(out["diagnostics"]["per_sample"]["outfit_maid"]) == 2  # 透明性のため全サンプルを残す


def test_adaptive_failed_first_read_triggers_and_counts_reads_used():
    first = {"style": None, "outfit_maid": {"yes": 0.9, "no": 0.1}, "outfit_swim": {"yes": 0.9, "no": 0.1}}
    second = {"style": {"A": 0.7, "B": 0.2, "C": 0.1}, "outfit_maid": {"yes": 0.1, "no": 0.9}, "outfit_swim": {"yes": 0.1, "no": 0.9}}
    eng, fake = _seq_engine([first, second])
    out = eng.handle(_body(adaptive_threshold=1.0, adaptive_max=3))
    ad = out["diagnostics"]["adaptive"]
    assert ad["triggered"] is True and ad["reads_used"] == 2 and len(fake.payloads) == 2  # 2回目で style がしきい値未満になり止まる
    assert ad["replaced"] == ["style"] and ad["trigger_questions"] == {"style": "failed"}
    assert out["answers"]["style"]["choice"] == "anime" and out["errors"] == {}
    assert out["answers"]["outfit_maid"]["noul"] == pytest.approx(0.9)  # 1回目のまま
    assert out["diagnostics"]["reads"][0]["canvas_width"] % 32 == 0 and len(out["diagnostics"]["reads"][0]["ms_per_sample"]) == 2
    assert ad["entropy_trace"]["style"][0] is None and ad["entropy_trace"]["style"][1] < 1.0 and ad["unresolved"] == []


SURE_STYLE = {"A": 0.98, "B": 0.01, "C": 0.01}
SURE_SWIM = {"yes": 0.01, "no": 0.99}


def test_adaptive_stops_early_when_running_average_converges():
    shaky = {"style": SURE_STYLE, "outfit_maid": {"yes": 0.8, "no": 0.2}, "outfit_swim": SURE_SWIM}
    sure = {"style": SURE_STYLE, "outfit_maid": {"yes": 0.999, "no": 0.001}, "outfit_swim": SURE_SWIM}
    eng, fake = _seq_engine([shaky, sure, sure, sure])
    out = eng.handle(_body(adaptive_threshold=0.5, adaptive_max=4))
    ad = out["diagnostics"]["adaptive"]
    assert len(fake.payloads) == 2 and ad["reads_used"] == 2 and ad["unresolved"] == []  # 平均 0.9 のエントロピー(約0.47) < 0.5
    assert set(ad["trigger_questions"]) == {"outfit_maid"} and ad["replaced"] == ["outfit_maid"]
    t = ad["entropy_trace"]["outfit_maid"]
    assert len(t) == 2 and t[0] > 0.5 > t[1]
    assert out["answers"]["outfit_maid"]["noul"] == pytest.approx((0.8 + 0.999) / 2)
    assert out["answers"]["outfit_swim"]["noul"] == pytest.approx(0.01)


def test_adaptive_runs_to_max_when_samples_disagree_and_lists_unresolved():
    up = {"style": SURE_STYLE, "outfit_maid": {"yes": 0.9, "no": 0.1}, "outfit_swim": SURE_SWIM}
    down = {"style": SURE_STYLE, "outfit_maid": {"yes": 0.1, "no": 0.9}, "outfit_swim": SURE_SWIM}
    eng, fake = _seq_engine([up, down])
    out = eng.handle(_body(adaptive_threshold=0.4, adaptive_max=4))
    ad = out["diagnostics"]["adaptive"]
    assert len(fake.payloads) == 4 and ad["reads_used"] == 4
    assert ad["unresolved"] == ["outfit_maid"] and len(ad["entropy_trace"]["outfit_maid"]) == 4
    assert ad["entropy_trace"]["outfit_maid"][1] == pytest.approx(1.0)  # 0.9 と 0.1 の平均 0.5
    assert out["answers"]["outfit_maid"]["noul"] == pytest.approx((0.9 + 0.1 + 0.1 + 0.1) / 4)


def test_adaptive_question_failing_in_every_read_is_unresolved_error_and_not_replaced():
    bad = {"style": None, "outfit_maid": {"yes": 0.99, "no": 0.01}, "outfit_swim": SURE_SWIM}
    eng, fake = _seq_engine([bad])
    out = eng.handle(_body(adaptive_threshold=0.5, adaptive_max=3))
    ad = out["diagnostics"]["adaptive"]
    assert len(fake.payloads) == 3 and ad["unresolved"] == ["style"] and ad["replaced"] == []
    assert ad["trigger_questions"] == {"style": "failed"} and ad["entropy_trace"]["style"] == [None, None, None]
    assert "style" not in out["answers"] and out["errors"]["style"]["type"] == "missing_label_logprob"



def test_steps_payload_pins_everything_except_label_slots():
    eng, fake = _engine(DIST)
    eng.handle(_body())
    eng.handle(_body(steps=4))
    d, s = fake.payloads
    assert d["vllm_xargs"] == {
        "diffusion_seed_canvas": d["vllm_xargs"]["diffusion_seed_canvas"],
        "diffusion_canvas_length": d["vllm_xargs"]["diffusion_canvas_length"],
        "diffusion_max_steps": 1,
        "diffusion_read_only": True,
    }  # 既定は従来どおり(pinned なし)
    x = s["vllm_xargs"]
    width = x["diffusion_canvas_length"]
    slots = {sl.pos for sl in eng.last_plans[0].template.slots}
    assert x["diffusion_max_steps"] == 4 and x["diffusion_read_only"] is True
    assert x["diffusion_pinned"] == [i for i in range(width) if i not in slots]
    assert len(x["diffusion_pinned"]) == width - len(slots) and slots.isdisjoint(x["diffusion_pinned"])
    assert len(x["diffusion_seed_canvas"]) == width


def test_steps_validation_diagnostics_and_adaptive_compat():
    for bad in (0, -1, 1.5, "2", True):
        with pytest.raises(ds.RequestError) as e:
            ds.parse_request(_body(steps=bad))
        assert e.value.status == 400
    assert ds.parse_request(_body()).steps == 1
    eng, fake = _engine(DIST)
    out = eng.handle(_body(steps=3, adaptive_threshold=0.95))
    assert out["diagnostics"]["steps"] == 3 and out["diagnostics"]["adaptive"]["reads_used"] == 1
    assert all(p["vllm_xargs"]["diffusion_max_steps"] == 3 for p in fake.payloads)
