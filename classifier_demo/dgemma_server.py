"""DiffusionGemma(vLLM)用の自前の判定サーバー(`POST /v1/systemone`)。stdlib のみ。

画像1枚・全質問を「1キャンバス・1回の読み出し」で聞くのが目的(拡散LMなので1回の順伝播で全スロットの分布が出る)。
キャンバスは「空の思考ブロック + 質問ごとの1行 `<key>: <label>\\n` + ターン終端 + PAD」で、
キー部分は固定テキスト、ラベル1トークンの位置(スロット)だけを語彙一様のノイズにして、1ステップ(read_only)で
各スロットの logprob を読む。各質問のスコアは、その質問のラベル集合内で再正規化した相対スコア(正答確率ではない)。
応答のフィールド名 `probabilities` は Jev API の形に合わせた外部仕様の名前で、中身は相対スコア。

失敗は明示する: ラベルの logprob 欠落・ラベル質量不足のサンプルは無効、有効サンプルが 0 の質問は
`errors` に理由を書き、任意の選択肢には強制しない。入力不正は 400、テンプレートが幅に収まらない/
トークン検証の失敗は 422、上流(vLLM)の失敗は 502。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
CHOICE_LABELS = list(string.ascii_uppercase)
NOUL_LABELS = ["yes", "no"]
SCAFFOLD_TEXT = "<|channel>thought\n<channel|>"  # enable_thinking=false 時にモデルが先頭に出す空の思考ブロック
TURN_CLOSE_TEXT = "<turn|>"
DEFAULT_PAD_ID = 0
DEFAULT_VOCAB_SIZE = 262144
WIDTH_STEP = 32
MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_IMAGES = 8
MAX_LOGPROB_TOKEN_IDS = 128  # vLLM の上限
SYSTEM_PROMPT = (
    "You label an image by answering questions. Each question below has a key and a list of allowed labels. "
    "Reply with exactly one line per question, in the order given, written as '<key>: <label>', "
    "using only a label that question allows. Never write prose or anything else."
)
TEMPLATES = ("keyed", "numbered")
INSTRUCTIONS = ("default", "strict", "strict_sys")
YN_STYLES = ("slash", "lines", "letters", "yn")
STRICT_SYS_ONLY_PROMPT = (
    "You label an image by answering questions. Each question below has a key, and its options are listed as "
    "'<label>: <meaning>'. Answer every question with exactly one label and nothing else: "
    "a single capital letter for a multiple-choice question, and for a presence question the label shown "
    "(yes or no, or a letter when the options are lettered). "
    "Never write an option's name, its description or any other word. "
    "Reply with exactly one line per question, in the order given, in the form '<key>: <label>'."
)
STRICT_SYSTEM_PROMPT = (
    "You label an image by answering questions. Each question below has a key and a list of allowed labels, "
    "and every option is written as '<label>: <description>'. Your answer for a question must be exactly one label: "
    "a single capital letter for a multiple-choice question, or yes or no for a yes/no question. "
    "Never write the option's name, a description or any other word. "
    "Reply with exactly one line per question, in the order given, in the form '<key>: <label>'. Nothing else."
)


class RequestError(Exception):
    """クライアントへ返すエラー(400 入力不正 / 422 テンプレート・トークンの問題)。"""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class UpstreamError(Exception):
    """vLLM への通信・HTTP失敗(502)。"""


@dataclass
class Config:
    canvas: int = 256
    max_per_read: int = 0
    min_label_mass: float = 0.5
    vocab_size: int = DEFAULT_VOCAB_SIZE
    pad_id: int = DEFAULT_PAD_ID
    model: str = "dgemma"
    timeout: float = 300.0


@dataclass
class Question:
    qid: str
    kind: str  # "choice" | "noul"
    instructions: str
    options: list[tuple[str, str]]  # choice: (名前, 説明)。noul は空
    subject: str = ""  # noul: 在否を聞く対象(lines/letters の質問文に使う)
    yn_style: str = "slash"  # noul の描き方(slash|lines|letters|yn)

    @property
    def labels(self) -> list[str]:
        if self.kind == "choice":
            return CHOICE_LABELS[: len(self.options)]
        if self.yn_style == "letters":
            return ["A", "B"]
        return ["Y", "N"] if self.yn_style == "yn" else NOUL_LABELS

    @property
    def yes_label(self) -> str:
        return self.labels[0]  # noul の「ある」側(letters では A、yn では Y)


@dataclass
class Request:
    model: str | None
    state: str
    questions: list[Question]
    images: list[str]
    samples: int
    seed: int
    template: str
    max_per_read: int | None
    mm_processor_kwargs: dict | None
    instruction: str = "default"
    yn_style: str = "slash"
    adaptive_threshold: float | None = None
    adaptive_max: int = 3
    steps: int = 1


@dataclass
class Slot:
    qid: str
    pos: int
    label_ids: dict[str, int]  # ラベル(A/B/..、yes/no) -> 正規トークン id
    aliases: dict[str, list[int]] | None = None  # ラベル -> 数える id(正規 id + 名寄せの別表記)。None なら正規 id のみ
    conflicts: list[int] | None = None  # 複数の選択肢に取り合われて外した id

    def alias_ids(self, label: str) -> list[int]:
        return self.aliases[label] if self.aliases else [self.label_ids[label]]

    def all_ids(self) -> list[int]:
        return sorted({i for lab in self.label_ids for i in self.alias_ids(lab)})


@dataclass
class Template:
    tokens: list[int]  # スロット位置にはプレースホルダ(ベースラベルの id)が入っている
    slots: list[Slot]
    keys: list[str]  # スロットごとのキー(keyed: qid、numbered: Q<n>)


# ---------------------------------------------------------------- 入力の検証


def clean_text(value, limit: int, what: str) -> str:
    """信頼できない文字列。制御文字・改行は空白にして畳み、長さを制限する。"""
    if not isinstance(value, str):
        raise RequestError(400, f"{what} must be a string")
    s = "".join(" " if unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp") else c for c in value)
    return " ".join(s.split())[:limit]


def _int_field(body: dict, name: str, default: int, minimum: int) -> int:
    v = body.get(name, default)
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
        raise RequestError(400, f"'{name}' must be an integer >= {minimum}")
    return v


def parse_request(body) -> Request:
    if not isinstance(body, dict):
        raise RequestError(400, "request body must be a JSON object")
    raw_q = body.get("questions")
    if not isinstance(raw_q, dict) or not raw_q:
        raise RequestError(400, "'questions' must be a non-empty object")
    yn_style = body.get("yn_style", "slash")
    if not isinstance(yn_style, str) or yn_style not in YN_STYLES:
        raise RequestError(400, f"'yn_style' must be one of {YN_STYLES}")
    questions: list[Question] = []
    for qid, spec in raw_q.items():
        if not KEY_RE.match(qid):
            raise RequestError(400, f"question id {qid!r} must match {KEY_RE.pattern}")
        if not isinstance(spec, dict):
            raise RequestError(400, f"question {qid!r} must be an object")
        kind = spec.get("type")
        instructions = clean_text(spec.get("instructions", ""), 1000, f"question {qid!r} instructions")
        if kind == "choice":
            crit = spec.get("criteria")
            if not isinstance(crit, dict) or len(crit) < 2:
                raise RequestError(400, f"question {qid!r}: 'criteria' must be an object with >= 2 options")
            if len(crit) > len(CHOICE_LABELS):
                raise RequestError(400, f"question {qid!r}: at most {len(CHOICE_LABELS)} options (got {len(crit)})")
            options = []
            for name, desc in crit.items():
                name = clean_text(name, 100, f"question {qid!r} option name")
                if not name:
                    raise RequestError(400, f"question {qid!r}: empty option name")
                options.append((name, clean_text(desc if desc is not None else "", 500, f"question {qid!r} option description")))
            if len({n for n, _ in options}) != len(options):
                raise RequestError(400, f"question {qid!r}: duplicate option names after cleaning")
            questions.append(Question(qid, "choice", instructions, options))
        elif kind == "noul":
            subject = clean_text(spec.get("subject", ""), 600, f"question {qid!r} subject")
            questions.append(Question(qid, "noul", instructions, [], subject, yn_style))
        else:
            raise RequestError(400, f"question {qid!r}: type must be 'choice' or 'noul'")
    images = body.get("images")
    if (
        not isinstance(images, list)
        or not 1 <= len(images) <= MAX_IMAGES
        or not all(isinstance(i, str) and i.startswith("data:image/") for i in images)
    ):
        raise RequestError(400, f"'images' must be 1..{MAX_IMAGES} data:image/ URLs")
    template = body.get("template", "keyed")
    if not isinstance(template, str) or template not in TEMPLATES:
        raise RequestError(400, f"'template' must be one of {TEMPLATES}")
    instruction = body.get("instruction", "default")
    if not isinstance(instruction, str) or instruction not in INSTRUCTIONS:
        raise RequestError(400, f"'instruction' must be one of {INSTRUCTIONS}")
    mpr = body.get("max_per_read")
    if mpr is not None:
        mpr = _int_field(body, "max_per_read", 0, 0)
    mm = body.get("mm_processor_kwargs")
    if mm is not None and not isinstance(mm, dict):
        raise RequestError(400, "'mm_processor_kwargs' must be an object")
    thr = body.get("adaptive_threshold")
    if thr is not None and (isinstance(thr, bool) or not isinstance(thr, (int, float)) or not 0 < thr <= 1):
        raise RequestError(400, "'adaptive_threshold' must be a number in (0, 1]")
    amax = _int_field(body, "adaptive_max", 3, 2) if thr is not None else 3  # 適応が無効なら無視
    if thr is not None and body.get("samples", 1) != 1:
        raise RequestError(400, "'adaptive_threshold' cannot be combined with samples > 1")
    steps = _int_field(body, "steps", 1, 1)
    model = body.get("model")
    return Request(
        model=model if isinstance(model, str) else None,
        state=clean_text(body.get("state", ""), 1000, "state"),
        questions=questions,
        images=images,
        samples=_int_field(body, "samples", 1, 1),
        seed=_int_field(body, "seed", 0, -(2**63)),
        template=template,
        max_per_read=mpr,
        mm_processor_kwargs=mm,
        instruction=instruction,
        yn_style=yn_style,
        adaptive_threshold=float(thr) if thr is not None else None,
        adaptive_max=amax,
        steps=steps,
    )


def split_reads(questions: list[Question], max_per_read: int) -> list[list[Question]]:
    """0 = 全質問を1回で。>0 = リクエスト順に連続して最大その数ずつ(型での再グループ化はしない)。"""
    if max_per_read <= 0:
        return [list(questions)]
    return [questions[i : i + max_per_read] for i in range(0, len(questions), max_per_read)]


# ---------------------------------------------------------------- トークナイズとテンプレート


class Tokenizer:
    """`tokenize_fn(text) -> list[int]`(vLLM の /tokenize)にキャッシュと所要時間の積算を足したもの。"""

    def __init__(self, tokenize_fn: Callable[[str], list[int]]) -> None:
        self._fn = tokenize_fn
        self._cache: dict[str, list[int]] = {}
        self._lock = threading.Lock()
        self.total_ms = 0.0

    def __call__(self, text: str) -> list[int]:
        with self._lock:
            hit = self._cache.get(text)
        if hit is not None:
            return hit
        start = time.perf_counter_ns()
        ids = list(self._fn(text))
        elapsed = (time.perf_counter_ns() - start) / 1e6
        with self._lock:
            self._cache[text] = ids
            self.total_ms += elapsed
        return ids


def line_text(key: str, label: str) -> str:
    return f"{key}: {label}\n"


def line_tokens(tok: Tokenizer, key: str, labels: list[str]) -> tuple[list[int], int, dict[str, int]]:
    """1行(`<key>: <label>\\n`)のトークン列(先頭ラベルの版)、スロット位置、ラベル->id。
    全ラベルで「同じ長さ・ちょうど1か所だけ違う(全ラベルで同じ位置)・id が互いに異なる」を検証する。"""
    base = tok(line_text(key, labels[0]))
    variants = {labels[0]: base}
    for lab in labels[1:]:
        variants[lab] = tok(line_text(key, lab))
    positions: set[int] = set()
    for lab, ids in variants.items():
        if len(ids) != len(base):
            raise RequestError(422, f"label {lab!r} on line {key!r} changes the token count ({len(ids)} vs {len(base)})")
        positions.update(i for i, (a, b) in enumerate(zip(base, ids)) if a != b)
    if len(positions) != 1:
        raise RequestError(422, f"labels on line {key!r} must differ at exactly one token position (got {sorted(positions)})")
    pos = next(iter(positions))
    label_ids = {lab: ids[pos] for lab, ids in variants.items()}
    if len(set(label_ids.values())) != len(label_ids):
        raise RequestError(422, f"label token ids are not distinct on line {key!r}")
    return base, pos, label_ids


def label_variants(q: Question) -> dict[str, list[str]]:
    """ラベルごとの別表記(文字列)。モデルが記号の代わりに選択肢名や yes/Yes を書く場合を数える(名寄せ)。"""
    if q.kind == "noul":
        if q.yn_style == "letters":
            return {"A": ["A", "yes", "Yes", "YES"], "B": ["B", "no", "No", "NO"]}
        if q.yn_style == "yn":
            return {"Y": ["Y", "yes", "Yes", "YES"], "N": ["N", "no", "No", "NO"]}
        return {"yes": ["yes", "Yes", "YES"], "no": ["no", "No", "NO"]}
    out = {}
    for lab, (name, _) in zip(q.labels, q.options):
        forms = [lab, name, name.lower(), name.capitalize()]
        out[lab] = list(dict.fromkeys(forms))
    return out


def build_aliases(
    tok: Tokenizer, key: str, q: Question, base: list[int], pos: int, label_ids: dict[str, int]
) -> tuple[dict[str, list[int]], list[int]]:
    """(ラベル -> 数える id, 取り合いで外した id)。別表記は「同じ行文脈でスロット位置に来る最初のトークン」。
    スロットより前のトークンがベース行と違う表記は使わない。別の選択肢にも属する id は全員から外す
    (他ラベルの正規 id は、その持ち主だけが残す)。"""
    claims: dict[int, set[str]] = {}
    for lab, forms in label_variants(q).items():
        claims.setdefault(label_ids[lab], set()).add(lab)
        for v in forms:
            ids = tok(line_text(key, v))
            if len(ids) > pos and ids[:pos] == base[:pos]:
                claims.setdefault(ids[pos], set()).add(lab)
    canonical_owner = {tid: lab for lab, tid in label_ids.items()}
    aliases: dict[str, set[int]] = {lab: set() for lab in label_ids}
    conflicts = []
    for tid, labs in claims.items():
        owner = canonical_owner.get(tid)
        if owner is not None:
            aliases[owner].add(tid)
            if len(labs) > 1:
                conflicts.append(tid)
        elif len(labs) == 1:
            aliases[next(iter(labs))].add(tid)
        else:
            conflicts.append(tid)
    return {lab: sorted(ids) for lab, ids in aliases.items()}, sorted(conflicts)


def build_template(tok: Tokenizer, keys: list[str], questions: list[Question]) -> Template:
    head = tok(SCAFFOLD_TEXT)
    turn = tok(TURN_CLOSE_TEXT)
    if len(turn) != 1:
        raise RequestError(422, f"{TURN_CLOSE_TEXT!r} must be a single token (got {len(turn)})")
    tokens = list(head)
    slots: list[Slot] = []
    for key, q in zip(keys, questions):
        ids, pos, label_ids = line_tokens(tok, key, q.labels)
        aliases, conflicts = build_aliases(tok, key, q, ids, pos, label_ids)
        slots.append(Slot(q.qid, len(tokens) + pos, label_ids, aliases, conflicts))
        tokens.extend(ids)
    tokens.extend(turn)
    # 行ごとの連結が全文の一括トークナイズと一致すること(BPE の境界ずれの検出)。
    whole = SCAFFOLD_TEXT + "".join(line_text(k, q.labels[0]) for k, q in zip(keys, questions)) + TURN_CLOSE_TEXT
    if tok(whole) != tokens:
        raise RequestError(422, "per-line token concatenation differs from tokenizing the whole template")
    return Template(tokens=tokens, slots=slots, keys=list(keys))


def canvas_width(template_len: int, canvas: int) -> int:
    """テンプレート長 + 1(最低1つの PAD)を 32 の倍数に切り上げる(canvas を超えるならその値に丸め、
    テンプレート自体が収まらないときは 422)。"""
    need = template_len + 1
    if need > canvas:
        raise RequestError(422, f"template needs {need} canvas positions but the served canvas is {canvas}")
    return min(math.ceil(need / WIDTH_STEP) * WIDTH_STEP, canvas)


def build_canvas(template: Template, width: int, rng: random.Random, vocab_size: int, pad_id: int) -> list[int]:
    ids = list(template.tokens)
    for slot in template.slots:
        ids[slot.pos] = rng.randrange(vocab_size)
    return ids + [pad_id] * (width - len(ids))


# ---------------------------------------------------------------- 読み出し結果の解釈


def parse_token_id(token) -> int | None:
    if isinstance(token, str) and token.startswith("token_id:"):
        try:
            return int(token[len("token_id:") :])
        except ValueError:
            return None
    return None


def read_slot(content: list, slot: Slot, labels: list[str], min_mass: float):
    """1サンプル・1質問の読み。(probs|None, mass|None, reason|None)。probs は label -> 相対スコア。"""
    if slot.pos >= len(content):
        return None, None, "short_content"
    seen: dict[int, float] = {}
    slot_entry = content[slot.pos]
    tops = slot_entry.get("top_logprobs") if isinstance(slot_entry, dict) else None
    for entry in tops if isinstance(tops, list) else []:
        if not isinstance(entry, dict):
            continue
        tid = parse_token_id(entry.get("token"))
        if tid is not None and isinstance(entry.get("logprob"), (int, float)):
            seen[tid] = float(entry["logprob"])
    weights = {}
    for lab in labels:
        if slot.label_ids[lab] not in seen:  # 正規ラベルの id は必ず要る(別表記の欠けは 0 とみなす)
            return None, None, "missing_label_logprob"
        weights[lab] = sum(math.exp(seen[i]) for i in slot.alias_ids(lab) if i in seen)
    mass = sum(weights.values())
    if mass < min_mass:
        return None, mass, "label_mass_low"
    if mass <= 0:
        return None, mass, "label_mass_low"
    return {lab: w / mass for lab, w in weights.items()}, mass, None


def normalized_entropy(probs: list[float]) -> float:
    """相対スコア分布の正規化エントロピー(0=確定、1=一様)。ラベル数は2以上。"""
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, h / math.log(len(probs)))


def aggregate_question(q: Question, samples: list[tuple]) -> tuple[dict | None, dict | None]:
    """有効サンプルを平均して (answer, error) のどちらかを返す。"""
    valid = [s[0] for s in samples if s[0] is not None]
    if not valid:
        reasons = Counter(s[2] or "unknown" for s in samples)
        top = reasons.most_common(1)[0][0]
        return None, {"type": top, "detail": f"no valid sample out of {len(samples)}: {dict(reasons)}"}
    avg = {lab: sum(p[lab] for p in valid) / len(valid) for lab in valid[0]}
    if q.kind == "noul":
        return {"type": "noul", "noul": avg[q.yes_label]}, None
    best = max(q.labels, key=lambda lab: avg[lab])  # 同点は先頭
    names = {lab: name for lab, (name, _) in zip(q.labels, q.options)}
    return {
        "type": "choice",
        "choice": names[best],
        "probabilities": {names[lab]: avg[lab] for lab in q.labels},  # Jev の外部仕様の名前。中身は相対スコア
    }, None


# ---------------------------------------------------------------- プロンプト


def question_block(keys: list[str], questions: list[Question], instruction: str = "default") -> str:
    lines = []
    for key, q in zip(keys, questions):
        styled = q.kind == "noul" and q.yn_style in ("lines", "letters")  # yn は質問文のまま
        lines.append(f"{key}: {(q.subject or q.instructions) if styled else q.instructions}")
        if q.kind == "choice":
            for lab, (name, desc) in zip(q.labels, q.options):
                lines.append(f"  {lab}: {name} ({desc})" if desc else f"  {lab}: {name}")
            if instruction == "strict":
                letters = q.labels
                lines.append(f"  Answer with one letter: {', '.join(letters[:-1])} or {letters[-1]}")
        elif q.yn_style == "slash":
            lines.append("  yes / no")
            if instruction == "strict":
                lines.append("  Answer with yes or no")
        elif q.yn_style == "yn":
            lines.append("  Y: yes")
            lines.append("  N: no")
            if instruction == "strict":
                lines.append("  Answer with one letter: Y or N")
        else:
            yes_l, no_l = q.labels[0], q.labels[1]
            lines.append(f"  {yes_l}: present in the image")
            lines.append(f"  {no_l}: not present in the image")
            if instruction == "strict":
                lines.append(f"  Answer with {'one letter: A or B' if q.yn_style == 'letters' else 'yes or no'}")
    return "\n".join(lines)


def read_keys(template: str, questions: list[Question]) -> list[str]:
    if template == "numbered":
        return [f"Q{i}" for i in range(1, len(questions) + 1)]
    return [q.qid for q in questions]


def build_messages(req: Request, block: str) -> list[dict]:
    content = [{"type": "image_url", "image_url": {"url": u}} for u in req.images]
    content.append({"type": "text", "text": req.state})
    content.append({"type": "text", "text": block})
    system = {"strict": STRICT_SYSTEM_PROMPT, "strict_sys": STRICT_SYS_ONLY_PROMPT}.get(req.instruction, SYSTEM_PROMPT)
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


# ---------------------------------------------------------------- エンジン本体


@dataclass
class ReadPlan:
    questions: list[Question]
    template: Template
    width: int
    label_ids: list[int]
    block: str


class Engine:
    """tokenize_fn(text)->ids と chat_fn(payload)->response dict を注入できる(テスト用)。"""

    def __init__(self, cfg: Config, tokenize_fn: Callable[[str], list[int]], chat_fn: Callable[[dict], dict]) -> None:
        self.cfg = cfg
        self.tok = Tokenizer(tokenize_fn)
        self.chat_fn = chat_fn

    def plan_reads(self, req: Request) -> list[ReadPlan]:
        mpr = self.cfg.max_per_read if req.max_per_read is None else req.max_per_read
        plans = []
        for group in split_reads(req.questions, mpr):
            keys = read_keys(req.template, group)
            template = build_template(self.tok, keys, group)
            width = canvas_width(len(template.tokens), self.cfg.canvas)
            ids = sorted({i for s in template.slots for i in s.all_ids()})
            if len(ids) > MAX_LOGPROB_TOKEN_IDS:
                raise RequestError(422, f"too many distinct label token ids ({len(ids)} > {MAX_LOGPROB_TOKEN_IDS})")
            plans.append(ReadPlan(group, template, width, ids, question_block(keys, group, req.instruction)))
        return plans

    def run_one(self, req: Request, plan: ReadPlan, read_idx: int, k: int) -> dict:
        rng = random.Random(f"{req.seed}:{read_idx}:{k}")
        canvas = build_canvas(plan.template, plan.width, rng, self.cfg.vocab_size, self.cfg.pad_id)
        payload = {
            "model": self.cfg.model,
            "messages": build_messages(req, plan.block),
            "max_tokens": plan.width,
            "logprobs": True,
            "logprob_token_ids": plan.label_ids,
            "return_tokens_as_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "vllm_xargs": {
                "diffusion_seed_canvas": canvas,
                "diffusion_canvas_length": plan.width,
                "diffusion_max_steps": 1,
                "diffusion_read_only": True,
            },
        }
        if req.steps > 1:
            # 複数ステップ: ラベルのスロット以外の全位置(足場・キー・改行・ターン終端・PAD)を固定し、スロットだけを動かす
            payload["vllm_xargs"]["diffusion_max_steps"] = req.steps
            slot_pos = {s.pos for s in plan.template.slots}
            payload["vllm_xargs"]["diffusion_pinned"] = [i for i in range(plan.width) if i not in slot_pos]
        if req.mm_processor_kwargs is not None:
            payload["mm_processor_kwargs"] = req.mm_processor_kwargs
        start = time.perf_counter_ns()
        resp = self.chat_fn(payload)
        ms = (time.perf_counter_ns() - start) / 1e6
        return {"resp": resp, "ms": ms}

    def handle(self, body) -> dict:
        t0 = time.perf_counter_ns()
        req = parse_request(body)
        tok_before = self.tok.total_ms
        plans = self.plan_reads(req)
        tokenize_ms = self.tok.total_ms - tok_before

        results: dict[tuple[int, int], dict | Exception] = {}

        def run(task):
            try:
                res = self.run_one(req, plans[task[0]], *task)
                resp = res["resp"]
                if not isinstance(resp, dict):
                    raise UpstreamError(f"upstream response is not an object: {type(resp).__name__}")
                u = resp.get("usage")
                if u is not None and not (
                    isinstance(u, dict) and all(isinstance(u.get(n) or 0, (int, float)) for n in ("prompt_tokens", "completion_tokens", "total_tokens"))
                ):
                    raise UpstreamError("upstream 'usage' has an unexpected shape")
                results[task] = res
            except UpstreamError as e:
                results[task] = e
            except Exception as e:  # 想定外の形・例外も、そのサンプルの失敗として残す(ハンドラを落とさない)
                results[task] = UpstreamError(f"unexpected {type(e).__name__}: {e}")

        def run_all(tasks):
            run(tasks[0])  # 先頭の1回で画像プレフィックスをキャッシュに載せてから残りを並列に
            if len(tasks) > 1:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(run, tasks[1:]))

        def evaluate(res, plan):
            """1サンプルの結果 -> (質問ごとの (probs, mass, reason), sample_error)。"""
            if isinstance(res, Exception):
                return {q.qid: (None, None, "upstream_error") for q in plan.questions}, str(res)[:300]
            try:
                content = res["resp"]["choices"][0]["logprobs"]["content"]
                if not isinstance(content, list):
                    raise TypeError("content is not a list")
            except (KeyError, IndexError, TypeError) as e:
                return {q.qid: (None, None, "no_logprobs") for q in plan.questions}, f"no logprobs.content in response ({type(e).__name__})"
            return {
                q.qid: read_slot(content, slot, q.labels, self.cfg.min_label_mass)
                for q, slot in zip(plan.questions, plan.template.slots)
            }, None

        n_samples = [req.samples] * len(plans)
        run_all([(r, k) for r in range(len(plans)) for k in range(req.samples)])
        adaptive_diag = None
        if req.adaptive_threshold is not None:
            thr = req.adaptive_threshold
            first_entropy: dict = {}
            trigger: dict = {}
            replaced: list = []
            trace: dict = {}
            unresolved: list = []
            for r, plan in enumerate(plans):
                tuples, _ = evaluate(results[(r, 0)], plan)
                targets = []
                valid: dict[str, list] = {}  # 対象の質問 -> 有効サンプルの相対スコア
                for q in plan.questions:
                    probs = tuples[q.qid][0]
                    if probs is None:
                        trigger[q.qid] = "failed"  # 1回目に失敗(エントロピーは無い)
                        targets.append(q)
                        valid[q.qid] = []
                        trace[q.qid] = [None]
                        continue
                    h = normalized_entropy([probs[lab] for lab in q.labels])
                    first_entropy[q.qid] = h
                    if h >= thr:
                        trigger[q.qid] = h
                        targets.append(q)
                        valid[q.qid] = [probs]
                        trace[q.qid] = [h]
                if not targets:
                    continue
                still = set(q.qid for q in targets)
                for k in range(1, req.adaptive_max):  # 1回ずつ読み、対象が全員しきい値未満になったら止める
                    run((r, k))
                    n_samples[r] = k + 1
                    tuples, _ = evaluate(results[(r, k)], plan)
                    still = set()
                    for q in targets:
                        if tuples[q.qid][0] is not None:
                            valid[q.qid].append(tuples[q.qid][0])
                        vs = valid[q.qid]
                        if not vs:
                            trace[q.qid].append(None)
                            still.add(q.qid)
                            continue
                        avg = [sum(v[lab] for v in vs) / len(vs) for lab in q.labels]
                        h = normalized_entropy(avg)
                        trace[q.qid].append(h)
                        if h >= thr:
                            still.add(q.qid)
                    if not still:
                        break
                replaced.extend(q.qid for q in targets if valid[q.qid])  # 有効サンプルが無いままの質問は置き換わらない(エラーのまま)
                unresolved.extend(q.qid for q in targets if q.qid in still)
            adaptive_diag = {
                "threshold": thr,
                "max": req.adaptive_max,
                "triggered": bool(trigger),
                "trigger_questions": trigger,
                "first_read_entropy": first_entropy,
                "reads_used": sum(n_samples),
                "replaced": replaced,  # 再読み出しの平均で置き換えた質問(他は1回目の値のまま)
                "entropy_trace": trace,  # 対象の質問の、各読み出し後の累積平均の正規化エントロピー(None=有効サンプルなし)
                "unresolved": unresolved,  # 上限まで読んでもしきい値以上(または失敗)のまま
            }
        if all(isinstance(v, Exception) for v in results.values()):
            raise UpstreamError(str(next(iter(results.values()))))

        answers: dict = {}
        errors: dict = {}
        label_mass: dict = {}
        per_sample: dict = {}
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        reads_diag = []
        for r, plan in enumerate(plans):
            per_q: dict[str, list[tuple]] = {q.qid: [] for q in plan.questions}
            ms_list, cached, sample_errors = [], [], {}
            prompt_tokens = None
            for k in range(n_samples[r]):
                res = results[(r, k)]
                tuples, err = evaluate(res, plan)
                if err is not None:
                    sample_errors[k] = err
                for q in plan.questions:
                    per_q[q.qid].append(tuples[q.qid])
                if isinstance(res, Exception):
                    ms_list.append(None)
                    cached.append(None)
                    continue
                resp = res["resp"]
                ms_list.append(res["ms"])
                u = resp.get("usage") or {}
                for name in usage:
                    usage[name] += u.get(name) or 0
                if prompt_tokens is None:
                    prompt_tokens = u.get("prompt_tokens")
                cached.append((u.get("prompt_tokens_details") or {}).get("cached_tokens"))
            for q in plan.questions:
                use = per_q[q.qid]
                if adaptive_diag is not None and q.qid not in adaptive_diag["replaced"]:
                    use = use[:1]  # 適応: 置き換え対象でない質問は1回目の値
                answer, error = aggregate_question(q, use)
                if answer is not None:
                    answers[q.qid] = answer
                else:
                    errors[q.qid] = error
                label_mass[q.qid] = [s[1] for s in per_q[q.qid]]
                per_sample[q.qid] = [
                    None if s[0] is None else (s[0][q.yes_label] if q.kind == "noul" else dict(zip([n for n, _ in q.options], [s[0][lab] for lab in q.labels])))
                    for s in per_q[q.qid]
                ]
            diag = {
                "questions": [q.qid for q in plan.questions],
                "canvas_width": plan.width,
                "ms_per_sample": ms_list,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached,
            }
            if sample_errors:
                diag["sample_errors"] = sample_errors
            reads_diag.append(diag)
        aliases_diag: dict = {}
        conflicts_diag: dict = {}
        for plan in plans:
            for q, slot in zip(plan.questions, plan.template.slots):
                names = {lab: n for lab, (n, _) in zip(q.labels, q.options)} if q.kind == "choice" else {}
                aliases_diag[q.qid] = {names.get(lab, lab): slot.alias_ids(lab) for lab in q.labels}
                if slot.conflicts:
                    conflicts_diag[q.qid] = slot.conflicts
        return {
            "model": req.model or self.cfg.model,
            "answers": {q.qid: answers[q.qid] for q in req.questions if q.qid in answers},
            "errors": errors,
            "usage": usage,
            "diagnostics": {
                "template": req.template,
                "seed": req.seed,
                "samples": req.samples,
                "steps": req.steps,
                "reads": reads_diag,
                "alias_ids": aliases_diag,
                "alias_conflicts": conflicts_diag,
                "label_mass": label_mass,
                "per_sample": per_sample,
                "tokenize_ms": tokenize_ms,
                **({"adaptive": adaptive_diag} if adaptive_diag is not None else {}),
                "total_ms": (time.perf_counter_ns() - t0) / 1e6,
            },
        }


# ---------------------------------------------------------------- HTTP(vLLM 側と自前の受け口)


def post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise UpstreamError(f"upstream HTTP {e.code} from {url}: {detail}") from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise UpstreamError(f"upstream request to {url} failed: {type(e).__name__}: {e}") from e


def make_engine(cfg: Config, vllm_url: str) -> Engine:
    base = vllm_url.rstrip("/")

    def tokenize(text: str) -> list[int]:
        out = post_json(f"{base}/tokenize", {"prompt": text, "add_special_tokens": False}, cfg.timeout)
        tokens = out.get("tokens")
        if not isinstance(tokens, list):
            raise UpstreamError(f"/tokenize returned no 'tokens': {str(out)[:200]}")
        return tokens

    def chat(payload: dict) -> dict:
        return post_json(f"{base}/v1/chat/completions", payload, cfg.timeout)

    return Engine(cfg, tokenize, chat)


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, obj: dict) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            if self.path != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > MAX_BODY_BYTES:
                    raise RequestError(400, f"Content-Length must be 1..{MAX_BODY_BYTES}")
                try:
                    body = json.loads(self.rfile.read(n).decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as e:
                    raise RequestError(400, f"invalid JSON: {e}") from e
                self._send(200, engine.handle(body))
            except RequestError as e:
                self._send(e.status, {"error": e.message})
            except UpstreamError as e:
                self._send(502, {"error": str(e)[:1000]})
            except Exception as e:  # 想定外でも必ず JSON のエラーを返す
                self._send(500, {"error": f"internal error: {type(e).__name__}: {e}"[:1000]})

        def log_message(self, fmt, *args) -> None:  # 標準エラーへ1行(既定の形式)
            super().log_message(fmt, *args)

    return Handler


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--vllm", default="http://127.0.0.1:8000", help="vLLM のベースURL(/tokenize と /v1/chat/completions)")
    p.add_argument("--model", default="dgemma", help="vLLM に送る model 名")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8012)
    p.add_argument("--canvas", type=int, default=256, help="vLLM 側の canvas_length(serve 時の値)")
    p.add_argument("--max-per-read", type=int, default=0, help="1回の読み出しに入れる質問数の上限。0=全質問を1回で")
    p.add_argument("--min-label-mass", type=float, default=0.5, help="質問のラベル質量がこの値未満のサンプルは無効")
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="ノイズを引く語彙サイズ")
    p.add_argument("--pad-id", type=int, default=DEFAULT_PAD_ID)


def serve(args: argparse.Namespace) -> int:
    cfg = Config(
        canvas=args.canvas,
        max_per_read=args.max_per_read,
        min_label_mass=args.min_label_mass,
        vocab_size=args.vocab_size,
        pad_id=args.pad_id,
        model=args.model,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(make_engine(cfg, args.vllm)))
    print(f"dgemma-server on http://{args.host}:{args.port}/v1/systemone -> {args.vllm}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
