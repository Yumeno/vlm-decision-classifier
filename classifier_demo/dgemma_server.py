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

    @property
    def labels(self) -> list[str]:
        return CHOICE_LABELS[: len(self.options)] if self.kind == "choice" else NOUL_LABELS


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


@dataclass
class Slot:
    qid: str
    pos: int
    label_ids: dict[str, int]


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
            questions.append(Question(qid, "noul", instructions, []))
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
    if template not in TEMPLATES:
        raise RequestError(400, f"'template' must be one of {TEMPLATES}")
    mpr = body.get("max_per_read")
    if mpr is not None:
        mpr = _int_field(body, "max_per_read", 0, 0)
    mm = body.get("mm_processor_kwargs")
    if mm is not None and not isinstance(mm, dict):
        raise RequestError(400, "'mm_processor_kwargs' must be an object")
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


def build_template(tok: Tokenizer, keys: list[str], questions: list[Question]) -> Template:
    head = tok(SCAFFOLD_TEXT)
    turn = tok(TURN_CLOSE_TEXT)
    if len(turn) != 1:
        raise RequestError(422, f"{TURN_CLOSE_TEXT!r} must be a single token (got {len(turn)})")
    tokens = list(head)
    slots: list[Slot] = []
    for key, q in zip(keys, questions):
        ids, pos, label_ids = line_tokens(tok, key, q.labels)
        slots.append(Slot(q.qid, len(tokens) + pos, label_ids))
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
    for entry in content[slot.pos].get("top_logprobs") or []:
        tid = parse_token_id(entry.get("token"))
        if tid is not None and isinstance(entry.get("logprob"), (int, float)):
            seen[tid] = float(entry["logprob"])
    lps = {}
    for lab in labels:
        lp = seen.get(slot.label_ids[lab])
        if lp is None:
            return None, None, "missing_label_logprob"
        lps[lab] = lp
    weights = {lab: math.exp(lp) for lab, lp in lps.items()}
    mass = sum(weights.values())
    if mass < min_mass:
        return None, mass, "label_mass_low"
    if mass <= 0:
        return None, mass, "label_mass_low"
    return {lab: w / mass for lab, w in weights.items()}, mass, None


def aggregate_question(q: Question, samples: list[tuple]) -> tuple[dict | None, dict | None]:
    """有効サンプルを平均して (answer, error) のどちらかを返す。"""
    valid = [s[0] for s in samples if s[0] is not None]
    if not valid:
        reasons = Counter(s[2] or "unknown" for s in samples)
        top = reasons.most_common(1)[0][0]
        return None, {"type": top, "detail": f"no valid sample out of {len(samples)}: {dict(reasons)}"}
    avg = {lab: sum(p[lab] for p in valid) / len(valid) for lab in valid[0]}
    if q.kind == "noul":
        return {"type": "noul", "noul": avg["yes"]}, None
    best = max(q.labels, key=lambda lab: avg[lab])  # 同点は先頭
    names = {lab: name for lab, (name, _) in zip(q.labels, q.options)}
    return {
        "type": "choice",
        "choice": names[best],
        "probabilities": {names[lab]: avg[lab] for lab in q.labels},  # Jev の外部仕様の名前。中身は相対スコア
    }, None


# ---------------------------------------------------------------- プロンプト


def question_block(keys: list[str], questions: list[Question]) -> str:
    lines = []
    for key, q in zip(keys, questions):
        lines.append(f"{key}: {q.instructions}")
        if q.kind == "choice":
            for lab, (name, desc) in zip(q.labels, q.options):
                lines.append(f"  {lab}: {name} ({desc})" if desc else f"  {lab}: {name}")
        else:
            lines.append("  yes / no")
    return "\n".join(lines)


def read_keys(template: str, questions: list[Question]) -> list[str]:
    if template == "numbered":
        return [f"Q{i}" for i in range(1, len(questions) + 1)]
    return [q.qid for q in questions]


def build_messages(req: Request, block: str) -> list[dict]:
    content = [{"type": "image_url", "image_url": {"url": u}} for u in req.images]
    content.append({"type": "text", "text": req.state})
    content.append({"type": "text", "text": block})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


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
            ids = sorted({i for s in template.slots for i in s.label_ids.values()})
            if len(ids) > MAX_LOGPROB_TOKEN_IDS:
                raise RequestError(422, f"too many distinct label token ids ({len(ids)} > {MAX_LOGPROB_TOKEN_IDS})")
            plans.append(ReadPlan(group, template, width, ids, question_block(keys, group)))
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

        tasks = [(r, k) for r in range(len(plans)) for k in range(req.samples)]
        results: dict[tuple[int, int], dict | Exception] = {}

        def run(task):
            try:
                results[task] = self.run_one(req, plans[task[0]], *task)
            except UpstreamError as e:
                results[task] = e

        run(tasks[0])  # 先頭の1回で画像プレフィックスをキャッシュに載せてから残りを並列に
        if len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(run, tasks[1:]))
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
            for k in range(req.samples):
                res = results[(r, k)]
                if isinstance(res, Exception):
                    ms_list.append(None)
                    cached.append(None)
                    sample_errors[k] = str(res)[:300]
                    for q in plan.questions:
                        per_q[q.qid].append((None, None, "upstream_error"))
                    continue
                resp, ms = res["resp"], res["ms"]
                ms_list.append(ms)
                u = resp.get("usage") or {}
                for name in usage:
                    usage[name] += u.get(name) or 0
                if prompt_tokens is None:
                    prompt_tokens = u.get("prompt_tokens")
                cached.append((u.get("prompt_tokens_details") or {}).get("cached_tokens"))
                try:
                    content = resp["choices"][0]["logprobs"]["content"]
                    if not isinstance(content, list):
                        raise TypeError("content is not a list")
                except (KeyError, IndexError, TypeError) as e:
                    content = None
                    sample_errors[k] = f"no logprobs.content in response ({type(e).__name__})"
                for q, slot in zip(plan.questions, plan.template.slots):
                    if content is None:
                        per_q[q.qid].append((None, None, "no_logprobs"))
                    else:
                        per_q[q.qid].append(read_slot(content, slot, q.labels, self.cfg.min_label_mass))
            for q in plan.questions:
                answer, error = aggregate_question(q, per_q[q.qid])
                if answer is not None:
                    answers[q.qid] = answer
                else:
                    errors[q.qid] = error
                label_mass[q.qid] = [s[1] for s in per_q[q.qid]]
                per_sample[q.qid] = [
                    None if s[0] is None else (s[0]["yes"] if q.kind == "noul" else dict(zip([n for n, _ in q.options], [s[0][lab] for lab in q.labels])))
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
        return {
            "model": req.model or self.cfg.model,
            "answers": {q.qid: answers[q.qid] for q in req.questions if q.qid in answers},
            "errors": errors,
            "usage": usage,
            "diagnostics": {
                "template": req.template,
                "seed": req.seed,
                "samples": req.samples,
                "reads": reads_diag,
                "label_mass": label_mass,
                "per_sample": per_sample,
                "tokenize_ms": tokenize_ms,
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
