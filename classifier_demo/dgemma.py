"""DiffusionGemma(vLLM)用の評価方式 dgemma_choice(issue #4)。

自前の dgemma-server(`classifier_demo/dgemma_server.py`、`/v1/systemone`、Jev 形状)に、画像1枚につき全軸を
1リクエストで投げる。サーバーが選択肢ラベルの付与・1キャンバスでの読み出し・ノイズ draw の平均(samples)を行い、
`probabilities` / `noul` を返す(`probabilities` は Jev API の外部仕様の名前で、中身は相対スコア)。
値は候補内の相対スコア(単一選択)と P(yes)(複数選択の選択肢ごとの yes/no)で、正答確率ではない。
質問IDは単一選択=`<axis_id>`、複数選択の選択肢=`<axis_id>_<choice_id>`(サーバーの規則 `^[a-z][a-z0-9_]{0,47}$`)。

- 単一選択軸: `choice` 質問1つ。選択肢名は choice.name(説明は criteria)。allow_none の軸は
  "none of the above" を足し、NONE_ID に対応づける。
- 複数選択軸: 選択肢ごとに `noul` 質問1つ(文面は decision.yes_no と同じ)。P(yes) >= YES_THRESHOLD を採用
  (bundled の yn と同じ規則: catch_all も通常の候補、none は聞かない、1つも無ければ空)。
- `vision_only`: メタデータはモデルに渡さない(画像と質問文だけ)。

dgemma_json は既存の通常JSON方式(classify_json、constrained=False)を vLLM(:8000)へ向けるだけ。
vLLM は拡散モデルへの json_schema を拒否するため制約付きは使わず、DgemmaBackend が
`chat_template_kwargs.enable_thinking=false` を全リクエストに足す。
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from .backend import ChatBackend
from .decision import NONE_ID, YES_THRESHOLD, DecisionError, _data_url
from .taxonomy import Axis, Taxonomy

STATE_TEXT = "Classify the attached image."
NONE_NAME = "none of the above"
SYSTEMONE_PATH = "/v1/systemone"
QID_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")  # dgemma_server.KEY_RE と同じ


class DgemmaBackend(ChatBackend):
    """base_url は vLLM(`http://127.0.0.1:8000/v1`)、structured_url は自前の dgemma-server(`http://127.0.0.1:8012`)。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        structured_url: str | None = None,
        samples: int = 1,
        seed: int = 0,
        template: str = "keyed",
        max_per_read: int = 0,
        max_soft_tokens: int | None = None,
        instruction: str = "default",
        yn_style: str = "slash",
        order: str = "taxonomy",
        adaptive_threshold: float | None = None,
        adaptive_max: int = 3,
        steps: int = 1,
        catchall_style: str = "default",
        timeout: float = 300.0,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            timeout=timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.structured_url = structured_url.rstrip("/") if structured_url else None
        self.samples = samples
        self.seed = seed
        self.template = template
        self.instruction = instruction
        self.yn_style = yn_style
        self.order = order
        self.adaptive_threshold = adaptive_threshold
        self.adaptive_max = adaptive_max
        self.steps = steps
        self.catchall_style = catchall_style
        self.max_per_read = max_per_read
        self.max_soft_tokens = max_soft_tokens
        # vLLM は拡散モデルへの temperature 等を拒否(400)するため送らない。run.json に記録する。
        self.dropped_params = ["temperature"]

    def chat(self, messages: list, **params) -> tuple[dict, float]:
        for k in self.dropped_params:
            params.pop(k, None)
        return super().chat(messages, **params)

    def systemone(self, body: dict) -> tuple[dict, float]:
        """/v1/systemone に POST する。HTTP エラーは本文付きの DecisionError にする。"""
        if not self.structured_url:
            raise DecisionError("request_error", "dgemma-server URL is not set (--dgemma-url)")
        req = urllib.request.Request(
            self.structured_url + SYSTEMONE_PATH,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        self._increment_request_count()
        start = time.perf_counter_ns()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise DecisionError("http_error", f"HTTP {e.code}: {detail}") from e
        return out, (time.perf_counter_ns() - start) / 1_000_000


def _check_qid(qid: str) -> str:
    if not QID_RE.match(qid):
        raise DecisionError("request_error", f"question id {qid!r} must match {QID_RE.pattern} (fix the taxonomy ids)")
    return qid


def _join_names(names: list[str]) -> str:
    """["A"] -> "A"、["A","B"] -> "A and B"、3つ以上は "A, B and C"。"""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def build_request(
    taxonomy: Taxonomy,
    image_bytes: bytes,
    mime: str,
    model: str,
    samples: int = 1,
    seed: int = 0,
    template: str = "keyed",
    max_per_read: int = 0,
    max_soft_tokens: int | None = None,
    instruction: str = "default",
    yn_style: str = "slash",
    order: str = "taxonomy",
    adaptive_threshold: float | None = None,
    adaptive_max: int = 3,
    steps: int = 1,
    catchall_style: str = "default",
) -> tuple[dict, dict]:
    """(リクエスト本体, 返答を読むための対応表)を返す。対応表は軸id -> {"qid", "names"}(単一)か
    {"qids"}(複数)。"""
    questions: dict[str, dict] = {}
    plan: dict[str, dict] = {}

    def add(qid: str, spec: dict) -> str:
        if _check_qid(qid) in questions:
            raise DecisionError("request_error", f"duplicate question id {qid!r}")
        questions[qid] = spec
        return qid

    axes = list(taxonomy.axes)
    if order == "character_first":
        # character 軸を、最初の複数選択軸の直前へ移す(他の並びは変えない)
        chars = [a for a in axes if a.id == "character" and a.multi]
        rest = [a for a in axes if a not in chars]
        first_multi = next((i for i, a in enumerate(rest) if a.multi), len(rest))
        axes = rest[:first_multi] + chars + rest[first_multi:]
    for axis in axes:
        if axis.multi:
            qids = {}
            for c in axis.choices:
                same = c.criteria.strip().lower() == c.name.strip().lower()
                subject = c.name if same else f"{c.name} ({c.criteria})"
                instructions = f"Is {c.name} ({c.criteria}) present in the image?"
                others = [o.name for o in axis.choices if o is not c and not o.catch_all]
                if catchall_style == "criteria" and c.catch_all and others:
                    # criteria 中の "none of the above" を、同じ軸の他の選択肢の列挙に置き換えて「写っているか」と聞く
                    crit = re.sub("none of the above", f"none of these: {_join_names(others)}", c.criteria, flags=re.IGNORECASE)
                    instructions = f"Does the image show {crit}?"
                    subject = crit
                qids[c.id] = add(f"{axis.id}_{c.id}", {"type": "noul", "instructions": instructions, "subject": subject})
            plan[axis.id] = {"qids": qids}
        else:
            criteria: dict[str, str] = {}
            names: dict[str, str] = {}  # 選択肢名 -> choice id
            entries = [(c.name, c.criteria, c.id) for c in axis.choices]
            if axis.allow_none:
                entries.append((NONE_NAME, axis.none_criteria, NONE_ID))
            for name, desc, cid in entries:
                if name in names:
                    raise DecisionError("request_error", f"axis {axis.id}: duplicate option name {name!r}")
                criteria[name] = desc
                names[name] = cid
            qid = add(axis.id, {"type": "choice", "instructions": axis.question, "criteria": criteria})
            plan[axis.id] = {"qid": qid, "names": names}
    body = {
        "model": model,
        "state": STATE_TEXT,
        "questions": questions,
        "images": [_data_url(image_bytes, mime)],
        "samples": samples,
        "seed": seed,
        "template": template,
        "max_per_read": max_per_read,
    }
    if instruction != "default":
        body["instruction"] = instruction
    if yn_style != "slash":
        body["yn_style"] = yn_style
    if adaptive_threshold is not None:
        body["adaptive_threshold"] = adaptive_threshold
        body["adaptive_max"] = adaptive_max
    if steps > 1:
        body["steps"] = steps
    if max_soft_tokens is not None:
        body["mm_processor_kwargs"] = {"max_soft_tokens": max_soft_tokens}
    return body, plan


def _server_error(errors: dict, qid: str) -> None:
    err = errors.get(qid)
    if err is not None:
        raise ValueError(f"server error for question {qid!r}: {err}")


def _read_single(axis: Axis, spec: dict, answers: dict, errors: dict) -> dict:
    _server_error(errors, spec["qid"])
    ans = answers.get(spec["qid"])
    if not isinstance(ans, dict) or ans.get("type") != "choice":
        raise ValueError(f"no choice answer for question {spec['qid']!r}")
    probs = ans.get("probabilities")
    if not isinstance(probs, dict) or set(probs) != set(spec["names"]):
        raise ValueError(f"probabilities keys do not match the options of {spec['qid']!r}")
    scores = {spec["names"][n]: float(p) for n, p in probs.items()}
    chosen = ans.get("choice")
    if chosen not in spec["names"]:
        raise ValueError(f"unknown choice {chosen!r}")
    out = {"relative_scores": scores, "selected": spec["names"][chosen]}
    if "uncertain" in ans:  # 適応が有効なときだけサーバーが付ける。選択結果は変えない
        out["uncertain"] = bool(ans["uncertain"])
        out["entropy"] = ans.get("entropy")
    return out


def _read_multi(axis: Axis, spec: dict, answers: dict, errors: dict) -> dict:
    confirmations: dict[str, float] = {}
    uncertain: list[str] = []
    has_flag = False
    for cid, qid in spec["qids"].items():
        _server_error(errors, qid)  # 選択肢が1つでも失敗した複数選択軸は軸ごと失敗
        ans = answers.get(qid)
        if not isinstance(ans, dict) or ans.get("type") != "noul" or not isinstance(ans.get("noul"), (int, float)):
            raise ValueError(f"no noul answer for question {qid!r}")
        confirmations[cid] = float(ans["noul"])
        if "uncertain" in ans:
            has_flag = True
            if ans["uncertain"]:
                uncertain.append(cid)
    tags = sorted(
        (cid for cid, p in confirmations.items() if p >= YES_THRESHOLD),
        key=lambda cid: confirmations[cid],
        reverse=True,
    )
    extra = {"uncertain_options": uncertain} if has_flag else {}
    return {
        **extra,
        "relative_scores": {},
        "confirmations": confirmations,
        "confirmation_errors": {},
        "candidates": list(spec["qids"]),
        "tags": tags,
        "failed": False,
    }


def parse_response(taxonomy: Taxonomy, plan: dict, response: dict) -> dict[str, dict]:
    """軸id -> 判定結果 か {"error": {"type", "detail"}}。欠けた・形の違う回答はその軸だけ失敗にし、
    任意の選択肢には強制しない。"""
    answers = response.get("answers") if isinstance(response, dict) else None
    if not isinstance(answers, dict):
        raise DecisionError("dgemma_format_error", f"no 'answers' in response: {str(response)[:200]}")
    errors = response.get("errors")
    errors = errors if isinstance(errors, dict) else {}
    results: dict[str, dict] = {}
    for axis in taxonomy.axes:
        try:
            read = _read_multi if axis.multi else _read_single
            results[axis.id] = read(axis, plan[axis.id], answers, errors)
        except (ValueError, TypeError) as e:
            results[axis.id] = {"error": {"type": "dgemma_format_error", "detail": f"axis {axis.id}: {e}"}}
    return results


def decide_dgemma(backend: DgemmaBackend, image_bytes: bytes, mime: str, taxonomy: Taxonomy) -> dict:
    """1リクエストで全軸を判定する。通信・HTTP失敗は例外(呼び出し側 pipeline が全軸の失敗として記録)。"""
    body, plan = build_request(
        taxonomy, image_bytes, mime, backend.model, backend.samples, backend.seed,
        backend.template, backend.max_per_read, backend.max_soft_tokens, backend.instruction,
        backend.yn_style, backend.order, backend.adaptive_threshold, backend.adaptive_max, backend.steps,
        backend.catchall_style,
    )
    response, elapsed_ms = backend.systemone(body)
    results = parse_response(taxonomy, plan, response)
    diag = response.get("diagnostics") or {}
    reads = diag.get("reads") or []
    adaptive = diag.get("adaptive") or {}
    return {
        "axes": results,
        "elapsed_ms": elapsed_ms,
        "info": {
            "request_ms": elapsed_ms,
            "seed": backend.seed,
            "samples": backend.samples,
            "template": backend.template,
            "instruction": backend.instruction,
            "yn_style": backend.yn_style,
            "order": backend.order,
            "steps": backend.steps,
            "catchall_style": backend.catchall_style,
            "adaptive_threshold": backend.adaptive_threshold,
            "adaptive_max": backend.adaptive_max if backend.adaptive_threshold is not None else None,
            "adaptive_triggered": adaptive.get("triggered"),
            "adaptive_reads_used": adaptive.get("reads_used"),
            "adaptive_trigger_questions": adaptive.get("trigger_questions"),
            "adaptive_replaced": adaptive.get("replaced"),
            "adaptive_unresolved": adaptive.get("unresolved"),
            "max_per_read": backend.max_per_read,
            "max_soft_tokens": backend.max_soft_tokens,
            "reads_n": len(reads),
            "read_ms": [r.get("ms_per_sample") for r in reads],
            "cached_tokens": [r.get("cached_tokens") for r in reads],
            "prompt_tokens": [r.get("prompt_tokens") for r in reads],
            "canvas_width": [r.get("canvas_width") for r in reads],
            "server_total_ms": diag.get("total_ms"),
            "tokenize_ms": diag.get("tokenize_ms"),
            "server_errors": response.get("errors") or {},
            "usage": response.get("usage"),
        },
    }
