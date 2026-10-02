"""DiffusionGemma(vLLM)用の評価方式 dgemma_choice(issue #4)。

vLLM の example「structured server」(`/v1/systemone`、Jev 形状)に、画像1枚につき全軸を1リクエストで投げる。
サーバーが選択肢ラベルの付与・ノイズ draw の平均(samples)を行い、`probabilities` / `noul` を返す。
値は候補内の相対スコア(単一選択)と P(yes)(複数選択の選択肢ごとの yes/no)で、正答確率ではない。

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
import time
import urllib.error
import urllib.request

from .backend import ChatBackend
from .decision import NONE_ID, YES_THRESHOLD, DecisionError, _data_url
from .taxonomy import Axis, Taxonomy

STATE_TEXT = "Classify the attached image."
NONE_NAME = "none of the above"
SYSTEMONE_PATH = "/v1/systemone"


class DgemmaBackend(ChatBackend):
    """base_url は vLLM(`http://127.0.0.1:8000/v1`)、structured_url は example サーバー(`http://127.0.0.1:8011`)。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        structured_url: str | None = None,
        samples: str | int = "auto",
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
        # vLLM は拡散モデルへの temperature 等を拒否(400)するため送らない。run.json に記録する。
        self.dropped_params = ["temperature"]

    def chat(self, messages: list, **params) -> tuple[dict, float]:
        for k in self.dropped_params:
            params.pop(k, None)
        return super().chat(messages, **params)

    def systemone(self, body: dict) -> tuple[dict, float]:
        """/v1/systemone に POST する。HTTP エラーは本文付きの DecisionError にする。"""
        if not self.structured_url:
            raise DecisionError("request_error", "structured server URL is not set (--dgemma-url)")
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


def build_request(
    taxonomy: Taxonomy, image_bytes: bytes, mime: str, model: str, samples: str | int = "auto"
) -> tuple[dict, dict]:
    """(リクエスト本体, 返答を読むための対応表)を返す。対応表は軸id -> {"qid", "names"}(単一)か
    {"qids"}(複数)。"""
    questions: dict[str, dict] = {}
    plan: dict[str, dict] = {}

    def new_qid() -> str:
        # 質問が11以上だとサーバーは "id label" を空白区切りで書き、"outfit.swimsuit" のような
        # 長いIDではラベルが1トークンに分かれない(422)ため、短い q<連番> を送り plan で軸・選択肢idへ戻す。
        return f"q{len(questions)}"

    for axis in taxonomy.axes:
        if axis.multi:
            qids = {}
            for c in axis.choices:
                qid = new_qid()
                qids[c.id] = qid
                questions[qid] = {
                    "type": "noul",
                    "instructions": f"Is {c.name} ({c.criteria}) present in the image?",
                }
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
            qid = new_qid()
            questions[qid] = {"type": "choice", "instructions": axis.question, "criteria": criteria}
            plan[axis.id] = {"qid": qid, "names": names}
    body = {
        "model": model,
        "state": STATE_TEXT,
        "questions": questions,
        "images": [_data_url(image_bytes, mime)],
        "samples": samples,
    }
    return body, plan


def _read_single(axis: Axis, spec: dict, answers: dict) -> dict:
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
    return {"relative_scores": scores, "selected": spec["names"][chosen]}


def _read_multi(axis: Axis, spec: dict, answers: dict) -> dict:
    confirmations: dict[str, float] = {}
    for cid, qid in spec["qids"].items():
        ans = answers.get(qid)
        if not isinstance(ans, dict) or ans.get("type") != "noul" or not isinstance(ans.get("noul"), (int, float)):
            raise ValueError(f"no noul answer for question {qid!r}")
        confirmations[cid] = float(ans["noul"])
    tags = sorted(
        (cid for cid, p in confirmations.items() if p >= YES_THRESHOLD),
        key=lambda cid: confirmations[cid],
        reverse=True,
    )
    return {
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
    results: dict[str, dict] = {}
    for axis in taxonomy.axes:
        try:
            read = _read_multi if axis.multi else _read_single
            results[axis.id] = read(axis, plan[axis.id], answers)
        except (ValueError, TypeError) as e:
            results[axis.id] = {"error": {"type": "dgemma_format_error", "detail": f"axis {axis.id}: {e}"}}
    return results


def decide_dgemma(backend: DgemmaBackend, image_bytes: bytes, mime: str, taxonomy: Taxonomy) -> dict:
    """1リクエストで全軸を判定する。通信・HTTP失敗は例外(呼び出し側 pipeline が全軸の失敗として記録)。"""
    body, plan = build_request(taxonomy, image_bytes, mime, backend.model, backend.samples)
    response, elapsed_ms = backend.systemone(body)
    results = parse_response(taxonomy, plan, response)
    diag = response.get("diagnostics") or {}
    return {
        "axes": results,
        "elapsed_ms": elapsed_ms,
        "info": {
            "request_ms": elapsed_ms,
            "samples_setting": backend.samples,
            "samples_n": (diag.get("samples") or {}).get("n"),
            "reads": (diag.get("timing") or {}).get("reads"),
            "server_total_ms": (diag.get("timing") or {}).get("total_ms"),
            "usage": response.get("usage"),
        },
    }
