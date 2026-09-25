"""選択式判定: 1文字の回答ラベルを logprobs から集計し、相対スコアを得る。

失敗(選択肢トークンが出ない/thinkingが先に出る/logprobsが欠ける)は例外として
呼び出し側(pipeline)が軸ごとに捕捉し、記録する。任意の選択肢へ強制しない。
"""

from __future__ import annotations

import base64
import json
import math
import urllib.error

from .taxonomy import Axis, Choice

# yes/no の通信失敗を候補単位で捕捉する対象(pipeline側の軸単位の捕捉と同じ集合)
REQUEST_EXCEPTIONS = (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError)

LABELS = list("ABCDEFGHIJKLMNOPQRST")  # 最大20ラベル

SYSTEM_PROMPT = "You are an image classifier. Reply with one letter only."

CHOOSE_PARAMS = {"max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20}
YESNO_PARAMS = {"max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20}

# 複数選択(character軸)の定数
CANDIDATE_FLOOR_RATIO = 0.001
MAX_CANDIDATES = 4
YES_THRESHOLD = 0.5

NONE_ID = "__none__"


class DecisionError(Exception):
    def __init__(self, error_type: str, detail: str):
        self.error_type = error_type
        self.detail = detail
        super().__init__(f"{error_type}: {detail}")


def _data_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _messages(image_bytes: bytes, mime: str, user_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_url(image_bytes, mime)}},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def extract_top_logprobs(response: dict) -> list[dict]:
    """response から先頭トークンの top_logprobs を取り出す。
    thinking で埋まった/logprobs が欠けたケースは DecisionError を送出する。
    """
    choices = response.get("choices") or []
    if not choices:
        raise DecisionError("no_logprobs", "empty choices in response")

    top_choice = choices[0]
    logprobs = top_choice.get("logprobs")
    content = (logprobs or {}).get("content") or []
    if not content:
        message = top_choice.get("message") or {}
        reasoning_content = message.get("reasoning_content") or message.get("reasoning")
        usage = response.get("usage") or {}
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        if reasoning_content or reasoning_tokens:
            raise DecisionError("thinking_before_answer", "reasoning content/tokens present before answer")
        raise DecisionError("no_logprobs", "logprobs.content is empty")

    return content[0].get("top_logprobs") or []


def pool_labels(top_logprobs: list[dict], labels: list[str]) -> dict[str, float]:
    """`A`/` A` のような表記差を吸収して exp(logprob) を合算し、
    列挙した labels 内で再正規化した相対スコアを返す。
    """
    mass = {label: 0.0 for label in labels}
    for tok in top_logprobs:
        token = tok.get("token", "")
        letter = token.strip().upper()
        if letter in mass:
            try:
                mass[letter] += math.exp(tok["logprob"])
            except OverflowError:
                mass[letter] = math.inf

    total = sum(mass.values())
    if total == 0 or not math.isfinite(total):
        observed = [repr(t.get("token", "")) for t in top_logprobs[:6]]
        raise DecisionError("no_label_tokens", f"no listed label token observed; top tokens: {observed}")

    return {label: value / total for label, value in mass.items()}


def _build_choice_prompt(question: str, choices: list[Choice], labels: list[str], allow_none: bool) -> str:
    lines = [f"{labels[i]}. {c.name} - {c.criteria}" for i, c in enumerate(choices)]
    if allow_none:
        lines.append(f"{labels[len(choices)]}. none of the above")
    return f"{question}\n\n" + "\n".join(lines) + "\n\nAnswer with the single letter only."


def choose(
    backend,
    image_bytes: bytes,
    mime: str,
    question: str,
    choices: list[Choice],
    allow_none: bool,
) -> dict:
    """1回の choose リクエストを送り、候補id(+ allow_none なら NONE_ID)ごとの
    相対スコアと argmax の selected を返す。"""
    n = len(choices)
    total_labels = n + (1 if allow_none else 0)
    labels = LABELS[:total_labels]

    prompt = _build_choice_prompt(question, choices, labels, allow_none)
    messages = _messages(image_bytes, mime, prompt)

    response, elapsed_ms = backend.chat(messages, **CHOOSE_PARAMS)
    top_logprobs = extract_top_logprobs(response)
    scores_by_label = pool_labels(top_logprobs, labels)

    id_by_label = {labels[i]: choices[i].id for i in range(n)}
    if allow_none:
        id_by_label[labels[n]] = NONE_ID

    relative_scores = {id_by_label[label]: score for label, score in scores_by_label.items()}
    selected = max(relative_scores, key=relative_scores.get)
    return {"relative_scores": relative_scores, "selected": selected, "elapsed_ms": elapsed_ms}


def yes_no(backend, image_bytes: bytes, mime: str, name: str, criteria: str) -> dict:
    question = f"Is {name} ({criteria}) present in the image?"
    prompt = f"{question}\n\nA. yes\nB. no\n\nAnswer with the single letter only."
    messages = _messages(image_bytes, mime, prompt)

    response, elapsed_ms = backend.chat(messages, **YESNO_PARAMS)
    top_logprobs = extract_top_logprobs(response)
    scores = pool_labels(top_logprobs, ["A", "B"])
    return {"p_yes": scores["A"], "p_no": scores["B"], "elapsed_ms": elapsed_ms}


def decide_axis(backend, image_bytes: bytes, mime: str, axis: Axis) -> dict:
    """単一選択軸の判定(multi=False の軸すべてに共通)。"""
    result = choose(backend, image_bytes, mime, axis.question, axis.choices, axis.allow_none)
    return {
        "relative_scores": result["relative_scores"],
        "selected": result["selected"],
        "elapsed_ms": result["elapsed_ms"],
    }


def decide_multi_axis(backend, image_bytes: bytes, mime: str, axis: Axis) -> dict:
    """複数選択軸の判定(multi=True の軸すべてに共通。character のほか outfit 等も対象)。
    ランキング1回 + 候補ごとの yes/no。"""
    ranking = choose(backend, image_bytes, mime, axis.question, axis.choices, axis.allow_none)
    relative_scores = ranking["relative_scores"]

    catch_all_ids = {c.id for c in axis.choices if c.catch_all}
    non_none_ids = [c.id for c in axis.choices]  # __none__ はそもそも axis.choices に含まれない
    candidate_ids = [cid for cid in non_none_ids if cid not in catch_all_ids]

    max_score = max((relative_scores[cid] for cid in candidate_ids), default=0.0)
    floor = max_score * CANDIDATE_FLOOR_RATIO
    filtered = sorted(
        (cid for cid in candidate_ids if relative_scores[cid] >= floor),
        key=lambda cid: relative_scores[cid],
        reverse=True,
    )
    candidates = filtered[:MAX_CANDIDATES]

    by_id = {c.id: c for c in axis.choices}
    confirmations: dict[str, float] = {}
    confirmation_errors: dict[str, str] = {}
    total_elapsed_ms = ranking["elapsed_ms"]
    for cid in candidates:
        c = by_id[cid]
        try:
            yn = yes_no(backend, image_bytes, mime, c.name, c.criteria)
        except DecisionError as e:
            confirmation_errors[cid] = f"{e.error_type}: {e.detail}"
            continue
        except REQUEST_EXCEPTIONS as e:
            confirmation_errors[cid] = f"{type(e).__name__}: {e}"
            continue
        confirmations[cid] = yn["p_yes"]
        total_elapsed_ms += yn["elapsed_ms"]

    tags = [cid for cid in candidates if confirmations.get(cid, 0.0) >= YES_THRESHOLD]

    # 確認(yes/no)が1件でも失敗した場合、この軸は失敗として扱う。
    # 未確認のまま catch_all へフォールバックしない(確認できなかったことを隠さない)。
    failed = bool(confirmation_errors)

    if not tags and not failed:
        # __none__ を含む全体の argmax が catch_all の場合だけ、catch_all をフォールバックにする。
        # __none__ 自体が全体最高ならタグなしのままにする。
        overall_top = max(relative_scores, key=relative_scores.get)
        if overall_top in catch_all_ids:
            tags = [overall_top]

    return {
        "relative_scores": relative_scores,
        "confirmations": confirmations,
        "confirmation_errors": confirmation_errors,
        "candidates": candidates,
        "tags": tags,
        "failed": failed,
        "elapsed_ms": total_elapsed_ms,
    }
