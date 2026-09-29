"""選択式判定: 1文字の回答ラベルを logprobs から集計し、相対スコアを得る。

失敗(選択肢トークンが出ない/thinkingが先に出る/logprobsが欠ける)は例外として
呼び出し側(pipeline)が軸ごとに捕捉し、記録する。任意の選択肢へ強制しない。
"""

from __future__ import annotations

import base64
import json
import math
import re
import urllib.error

from .taxonomy import Axis, Choice

# yes/no の通信失敗を候補単位で捕捉する対象(pipeline側の軸単位の捕捉と同じ集合)
REQUEST_EXCEPTIONS = (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError)

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("abcdefghijklmnopqrstuvwxyz")  # 最大52ラベル
CASE_INSENSITIVE_LABEL_LIMIT = 26  # これ以下はA-Zのみで大文字小文字を区別しない照合(従来どおり)

SYSTEM_PROMPT = "You are an image classifier. Reply with one letter only."

CHOOSE_PARAMS = {"max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20}
YESNO_PARAMS = {"max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20}

# E9 ホットロード用: サーバーの画像キャッシュに載せるだけの準備リクエスト。
# システム文・画像部分は choose()/yes_no() と同じ _messages() を使うこと(キャッシュの先頭一致のため)。
PRIME_TEXT = "Reply with A."
PRIME_PARAMS = {"max_tokens": 1, "temperature": 0}

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


def _logprob_content(response: dict) -> list[dict]:
    """response から logprobs.content(出力トークンごとのエントリ)を取り出す。
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
    return content


def extract_top_logprobs(response: dict) -> list[dict]:
    """response から先頭トークンの top_logprobs を取り出す。"""
    return _logprob_content(response)[0].get("top_logprobs") or []


def pool_labels(top_logprobs: list[dict], labels: list[str]) -> dict[str, float]:
    """`A`/` A` のような表記差を吸収して exp(logprob) を合算し、
    列挙した labels 内で再正規化した相対スコアを返す。

    labels が26以下(A-Zのみ)なら、従来どおり strip + upper で大文字小文字を区別せずに
    照合する(E1〜E4等の既存実験の再現性のため)。27以上(a-zを含む)なら、'A'と'a'を
    区別する必要があるため strip のみで大文字小文字を区別して照合する。
    """
    case_insensitive = len(labels) <= CASE_INSENSITIVE_LABEL_LIMIT
    mass = {label: 0.0 for label in labels}
    for tok in top_logprobs:
        token = tok.get("token", "")
        letter = token.strip()
        if case_insensitive:
            letter = letter.upper()
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


CHOICE_ANSWER_SUFFIX = "\n\nAnswer with the single letter only."


def _build_choice_prompt(
    question: str,
    choices: list[Choice],
    labels: list[str],
    allow_none: bool,
    none_criteria: str | None = None,
) -> str:
    lines = [f"{labels[i]}. {c.name} - {c.criteria}" for i, c in enumerate(choices)]
    if allow_none:
        none_label = labels[len(choices)]
        if none_criteria:
            lines.append(f"{none_label}. none of the above - {none_criteria}")
        else:
            lines.append(f"{none_label}. none of the above")
    return f"{question}\n\n" + "\n".join(lines) + CHOICE_ANSWER_SUFFIX


def choose(
    backend,
    image_bytes: bytes,
    mime: str,
    question: str,
    choices: list[Choice],
    allow_none: bool,
    none_criteria: str | None = None,
) -> dict:
    """1回の choose リクエストを送り、候補id(+ allow_none なら NONE_ID)ごとの
    相対スコアと argmax の selected を返す。"""
    n = len(choices)
    total_labels = n + (1 if allow_none else 0)
    labels = LABELS[:total_labels]

    prompt = _build_choice_prompt(question, choices, labels, allow_none, none_criteria)
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


def prime(backend, image_bytes: bytes, mime: str) -> dict:
    """E9 ホットロード用: 画像をサーバーのキャッシュに載せるだけの準備リクエストを送る。
    choose()/yes_no() と同じ _messages() で system文+画像を組み立てるため、後続の判定
    リクエストとキャッシュの先頭が一致する。logprobs は不要(判定に使わないため)。
    失敗は例外のまま呼び出し側(evaluate/server)に投げる。"""
    messages = _messages(image_bytes, mime, PRIME_TEXT)
    _response, elapsed_ms = backend.chat(messages, **PRIME_PARAMS)
    return {"elapsed_ms": elapsed_ms}


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
    result = choose(
        backend, image_bytes, mime, axis.question, axis.choices, axis.allow_none, axis.none_criteria
    )
    return {
        "relative_scores": result["relative_scores"],
        "selected": result["selected"],
        "elapsed_ms": result["elapsed_ms"],
    }


def _rank_threshold_candidates(axis: Axis, relative_scores: dict[str, float], rank_threshold: float) -> list[str]:
    """確認オフ時の採用候補: 相対スコアが rank_threshold 以上の非 catch_all をスコア降順で。"""
    return sorted(
        (c.id for c in axis.choices if not c.catch_all and relative_scores[c.id] >= rank_threshold),
        key=lambda cid: relative_scores[cid],
        reverse=True,
    )


def _catch_all_fallback(axis: Axis, relative_scores: dict[str, float]) -> list[str]:
    """タグが空のとき、__none__ を含む全体の argmax が catch_all の場合だけ catch_all を返す。
    __none__ 自体が全体最高ならタグなし(空)のまま。"""
    overall_top = max(relative_scores, key=relative_scores.get)
    if overall_top in {c.id for c in axis.choices if c.catch_all}:
        return [overall_top]
    return []


def decide_multi_axis(
    backend, image_bytes: bytes, mime: str, axis: Axis, confirm: bool, rank_threshold: float = 0.5
) -> dict:
    """複数選択軸の判定(multi=True の軸すべてに共通。character のほか outfit 等も対象)。

    confirm=True: ランキング1回 + 候補ごとの yes/no 確認(E1〜E4等の既存実験と完全に同じ判定)。
    confirm=False: yes/no を送らず、ランキングの相対スコアが rank_threshold 以上の
    (catch_all でない)候補をスコア降順で採用する。確認を行わないため confirmations は
    常に空で、failed は常に False(通信失敗の余地がランキング1回分しかなく、それは
    decide_multi_axis 自体の例外として呼び出し側に伝播する)。
    """
    ranking = choose(
        backend, image_bytes, mime, axis.question, axis.choices, axis.allow_none, axis.none_criteria
    )
    relative_scores = ranking["relative_scores"]

    catch_all_ids = {c.id for c in axis.choices if c.catch_all}
    non_none_ids = [c.id for c in axis.choices]  # __none__ はそもそも axis.choices に含まれない
    candidate_ids = [cid for cid in non_none_ids if cid not in catch_all_ids]

    if confirm:
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
    else:
        candidates = _rank_threshold_candidates(axis, relative_scores, rank_threshold)
        confirmations = {}
        confirmation_errors = {}
        total_elapsed_ms = ranking["elapsed_ms"]
        tags = list(candidates)
        failed = False

    if not tags and not failed:
        tags = _catch_all_fallback(axis, relative_scores)

    return {
        "relative_scores": relative_scores,
        "confirmations": confirmations,
        "confirmation_errors": confirmation_errors,
        "candidates": candidates,
        "tags": tags,
        "failed": failed,
        "elapsed_ms": total_elapsed_ms,
    }


BUNDLED_INSTRUCTION = (
    "Answer each question above on its own line, one line per question, in the order of the questions. "
    "Write each line as the question number, a colon, a single space, and the answer letter "
    "(format: \"<question number>: <letter>\"). Write nothing else."
)


BUNDLED_YN_INSTRUCTION = (
    "Answer with exactly the following lines, in this order, "
    "replacing each ? with the letter of your answer. Write nothing else."
)


def _axis_labels(axis: Axis) -> list[str]:
    return LABELS[: len(axis.choices) + (1 if axis.allow_none else 0)]


def _sub_key(k: int, j: int) -> str:
    """yn の複数選択軸のサブ質問番号: 軸番号 + 小文字(6a, 6b, ...)。"""
    return f"{k}{chr(ord('a') + j)}"


def _build_bundled_prompt(axes: list[Axis], multi_mode: str = "rank") -> str:
    """全軸を番号付きで並べる。各軸の質問・選択肢の書式は _build_choice_prompt と同じ
    (末尾の "Answer with the single letter only." だけ除き、束ね用の指示を最後に1回置く)。
    multi_mode="yn" の複数選択軸は、選択肢ごとの yes/no サブ質問(yes_no と同じ文面)を並べる。"""
    blocks = []
    keys = []
    for i, axis in enumerate(axes, start=1):
        if multi_mode == "yn" and axis.multi:
            for j, c in enumerate(axis.choices):
                keys.append(_sub_key(i, j))
                blocks.append(
                    f"Question {_sub_key(i, j)}:\n"
                    f"Is {c.name} ({c.criteria}) present in the image?\n\nA. yes\nB. no"
                )
            continue
        body = _build_choice_prompt(
            axis.question, axis.choices, _axis_labels(axis), axis.allow_none, axis.none_criteria
        )
        body = body.removesuffix(CHOICE_ANSWER_SUFFIX)
        keys.append(str(i))
        blocks.append(f"Question {i}:\n" + body)
    if multi_mode == "yn":
        # 欄が多く番号に小文字が混ざるので、回答の雛形を明示する
        instruction = BUNDLED_YN_INSTRUCTION + "\n" + "\n".join(f"{k}: ?" for k in keys)
    else:
        instruction = BUNDLED_INSTRUCTION
    return "\n\n".join(blocks) + "\n\n" + instruction


def _normalize_label_token(token: str, labels: list[str]) -> str:
    """pool_labels と同じ照合規則(26以下は大文字小文字を区別しない)でトークンを正規化する。"""
    letter = token.strip()
    return letter.upper() if len(labels) <= CASE_INSENSITIVE_LABEL_LIMIT else letter


def decide_bundled(
    backend, image_bytes: bytes, mime: str, taxonomy, rank_threshold: float = 0.5, multi_mode: str = "rank"
) -> dict:
    """E7 束ね質問: 1リクエストで全軸を聞き、軸ごとに1トークンずつ答えさせる。
    各出力位置の top_logprobs を、その軸の選択肢の相対スコアとして読む(確認オフのルール)。

    multi_mode="yn"(E7e): 複数選択軸は選択肢ごとの yes/no サブ質問(6a, 6b, ...)にし、
    各サブ質問の A(yes) の相対スコアを P(yes) として YES_THRESHOLD 以上をすべて採用する
    (catch_all も通常の候補。1つも無ければ空)。サブ質問が1つでも見つからない軸は
    bundled_format_error(欠けた選択肢を detail に)。"rank" は従来どおり。

    ラベルが見つからなかった軸は error={"type": "bundled_format_error", ...} を持ち、
    任意の選択肢に強制しない(見つかった軸は判定する)。thinking/logprobs欠損は DecisionError。
    """
    if multi_mode not in ("rank", "yn"):
        raise ValueError(f"unknown multi_mode: {multi_mode}")
    axes = taxonomy.axes
    yn_axis = {a.id: multi_mode == "yn" and a.multi for a in axes}
    prompt = _build_bundled_prompt(axes, multi_mode)
    messages = _messages(image_bytes, mime, prompt)
    n_fields = sum(len(a.choices) if yn_axis[a.id] else 1 for a in axes)
    params = {**CHOOSE_PARAMS, "max_tokens": 5 * n_fields}

    response, elapsed_ms = backend.chat(messages, **params)
    content = _logprob_content(response)
    raw_text = "".join(entry.get("token", "") for entry in content)

    # 質問番号(キー)ごとの、ラベルとして読める記号の集合。yn の複数選択軸は "6a" 等のサブ質問。
    key_labels: dict[str, list[str]] = {}
    for i, axis in enumerate(axes, start=1):
        if yn_axis[axis.id]:
            for j in range(len(axis.choices)):
                key_labels[_sub_key(i, j)] = ["A", "B"]
        else:
            key_labels[str(i)] = _axis_labels(axis)

    # 出力を先頭から連結しながら見て、ラベルのトークンで、直前までの連結テキストが
    # "<番号>:"(末尾の空白は無視。番号は 3 や 6a)で終わるものをその質問の答えとする。
    # 同じ番号は最初のものを採用。
    key_pos: dict[str, int] = {}
    prefix = ""
    for pos, entry in enumerate(content):
        token = entry.get("token", "")
        m = re.search(r"(\d+[a-z]?):$", prefix.rstrip())
        if m and m.group(1) in key_labels:
            labels = key_labels[m.group(1)]
            if m.group(1) not in key_pos and _normalize_label_token(token, labels) in labels:
                key_pos[m.group(1)] = pos
        prefix += token

    results: dict[str, dict] = {}
    for i, axis in enumerate(axes, start=1):
        if yn_axis[axis.id]:
            results[axis.id] = _read_yn_axis(axis, i, content, key_pos, raw_text)
            continue
        if str(i) not in key_pos:
            results[axis.id] = {
                "error": {
                    "type": "bundled_format_error",
                    "detail": f"no label token found for axis {axis.id!r}; output: {raw_text!r}",
                }
            }
            continue
        pos = key_pos[str(i)]
        labels = _axis_labels(axis)
        try:
            scores_by_label = pool_labels(content[pos].get("top_logprobs") or [], labels)
        except DecisionError as e:
            results[axis.id] = {"error": {"type": e.error_type, "detail": e.detail}}
            continue
        id_by_label = {labels[i]: c.id for i, c in enumerate(axis.choices)}
        if axis.allow_none:
            id_by_label[labels[len(axis.choices)]] = NONE_ID
        relative_scores = {id_by_label[lb]: sc for lb, sc in scores_by_label.items()}
        entry = {"relative_scores": relative_scores, "position": pos, "error": None}
        if axis.multi:
            candidates = _rank_threshold_candidates(axis, relative_scores, rank_threshold)
            entry["candidates"] = candidates
            entry["tags"] = list(candidates) or _catch_all_fallback(axis, relative_scores)
        else:
            entry["selected"] = max(relative_scores, key=relative_scores.get)
        results[axis.id] = entry

    return {"axes": results, "raw_text": raw_text, "elapsed_ms": elapsed_ms}


def _read_yn_axis(axis: Axis, k: int, content: list[dict], key_pos: dict[str, int], raw_text: str) -> dict:
    """E7e: 複数選択軸のサブ質問(<k>a, <k>b, ...)から P(yes) を読み、YES_THRESHOLD 以上を採用する。"""
    keys = {c.id: _sub_key(k, j) for j, c in enumerate(axis.choices)}
    missing = [cid for cid, key in keys.items() if key not in key_pos]
    if missing:
        return {
            "error": {
                "type": "bundled_format_error",
                "detail": f"no label token found for sub-questions of axis {axis.id!r}: "
                f"{[keys[c] + '=' + c for c in missing]}; output: {raw_text!r}",
            }
        }
    confirmations: dict[str, float] = {}
    for cid, key in keys.items():
        try:
            scores = pool_labels(content[key_pos[key]].get("top_logprobs") or [], ["A", "B"])
        except DecisionError as e:
            return {"error": {"type": e.error_type, "detail": f"sub-question {key} ({cid}): {e.detail}"}}
        confirmations[cid] = scores["A"]
    tags = sorted(
        (cid for cid, p in confirmations.items() if p >= YES_THRESHOLD),
        key=lambda cid: confirmations[cid],
        reverse=True,
    )
    return {
        "relative_scores": {},
        "confirmations": confirmations,
        "candidates": list(keys),
        "tags": tags,
        "position": {cid: key_pos[key] for cid, key in keys.items()},
        "error": None,
    }
