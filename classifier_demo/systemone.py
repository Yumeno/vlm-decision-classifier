"""Jev(TypeSafe System One)風の呼び出し形状を、ローカル VLM(OpenAI互換サーバー)で再現する最小クライアント(実験的)。

TypeSafe 社および Jev とは無関係。`typesafe-sdk` には依存せず、名前とフィールドだけを写した。
写した名前と出典(typesafe-ai/typesafe-sdk-python 0.7.2、MIT。2026-10-01 取得):
- `client.system_one(state, questions, ...)`  <- _core/client/sync/client.py
- 質問 dict: {"type": "choice"|"noul"|"score", "instructions", "criteria"}  <- _core/question_types.py
  (choice の criteria は {ラベル名: 説明}、score は順序付きリスト、noul の criteria は {"true","false"})。
  SDK の Choice/Noul/Score ヘルパークラスは受け付けず、SDK の形の素の dict だけを受ける。
- 応答: `.model` `.answers`(質問名キー)`.usage`、`.choices` `.nouls` `.scores`(0.7.2 ではプロパティ)
  <- _core/response_types.py。回答の型: ChoiceAnswer(choice, probabilities) / NoulAnswer(noul) /
  ScoreAnswer(score, legend, probabilities。キーは整数) <- _schemas/models.py
- usage: SDK は input_tokens / output_tokens。本実装は加えて request_count / total_elapsed_ms /
  prime_elapsed_ms を持つ。
- 誤り: SDK は TypeSafeError を送出する。本実装は SystemOneError を送出する(回答ごとのエラー欄はない)。

SDK と違う点(意図的):
- `images` は本実装の拡張(Jev は画像非対応)。形式は razorback16/openjev(Apache-2.0、api.py 2026-10-01)の
  `images: list[str | {"content_type", "base64"}]` と同じ(str は `data:image/...;base64,` URL)。
  加えて利便としてファイルパスと bytes も受ける。上限は openjev に揃える(8枚、デコード後5MiB、
  jpeg/png/webp/gif。違反は明示エラー)。openjev はサーバー側でリサイズしないが、本実装は
  既存実験と結果を揃えるため送信前に prepare_image(長辺 max_edge、既定 jpeg q90)を通す。
- `confidence` は返さない。公式文書は「probabilities の広がりから計算」としか書いておらず式を確認できないため。
  必要なら probabilities から自分で計算する。
- `probabilities` は候補内で再正規化した**相対スコア**(Jev の「較正済み確率」ではなく、正答確率でもない)。
  全候補を観測できるのは top_logprobs の上位20件に収まる20候補以内。`A` と ` A` のような表記ゆれも
  20枠を使う。choice の上限は52ラベル(Jev は255)。
- 質問は互いに見えない。1質問=1リクエスト(束ねない)。先頭(system → 画像 → state)を全質問で共有し、
  サーバーのプレフィックスキャッシュを使えるようにする。
"""

from __future__ import annotations

import base64
import io
import os
import json
import re
from dataclasses import dataclass
from functools import cached_property

from PIL import Image

from .backend import ChatBackend
from .decision import LABELS, DecisionError, _data_url, extract_top_logprobs, pool_labels
from .image import prepare_image

SYSTEM_TEXT = "You answer a question about the given state. Reply with one character only."
SCORE_MAX_LEVELS = 10  # ラベルは数字 "0".."9" の1トークン
MAX_STATE_CHARS = 100_000
MAX_TEXT_CHARS = 8_000  # instructions と各説明の上限。超過は切り捨てずエラー
MAX_IMAGES = 8  # 以下3つは openjev の既定に揃える
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # デコード後
IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
PARAMS = {"max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20}
PRIME_PARAMS = {"max_tokens": 1, "temperature": 0}

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class SystemOneError(Exception):
    """入力不正・推論失敗(ラベルトークンが出ない、thinking が先に出る、logprobs 欠損)。"""


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]  # 候補内で再正規化した相対スコア(較正されていない)
    type: str = "choice"


@dataclass(frozen=True)
class NoulAnswer:
    noul: float  # yes/no の2択内での yes の相対スコア
    type: str = "noul"


@dataclass(frozen=True)
class ScoreAnswer:
    score: float  # 相対スコアで重み付けした段階の期待値
    legend: dict[int, object]
    probabilities: dict[int, float]
    type: str = "score"


@dataclass(frozen=True)
class Usage:
    request_count: int
    total_elapsed_ms: float  # 質問へのリクエストの合計(prime は含まない)
    input_tokens: int | None = None
    output_tokens: int | None = None
    prime_elapsed_ms: float | None = None


@dataclass(frozen=True)
class SystemOneResponse:
    model: str
    answers: dict[str, object]
    usage: Usage

    @cached_property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, ChoiceAnswer)}

    @cached_property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, NoulAnswer)}

    @cached_property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, ScoreAnswer)}


def _clean(text: str, limit: int, what: str) -> str:
    """信頼できない文字列: 制御文字を除き、長さ超過は切り捨てずエラーにする。"""
    text = _CONTROL_CHARS_RE.sub("", text)
    if len(text) > limit:
        raise SystemOneError(f"{what} is too long ({len(text)} > {limit} chars)")
    return text


def _single_line(text: str, limit: int, what: str) -> str:
    """選択肢行への注入を防ぐため、改行・タブ・制御文字を空白にして1行にする(state 以外はすべてこれを通す)。"""
    text = re.sub(r"[\x00-\x1f\x7f]+|\s+", " ", text).strip()
    return _clean(text, limit, what)


def _content_text(value: object, what: str) -> str:
    """instructions / 説明: 文字列、または JSON として直列化できる object / array。"""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError as e:
            raise SystemOneError(f"{what} is not JSON-serializable: {e}") from e
    return _single_line(text, MAX_TEXT_CHARS, what)


def _state_text(state: object) -> str:
    if isinstance(state, str):
        text = state
    else:
        try:
            text = json.dumps(state, ensure_ascii=False)
        except TypeError as e:
            raise SystemOneError(f"state is not JSON-serializable: {e}") from e
    return _clean(text, MAX_STATE_CHARS, "state")


def _decode_image(item: object) -> bytes:
    """1枚を生バイトにする。openjev 形式(data URL 文字列 / {content_type, base64})のほか、
    クライアントの利便としてファイルパス(data: で始まらない str)と bytes も受ける。"""
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    if isinstance(item, dict):
        mime, b64 = item.get("content_type"), item.get("base64")
    elif isinstance(item, str) and item.startswith("data:"):
        m = re.fullmatch(r"data:([^;,]+);base64,(.*)", item, re.DOTALL)
        if not m:
            raise SystemOneError("image data URL must be 'data:<mime>;base64,<data>'")
        mime, b64 = m.group(1), m.group(2)
    elif isinstance(item, str):
        try:
            if os.path.getsize(item) > MAX_IMAGE_BYTES:
                raise SystemOneError(f"image file is too large (> {MAX_IMAGE_BYTES} bytes)")
            with open(item, "rb") as f:
                return f.read()
        except OSError as e:
            raise SystemOneError(f"cannot read image file: {e}") from e
    else:
        raise SystemOneError(f"unsupported image item: {type(item).__name__}")
    if mime not in IMAGE_MIME_TYPES:
        raise SystemOneError(f"unsupported image MIME type {mime!r} (allowed: {sorted(IMAGE_MIME_TYPES)})")
    try:
        return base64.b64decode(b64, validate=True)
    except (ValueError, TypeError) as e:
        raise SystemOneError(f"invalid base64 image data: {e}") from e


def _image_parts(images: list | None, max_edge: int, image_format: str) -> list[dict]:
    """画像(本実装の拡張。形式は openjev の `images` と同じ)を、既存の prepare_image で
    縮小・再エンコードしてから chat の image_url(data URL)にして配列順に返す。"""
    images = list(images or [])
    if len(images) > MAX_IMAGES:
        raise SystemOneError(f"too many images ({len(images)} > {MAX_IMAGES})")
    parts = []
    for i, item in enumerate(images):
        raw = _decode_image(item)
        if len(raw) > MAX_IMAGE_BYTES:
            raise SystemOneError(f"image {i} is too large ({len(raw)} > {MAX_IMAGE_BYTES} bytes)")
        try:
            with Image.open(io.BytesIO(raw)) as probe:
                actual = probe.format
            if actual not in ("JPEG", "PNG", "WEBP", "GIF"):
                raise SystemOneError(f"image {i} has unsupported actual format {actual!r}")
            image_bytes, mime, _orig, _sent = prepare_image(io.BytesIO(raw), max_edge, image_format)
        except (OSError, ValueError) as e:  # PIL の UnidentifiedImageError は OSError
            raise SystemOneError(f"image {i} cannot be decoded: {e}") from e
        parts.append({"type": "image_url", "image_url": {"url": _data_url(image_bytes, mime)}})
    return parts


def _label_line(label: str, name: str, description: object) -> str:
    line = f"{label}. {name}"
    if description is not None:
        line += " - " + _content_text(description, f"description of {name!r}")
    return line


class SystemOneClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        model: str = "",
        max_edge: int = 1024,
        image_format: str = "jpeg",
        timeout: float = 120.0,
        backend=None,  # テスト用の差し替え
    ) -> None:
        self.backend = backend or ChatBackend(base_url=base_url, model=model, timeout=timeout)
        self.max_edge = max_edge
        self.image_format = image_format

    def _build(self, question: object, qid: str):
        """質問の本文・使うラベル・(ラベル別スコア → 回答)の組み立て関数を返す。"""
        if not isinstance(question, dict):
            raise SystemOneError(f"question {qid!r}: must be a dict")
        qtype = question.get("type")
        instructions = question.get("instructions")
        head = _content_text(instructions, f"instructions of {qid!r}") if instructions is not None else None
        criteria = question.get("criteria")

        if qtype == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise SystemOneError(f"question {qid!r}: choice criteria must be a dict of 2+ labels")
            if len(criteria) > len(LABELS):
                raise SystemOneError(
                    f"question {qid!r}: {len(criteria)} choices exceed the {len(LABELS)}-label limit"
                )
            names = list(criteria)
            labels = LABELS[: len(names)]
            lines = [
                _label_line(lb, _single_line(str(n), 200, "choice name"), criteria[n])
                for lb, n in zip(labels, names)
            ]
            body = "\n".join(lines) + "\n\nAnswer with the single letter only."

            def make(scores: dict[str, float]):
                probs = {n: scores[lb] for lb, n in zip(labels, names)}
                return ChoiceAnswer(choice=max(probs, key=probs.get), probabilities=probs)

        elif qtype == "noul":
            labels = ["A", "B"]
            lines = ["A. yes", "B. no"]
            if isinstance(criteria, dict):
                for i, key in enumerate(("true", "false")):
                    if criteria.get(key) is not None:
                        lines[i] += " - " + _content_text(criteria[key], f"criteria.{key} of {qid!r}")
            body = "\n".join(lines) + "\n\nAnswer with the single letter only."
            head = head or "Is the statement true?"

            def make(scores: dict[str, float]):
                return NoulAnswer(noul=scores["A"])

        elif qtype == "score":
            if not isinstance(criteria, (list, tuple)) or not 2 <= len(criteria) <= SCORE_MAX_LEVELS:
                raise SystemOneError(
                    f"question {qid!r}: score criteria must be a list of 2-{SCORE_MAX_LEVELS} levels"
                )
            labels = [str(i) for i in range(len(criteria))]
            body = "\n".join(_label_line(lb, f"level {lb}", d) for lb, d in zip(labels, criteria))
            body += "\n\nAnswer with the single digit only."

            def make(scores: dict[str, float]):
                probs = {i: scores[lb] for i, lb in enumerate(labels)}
                return ScoreAnswer(
                    score=sum(i * p for i, p in probs.items()),
                    legend=dict(enumerate(criteria)),
                    probabilities=probs,
                )

        else:
            raise SystemOneError(f"question {qid!r}: unknown type {qtype!r}")

        text = (head + "\n\n" if head else "") + body
        return text, labels, make

    @staticmethod
    def _messages(prefix_parts: list[dict], question_text: str) -> list[dict]:
        # system → 画像 → state → 質問。質問の前までは全質問で同一(プレフィックスキャッシュの共有)。
        return [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": prefix_parts + [{"type": "text", "text": question_text}]},
        ]

    def system_one(
        self,
        state: object,
        questions: dict[str, dict],
        *,
        images: list[str] | None = None,
        prime: bool = False,
    ) -> SystemOneResponse:
        """questions は SDK の生 dict({"type": "choice"|"noul"|"score", ...})の {質問名: dict}。
        prime=True なら先に system+画像+state だけの1リクエストを送る(時間は usage.prime_elapsed_ms)。"""
        if not questions:
            raise SystemOneError("questions is empty")
        # 入力検証はリクエストを送る前にすべて済ませる
        built = {qid: self._build(q, qid) for qid, q in questions.items()}
        prefix = _image_parts(images, self.max_edge, self.image_format) + [
            {"type": "text", "text": "State:\n" + _state_text(state)}
        ]

        count0 = self.backend.request_count
        prime_ms = None
        if prime:
            _r, prime_ms = self.backend.chat(self._messages(prefix, "Reply with A."), **PRIME_PARAMS)

        answers: dict[str, object] = {}
        total_ms = 0.0
        in_tok = out_tok = 0
        have_usage = False
        for qid, (text, labels, make) in built.items():
            response, elapsed = self.backend.chat(self._messages(prefix, text), **PARAMS)
            total_ms += elapsed
            usage = response.get("usage") or {}
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                have_usage = True
                in_tok += usage["prompt_tokens"]
                out_tok += usage["completion_tokens"]
            try:
                scores = pool_labels(extract_top_logprobs(response), labels)
            except DecisionError as e:
                raise SystemOneError(f"question {qid!r}: {e}") from e
            answers[qid] = make(scores)

        return SystemOneResponse(
            model=self.backend.model,
            answers=answers,
            usage=Usage(
                request_count=self.backend.request_count - count0,
                total_elapsed_ms=total_ms,
                input_tokens=in_tok if have_usage else None,
                output_tokens=out_tok if have_usage else None,
                prime_elapsed_ms=prime_ms,
            ),
        )
