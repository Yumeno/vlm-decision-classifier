"""テスト用の偽バックエンド・レスポンス生成ヘルパー。ネットワークを一切使わない。"""

from __future__ import annotations

import math
import threading


def make_logprobs_response(label_probs: dict[str, float]) -> dict:
    """label -> 確率 から choose()/yes_no() 用の chat レスポンスを作る。
    `exp(logprob) == probability` になるよう `logprob = math.log(probability)` を使う。
    """
    top_logprobs = [
        {"token": label, "logprob": math.log(p)} for label, p in label_probs.items() if p > 0
    ]
    return {
        "choices": [
            {
                "message": {"content": None},
                "logprobs": {"content": [{"token": "", "top_logprobs": top_logprobs}]},
            }
        ],
        "usage": {},
    }


def make_text_response(text: str, completion_tokens: int = 10) -> dict:
    return {
        "choices": [{"message": {"content": text}, "logprobs": None}],
        "usage": {"completion_tokens": completion_tokens},
    }


def make_thinking_response(reasoning: str = "thinking...") -> dict:
    return {
        "choices": [
            {
                "message": {"content": "", "reasoning_content": reasoning},
                "logprobs": {"content": []},
            }
        ],
        "usage": {},
    }


def make_no_logprobs_response() -> dict:
    return {
        "choices": [{"message": {"content": ""}, "logprobs": {"content": []}}],
        "usage": {},
    }


class FakeBackend:
    """事前に用意したレスポンス列を順番に返すだけの偽バックエンド。"""

    def __init__(self, responses: list[dict]):
        self.model = "fake-model"
        self.base_url = "http://fake"
        self.request_count = 0
        self.dropped_reasoning_effort = False
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, **params):
        self.request_count += 1
        self.calls.append({"messages": messages, "params": params})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, 1.0

    def list_models(self):
        return {"data": [{"id": self.model}]}


class KeyedFakeBackend:
    """プロンプト中の目印文字列(axisの質問文・候補criteriaなど)で応答を振り分ける偽バックエンド。

    axis_concurrency のテストのように、リクエストの到着順が保証されない(FakeBackendの
    ような「順番に1件ずつpop」が使えない)場合に使う。request_countの加算はロックで保護する。
    """

    def __init__(self, responses_by_marker: dict[str, object]):
        self.model = "fake-model"
        self.base_url = "http://fake"
        self.request_count = 0
        self.dropped_reasoning_effort = False
        self._responses_by_marker = responses_by_marker
        self._lock = threading.Lock()

    def chat(self, messages, **params):
        with self._lock:
            self.request_count += 1
        text = messages[1]["content"][1]["text"]
        for marker, response in self._responses_by_marker.items():
            if marker in text:
                if isinstance(response, Exception):
                    raise response
                return response, 1.0
        raise AssertionError(f"no fake response registered for prompt: {text!r}")

    def list_models(self):
        return {"data": [{"id": self.model}]}
