"""テスト用の偽バックエンド・レスポンス生成ヘルパー。ネットワークを一切使わない。"""

from __future__ import annotations

import math


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

    def chat(self, messages, **params):
        self.request_count += 1
        response = self._responses.pop(0)
        return response, 1.0

    def list_models(self):
        return {"data": [{"id": self.model}]}
