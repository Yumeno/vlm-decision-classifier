"""OpenAI互換 chat completions クライアント(stdlib urllib のみ使用)。

モデルの自動ダウンロード・サーバーの自動起動/終了はしない。ユーザーが起動した
LM Studio / llama-server に接続するだけ。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class ChatBackend:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        model: str = "",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.request_count = 0
        self.dropped_reasoning_effort = False

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        self.request_count += 1
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def chat(self, messages: list, **params) -> tuple[dict, float]:
        payload = {"model": self.model, "messages": messages, "reasoning_effort": "none", **params}
        start = time.perf_counter_ns()
        try:
            response = self._post("/chat/completions", payload)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 400 and "reasoning_effort" in body:
                payload.pop("reasoning_effort", None)
                response = self._post("/chat/completions", payload)
                self.dropped_reasoning_effort = True
            else:
                raise
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        return response, elapsed_ms

    def list_models(self) -> dict:
        req = urllib.request.Request(self.base_url + "/models", method="GET")
        self.request_count += 1
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
