"""デモUIサーバーの最小テスト。偽バックエンドのみを使い、ネットワーク越しの実サーバーには接続しない
(接続先はテストが自分で立てたループバックサーバーのみ)。"""

from __future__ import annotations

import base64
import io
import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from PIL import Image

from classifier_demo import server
from tests.fakes import FakeBackend, make_logprobs_response, make_no_logprobs_response, make_text_response
from tests.test_pipeline import _test_taxonomy


def _png_base64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def running_server():
    taxonomy = _test_taxonomy()
    state = {"factory": None}

    def backend_factory(base_url, model):
        return state["factory"]()

    handler_cls = server.make_handler(taxonomy, backend_factory=backend_factory)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}", state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _post_ndjson(base_url: str, payload: dict) -> list[dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/api/classify",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events = []
    with urllib.request.urlopen(req, timeout=10) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if line:
                events.append(json.loads(line))
    return events


def test_get_index_serves_html(running_server):
    base_url, _ = running_server
    with urllib.request.urlopen(base_url + "/", timeout=10) as resp:
        assert resp.status == 200
        body = resp.read().decode("utf-8")
    assert "<html" in body.lower()


def test_classify_choice_mode_streams_metadata_axis_and_done_events(running_server):
    base_url, state = running_server
    state["factory"] = lambda: FakeBackend(
        [
            make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),  # image_type
            make_logprobs_response({"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}),  # art_style
            make_logprobs_response(
                {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.05, "E": 0.03, "F": 0.02}
            ),  # subject
            make_no_logprobs_response(),  # character axis fails
        ]
    )

    events = _post_ndjson(
        base_url,
        {
            "image_base64": _png_base64(),
            "filename": "img.png",
            "mode": "choice",
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "ignored",
        },
    )

    assert events[0]["type"] == "metadata"

    axis_events = [e for e in events if e["type"] == "axis"]
    assert [e["axis_id"] for e in axis_events] == ["image_type", "art_style", "subject", "character"]
    assert axis_events[0]["selected"] == "illustration"
    assert axis_events[-1]["error"]["type"] == "no_logprobs"

    assert events[-1]["type"] == "done"
    assert events[-1]["request_count"] == 4
    assert len(events[-1]["errors"]) == 1


def test_classify_json_mode_streams_json_result_and_done(running_server):
    base_url, state = running_server
    state["factory"] = lambda: FakeBackend(
        [
            make_text_response(
                json.dumps(
                    {
                        "image_type": "illustration",
                        "art_style": "anime_2d",
                        "subject": "person",
                        "character": [],
                    }
                )
            )
        ]
    )

    events = _post_ndjson(
        base_url,
        {
            "image_base64": _png_base64(),
            "filename": "img.png",
            "mode": "json",
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "ignored",
        },
    )

    assert events[0]["type"] == "metadata"
    json_events = [e for e in events if e["type"] == "json_result"]
    assert len(json_events) == 1
    assert json_events[0]["tags"]["image_type"] == ["illustration"]

    assert events[-1]["type"] == "done"
    assert events[-1]["mode"] == "json"
    assert events[-1]["errors"] == []


def test_classify_prime_true_emits_prime_event_before_metadata(running_server):
    # E9: prime:true を送ると、判定前に decision.prime() が1回呼ばれ、
    # {"type": "prime", "elapsed_ms": ...} イベントが metadata より前に届く。
    base_url, state = running_server
    state["factory"] = lambda: FakeBackend(
        [
            make_logprobs_response({"A": 1.0}),  # prime
            make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),  # image_type
            make_logprobs_response({"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}),  # art_style
            make_logprobs_response(
                {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.05, "E": 0.03, "F": 0.02}
            ),  # subject
            make_no_logprobs_response(),  # character axis fails
        ]
    )

    events = _post_ndjson(
        base_url,
        {
            "image_base64": _png_base64(),
            "filename": "img.png",
            "mode": "choice",
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "ignored",
            "prime": True,
        },
    )

    assert events[0]["type"] == "prime"
    assert events[0]["elapsed_ms"] == 1.0
    assert events[1]["type"] == "metadata"
    assert events[-1]["type"] == "done"
    assert events[-1]["request_count"] == 4  # prime分はrequest_countに含めない(判定のみ)


def test_classify_without_prime_flag_emits_no_prime_event(running_server):
    base_url, state = running_server
    state["factory"] = lambda: FakeBackend(
        [
            make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),
            make_logprobs_response({"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}),
            make_logprobs_response(
                {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.05, "E": 0.03, "F": 0.02}
            ),
            make_no_logprobs_response(),
        ]
    )

    events = _post_ndjson(
        base_url,
        {
            "image_base64": _png_base64(),
            "filename": "img.png",
            "mode": "choice",
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "ignored",
        },
    )

    assert all(e["type"] != "prime" for e in events)
    assert events[0]["type"] == "metadata"


def test_classify_axis_concurrency_is_forwarded_and_still_completes(running_server):
    # E9: axis_concurrency をサーバー経由で渡しても判定が最後まで完了し、全軸のイベントが届く。
    base_url, state = running_server
    state["factory"] = lambda: FakeBackend(
        [
            make_logprobs_response({"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}),
            make_logprobs_response({"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}),
            make_logprobs_response(
                {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.05, "E": 0.03, "F": 0.02}
            ),
            make_no_logprobs_response(),
        ]
    )

    events = _post_ndjson(
        base_url,
        {
            "image_base64": _png_base64(),
            "filename": "img.png",
            "mode": "choice",
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "ignored",
            "axis_concurrency": 4,
        },
    )

    # FakeBackend自体はスレッド安全な応答順マッチではないため(request_countのロックは
    # ChatBackend側、応答リストのpop順はスレッド間で非決定)、ここでは全軸が完了することだけを
    # 見る。並列時の内容・順序・request_countの正しさは KeyedFakeBackend を使う
    # test_pipeline.py 側のテストで検証済み。
    axis_events = [e for e in events if e["type"] == "axis"]
    assert {e["axis_id"] for e in axis_events} == {"image_type", "art_style", "subject", "character"}
    assert events[-1]["type"] == "done"


def test_classify_bad_request_returns_400(running_server):
    base_url, _ = running_server
    data = json.dumps({"mode": "choice"}).encode("utf-8")  # image_base64等が欠けている
    req = urllib.request.Request(
        base_url + "/api/classify",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=10)
    assert excinfo.value.code == 400


def test_classify_rejects_non_loopback_base_url(running_server):
    # 中継先(base_url)はこのPC上のサーバーだけに限る(最低限のSSRF対策)。
    base_url, state = running_server
    state["factory"] = lambda: FakeBackend([])
    data = json.dumps(
        {
            "image_base64": _png_base64(),
            "filename": "img.png",
            "mode": "choice",
            "base_url": "http://example.com/v1",
            "model": "m",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/api/classify",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=10)
    assert excinfo.value.code == 400


def test_classify_rejects_foreign_origin(running_server):
    # 別オリジンからのPOST(中継の悪用/CSRF的な経路)を拒否する。
    base_url, state = running_server
    state["factory"] = lambda: FakeBackend([])
    data = json.dumps(
        {
            "image_base64": _png_base64(),
            "filename": "img.png",
            "mode": "choice",
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "m",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/api/classify",
        data=data,
        headers={"Content-Type": "application/json", "Origin": "http://evil.example"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=10)
    assert excinfo.value.code == 403


def test_classify_rejects_non_json_content_type(running_server):
    # プリフライトなしの単純リクエスト(他サイトのフォーム送信など)を弾く。
    base_url, state = running_server
    state["factory"] = lambda: FakeBackend([])
    data = json.dumps(
        {
            "image_base64": _png_base64(),
            "filename": "img.png",
            "mode": "choice",
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "m",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/api/classify",
        data=data,
        headers={"Content-Type": "text/plain"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=10)
    assert excinfo.value.code == 415


def test_classify_rejects_invalid_content_length(running_server):
    # Content-Length が整数でなければ400(本文は読まずに拒否する)。
    # urllib.request(http.client)経由だと、ヘッダーを不正な値で上書きしても実際の
    # 送信バイト数は変わらず、サーバーが本文を読まずに接続を閉じた際にOS側でRSTになり
    # テストが不安定になるため、生ソケットでヘッダーのみ(本文なし)のリクエストを送る。
    base_url, state = running_server
    state["factory"] = lambda: FakeBackend([])
    host, port_str = base_url.replace("http://", "").split(":")
    port = int(port_str)

    request = (
        "POST /api/classify HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: not-a-number\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii")

    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(request)
        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk

    status_line = response.split(b"\r\n", 1)[0].decode("ascii")
    assert " 400 " in status_line


def test_serve_fails_when_port_in_use():
    from classifier_demo.server import _ExclusiveHTTPServer

    first = _ExclusiveHTTPServer(("127.0.0.1", 0), server.BaseHTTPRequestHandler)
    try:
        port = first.server_address[1]
        try:
            _ExclusiveHTTPServer(("127.0.0.1", port), server.BaseHTTPRequestHandler).server_close()
            raise AssertionError("2つ目の bind が成功してしまった")
        except OSError:
            pass
    finally:
        first.server_close()
