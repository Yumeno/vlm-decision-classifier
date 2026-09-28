"""動画収録用デモUIのサーバー(標準ライブラリのみ、ループバック専用)。

静的ファイル(web/index.html)を配信し、POST /api/classify で1件の画像分類を
NDJSON(1行1イベント)でストリーム配信する。モデルの自動起動・外部公開はしない。
比較(選択式 vs JSON、サーバー1 vs サーバー2)はUI側がこのAPIを順番に2回呼んで行う。
"""

from __future__ import annotations

import base64
import json
import os
import queue
import tempfile
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import decision, pipeline
from .backend import ChatBackend
from .image import prepare_image
from .taxonomy import Taxonomy

WEB_DIR = Path(__file__).parent / "web"
INDEX_PATH = WEB_DIR / "index.html"


def _write_ndjson_line(wfile, event: dict) -> None:
    line = json.dumps(event, ensure_ascii=False) + "\n"
    wfile.write(line.encode("utf-8"))
    wfile.flush()


def make_handler(taxonomy: Taxonomy, backend_factory=ChatBackend):
    """`backend_factory(base_url=..., model=...)` はテスト用の差し替え口
    (既定はChatBackend。テストではFakeBackendを返す関数を渡す)。"""

    class Handler(BaseHTTPRequestHandler):
        server_version = "classifier-demo-ui/0.1"

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - stdlib signature
            pass  # 収録中の標準エラー出力を抑える。動作確認はテスト/手動確認で行う。

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _allowed_hosts(self) -> set[str]:
            # このサーバー自身がbindしているport。DNSリバインディング対策(Hostヘッダー検査)用。
            port = self.server.server_address[1]
            return {f"127.0.0.1:{port}", f"localhost:{port}"}

        def _host_header_allowed(self) -> bool:
            return self.headers.get("Host", "") in self._allowed_hosts()

        def _origin_allowed(self) -> bool:
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            allowed = {f"http://{h}" for h in self._allowed_hosts()}
            return origin in allowed

        @staticmethod
        def _is_loopback_base_url(base_url) -> bool:
            # 中継先(LM Studio/llama-server)をこのPC上に限定する(最低限のSSRF対策)。
            if not isinstance(base_url, str):
                return False
            try:
                parsed = urllib.parse.urlsplit(base_url)
            except ValueError:
                return False
            if parsed.scheme not in ("http", "https"):
                return False
            return parsed.hostname in ("127.0.0.1", "localhost", "::1")

        def do_GET(self) -> None:  # noqa: N802 - stdlib method name
            if self.path in ("/", "/index.html"):
                body = INDEX_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def _drain_body_best_effort(self) -> None:
            # 本文を読まずに接続を閉じると、未読データが残ったソケットとしてOS側で
            # RSTになり、直前に送った拒否レスポンスがクライアントに届かないことがある
            # (Windowsで確認)。ヘッダー検査による早期拒否のあとに、分かる範囲で読み捨てる。
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return
            if length > 0:
                try:
                    self.rfile.read(length)
                except Exception:
                    pass

        def do_POST(self) -> None:  # noqa: N802 - stdlib method name
            if self.path != "/api/classify":
                self.send_error(404)
                return

            # ヘッダーだけで判定できるものは本文を読む前に弾く(不要な読み込みをしない)。
            # DNSリバインディング対策: Hostヘッダーがこのサーバー自身以外なら拒否。
            if not self._host_header_allowed():
                self._send_json(403, {"error": "forbidden host"})
                self._drain_body_best_effort()
                return
            # ブラウザ単純リクエストでの他オリジンからのPOST(CSRF的な中継悪用)を拒否。
            if not self._origin_allowed():
                self._send_json(403, {"error": "forbidden origin"})
                self._drain_body_best_effort()
                return
            # プリフライトなしの単純リクエスト(例: 他サイトのHTMLフォーム送信)を弾く。
            content_type = self.headers.get("Content-Type", "")
            if content_type.split(";")[0].strip().lower() != "application/json":
                self._send_json(415, {"error": "expected application/json"})
                self._drain_body_best_effort()
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "invalid Content-Length"})
                return
            # サイズ上限は付けない(ループバック限定の録画用途で、画像は数MB程度のため)。
            raw = self.rfile.read(length)

            try:
                req = json.loads(raw.decode("utf-8"))
                image_bytes = base64.b64decode(req["image_base64"])
                filename = str(req.get("filename") or "image.png")
                mode = req.get("mode", "choice")
                base_url = req["base_url"]
                model = req["model"]
                max_edge = int(req.get("max_edge", 1024))
                image_format = req.get("image_format", "jpeg")
                prime = bool(req.get("prime", False))
                axis_concurrency = int(req.get("axis_concurrency", 1))
                confirm = bool(req.get("confirm", False))
                rank_threshold = float(req.get("rank_threshold", 0.5))
            except Exception as e:
                self._send_json(400, {"error": f"bad request: {type(e).__name__}: {e}"})
                return

            if mode not in ("choice", "json"):
                self._send_json(400, {"error": f"unknown mode: {mode}"})
                return
            if image_format not in ("jpeg", "png"):
                self._send_json(400, {"error": f"unknown image_format: {image_format}"})
                return
            if not self._is_loopback_base_url(base_url):
                self._send_json(
                    400, {"error": "base_url must be a loopback (127.0.0.1/localhost/::1) http(s) URL"}
                )
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()

            suffix = os.path.splitext(filename)[1] or ".png"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(image_bytes)

                backend = backend_factory(base_url=base_url, model=model)

                # イベントの書き出し(ソケットI/O)を判定の測定範囲から外すため、on_progress は
                # queue.Queue に積むだけにし、専用の送信スレッドがソケットへ書く。クライアント
                # 切断などで書き込みが失敗したら送信スレッドは止まるが、判定自体は最後まで走らせて
                # そのまま結果を捨てる(判定をソケットの状態に引きずられて止めない)。
                event_queue: queue.Queue = queue.Queue()

                def writer_loop() -> None:
                    while True:
                        item = event_queue.get()
                        if item is None:
                            return
                        try:
                            _write_ndjson_line(self.wfile, item)
                        except Exception:
                            return  # クライアント切断等。以降キューに積まれても捨てるだけ。

                writer_thread = threading.Thread(target=writer_loop, daemon=True)
                writer_thread.start()

                if prime:
                    prime_n = axis_concurrency if axis_concurrency >= 2 else 1
                    prime_start = time.perf_counter_ns()
                    try:
                        prime_image_bytes, prime_mime, _, _ = prepare_image(
                            tmp_path, max_edge=max_edge, image_format=image_format
                        )
                        if prime_n == 1:
                            prime_result = decision.prime(backend, prime_image_bytes, prime_mime)
                            prime_ms = prime_result["elapsed_ms"]
                        else:
                            with ThreadPoolExecutor(max_workers=prime_n) as prime_executor:
                                prime_futures = [
                                    prime_executor.submit(decision.prime, backend, prime_image_bytes, prime_mime)
                                    for _ in range(prime_n)
                                ]
                                for pf in prime_futures:
                                    pf.result()
                            prime_ms = (time.perf_counter_ns() - prime_start) / 1e6
                        event_queue.put({"type": "prime", "elapsed_ms": prime_ms})
                    except Exception as e:
                        event_queue.put({"type": "prime", "error": f"{type(e).__name__}: {e}"})

                def on_progress(event: dict) -> None:
                    event_queue.put(event)

                try:
                    result = pipeline.classify(
                        tmp_path,
                        taxonomy,
                        backend,
                        mode=mode,
                        max_edge=max_edge,
                        image_format=image_format,
                        on_progress=on_progress,
                        axis_concurrency=axis_concurrency,
                        confirm=confirm,
                        rank_threshold=rank_threshold,
                    )
                    event_queue.put(
                        {
                            "type": "done",
                            "mode": mode,
                            "classification_wall_ms": result["timing_ms"]["classification_wall_ms"],
                            "request_count": result["request_count"],
                            "errors": result["errors"],
                            "combined_evidence": result["combined_evidence"],
                        }
                    )
                except Exception as e:
                    # 予期しない例外もストリームを開いたまま落とさず、イベントとして返す
                    # (不変条件: 失敗は失敗として記録し、別方式へ黙ってフォールバックしない)。
                    event_queue.put({"type": "error", "detail": f"{type(e).__name__}: {e}"})
                finally:
                    event_queue.put(None)  # 送信スレッドへ終了合図
                    writer_thread.join()
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    return Handler


def serve(port: int, taxonomy: Taxonomy, backend_factory=ChatBackend) -> None:
    """待ち受け先は常に127.0.0.1固定(外部公開しない。動画収録用のループバック専用サーバー)。"""
    handler_cls = make_handler(taxonomy, backend_factory=backend_factory)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    print(f"serving demo UI on http://127.0.0.1:{port} (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
