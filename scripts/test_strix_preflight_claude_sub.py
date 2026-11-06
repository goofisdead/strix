#!/usr/bin/env python3
"""End-to-end preflight: strix uses Claude subscription auth with no API key.

Stands up a fake api.anthropic.com, points the proxy at it, drives strix's own
``preflight_model_connection`` with ``STRIX_LLM=claude-code/...`` and no
``LLM_API_KEY``, and checks the preflight succeeds. Validates that the
upstream request carries Bearer auth, the OAuth beta flags, and the Claude
Code system-prompt prefix.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_fake_anthropic(port: int, captured: dict[str, object]) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: object) -> None:
            pass

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            captured["headers"] = {k: v for k, v in self.headers.items()}
            captured["body"] = json.loads(raw.decode("utf-8")) if raw else {}
            body = captured["body"]
            model = body.get("model", "claude-opus-4-5-20250929")
            msg_id = f"msg_{int(time.time()*1000)}"
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                events = [
                    "event: message_start\n"
                    "data: " + json.dumps({
                        "type": "message_start",
                        "message": {
                            "id": msg_id, "type": "message", "role": "assistant",
                            "model": model,
                            "content": [], "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": 5, "output_tokens": 0},
                        },
                    }) + "\n\n",
                    "event: content_block_start\n"
                    "data: " + json.dumps({
                        "type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }) + "\n\n",
                    "event: content_block_delta\n"
                    "data: " + json.dumps({
                        "type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": "OK"},
                    }) + "\n\n",
                    "event: content_block_stop\n"
                    "data: " + json.dumps({"type": "content_block_stop", "index": 0}) + "\n\n",
                    "event: message_delta\n"
                    "data: " + json.dumps({
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 1},
                    }) + "\n\n",
                    "event: message_stop\n"
                    "data: " + json.dumps({"type": "message_stop"}) + "\n\n",
                ]
                for evt in events:
                    data = evt.encode("utf-8")
                    self.wfile.write(f"{len(data):x}\r\n".encode())
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                return
            resp = {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }
            payload = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="strix-pre-")
    os.environ["HOME"] = tmp
    (Path(tmp) / ".strix").mkdir()
    cl = Path(tmp) / ".claude"
    cl.mkdir()
    (cl / ".credentials.json").write_text(
        json.dumps({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-PRE",
                "refreshToken": "sk-ant-ort01-PRE",
                "expiresAt": int((time.time() + 7200) * 1000),
                "scopes": ["user:profile", "user:inference"],
            }
        })
    )

    os.environ["STRIX_LLM"] = "claude-code/claude-opus-4-5-20250929"
    os.environ.pop("LLM_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("ANTHROPIC_API_KEY", None)

    port = free_port()
    captured: dict[str, object] = {}
    httpd = start_fake_anthropic(port, captured)

    from strix.config import claude_proxy, load_settings
    from strix.config.models import configure_sdk_model_defaults
    claude_proxy._UPSTREAM = f"http://127.0.0.1:{port}/v1/messages"
    claude_proxy._UPSTREAM_HOST = "127.0.0.1"

    settings = load_settings()
    configure_sdk_model_defaults(settings)

    from strix.interface.scan_setup import preflight_model_connection

    asyncio.run(preflight_model_connection("claude-code/claude-opus-4-5-20250929", settings=settings))

    hdrs = captured["headers"]
    body = captured["body"]
    assert hdrs.get("Authorization") == "Bearer sk-ant-oat01-PRE", hdrs
    assert "oauth-2025-04-20" in hdrs.get("anthropic-beta", ""), hdrs
    assert "claude-code-20250219" in hdrs.get("anthropic-beta", ""), hdrs
    assert hdrs.get("anthropic-version") == "2023-06-01", hdrs
    sys_field = body.get("system")
    if isinstance(sys_field, list):
        assert sys_field[0]["text"].startswith("You are Claude Code"), sys_field
    else:
        assert str(sys_field).startswith("You are Claude Code"), sys_field
    httpd.shutdown()
    httpd.server_close()
    claude_proxy.stop()
    print("preflight succeeded with Claude subscription, no API key")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
