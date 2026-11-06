#!/usr/bin/env python3
"""End-to-end check for the Claude subscription proxy.

Runs through:
  1. Reading token records from our own store and from a fake Claude Code store.
  2. Launching the local proxy.
  3. Stubbing api.anthropic.com with a local HTTPServer (via monkey-patching
     the proxy's upstream URL) and verifying the forwarded request gets the
     Bearer token, the required anthropic-beta flags, and the Claude Code
     identifier prepended to the system prompt.
  4. Verifying streaming (SSE) passes through byte-for-byte.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _start_fake_upstream(port: int, captured: dict[str, object]) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a, **k):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            captured["headers"] = {k: v for k, v in self.headers.items()}
            captured["body"] = json.loads(body.decode("utf-8")) if body else {}
            if self.path.endswith("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for evt in (
                    b'event: message_start\ndata: {"type":"message_start"}\n\n',
                    b'event: content_block_delta\ndata: {"delta":{"text":"hi"}}\n\n',
                    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
                ):
                    chunk = f"{len(evt):x}\r\n".encode()
                    self.wfile.write(chunk)
                    self.wfile.write(evt)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                return
            payload = json.dumps(
                {"id": "msg_test", "role": "assistant", "content": [{"type": "text", "text": "ok"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="strix-test-")
    os.environ["HOME"] = tmp
    (Path(tmp) / ".strix").mkdir()

    # Pre-stage a Claude Code style credentials file so claude_sub picks it up.
    claude_dir = Path(tmp) / ".claude"
    claude_dir.mkdir()
    (claude_dir / ".credentials.json").write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "sk-ant-oat01-TEST",
                    "refreshToken": "sk-ant-ort01-TEST",
                    "expiresAt": int((time.time() + 3600) * 1000),
                    "scopes": ["user:profile", "user:inference"],
                }
            }
        )
    )

    # Import after setting HOME so claude_sub resolves the fake cred paths.
    from strix.config import claude_proxy, claude_sub

    assert claude_sub.is_authenticated(), "fake Claude Code creds should be visible"
    assert claude_sub.get_valid_token() == "sk-ant-oat01-TEST"
    assert claude_sub.external_source_summary() is not None

    # Stand up the fake upstream and point the proxy at it.
    upstream_port = _free_port()
    captured: dict[str, object] = {}
    fake = _start_fake_upstream(upstream_port, captured)
    claude_proxy._UPSTREAM = f"http://127.0.0.1:{upstream_port}/v1/messages"
    claude_proxy._UPSTREAM_HOST = "127.0.0.1"

    base = claude_proxy.start()
    # Non-streaming request
    req = urllib.request.Request(
        f"{base}/v1/messages",
        method="POST",
        data=json.dumps(
            {
                "model": "claude-opus-4-5-20250929",
                "system": "You are a helpful penetration tester.",
                "messages": [{"role": "user", "content": "say ok"}],
            }
        ).encode(),
        headers={"x-api-key": "placeholder", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read())
    assert body["content"][0]["text"] == "ok", body

    headers = captured["headers"]
    assert headers.get("Authorization") == "Bearer sk-ant-oat01-TEST", headers
    assert "x-api-key" not in {k.lower() for k in headers}, headers
    assert "oauth-2025-04-20" in headers.get("anthropic-beta", "")
    assert "claude-code-20250219" in headers.get("anthropic-beta", "")
    assert headers.get("anthropic-version") == "2023-06-01"
    body_fwd = captured["body"]
    assert body_fwd["system"].startswith(claude_sub.CLAUDE_CODE_SYSTEM_IDENTIFIER)
    assert "You are a helpful penetration tester." in body_fwd["system"]

    # Streaming request: path tweaked so the fake switches to SSE.
    claude_proxy._UPSTREAM = f"http://127.0.0.1:{upstream_port}/v1/messages/stream"
    req = urllib.request.Request(
        f"{base}/v1/messages",
        method="POST",
        data=json.dumps(
            {
                "model": "claude-opus-4-5-20250929",
                "messages": [{"role": "user", "content": "stream"}],
                "stream": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = resp.read().decode()
    assert "message_start" in payload and "message_stop" in payload, payload

    # List-shaped system prompt: identifier should prepend as a block.
    claude_proxy._UPSTREAM = f"http://127.0.0.1:{upstream_port}/v1/messages"
    captured.clear()
    req = urllib.request.Request(
        f"{base}/v1/messages",
        method="POST",
        data=json.dumps(
            {
                "model": "claude-opus-4-5-20250929",
                "system": [
                    {"type": "text", "text": "Act as a red teamer.", "cache_control": {"type": "ephemeral"}}
                ],
                "messages": [{"role": "user", "content": "ok"}],
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()
    sysf = captured["body"]["system"]
    assert isinstance(sysf, list) and sysf[0]["text"] == claude_sub.CLAUDE_CODE_SYSTEM_IDENTIFIER

    claude_proxy.stop()
    fake.shutdown()
    fake.server_close()
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
