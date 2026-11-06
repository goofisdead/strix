"""Local Anthropic Messages API proxy, OAuth-signed for Claude subscription use.

Pointing LiteLLM at this proxy (``ANTHROPIC_API_BASE=http://127.0.0.1:<port>``)
lets strix run a ``claude-code/<model>`` on the user's Claude Pro/Max
subscription without changing LiteLLM or the agents SDK. The proxy accepts
``POST /v1/messages`` from LiteLLM, rewrites it to carry a fresh OAuth bearer
token plus the ``oauth-2025-04-20,claude-code-20250219`` beta flags, prepends
the Claude Code identifier to the system prompt, forwards to
``api.anthropic.com``, and streams the response back byte-for-byte.

Binds to 127.0.0.1 on an ephemeral port in a daemon ThreadingHTTPServer.
"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import requests

from strix.config import claude_sub


logger = logging.getLogger(__name__)


_LOCAL_HOST = "127.0.0.1"
_UPSTREAM = claude_sub.ANTHROPIC_MESSAGES_URL
_UPSTREAM_HOST = urlparse(_UPSTREAM).hostname or "api.anthropic.com"
_READ_TIMEOUT = 600
_CONNECT_TIMEOUT = 30

# Hop-by-hop headers plus LiteLLM-sent auth we must not forward.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
        "authorization",
        "x-api-key",
        "accept-encoding",
    }
)

_RESPONSE_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-encoding",
        "content-length",
    }
)


_server_lock = threading.Lock()
_server: ThreadingHTTPServer | None = None
_base_url: str | None = None


def start() -> str:
    """Idempotently start the proxy and return its base URL (``http://127.0.0.1:PORT``)."""
    global _server, _base_url  # noqa: PLW0603
    with _server_lock:
        if _base_url is not None:
            return _base_url
        session = requests.Session()
        handler_cls = _make_handler(session)
        port = _find_free_port()
        server = ThreadingHTTPServer((_LOCAL_HOST, port), handler_cls)
        thread = threading.Thread(
            target=server.serve_forever,
            name="strix-claude-proxy",
            daemon=True,
        )
        thread.start()
        _server = server
        _base_url = f"http://{_LOCAL_HOST}:{port}"
        logger.info("Claude subscription proxy listening on %s", _base_url)
        return _base_url


def stop() -> None:
    """Shut the proxy down (idempotent)."""
    global _server, _base_url  # noqa: PLW0603
    with _server_lock:
        if _server is None:
            return
        try:
            _server.shutdown()
            _server.server_close()
        finally:
            _server = None
            _base_url = None


def base_url() -> str | None:
    return _base_url


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_LOCAL_HOST, 0))
        return sock.getsockname()[1]


def _make_handler(session: requests.Session) -> type[BaseHTTPRequestHandler]:
    class _ClaudeProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # silence default stderr
            logger.debug("claude-proxy: %s", fmt % args if args else fmt)

        def do_GET(self) -> None:
            if self.path.startswith("/healthz"):
                payload = b'{"ok":true,"service":"strix-claude-proxy"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self._error(404, "not_found", "only POST /v1/messages is supported")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path not in ("/v1/messages", "/messages"):
                self._error(404, "not_found", f"no route for {parsed.path}")
                return
            try:
                self._forward_messages(parsed)
            except claude_sub.ClaudeAuthError as exc:
                self._error(401, exc.code, str(exc))
            except Exception as exc:
                logger.exception("claude-proxy forwarding failed")
                self._error(502, "upstream_failed", str(exc))

        def _forward_messages(self, parsed: Any) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body: dict[str, Any] = json.loads(raw) if raw else {}
            except json.JSONDecodeError as exc:
                self._error(400, "bad_request", f"request body is not JSON: {exc}")
                return
            if not isinstance(body, dict):
                self._error(400, "bad_request", "request body must be a JSON object")
                return
            _prepend_claude_identifier(body)
            body_bytes = json.dumps(body).encode("utf-8")

            token = claude_sub.get_valid_token()
            headers = _build_upstream_headers(self.headers, token, len(body_bytes))

            upstream_url = _UPSTREAM
            if parsed.query:
                upstream_url = f"{upstream_url}?{parsed.query}"

            with session.post(
                upstream_url,
                data=body_bytes,
                headers=headers,
                stream=True,
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            ) as response:
                self.send_response(response.status_code)
                for name, value in response.headers.items():
                    if name.lower() in _RESPONSE_HOP_BY_HOP:
                        continue
                    self.send_header(name, value)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    self._write_chunk(chunk)
                self._write_chunk(b"")  # terminate chunked stream

        def _write_chunk(self, chunk: bytes) -> None:
            size = f"{len(chunk):x}\r\n".encode("ascii")
            self.wfile.write(size)
            if chunk:
                self.wfile.write(chunk)
            self.wfile.write(b"\r\n")
            with contextlib.suppress(OSError):
                self.wfile.flush()

        def _error(self, status: int, code: str, message: str) -> None:
            payload = json.dumps(
                {"error": {"type": code, "message": message}}
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)

    return _ClaudeProxyHandler


def _build_upstream_headers(
    incoming: Any, token: str, content_length: int
) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key in incoming:
        lower = key.lower()
        if lower in _HOP_BY_HOP:
            continue
        value = incoming[key]
        if value is None:
            continue
        headers[key] = value

    headers["Host"] = _UPSTREAM_HOST
    headers["Authorization"] = f"Bearer {token}"
    headers["Content-Type"] = "application/json"
    headers["Content-Length"] = str(content_length)
    headers.setdefault("anthropic-version", claude_sub.ANTHROPIC_VERSION)
    headers["anthropic-beta"] = _merge_beta_header(headers.get("anthropic-beta"))
    headers.setdefault("User-Agent", "claude-cli/strix-sub")
    return headers


def _merge_beta_header(existing: str | None) -> str:
    """Union of the OAuth beta flags with whatever LiteLLM already asked for."""
    wanted = [flag.strip() for flag in claude_sub.ANTHROPIC_BETA.split(",") if flag.strip()]
    present: list[str] = []
    if existing:
        present = [p.strip() for p in existing.split(",") if p.strip()]
    for flag in wanted:
        if flag not in present:
            present.append(flag)
    return ",".join(present)


def _prepend_claude_identifier(body: dict[str, Any]) -> None:
    """Make the ``system`` field start with the Claude Code identifier line.

    Required by the OAuth-gated backend; the agent's own system prompt follows
    unchanged.
    """
    system = body.get("system")
    marker = claude_sub.CLAUDE_CODE_SYSTEM_IDENTIFIER
    if isinstance(system, str):
        if marker in system[: len(marker) + 20]:
            return
        body["system"] = f"{marker}\n\n{system}" if system else marker
        return
    if isinstance(system, list):
        for block in system:
            if (
                isinstance(block, dict)
                and isinstance(block.get("text"), str)
                and marker in block["text"][: len(marker) + 20]
            ):
                return
        body["system"] = [{"type": "text", "text": marker}, *system]
        return
    body["system"] = marker


__all__ = ["base_url", "start", "stop"]
