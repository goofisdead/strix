"""Claude subscription auth: OAuth login, token refresh, and credential discovery.

Mirrors the Claude Code CLI OAuth flow against ``claude.ai``. Stores tokens at
``~/.strix/subscription-auth.json`` under the ``claude`` provider key, and
falls back to ``~/.claude/.credentials.json`` (Claude Code) and OpenCode's
auth.json so an already-signed-in CLI user does not need to re-authenticate.

Inference goes to ``https://api.anthropic.com/v1/messages`` with a Bearer token
and the ``oauth-2025-04-20,claude-code-20250219`` anthropic-beta header; see
:mod:`strix.config.claude_proxy` for the request-rewriting proxy.

Note: this consumer OAuth flow is not an officially supported integration path.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from strix.utils.secret_files import write_secret_text


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


logger = logging.getLogger(__name__)


PROVIDER = "claude"

# Values from the Claude Code CLI binary; not a stable public API.
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"  # noqa: S105  # nosec B105 - URL, not a secret
MANUAL_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
CALLBACK_HOST = "localhost"
CALLBACK_PORT = 54545
CALLBACK_PATH = "/callback"
LOOPBACK_REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
SCOPE = "org:create_api_key user:profile user:inference"

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_BETA = "oauth-2025-04-20,claude-code-20250219"
CLAUDE_CODE_SYSTEM_IDENTIFIER = "You are Claude Code, Anthropic's official CLI for Claude."

_TOKEN_TIMEOUT = 30
_EXPIRY_SKEW_S = 300

_refresh_lock = threading.Lock()

# Kept separate from cli-config.json so OAuth tokens never land in the env-var config.
AUTH_PATH = Path.home() / ".strix" / "subscription-auth.json"

# Credential stores we read as a fallback to our own.
def _external_credential_paths() -> list[Path]:
    home = Path.home()
    candidates: list[Path] = [
        home / ".claude" / ".credentials.json",
        home / ".config" / "opencode" / "auth.json",
        home / ".local" / "share" / "opencode" / "auth.json",
    ]
    win_profile = os.environ.get("STRIX_WINDOWS_PROFILE", "").strip()
    if win_profile:
        candidates.append(Path(win_profile) / ".claude" / ".credentials.json")
    return candidates


def _read_store() -> dict[str, Any]:
    try:
        data = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_store(data: dict[str, Any]) -> None:
    write_secret_text(AUTH_PATH, json.dumps(data, indent=2))


def read_record() -> dict[str, Any] | None:
    """Return the active Claude OAuth record from any source, or ``None``."""
    record = _read_strix_record()
    if record is not None:
        return record
    for path in _external_credential_paths():
        record = _read_external_record(path)
        if record is not None:
            return record
    return None


def _read_strix_record() -> dict[str, Any] | None:
    record = _read_store().get(PROVIDER)
    if not isinstance(record, dict) or record.get("type") != "oauth":
        return None
    if not (record.get("access") and record.get("refresh")):
        return None
    return record


def _read_external_record(path: Path) -> dict[str, Any] | None:
    """Parse a Claude Code / OpenCode credentials file into our record shape."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("ignoring unreadable claude-sub credentials at %s", path)
        return None
    if not isinstance(data, dict):
        return None
    return _record_from_claude_code(data, path) or _record_from_opencode(data, path)


def _record_from_claude_code(data: dict[str, Any], path: Path) -> dict[str, Any] | None:
    block = data.get("claudeAiOauth")
    if not isinstance(block, dict):
        return None
    access = block.get("accessToken")
    refresh = block.get("refreshToken")
    expires_at = _coerce_expiry(block.get("expiresAt"))
    if not isinstance(access, str) or not access:
        return None
    if not isinstance(refresh, str):
        refresh = ""
    return {
        "type": "oauth",
        "provider": PROVIDER,
        "source": "claude-code",
        "source_path": str(path),
        "access": access,
        "refresh": refresh,
        "expires_at": expires_at,
    }


def _record_from_opencode(data: dict[str, Any], path: Path) -> dict[str, Any] | None:
    """Convert an OpenCode auth.json entry for an Anthropic-style provider.

    Field names vary across OpenCode builds; any entry with a Bearer access
    token plus (optionally) a refresh token and an expiry stamp is usable.
    """
    for provider_key in ("anthropic", "claude", "anthropic-sub", "claude-pro"):
        block = data.get(provider_key)
        if not isinstance(block, dict):
            continue
        kind = (block.get("type") or "").lower()
        if kind and kind not in ("oauth", "auth"):
            continue
        access = block.get("access") or block.get("accessToken") or block.get("token")
        refresh = block.get("refresh") or block.get("refreshToken") or ""
        expires = block.get("expires") or block.get("expiresAt") or block.get("expires_at")
        if not isinstance(access, str) or not access:
            continue
        return {
            "type": "oauth",
            "provider": PROVIDER,
            "source": "opencode",
            "source_path": str(path),
            "access": access,
            "refresh": refresh if isinstance(refresh, str) else "",
            "expires_at": _coerce_expiry(expires),
        }
    return None


def _coerce_expiry(value: Any) -> float:
    """POSIX seconds from an int (s), int (ms), or ISO-8601 string; 0 for junk."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        # Values > 1e12 are ms since epoch (Claude Code's format).
        return float(value) / 1000.0 if value > 1e12 else float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0
        with contextlib.suppress(ValueError):
            return float(text)
        with contextlib.suppress(ValueError):
            iso = text.replace("Z", "+00:00") if text.endswith("Z") else text
            return datetime.fromisoformat(iso).timestamp()
    return 0.0


def is_authenticated() -> bool:
    return read_record() is not None


def save_record(record: dict[str, Any]) -> None:
    data = _read_store()
    data[PROVIDER] = record
    _write_store(data)


def logout() -> None:
    data = _read_store()
    if PROVIDER not in data:
        return
    del data[PROVIDER]
    if data:
        _write_store(data)
        return
    with contextlib.suppress(OSError):
        AUTH_PATH.unlink()


try:
    import fcntl as _fcntl  # Linux only; Windows falls back to the in-process lock
except ImportError:
    _fcntl = None  # type: ignore[assignment]


@contextlib.contextmanager
def _refresh_guard() -> Iterator[None]:
    """Serialize token refresh within (lock) and across (flock) Strix processes."""
    with _refresh_lock:
        if _fcntl is None:
            yield
            return
        try:
            lock_path = AUTH_PATH.with_suffix(".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = lock_path.open("w")
        except OSError:
            yield
            return
        try:
            with contextlib.suppress(OSError):
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
            handle.close()


class ClaudeAuthError(Exception):
    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def generate_pkce() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(64))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def create_state() -> str:
    return secrets.token_hex(16)


def build_authorize_url(challenge: str, state: str, *, manual: bool) -> str:
    """Return the authorize URL. ``manual=True`` renders code#state on a page
    instead of redirecting to localhost."""
    redirect_uri = MANUAL_REDIRECT_URI if manual else LOOPBACK_REDIRECT_URI
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    if manual:
        params["code"] = "true"
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def parse_redirect_input(value: str) -> tuple[str | None, str | None]:
    """Extract ``(code, state)`` from ``CODE#STATE``, a redirect URL, or a bare code."""
    value = (value or "").strip()
    if not value:
        return None, None
    with contextlib.suppress(ValueError):
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme and parsed.query:
            query = urllib.parse.parse_qs(parsed.query)
            return _first(query, "code"), _first(query, "state")
    if "#" in value:
        code, _, state = value.partition("#")
        return code or None, state or None
    if "code=" in value:
        query = urllib.parse.parse_qs(value)
        return _first(query, "code"), _first(query, "state")
    return value, None


def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def _post_token(payload: dict[str, str]) -> dict[str, Any]:
    """Call the OAuth token endpoint; retry as JSON if form-urlencoded is rejected."""
    detail = ""
    status_code = 0
    body = b""
    try:
        with requests.post(
            TOKEN_URL,
            data=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=_TOKEN_TIMEOUT,
        ) as response:
            status_code = response.status_code
            body = response.content
            if status_code >= 400:
                detail = response.text[:300]
    except requests.RequestException as exc:
        raise ClaudeAuthError("unavailable", str(exc)) from exc
    if status_code >= 400:
        try:
            with requests.post(
                TOKEN_URL,
                json=payload,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                timeout=_TOKEN_TIMEOUT,
            ) as response:
                status_code = response.status_code
                body = response.content
                if status_code >= 400:
                    detail = response.text[:300]
        except requests.RequestException as exc:
            raise ClaudeAuthError("unavailable", str(exc)) from exc
    if status_code >= 400:
        raise ClaudeAuthError("token_http_error", f"HTTP {status_code}: {detail}")
    data = json.loads(body or b"{}")
    if not isinstance(data, dict):
        raise ClaudeAuthError("bad_response", "token endpoint returned non-object")
    return data


def _record_from_token_response(
    data: dict[str, Any], refresh_fallback: str | None = None
) -> dict[str, Any]:
    access = data.get("access_token")
    refresh = data.get("refresh_token") or refresh_fallback
    expires_in = data.get("expires_in")
    if not isinstance(access, str) or not access:
        raise ClaudeAuthError("bad_response", "token response missing access_token")
    if not isinstance(refresh, str) or not refresh:
        raise ClaudeAuthError("bad_response", "token response missing refresh_token")
    ttl = float(expires_in) if isinstance(expires_in, int | float) else 3600.0
    return {
        "type": "oauth",
        "provider": PROVIDER,
        "source": "strix",
        "access": access,
        "refresh": refresh,
        "expires_at": time.time() + ttl,
    }


def exchange_code(
    code: str, verifier: str, state: str, *, manual: bool
) -> dict[str, Any]:
    redirect_uri = MANUAL_REDIRECT_URI if manual else LOOPBACK_REDIRECT_URI
    data = _post_token(
        {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "state": state,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
        }
    )
    return _record_from_token_response(data)


def refresh_tokens(refresh_token: str) -> dict[str, Any]:
    data = _post_token(
        {
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token,
        }
    )
    return _record_from_token_response(data, refresh_fallback=refresh_token)


def _near_expiry(record: dict[str, Any]) -> bool:
    expires_at = record.get("expires_at")
    if not isinstance(expires_at, int | float):
        return True
    return expires_at - _EXPIRY_SKEW_S <= time.time()


def get_valid_token() -> str:
    """Return a fresh access token, refreshing under the cross-process guard if needed."""
    record = read_record()
    if record is None:
        raise ClaudeAuthError(
            "not_authenticated",
            "not signed in. Run `strix auth login claude` or sign into Claude Code first.",
        )
    if not _near_expiry(record):
        return str(record["access"])
    refresh = record.get("refresh") or ""
    if not refresh:
        # No refresh token available (some OpenCode configs); try the current
        # access token and surface a re-auth hint on 401.
        return str(record["access"])
    with _refresh_guard():
        record = read_record() or record
        if not _near_expiry(record):
            return str(record["access"])
        try:
            refreshed = refresh_tokens(record["refresh"])
        except ClaudeAuthError:
            latest = read_record()
            if (
                latest
                and latest.get("refresh") != record.get("refresh")
                and not _near_expiry(latest)
            ):
                return str(latest["access"])
            raise
        save_record(refreshed)
        return refreshed["access"]


SUBSCRIPTION_PREFIXES: tuple[str, ...] = ("claude-code/", "claude-sub/", "claude-max/")


def subscription_model(model_name: str | None) -> str | None:
    """Anthropic slug behind a ``claude-code/<model>`` STRIX_LLM, or ``None``."""
    name = (model_name or "").strip()
    lowered = name.lower()
    for prefix in SUBSCRIPTION_PREFIXES:
        if lowered.startswith(prefix):
            stripped = name[len(prefix):]
            return stripped or None
    return None


def auth_mode(model_name: str | None) -> str:
    return "subscription" if subscription_model(model_name) else "api_key"


def external_source_summary() -> str | None:
    """Which external store supplies the active record, or ``None`` for strix's own."""
    if _read_strix_record() is not None:
        return None
    for path in _external_credential_paths():
        record = _read_external_record(path)
        if record is not None:
            return f"{record['source']} ({path})"
    return None


def scopes_from_record(record: dict[str, Any]) -> Iterable[str]:
    scopes = record.get("scopes")
    if isinstance(scopes, list):
        return [str(s) for s in scopes]
    return []


__all__ = [
    "ANTHROPIC_BETA",
    "ANTHROPIC_MESSAGES_URL",
    "ANTHROPIC_VERSION",
    "AUTHORIZE_URL",
    "CALLBACK_HOST",
    "CALLBACK_PATH",
    "CALLBACK_PORT",
    "CLAUDE_CODE_SYSTEM_IDENTIFIER",
    "CLIENT_ID",
    "LOOPBACK_REDIRECT_URI",
    "MANUAL_REDIRECT_URI",
    "PROVIDER",
    "SCOPE",
    "SUBSCRIPTION_PREFIXES",
    "TOKEN_URL",
    "ClaudeAuthError",
    "auth_mode",
    "build_authorize_url",
    "create_state",
    "exchange_code",
    "external_source_summary",
    "generate_pkce",
    "get_valid_token",
    "is_authenticated",
    "logout",
    "parse_redirect_input",
    "read_record",
    "refresh_tokens",
    "save_record",
    "scopes_from_record",
    "subscription_model",
]
