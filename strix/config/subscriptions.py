"""Unified view over per-vendor subscription auth: codex (ChatGPT) and claude."""

from __future__ import annotations

from strix.config import claude_sub, codex


def is_subscription_model(model_name: str | None) -> bool:
    return bool(codex.subscription_model(model_name) or claude_sub.subscription_model(model_name))


def subscription_auth_mode(model_name: str | None) -> str:
    return "subscription" if is_subscription_model(model_name) else "api_key"


def subscription_vendor(model_name: str | None) -> str | None:
    if codex.subscription_model(model_name):
        return "chatgpt"
    if claude_sub.subscription_model(model_name):
        return "claude"
    return None


def is_authenticated_for(model_name: str | None) -> bool:
    if codex.subscription_model(model_name):
        return codex.is_authenticated()
    if claude_sub.subscription_model(model_name):
        return claude_sub.is_authenticated()
    return False


def login_command(model_name: str | None) -> str | None:
    vendor = subscription_vendor(model_name)
    if vendor == "chatgpt":
        return "strix auth login chatgpt"
    if vendor == "claude":
        return "strix auth login claude"
    return None


__all__ = [
    "is_authenticated_for",
    "is_subscription_model",
    "login_command",
    "subscription_auth_mode",
    "subscription_vendor",
]
