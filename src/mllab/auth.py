"""Auth check for Caddy forward_auth.

A request passes only with BOTH Henry's Tailscale login header and the shared secret.
The header alone is spoofable by host-network containers, so it is never sufficient.
"""
from __future__ import annotations

import hmac
import os

# Account domains the lab must never use (for example an employer's), read from the environment so
# no domain is written into the code. Set MLLAB_BLOCKED_ACCOUNT_DOMAINS (comma-separated) on any
# machine that is also signed in to such an account.
BLOCKED_DOMAINS_ENV = "MLLAB_BLOCKED_ACCOUNT_DOMAINS"


def blocked_domains() -> tuple[str, ...]:
    raw = os.environ.get(BLOCKED_DOMAINS_ENV, "")
    return tuple(d.strip().lower().lstrip("@") for d in raw.split(",") if d.strip())


def is_blocked_account(email: str | None) -> bool:
    if not email or "@" not in email:
        return False
    return email.rsplit("@", 1)[-1].strip().lower() in blocked_domains()


def check(login: str | None, secret: str | None, *, allowed_login: str, expected_secret: str) -> bool:
    if not login or not secret or not allowed_login or not expected_secret:
        return False
    login_ok = hmac.compare_digest(login.strip().lower().encode(), allowed_login.strip().lower().encode())
    secret_ok = hmac.compare_digest(secret.encode(), expected_secret.encode())
    return login_ok and secret_ok


def assert_not_work_account(email: str) -> None:
    if is_blocked_account(email):
        raise RuntimeError(f"refusing to use work-domain credential: {email}")


def check_login(login: str | None, *, allowed_login: str) -> bool:
    if not login or not allowed_login:
        return False
    return hmac.compare_digest(login.strip().lower().encode(), allowed_login.strip().lower().encode())


def needs_secret(uri: str | None) -> bool:
    """Programmatic paths (/api/...) require the shared secret; browser routes cannot send it
    and rely on Jupyter's own token as the second factor instead."""
    return bool(uri) and uri.split("?", 1)[0].startswith("/api/")


def from_env(login: str | None, secret: str | None, uri: str | None = None) -> bool:
    allowed = os.environ.get("MLLAB_ALLOWED_LOGIN", "")
    if not needs_secret(uri):
        return check_login(login, allowed_login=allowed)
    return check(login, secret, allowed_login=allowed, expected_secret=os.environ.get("MLLAB_SECRET", ""))
