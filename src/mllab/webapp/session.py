"""Who may do what in the web app, decided by the app itself.

Caddy checks the Tailscale login before it proxies, but the app listens on 127.0.0.1:8897 and any container on this host can connect there directly,
without Caddy and with any header it likes. So nothing here trusts Caddy:

    read   the Tailscale login header must name the allowed login (a container can forge it, so reads show nothing worse than status)
    write  reading's check, PLUS the unlock cookie (an HMAC under MLLAB_SECRET, which no container has), PLUS a same-origin request carrying a custom
           header (a page on another site cannot send one without a CORS preflight, which the app never grants)

(Setting MLLAB_APP_NO_UNLOCK=1 removes the unlock for people who trust everything running on the host: a process here can then forge the login header and change
modes. The typed-name and Cloud Scheduler gates for going Live are unaffected.)

The unlock is a password, once per device: the secret in /etc/mllab/env, entered at the app's lock screen, remembered by a 30-day cookie."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from urllib.parse import urlparse

from .. import auth

COOKIE = "mllab_unlock"
TTL = 30 * 24 * 3600
CSRF_HEADER = "x-mllab-app"
MAX_FAILS, WINDOW = 5, 60  # wrong passwords per minute before the lock screen refuses to look


class Denied(Exception):
    def __init__(self, status: int, code: str, detail: str):
        super().__init__(detail)
        self.status, self.code, self.detail = status, code, detail


def _mac(secret: str, login: str, ts: int) -> str:
    return hmac.new(secret.encode(), f"unlock|{login.lower()}|{ts}".encode(), hashlib.sha256).hexdigest()


def make_cookie(secret: str, login: str, now: float | None = None) -> str:
    ts = int(now if now is not None else time.time())
    return base64.urlsafe_b64encode(f"{ts}.{_mac(secret, login, ts)}".encode()).decode()


def valid_cookie(secret: str, login: str, cookie: str | None, now: float | None = None) -> bool:
    if not secret or not cookie:
        return False
    try:
        ts_s, mac = base64.urlsafe_b64decode(cookie.encode()).decode().split(".", 1)
        ts = int(ts_s)
    except (ValueError, UnicodeError):
        return False
    age = (now if now is not None else time.time()) - ts
    return 0 <= age < TTL and hmac.compare_digest(mac, _mac(secret, login, ts))


def same_origin(origin: str | None, host: str | None) -> bool:
    """A browser always sends Origin on a POST; its host must be the host the request was addressed to."""
    if not origin or not host:
        return False
    return urlparse(origin).netloc.lower() == host.lower()


class Throttle:
    def __init__(self, clock=time.monotonic):
        self.clock, self.fails = clock, []

    def blocked(self) -> bool:
        now = self.clock()
        self.fails = [t for t in self.fails if now - t < WINDOW]
        return len(self.fails) >= MAX_FAILS

    def fail(self) -> None:
        self.fails.append(self.clock())


def reader(headers: dict) -> str:
    """The login, if it is the allowed one. `headers` keys are lower-case."""
    login = headers.get("tailscale-user-login")
    if not auth.check_login(login, allowed_login=os.environ.get("MLLAB_ALLOWED_LOGIN", "")):
        raise Denied(401, "login", "not signed in with the allowed Tailscale login")
    return login


def open_access() -> bool:
    """MLLAB_APP_NO_UNLOCK=1 in /etc/mllab/env turns the password off. Off by default, and only that exact value counts: a typo leaves the lock on."""
    return os.environ.get("MLLAB_APP_NO_UNLOCK") == "1"


def unlocked(headers: dict, cookies: dict) -> bool:
    if open_access():
        return True  # the login and same-origin checks still apply to every change; only the password is skipped
    login = headers.get("tailscale-user-login") or ""
    return valid_cookie(os.environ.get("MLLAB_SECRET", ""), login, cookies.get(COOKIE))


def writer(headers: dict, cookies: dict) -> str:
    login = reader(headers)
    if headers.get(CSRF_HEADER) != "1" or not same_origin(headers.get("origin"), headers.get("host")):
        raise Denied(403, "origin", "requests must come from the app itself")
    if not unlocked(headers, cookies):
        raise Denied(401, "locked", "locked: enter the lab password to make changes")
    return login


def unlocked_reader(headers: dict, cookies: dict) -> str:
    """For reads that show more than status (run logs): the login plus the unlock, but no same-origin header, since a plain GET carries none."""
    login = reader(headers)
    if not unlocked(headers, cookies):
        raise Denied(401, "locked", "locked: enter the lab password to view this")
    return login
