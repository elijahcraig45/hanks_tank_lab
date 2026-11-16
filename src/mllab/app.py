"""Small FastAPI service. Bind to 127.0.0.1 only; Caddy fronts it with forward_auth."""
from __future__ import annotations

from fastapi import FastAPI, Header, Response

from . import auth
from .webapp import routes as webapp_routes

app = FastAPI(title="mllab", docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/auth")
def forward_auth(
    response: Response,
    tailscale_user_login: str | None = Header(default=None),
    x_mllab_secret: str | None = Header(default=None),
    x_forwarded_uri: str | None = Header(default=None),
):
    if auth.from_env(tailscale_user_login, x_mllab_secret, x_forwarded_uri):
        return {"ok": True}
    response.status_code = 401
    return {"ok": False}


def _scheduler():
    try:
        from .health_collect import ApiScheduler
        return ApiScheduler()
    except Exception:  # noqa: BLE001  the gate treats an unavailable scheduler as a refusal, so the app still starts
        return None


# The control app (page at /app/, API at /app-api/). It does its own access checks: this service is reachable on loopback without Caddy.
app.include_router(webapp_routes.build(webapp_routes.Config.from_env(), scheduler=_scheduler()))
