"""HTTP surface of the control app: the page at /app/ and its JSON API at /app-api/.

Every handler runs the app's own checks (session.py) because the service is reachable on loopback without Caddy. Reads need the allowed Tailscale login;
anything that changes state also needs the unlock cookie and a same-origin request. Responses are never cached, and the page ships with a strict
Content-Security-Policy: its own script and stylesheet only, no inline code, no framing."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, Body, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from .. import audit
from ..jobs import LocalRunner
from ..pipeline import broker
from ..registry import Registry
from .. import datamap
from . import session, views
from .pipeline_view import PipelineView
from .views import Refused

STATIC = Path(__file__).parent / "static"
ASSETS = {"app.css": "text/css; charset=utf-8", "app.js": "text/javascript; charset=utf-8"}
CSP = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


@dataclass
class Config:
    root: Path
    registry: Path
    status_dir: Path
    spool: Path
    results: Path
    audit_path: Path

    @classmethod
    def from_env(cls) -> "Config":
        root = Path(os.environ.get("MLLAB_ROOT", "/mnt/appdata/mllab"))
        return cls(root=root, registry=Path(os.environ.get("MLLAB_REGISTRY", root / "registry")), status_dir=broker.STATUS_DIR, spool=broker.SPOOL,
                   results=broker.RESULTS, audit_path=audit.log_path())


def _headers(request: Request) -> dict:
    return {k.lower(): v for k, v in request.headers.items()}


def build(cfg: Config, *, scheduler=None, throttle: session.Throttle | None = None, runner: LocalRunner | None = None, bq=None, catalog_dir: Path | None = None,
          clock=None, control_store=None) -> APIRouter:
    router = APIRouter()
    pipe = PipelineView(cfg.status_dir, cfg.spool, cfg.results, scheduler=scheduler, audit_path=cfg.audit_path)
    reg = Registry(cfg.registry)
    runner = runner or LocalRunner(cfg.root)
    throttle = throttle or session.Throttle()
    data_cache = datamap.Cache(60.0)
    catalog: list = []  # loaded on first use, so a bad catalog file reports itself on the Data page instead of stopping the app from starting

    def reply(data: dict, status: int = 200) -> JSONResponse:
        return JSONResponse(data, status_code=status, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    def call(request: Request, fn, *, write: bool = False, unlocked: bool = False) -> JSONResponse:
        """Run fn(login) after the access check (read, unlocked read, or write); turn refusals into JSON the page can show."""
        try:
            h, c = _headers(request), request.cookies
            login = session.writer(h, c) if write else session.unlocked_reader(h, c) if unlocked else session.reader(h)
            return reply(fn(login))
        except session.Denied as e:
            return reply({"error": e.code, "reasons": [e.detail]}, e.status)
        except Refused as e:
            return reply({"error": "refused", "reasons": e.reasons}, e.status)

    # ------------------------------------------------------------------------------------------------------------------------- the page
    def page(name: str, ctype: str) -> Response:
        return FileResponse(STATIC / name, media_type=ctype, headers={"Cache-Control": "no-store", "Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff",
                                                                      "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY"})

    @router.get("/app")
    @router.get("/app/")
    def app_page(request: Request):
        try:
            session.reader(_headers(request))
        except session.Denied as e:
            return Response(f"{e.detail}\n", status_code=e.status, media_type="text/plain")
        return page("index.html", "text/html; charset=utf-8")

    @router.get("/app/static/{name}")
    def app_asset(name: str, request: Request):
        if name not in ASSETS:  # a fixed list, never a path built from the request
            return Response("not found\n", status_code=404, media_type="text/plain")
        try:
            session.reader(_headers(request))
        except session.Denied as e:
            return Response(f"{e.detail}\n", status_code=e.status, media_type="text/plain")
        return page(name, ASSETS[name])

    # ---------------------------------------------------------------------------------------------------------------------- session
    @router.get("/app-api/session")
    def get_session(request: Request):
        return call(request, lambda login: {"login": login, "unlocked": session.unlocked(_headers(request), request.cookies), "open": session.open_access()})

    @router.post("/app-api/unlock")
    def unlock(request: Request, body: dict = Body(...)):
        h = _headers(request)
        try:
            login = session.reader(h)
            if h.get(session.CSRF_HEADER) != "1" or not session.same_origin(h.get("origin"), h.get("host")):
                raise session.Denied(403, "origin", "requests must come from the app itself")
        except session.Denied as e:
            return reply({"error": e.code, "reasons": [e.detail]}, e.status)
        if session.open_access():
            return reply({"unlocked": True, "open": True})  # nothing to unlock
        if throttle.blocked():
            return reply({"error": "throttled", "reasons": ["too many wrong passwords; wait a minute"]}, 429)
        import hmac
        secret = os.environ.get("MLLAB_SECRET", "")
        given = body.get("password")
        if not secret or not isinstance(given, str) or not hmac.compare_digest(given.encode(), secret.encode()):
            throttle.fail()
            audit.record("app_unlock_failed", path=cfg.audit_path, by=login)
            return reply({"error": "password", "reasons": ["wrong password"]}, 401)
        res = reply({"unlocked": True})
        res.set_cookie(session.COOKIE, session.make_cookie(secret, login), max_age=session.TTL, httponly=True, samesite="strict", path="/app-api",
                       secure=h.get("x-forwarded-proto") == "https")
        audit.record("app_unlock", path=cfg.audit_path, by=login)
        return res

    @router.post("/app-api/lock")
    def lock(request: Request):
        res = reply({"unlocked": False})
        res.delete_cookie(session.COOKIE, path="/app-api")
        return res

    # ------------------------------------------------------------------------------------------------------------------------ pipeline
    @router.get("/app-api/pipeline")
    def get_pipeline(request: Request, fresh: bool = False):
        return call(request, lambda login: pipe.state(fresh=fresh))

    @router.get("/app-api/pipeline/log/{job}")
    def get_pipeline_log(job: str, request: Request):
        return call(request, lambda login: pipe.log(job), unlocked=True)

    @router.post("/app-api/pipeline/mode")
    def post_mode(request: Request, body: dict = Body(...)):
        return call(request, lambda login: {"request": pipe.set_mode(str(body.get("job")), str(body.get("mode")), login, body.get("confirm"),
                                                                      body.get("acks") if isinstance(body.get("acks"), list) else [])}, write=True)

    @router.post("/app-api/pipeline/rollback")
    def post_rollback(request: Request, body: dict = Body(...)):
        return call(request, lambda login: {"request": pipe.rollback(str(body.get("job")), login, body.get("confirm"),
                                                                      body.get("acks") if isinstance(body.get("acks"), list) else [])}, write=True)

    @router.post("/app-api/pipeline/run")
    def post_run(request: Request, body: dict = Body(...)):
        return call(request, lambda login: {"request": pipe.run(str(body.get("job")), body.get("dry") is not False, login, body.get("confirm"),
                                                                 body.get("acks") if isinstance(body.get("acks"), list) else [])}, write=True)

    @router.get("/app-api/pipeline/request/{rid}")
    def get_request(rid: str, request: Request):
        return call(request, lambda login: pipe.request(rid))

    # ------------------------------------------------------------------------------------------------------------------------- models
    @router.get("/app-api/models")
    def get_models(request: Request):
        return call(request, lambda login: views.models(reg, cfg.audit_path))

    @router.post("/app-api/models/promote")
    def post_promote(request: Request, body: dict = Body(...)):
        return call(request, lambda login: views.promote(reg, str(body.get("project")), str(body.get("model")), str(body.get("version")),
                                                         str(body.get("channel")), login, body.get("confirm"), cfg.audit_path), write=True)

    @router.post("/app-api/models/rollback")
    def post_model_rollback(request: Request, body: dict = Body(...)):
        return call(request, lambda login: views.rollback_channel(reg, str(body.get("project")), str(body.get("model")), str(body.get("channel")), login,
                                                                   body.get("confirm"), cfg.audit_path), write=True)

    # --------------------------------------------------------------------------------------------------------------------------- jobs
    @router.get("/app-api/jobs")
    def get_jobs(request: Request):
        return call(request, lambda login: views.jobs(runner))

    @router.get("/app-api/jobs/{job_id}/log")
    def get_job_log(job_id: str, request: Request, offset: int = 0):
        return call(request, lambda login: views.job_log(runner, job_id, offset), unlocked=True)

    @router.post("/app-api/jobs/{job_id}/cancel")
    def post_cancel(job_id: str, request: Request):
        return call(request, lambda login: views.cancel(runner, job_id, login, cfg.audit_path), write=True)

    # -------------------------------------------------------------------------------------------------------------- health, gcp, audit
    @router.get("/app-api/health")
    def get_health(request: Request):
        return call(request, lambda login: views.health(cfg.root))

    @router.get("/app-api/gcp")
    def get_gcp(request: Request):
        def go(login):
            jobs, err = pipe.gcp()
            raw, _ = pipe._snapshot()
            modes = {j["name"]: j["mode"] for j in (raw or {}).get("jobs", [])}
            if raw:
                modes["mlb-2026-pregame-schedule"] = raw["pregame"]["mode"]
            return views.gcp(jobs, err, modes)
        return call(request, go)

    @router.get("/app-api/data")
    def get_data(request: Request, fresh: bool = False):
        def go(login):
            from datetime import datetime, timezone
            try:
                if not catalog:
                    catalog.extend(datamap.load(catalog_dir or datamap.find_catalog()))

                def compute():
                    runner_ = bq
                    if runner_ is None:
                        from ..health_collect import BqClientRunner
                        runner_ = BqClientRunner()
                    facts, errors = datamap.collect_facts(runner_, {t.id.split(".")[0] for t in catalog})
                    raw, _ = pipe._snapshot()
                    jobs_ = (raw or {}).get("jobs", [])
                    runs = {j["name"]: j["last"] for j in jobs_ if j.get("last")}
                    modes = {j["name"]: j["mode"] for j in jobs_}
                    return datamap.build(catalog, facts, clock() if clock else datetime.now(timezone.utc), runs, modes, errors)

                if fresh:
                    data_cache._at = None
                return {"available": True, **data_cache.get(compute)}
            except datamap.CatalogError as e:
                return {"available": False, "error": f"catalog: {e}", "sports": {}, "stages": datamap.STAGES}
            except Exception as e:  # noqa: BLE001  BigQuery unreachable or not permitted: say so, do not call every table missing
                return {"available": False, "error": f"{type(e).__name__}: {e}"[:300], "sports": {}, "stages": datamap.STAGES}
        return call(request, go)

    # ------------------------------------------------------------------------------------------------------------------------- the site
    def store():
        if control_store is not None:
            return control_store
        from ..control import BqClientStore
        return BqClientStore()

    def bq_runner():
        if bq is not None:
            return bq
        from ..health_collect import BqClientRunner
        return BqClientRunner()

    @router.get("/app-api/site")
    def get_site(request: Request):
        def go(login):
            from ..health_collect import control_from_view

            def columns():
                rows = bq_runner().query("SELECT column_name FROM `hankstank.control.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = 'model_control_current'")
                return [r["column_name"] for r in rows]

            return views.site_rankings(lambda: control_from_view(bq_runner()), columns)
        return call(request, go)

    @router.post("/app-api/site/rankings")
    def post_site_rankings(request: Request, body: dict = Body(...)):
        def pick(key):
            if key not in body:
                return ...
            v = body[key]
            if v is None:
                return None
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise views.Refused([f"{key} must be a list of names, or null for the default"], 400)
            return v

        def pick_primary():
            if "primary" not in body:
                return ...
            v = body["primary"]
            if v is None or isinstance(v, str):
                return v
            raise views.Refused(["primary must be 'resume', 'rating', or null for the default"], 400)

        return call(request, lambda login: views.set_site_rankings(store(), str(body.get("sport")), pick("show"), pick("media"), login, body.get("confirm"), cfg.audit_path,
                                                                   primary=pick_primary()),
                    write=True)

    @router.get("/app-api/audit")
    def get_audit(request: Request):
        return call(request, lambda login: views.audit_tail(cfg.audit_path))

    @router.get("/app-api/notebooks")
    def get_notebooks(request: Request):
        return call(request, lambda login: views.notebooks(cfg.root / "work"))

    return router
