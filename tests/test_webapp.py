"""The control app: who may read, who may change, what the gates refuse, and that a request reaches the broker in the shape it validates.

The service listens on loopback without Caddy in front, so these tests talk to it the way a hostile local process would: no Tailscale header, a forged
one, no unlock cookie, a request from another origin."""
import json
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mllab.pipeline import broker
from mllab.pipeline.jobs import JOBS, PREGAME
from mllab.registry import Registry
from mllab.webapp import routes, session, views
from mllab.webapp.pipeline_view import PREGAME_SCHEDULER, PipelineView, gate

LOGIN = "henry@example.com"
SECRET = "s3cret-lab-password"
H = {"Tailscale-User-Login": LOGIN}
W = {**H, "X-Mllab-App": "1", "Origin": "http://testserver"}  # what the page itself sends


class FakeScheduler:
    def __init__(self, states=None, fail=False):
        self.states, self.fail = states or {}, fail

    def list(self):
        if self.fail:
            raise RuntimeError("no credentials")
        names = set(JOBS) | {PREGAME_SCHEDULER}
        return [{"name": n, "state": self.states.get(n, "PAUSED"), "schedule": "0 4 * * *", "tz": "America/New_York", "mode": "x", "body_ok": True,
                 "last_attempt_at": None, "last_attempt_ok": None, "next_run": None} for n in sorted(names)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MLLAB_ALLOWED_LOGIN", LOGIN)
    monkeypatch.setenv("MLLAB_SECRET", SECRET)
    for n in ("status", "spool", "results", "registry", "work"):
        (tmp_path / n).mkdir()
    (tmp_path / "jobs").mkdir()
    (tmp_path / "state").mkdir(); (tmp_path / "pjobs").mkdir()
    broker.cmd_snapshot(jobs_dir=tmp_path / "pjobs", state_dir=tmp_path / "state", release=tmp_path / "none", show=lambda u: {}, status_dir=tmp_path / "status")
    return tmp_path


def make(env, scheduler=None, throttle=None):
    cfg = routes.Config(root=env, registry=env / "registry", status_dir=env / "status", spool=env / "spool", results=env / "results", audit_path=env / "audit.log")
    app = FastAPI()
    app.include_router(routes.build(cfg, scheduler=scheduler if scheduler is not None else FakeScheduler(), throttle=throttle))
    return TestClient(app)


def unlock(c):
    r = c.post("/app-api/unlock", json={"password": SECRET}, headers=W)
    assert r.status_code == 200, r.text
    return r


# ------------------------------------------------------------------------------------------------------------------------------ access
@pytest.mark.parametrize("headers", [{}, {"Tailscale-User-Login": "intruder@example.com"}, {"Tailscale-User-Login": ""}])
def test_reads_need_the_allowed_login(env, headers):
    c = make(env)
    for path in ("/app-api/pipeline", "/app-api/models", "/app-api/gcp", "/app-api/audit", "/app-api/jobs", "/app/"):
        assert c.get(path, headers=headers).status_code == 401, path


def test_no_allowed_login_configured_means_nobody(env, monkeypatch):
    monkeypatch.setenv("MLLAB_ALLOWED_LOGIN", "")
    assert make(env).get("/app-api/pipeline", headers=H).status_code == 401


def test_a_change_without_the_unlock_is_refused_even_with_the_right_login(env):
    c = make(env)
    r = c.post("/app-api/pipeline/mode", json={"job": "nfl-weekly-predict", "mode": "dry"}, headers=W)
    assert r.status_code == 401 and r.json()["error"] == "locked"
    assert not list((env / "spool").glob("pipe-*"))


def test_a_forged_login_header_alone_cannot_change_anything(env):
    # what a container on this host can do: hit the port directly with a forged header, and no secret
    c = make(env)
    for path, body in (("/app-api/pipeline/mode", {"job": "nfl-weekly-predict", "mode": "live"}), ("/app-api/pipeline/run", {"job": "nfl-weekly-predict"}),
                       ("/app-api/models/promote", {}), ("/app-api/jobs/abcd1234-0000/cancel", {})):
        assert c.post(path, json=body, headers=W).status_code == 401, path


def test_a_change_from_another_origin_is_refused_even_when_unlocked(env):
    c = make(env)
    unlock(c)
    body = {"job": "nfl-weekly-predict", "mode": "dry"}
    assert c.post("/app-api/pipeline/mode", json=body, headers={**W, "Origin": "https://evil.example"}).status_code == 403
    assert c.post("/app-api/pipeline/mode", json=body, headers={k: v for k, v in W.items() if k != "Origin"}).status_code == 403
    assert c.post("/app-api/pipeline/mode", json=body, headers={k: v for k, v in W.items() if k != "X-Mllab-App"}).status_code == 403
    assert not list((env / "spool").glob("pipe-*"))


def test_the_wrong_password_does_not_unlock_and_is_throttled(env):
    c = make(env, throttle=session.Throttle())
    for _ in range(5):
        assert c.post("/app-api/unlock", json={"password": "nope"}, headers=W).status_code == 401
    r = c.post("/app-api/unlock", json={"password": SECRET}, headers=W)
    assert r.status_code == 429  # even the right one waits: a guesser gets no oracle
    assert session.COOKIE not in c.cookies


def test_unlock_sets_a_cookie_that_only_works_for_that_login(env):
    c = make(env)
    unlock(c)
    assert c.get("/app-api/session", headers=H).json() == {"login": LOGIN, "unlocked": True, "open": False}
    assert "samesite=strict" in c.post("/app-api/unlock", json={"password": SECRET}, headers=W).headers["set-cookie"].lower()
    cookie = session.make_cookie(SECRET, "someone-else@example.com")
    assert not session.valid_cookie(SECRET, LOGIN, cookie)


def test_the_cookie_expires_and_is_not_forgeable():
    now = time.time()
    c = session.make_cookie(SECRET, LOGIN, now)
    assert session.valid_cookie(SECRET, LOGIN, c, now + 10)
    assert not session.valid_cookie(SECRET, LOGIN, c, now + session.TTL + 1)
    assert not session.valid_cookie("other-secret", LOGIN, c, now)
    assert not session.valid_cookie(SECRET, LOGIN, "garbage") and not session.valid_cookie(SECRET, LOGIN, None) and not session.valid_cookie("", LOGIN, c)


def test_lock_clears_the_unlock(env):
    c = make(env)
    unlock(c)
    c.post("/app-api/lock", headers=W)
    assert c.get("/app-api/session", headers=H).json()["unlocked"] is False


def test_caddy_does_not_demand_the_secret_for_the_app_paths_but_does_for_api():
    from mllab import auth
    assert not auth.needs_secret("/app-api/pipeline") and not auth.needs_secret("/app/") and auth.needs_secret("/api/anything")


# ------------------------------------------------------------------------------------------------------------------------------- the page
def test_the_page_and_its_assets_are_served_with_a_strict_policy(env):
    c = make(env)
    r = c.get("/app/", headers=H)
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "script-src 'self'" in r.headers["content-security-policy"] and "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers["cache-control"] == "no-store"
    assert "<script" in r.text and "onclick=" not in r.text  # no inline handlers: the policy would block them
    for name in ("app.js", "app.css"):
        assert c.get(f"/app/static/{name}", headers=H).status_code == 200


def test_assets_come_from_a_fixed_list(env):
    c = make(env)
    for bad in ("../session.py", "..%2fsession.py", "index.html", "x.js"):
        assert c.get(f"/app/static/{bad}", headers=H).status_code == 404


# ----------------------------------------------------------------------------------------------------------------------------- pipeline
def test_pipeline_state_joins_the_snapshot_with_cloud_scheduler(env):
    c = make(env, FakeScheduler({"nfl-weekly-predict": "ENABLED"}))
    st = c.get("/app-api/pipeline", headers=H).json()
    by = {j["name"]: j for j in st["jobs"]}
    assert st["broker"]["fresh"] is True and set(by) == set(JOBS)
    assert by["nfl-weekly-predict"]["gcp"]["state"] == "ENABLED" and by["nfl-weekly-predict"]["gate"]["allowed"] is False
    assert by["nfl-weekly-score"]["gate"]["allowed"] is True  # its Cloud Scheduler job is paused
    assert st["pregame"]["gate"]["acks"]  # the two steps the app cannot see


def test_a_stale_snapshot_says_the_broker_is_not_running(env):
    old = json.loads((env / "status" / "state.json").read_text())
    old["taken_at"] = "2026-01-01T00:00:00+00:00"
    (env / "status" / "state.json").write_text(json.dumps(old))
    st = make(env).get("/app-api/pipeline", headers=H).json()
    assert st["broker"]["fresh"] is False and "not publishing" in st["broker"]["hint"]


def test_no_snapshot_at_all_is_reported_not_a_crash(env):
    (env / "status" / "state.json").unlink()
    st = make(env).get("/app-api/pipeline", headers=H).json()
    assert st["broker"]["available"] is False and st["jobs"] == []


def test_a_job_live_here_and_enabled_in_gcp_is_flagged_as_a_conflict(env):
    state = json.loads((env / "status" / "state.json").read_text())
    for j in state["jobs"]:
        if j["name"] == "nfl-weekly-predict":
            j["mode"] = "live"
    state["taken_at"] = broker._now()
    (env / "status" / "state.json").write_text(json.dumps(state))
    st = make(env, FakeScheduler({"nfl-weekly-predict": "ENABLED"})).get("/app-api/pipeline", headers=H).json()
    assert {j["name"]: j["conflict"] for j in st["jobs"]}["nfl-weekly-predict"] is True


def spooled(env):
    files = list((env / "spool").glob("pipe-*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text())


def test_a_dry_change_needs_no_gate_and_reaches_the_broker_in_the_shape_it_validates(env):
    c = make(env, FakeScheduler({"nfl-weekly-predict": "ENABLED"}))
    unlock(c)
    r = c.post("/app-api/pipeline/mode", json={"job": "nfl-weekly-predict", "mode": "dry"}, headers=W)
    assert r.status_code == 200
    req = spooled(env)
    assert req["id"] == r.json()["request"] and req["actor"] == LOGIN
    broker.validate(json.dumps(req).encode(), current_mode=lambda j: "off")  # the broker accepts exactly what the app writes


def test_going_live_is_refused_while_cloud_scheduler_still_runs_the_job(env):
    c = make(env, FakeScheduler({"nfl-weekly-predict": "ENABLED"}))
    unlock(c)
    r = c.post("/app-api/pipeline/mode", json={"job": "nfl-weekly-predict", "mode": "live", "confirm": "nfl-weekly-predict"}, headers=W)
    assert r.status_code == 409 and "ENABLED" in " ".join(r.json()["reasons"])
    assert not list((env / "spool").glob("pipe-*"))


def test_going_live_is_refused_when_cloud_scheduler_cannot_be_read(env):
    c = make(env, FakeScheduler(fail=True))
    unlock(c)
    r = c.post("/app-api/pipeline/mode", json={"job": "nfl-weekly-predict", "mode": "live", "confirm": "nfl-weekly-predict"}, headers=W)
    assert r.status_code == 409 and "refusing rather than guessing" in " ".join(r.json()["reasons"])


def test_going_live_needs_the_name_typed_and_then_goes_through_when_paused(env):
    c = make(env, FakeScheduler({"nfl-weekly-predict": "PAUSED"}))
    unlock(c)
    assert c.post("/app-api/pipeline/mode", json={"job": "nfl-weekly-predict", "mode": "live"}, headers=W).status_code == 400
    assert c.post("/app-api/pipeline/mode", json={"job": "nfl-weekly-predict", "mode": "live", "confirm": "typo"}, headers=W).status_code == 400
    r = c.post("/app-api/pipeline/mode", json={"job": "nfl-weekly-predict", "mode": "live", "confirm": "nfl-weekly-predict"}, headers=W)
    assert r.status_code == 200 and spooled(env)["mode"] == "live"


def test_pregame_live_needs_both_acknowledgements_as_well_as_the_scheduler(env):
    c = make(env, FakeScheduler({PREGAME_SCHEDULER: "PAUSED"}))
    unlock(c)
    body = {"job": PREGAME, "mode": "live", "confirm": PREGAME}
    r = c.post("/app-api/pipeline/mode", json=body, headers=W)
    assert r.status_code == 409 and "cron" in " ".join(r.json()["reasons"])
    assert c.post("/app-api/pipeline/mode", json={**body, "acks": ["cron"]}, headers=W).status_code == 409
    assert c.post("/app-api/pipeline/mode", json={**body, "acks": ["cron", "queues"]}, headers=W).status_code == 200
    broker.validate(json.dumps(spooled(env)).encode(), current_mode=lambda j: "off")


def test_a_run_defaults_to_dry_and_a_real_run_is_gated(env):
    c = make(env, FakeScheduler({"nfl-weekly-predict": "ENABLED"}))
    unlock(c)
    assert c.post("/app-api/pipeline/run", json={"job": "nfl-weekly-predict"}, headers=W).status_code == 200
    assert spooled(env)["dry"] is True
    for f in (env / "spool").glob("pipe-*"):
        f.unlink()
    r = c.post("/app-api/pipeline/run", json={"job": "nfl-weekly-predict", "dry": False, "confirm": "nfl-weekly-predict"}, headers=W)
    assert r.status_code == 409


def test_unknown_and_paused_jobs_are_refused(env):
    c = make(env)
    unlock(c)
    assert c.post("/app-api/pipeline/mode", json={"job": "../etc", "mode": "dry"}, headers=W).status_code == 404
    assert c.post("/app-api/pipeline/mode", json={"job": "mlb-2026-weekly-train-v10", "mode": "dry"}, headers=W).status_code == 409
    assert c.post("/app-api/pipeline/run", json={"job": PREGAME}, headers=W).status_code == 404


def test_request_status_follows_a_request_from_queued_to_done(env):
    c = make(env)
    unlock(c)
    rid = c.post("/app-api/pipeline/mode", json={"job": "nfl-weekly-predict", "mode": "dry"}, headers=W).json()["request"]
    assert c.get(f"/app-api/pipeline/request/{rid}", headers=H).json()["status"] == "queued"
    for f in (env / "spool").glob("pipe-*"):
        f.unlink()
    (env / "results" / f"{rid}.json").write_text(json.dumps({"id": rid, "status": "done", "detail": "ok"}))
    assert c.get(f"/app-api/pipeline/request/{rid}", headers=H).json()["status"] == "done"
    assert c.get("/app-api/pipeline/request/..%2f..%2fetc", headers=H).status_code in (400, 404)


def test_run_logs_need_the_unlock(env):
    (env / "status" / "logs").mkdir(exist_ok=True)
    (env / "status" / "logs" / "nfl-weekly-predict.txt").write_text("hello")
    c = make(env)
    assert c.get("/app-api/pipeline/log/nfl-weekly-predict", headers=H).status_code == 401
    unlock(c)
    assert c.get("/app-api/pipeline/log/nfl-weekly-predict", headers=H).json()["text"] == "hello"
    assert c.get("/app-api/pipeline/log/..%2fsecret", headers=H).status_code in (404, 400)


def test_gate_function_is_pure_and_fails_closed():
    assert gate("nfl-weekly-predict", None, "boom")["allowed"] is False
    assert gate("nfl-weekly-predict", {}, None)["allowed"] is False  # a name Cloud Scheduler does not have is not a green light
    assert gate("nfl-weekly-predict", {"nfl-weekly-predict": {"state": "PAUSED"}})["allowed"] is True
    assert gate(PREGAME, {PREGAME_SCHEDULER: {"state": "PAUSED"}})["commands"]  # the queue commands are shown even when the scheduler is already paused


# ------------------------------------------------------------------------------------------------------------------------ models, gcp, jobs
@pytest.fixture
def registered(env):
    art = env / "m.pkl"
    art.write_bytes(b"model")
    reg = Registry(env / "registry")
    reg.register("hankstank", "game_outcome", "v10", art, metrics={"acc": 0.54})
    reg.register("hankstank", "game_outcome", "v11", art, parent="v10")
    reg.promote("hankstank", "game_outcome", "v10", "prod")
    return reg


def test_models_list_versions_channels_and_metrics(env, registered):
    d = make(env).get("/app-api/models", headers=H).json()
    m = d["projects"][0]["models"][0]
    assert d["projects"][0]["project"] == "hankstank" and m["channels"] == {"prod": "v10"}
    v = {x["version"]: x for x in m["versions"]}
    assert v["v10"]["channels"] == ["prod"] and v["v10"]["metrics"] == {"acc": 0.54} and v["v11"]["parent"] == "v10"


def test_promoting_to_prod_needs_the_typed_phrase_and_is_audited(env, registered):
    c = make(env)
    unlock(c)
    body = {"project": "hankstank", "model": "game_outcome", "version": "v11", "channel": "prod"}
    assert c.post("/app-api/models/promote", json=body, headers=W).status_code == 400
    assert registered.channels("hankstank", "game_outcome")["prod"] == "v10"
    r = c.post("/app-api/models/promote", json={**body, "confirm": "game_outcome@prod"}, headers=W)
    assert r.status_code == 200 and r.json() == {"previous": "v10", "now": "v11"}
    assert registered.channels("hankstank", "game_outcome")["prod"] == "v11"
    assert json.loads((env / "audit.log").read_text().splitlines()[-1])["action"] == "promote"
    # a non-prod channel is a pointer experiment: no phrase
    assert c.post("/app-api/models/promote", json={**body, "channel": "staging"}, headers=W).status_code == 200


def test_a_bad_version_is_refused_not_a_crash(env, registered):
    c = make(env)
    unlock(c)
    r = c.post("/app-api/models/promote", json={"project": "hankstank", "model": "game_outcome", "version": "v99", "channel": "staging"}, headers=W)
    assert r.status_code == 400 and r.json()["error"] == "refused"


def test_gcp_marks_a_live_job_that_is_still_enabled(env):
    state = json.loads((env / "status" / "state.json").read_text())
    for j in state["jobs"]:
        if j["name"] == "cfb-weekly-predict":
            j["mode"] = "live"
    state["taken_at"] = broker._now()
    (env / "status" / "state.json").write_text(json.dumps(state))
    d = make(env, FakeScheduler({"cfb-weekly-predict": "ENABLED"})).get("/app-api/gcp", headers=H).json()
    row = {j["name"]: j for j in d["jobs"]}["cfb-weekly-predict"]
    assert row["conflict"] is True and "pause cfb-weekly-predict --confirm cfb-weekly-predict" in row["commands"][0]


def test_gcp_unavailable_is_reported(env):
    d = make(env, FakeScheduler(fail=True)).get("/app-api/gcp", headers=H).json()
    assert d["available"] is False and "no credentials" in d["error"]


def test_audit_tail_is_newest_first(env):
    (env / "audit.log").write_text("".join(json.dumps({"ts": f"t{i}", "action": "a", "actor": "x"}) + "\n" for i in range(3)))
    assert [e["ts"] for e in make(env).get("/app-api/audit", headers=H).json()["entries"]] == ["t2", "t1", "t0"]


def test_notebooks_are_listed_with_links_and_checkpoints_are_skipped(env):
    (env / "work" / "a.ipynb").write_text("{}")
    (env / "work" / ".ipynb_checkpoints").mkdir()
    (env / "work" / ".ipynb_checkpoints" / "a-checkpoint.ipynb").write_text("{}")
    d = make(env).get("/app-api/notebooks", headers=H).json()
    assert [n["path"] for n in d["notebooks"]] == ["a.ipynb"] and d["notebooks"][0]["link"] == "/lab/tree/a.ipynb"


def test_jobs_endpoint_survives_an_empty_lab(env):
    d = make(env).get("/app-api/jobs", headers=H).json()
    assert d == {"jobs": [], "experiments": []}


def test_views_refuse_a_cancel_of_a_finished_job(env):
    from mllab.jobs import LocalRunner
    r = LocalRunner(env)
    jd = env / "jobs" / "abcd1234-ffff"
    jd.mkdir(parents=True)
    (jd / "spec.json").write_text(json.dumps({"name": "x", "submitted_at": "2026-01-01T00:00:00+00:00"}))
    (jd / "status.json").write_text(json.dumps({"state": "finished"}))
    with pytest.raises(views.Refused, match="nothing to cancel"):
        views.cancel(r, "abcd1234-ffff", "me", env / "audit.log")


# ----------------------------------------------------------------------------------------------------------------------- password switched off
def test_the_password_is_on_unless_the_setting_is_exactly_1(env, monkeypatch):
    for val in (None, "", "0", "true", "yes", "1 "):
        monkeypatch.delenv("MLLAB_APP_NO_UNLOCK", raising=False)
        if val is not None:
            monkeypatch.setenv("MLLAB_APP_NO_UNLOCK", val)
        assert session.open_access() is False, repr(val)
        assert make(env).post("/app-api/pipeline/mode", json={"job": "nfl-weekly-predict", "mode": "dry"}, headers=W).status_code == 401


def test_with_the_password_off_a_change_needs_no_unlock_but_everything_else_still_applies(env, monkeypatch):
    monkeypatch.setenv("MLLAB_APP_NO_UNLOCK", "1")
    c = make(env, FakeScheduler({"nfl-weekly-predict": "ENABLED"}))
    assert c.get("/app-api/session", headers=H).json() == {"login": LOGIN, "unlocked": True, "open": True}
    body = {"job": "nfl-weekly-predict", "mode": "dry"}
    assert c.post("/app-api/pipeline/mode", json=body, headers=W).status_code == 200            # no cookie, no unlock
    assert c.post("/app-api/pipeline/mode", json=body, headers={"Tailscale-User-Login": "intruder@example.com", "X-Mllab-App": "1", "Origin": "http://testserver"}).status_code == 401
    assert c.post("/app-api/pipeline/mode", json=body, headers={**W, "Origin": "https://evil.example"}).status_code == 403   # another site still cannot act for you
    assert c.post("/app-api/pipeline/mode", json=body, headers={k: v for k, v in W.items() if k != "X-Mllab-App"}).status_code == 403
    # the gate that protects production does not depend on the password
    r = c.post("/app-api/pipeline/mode", json={**body, "mode": "live", "confirm": "nfl-weekly-predict"}, headers=W)
    assert r.status_code == 409 and "ENABLED" in " ".join(r.json()["reasons"])
    assert c.post("/app-api/pipeline/mode", json={**body, "mode": "live"}, headers=W).status_code == 400  # and the name still has to be typed


def test_with_the_password_off_logs_open_and_unlock_is_a_no_op(env, monkeypatch):
    monkeypatch.setenv("MLLAB_APP_NO_UNLOCK", "1")
    (env / "status" / "logs").mkdir(exist_ok=True)
    (env / "status" / "logs" / "nfl-weekly-predict.txt").write_text("hello")
    c = make(env)
    assert c.get("/app-api/pipeline/log/nfl-weekly-predict", headers=H).json()["text"] == "hello"
    r = c.post("/app-api/unlock", json={"password": "anything"}, headers=W)
    assert r.status_code == 200 and r.json()["open"] is True and "set-cookie" not in r.headers


# ------------------------------------------------------------------------------------------------------------------------------ rollback
from mllab.webapp import history  # noqa: E402


def audit_lines(env, *entries):
    with open(env / "audit.log", "a") as f:
        for e in entries:
            f.write(json.dumps({"ts": "2026-10-02T21:00:00+00:00", "actor": "pipeline-broker", **e}) + "\n")


def snapshot_mode(env, job, mode):
    st = json.loads((env / "status" / "state.json").read_text())
    for j in st["jobs"]:
        if j["name"] == job:
            j["mode"] = mode
    st["taken_at"] = broker._now()
    (env / "status" / "state.json").write_text(json.dumps(st))


def test_history_reads_structured_entries_and_the_older_sentence_form():
    entries = [{"action": "pipeline_set_mode", "job": "a", "detail": "a: off -> dry (takes effect at its next run)", "by": "me", "ts": "t1"},     # first entries: a sentence only
               {"action": "pipeline_set_mode", "job": "a", "before": "dry", "mode": "live", "by": "me", "ts": "t2"},
               {"action": "pipeline_set_mode", "job": "b", "before": "off", "mode": "live", "by": "me", "ts": "t3"},
               {"action": "pipeline_rejected", "job": "a", "ts": "t4"}, {"action": "pipeline_set_mode", "job": "a", "detail": "garbage", "ts": "t5"}]
    assert [(c["before"], c["mode"]) for c in history.mode_changes(entries, "a")] == [("off", "dry"), ("dry", "live")]
    assert history.previous_mode(entries, "a", "live") == {"to": "dry", "from": "live", "at": "t2", "by": "me"}
    assert history.previous_mode(entries, "a", "dry")["to"] == "off"      # looks back to the change that led to the CURRENT mode
    assert history.previous_mode(entries, "a", "off") is None            # nothing on record led to off
    assert history.previous_mode(entries, "zzz", "live") is None


def test_history_for_channels_reads_previous_and_the_old_was_key():
    e = [{"action": "promote", "project": "p", "model": "m", "channel": "prod", "version": "v10", "previous": None, "ts": "t1"},
         {"action": "promote", "project": "p", "model": "m", "channel": "prod", "version": "v11", "was": "v10", "ts": "t2"},
         {"action": "promote", "project": "p", "model": "m", "channel": "staging", "version": "v12", "previous": "v10", "ts": "t3"}]
    assert history.previous_version(e, "p", "m", "prod", "v11")["to"] == "v10"
    assert history.previous_version(e, "p", "m", "prod", "v10") is None          # its first-ever move: nothing earlier
    assert history.previous_version(e, "p", "m", "staging", "v12")["to"] == "v10"


def test_the_pipeline_state_says_what_a_job_can_roll_back_to(env):
    audit_lines(env, {"action": "pipeline_set_mode", "job": "nfl-weekly-predict", "before": "dry", "mode": "live", "by": LOGIN})
    snapshot_mode(env, "nfl-weekly-predict", "live")
    st = make(env, FakeScheduler({})).get("/app-api/pipeline", headers=H).json()
    by = {j["name"]: j for j in st["jobs"]}
    assert by["nfl-weekly-predict"]["rollback"]["to"] == "dry" and by["nfl-weekly-score"]["rollback"] is None


def test_rolling_a_live_job_back_to_dry_goes_through_and_touches_only_the_homelab(env):
    audit_lines(env, {"action": "pipeline_set_mode", "job": "nfl-weekly-predict", "before": "dry", "mode": "live", "by": LOGIN})
    snapshot_mode(env, "nfl-weekly-predict", "live")
    sched = FakeScheduler({"nfl-weekly-predict": "PAUSED"})
    c = make(env, sched)
    unlock(c)
    r = c.post("/app-api/pipeline/rollback", json={"job": "nfl-weekly-predict"}, headers=W)
    assert r.status_code == 200
    req = spooled(env)
    assert (req["action"], req["job"], req["mode"]) == ("set_mode", "nfl-weekly-predict", "dry")
    broker.validate(json.dumps(req).encode(), current_mode=lambda j: "live")          # the broker accepts exactly this
    assert sched.states == {"nfl-weekly-predict": "PAUSED"}                            # Cloud Scheduler was not touched


def test_nothing_to_roll_back_is_said_plainly(env):
    c = make(env)
    unlock(c)
    r = c.post("/app-api/pipeline/rollback", json={"job": "nfl-weekly-predict"}, headers=W)
    assert r.status_code == 409 and "nothing to roll back to" in " ".join(r.json()["reasons"])
    assert c.post("/app-api/pipeline/rollback", json={"job": "../x"}, headers=W).status_code == 404
    assert not list((env / "spool").glob("pipe-*"))


def test_a_rollback_needs_the_same_access_as_any_change(env):
    audit_lines(env, {"action": "pipeline_set_mode", "job": "nfl-weekly-predict", "before": "dry", "mode": "live", "by": LOGIN})
    snapshot_mode(env, "nfl-weekly-predict", "live")
    c = make(env)
    assert c.post("/app-api/pipeline/rollback", json={"job": "nfl-weekly-predict"}, headers=W).status_code == 401      # locked
    assert c.post("/app-api/pipeline/rollback", json={"job": "nfl-weekly-predict"}, headers={**W, "Origin": "https://evil.example"}).status_code in (401, 403)


def test_a_rollback_that_would_go_back_to_live_meets_the_same_gate(env):
    # the job was Live, someone set it to Dry; rolling that back means going Live again, so Cloud Scheduler must be paused and the name typed
    audit_lines(env, {"action": "pipeline_set_mode", "job": "nfl-weekly-predict", "before": "live", "mode": "dry", "by": LOGIN})
    snapshot_mode(env, "nfl-weekly-predict", "dry")
    c = make(env, FakeScheduler({"nfl-weekly-predict": "ENABLED"}))
    unlock(c)
    r = c.post("/app-api/pipeline/rollback", json={"job": "nfl-weekly-predict", "confirm": "nfl-weekly-predict"}, headers=W)
    assert r.status_code == 409 and "ENABLED" in " ".join(r.json()["reasons"])
    c2 = make(env, FakeScheduler({"nfl-weekly-predict": "PAUSED"}))
    unlock(c2)
    assert c2.post("/app-api/pipeline/rollback", json={"job": "nfl-weekly-predict"}, headers=W).status_code == 400            # name not typed
    assert c2.post("/app-api/pipeline/rollback", json={"job": "nfl-weekly-predict", "confirm": "nfl-weekly-predict"}, headers=W).status_code == 200
    assert spooled(env)["mode"] == "live"


def test_models_report_where_each_channel_can_roll_back_to_and_the_rollback_is_audited(env, registered):
    c = make(env)
    unlock(c)
    base = {"project": "hankstank", "model": "game_outcome"}
    m = c.get("/app-api/models", headers=H).json()["projects"][0]["models"][0]
    assert m["rollback"] == {}                                                       # prod was set directly in the registry: no move on record
    assert c.post("/app-api/models/promote", json={**base, "version": "v11", "channel": "prod", "confirm": "game_outcome@prod"}, headers=W).status_code == 200
    m = c.get("/app-api/models", headers=H).json()["projects"][0]["models"][0]
    assert m["rollback"]["prod"]["to"] == "v10"
    r = c.post("/app-api/models/rollback", json={**base, "channel": "prod"}, headers=W)
    assert r.status_code == 400 and "game_outcome@prod" in " ".join(r.json()["reasons"])                     # prod needs the phrase
    assert registered.channels("hankstank", "game_outcome")["prod"] == "v11"
    r = c.post("/app-api/models/rollback", json={**base, "channel": "prod", "confirm": "game_outcome@prod"}, headers=W)
    assert r.status_code == 200 and r.json()["now"] == "v10" and r.json()["rolled_back"] is True
    assert registered.channels("hankstank", "game_outcome")["prod"] == "v10"
    last = json.loads((env / "audit.log").read_text().splitlines()[-1])
    assert last["action"] == "promote" and last["previous"] == "v11" and last["version"] == "v10"
    r = c.post("/app-api/models/rollback", json={**base, "channel": "prod", "confirm": "game_outcome@prod"}, headers=W)    # roll back again: v11 was the earlier move
    assert r.status_code == 200 and r.json()["now"] == "v11"        # a rollback is itself a move on record, so rolling back again undoes it
    assert registered.channels("hankstank", "game_outcome")["prod"] == "v11"


def test_a_model_rollback_with_nothing_earlier_or_a_missing_version_is_refused(env, registered):
    c = make(env)
    unlock(c)
    base = {"project": "hankstank", "model": "game_outcome", "channel": "prod", "confirm": "game_outcome@prod"}
    r = c.post("/app-api/models/rollback", json=base, headers=W)
    assert r.status_code == 409 and "nothing to roll back to" in " ".join(r.json()["reasons"])
    audit_lines(env, {"action": "promote", "project": "hankstank", "model": "game_outcome", "channel": "prod", "version": "v10", "previous": "v999"})
    r = c.post("/app-api/models/rollback", json=base, headers=W)
    assert r.status_code == 409 and "no longer in the registry" in " ".join(r.json()["reasons"])


# ------------------------------------------------------------------------------------------------------------------------------- data map
def catalog_dir(tmp_path, tables=None):
    d = tmp_path / "catalog"
    d.mkdir(exist_ok=True)
    (d / "mlb.yaml").write_text(yaml_dump({"sport": "mlb", "tables": tables if tables is not None else [
        {"id": "mlb_ds.games", "stage": "raw", "label": "Games", "written_by": ["mlb-2026-daily"], "cadence": {"kind": "daily", "max_age_hours": 30}},
        {"id": "mlb_ds.features", "stage": "features", "label": "Features", "written_by": ["mlb-2026-daily"], "cadence": {"kind": "daily", "max_age_hours": 30}},
        {"id": "mlb_ds.shadow", "stage": "shadows", "optional": True, "cadence": {"kind": "weekly", "max_age_hours": 200}}]}))
    return d


def yaml_dump(x):
    import yaml
    return yaml.safe_dump(x)


class FakeBq:
    def __init__(self, tables=None, fail=False):
        self.fail, self.sql = fail, []
        now_ms = int(time.time() * 1000)
        self.tables = tables if tables is not None else {"games": now_ms - 2 * 3600_000, "features": now_ms - 3 * 86400_000}

    def query(self, sql, params=None):
        self.sql.append(sql)
        if self.fail:
            raise RuntimeError("403 Access Denied: bigquery.tables.list")
        return [{"table_id": k, "row_count": "10", "size_bytes": "99", "last_modified_time": str(v)} for k, v in self.tables.items()]


def data_client(env, tmp_path, bq=None, tables=None):
    cfg = routes.Config(root=env, registry=env / "registry", status_dir=env / "status", spool=env / "spool", results=env / "results", audit_path=env / "audit.log")
    app = FastAPI()
    app.include_router(routes.build(cfg, scheduler=FakeScheduler(), bq=bq or FakeBq(), catalog_dir=catalog_dir(tmp_path, tables)))
    return TestClient(app)


def test_the_data_endpoint_needs_the_allowed_login(env, tmp_path):
    assert data_client(env, tmp_path).get("/app-api/data", headers={}).status_code == 401


def test_the_data_map_judges_each_table_against_its_own_cadence(env, tmp_path):
    d = data_client(env, tmp_path).get("/app-api/data", headers=H).json()
    assert d["available"] is True
    by = {t["id"]: t for t in d["sports"]["mlb"]["tables"]}
    assert by["mlb_ds.games"]["status"] == "ok" and by["mlb_ds.features"]["status"] == "stale"          # 3 days old against a 30 h cadence
    assert by["mlb_ds.shadow"]["status"] == "idle" and "first run creates it" in by["mlb_ds.shadow"]["reasons"][0]   # optional, not created yet
    assert d["sports"]["mlb"]["status"] == "stale" and [t["stage"] for t in d["sports"]["mlb"]["tables"]] == ["raw", "features", "shadows"]


def test_an_unreadable_dataset_is_unknown_and_the_others_are_still_shown(env, tmp_path):
    class Partial(FakeBq):
        def query(self, sql, params=None):
            if "locked_ds" in sql:
                raise RuntimeError("403 Access Denied: bigquery.tables.list")
            return super().query(sql, params)

    d = data_client(env, tmp_path, bq=Partial(), tables=[
        {"id": "mlb_ds.games", "stage": "raw", "cadence": {"kind": "daily", "max_age_hours": 30}},
        {"id": "locked_ds.games", "stage": "raw", "cadence": {"kind": "daily", "max_age_hours": 30}}]).get("/app-api/data", headers=H).json()
    assert d["available"] is True and "locked_ds" in d["unreadable"] and "Access Denied" in d["unreadable"]["locked_ds"]
    by = {t["id"]: t for t in d["sports"]["mlb"]["tables"]}
    assert by["locked_ds.games"]["status"] == "unknown" and by["mlb_ds.games"]["status"] == "ok"        # never 'missing', and the readable one still answers


def test_a_broken_catalog_file_is_reported_on_the_page_not_a_crash(env, tmp_path):
    d = data_client(env, tmp_path, tables=[{"id": "nodot", "stage": "raw"}]).get("/app-api/data", headers=H).json()
    assert d["available"] is False and d["error"].startswith("catalog:") and "dataset.table" in d["error"]


def test_the_data_map_flags_a_live_job_whose_table_did_not_change(env, tmp_path):
    state = json.loads((env / "status" / "state.json").read_text())
    started = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    for j in state["jobs"]:
        if j["name"] == "mlb-2026-daily":
            j["mode"] = "live"
            j["last"] = {"job": j["name"], "mode": "live", "ok": True, "started": started, "finished": started, "seconds": 90}
    state["taken_at"] = broker._now()
    (env / "status" / "state.json").write_text(json.dumps(state))
    fresh_but_old = {"games": int(time.time() * 1000) - 5 * 3600_000, "features": int(time.time() * 1000) - 10 * 60_000}     # games last changed 5 h ago, before the run
    by = {t["id"]: t for t in data_client(env, tmp_path, bq=FakeBq(fresh_but_old)).get("/app-api/data", headers=H).json()["sports"]["mlb"]["tables"]}
    assert by["mlb_ds.games"]["status"] == "unchanged" and "finished OK" in by["mlb_ds.games"]["reasons"][0]
    assert by["mlb_ds.features"]["status"] == "ok"                          # this one the run did change


def test_the_data_map_is_cached_so_the_page_polling_does_not_query_bigquery_each_time(env, tmp_path):
    bq = FakeBq()
    c = data_client(env, tmp_path, bq=bq)
    c.get("/app-api/data", headers=H); c.get("/app-api/data", headers=H)
    assert len(bq.sql) == 1
    c.get("/app-api/data?fresh=true", headers=H)
    assert len(bq.sql) == 2
