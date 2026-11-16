"""Collectors, dashboard and self-healing, with fake runners: nothing touches BigQuery, gcloud or the site."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from mllab import dashboard, remedy
from mllab import health as H
from mllab import health_collect as C
from mllab.cli import main

NOW = datetime(2026, 9, 29, 19, 0, tzinfo=timezone.utc)  # 15:00 ET


class FakeBq:
    def __init__(self, fail_for=()):
        self.fail_for, self.sqls = fail_for, []

    def query(self, sql, params=None):
        self.sqls.append(sql)
        if any(f in sql for f in self.fail_for):
            raise C.Unavailable("permission denied")
        if "FROM `hankstank.mlb_2026_season.games`" in sql and "today_games" in sql:
            return [{"today_games": 15, "nearby": 90}]
        if "game_v10_features" in sql:
            return [{"newest": (NOW - timedelta(hours=1)).isoformat()}]
        if "WITH pred" in sql:
            return []
        return [{"rows_today": 15, "games_today": 15, "last_at": NOW.isoformat(), "n_recent": 200, "mean_p": 0.53, "std_p": 0.09,
                 "bad_rows": 0, "high_share": 0.1}]


@pytest.fixture(autouse=True)
def _healthy_lab(monkeypatch):
    """The lab check inspects the REAL host (systemd units, disk). On a laptop there is no systemd, so it reports nothing; on a CI runner it finds the real mllab units
    inactive and the snapshot is no longer 'healthy'. Tests inject a lab the way they inject BigQuery; a test that wants a specific lab passes lab_fn itself."""
    real = C.build_snapshot

    def with_lab(*a, **kw):
        kw.setdefault("lab_fn", lambda: {"disk_used_pct": 20.0, "services_ok": True})
        return real(*a, **kw)
    monkeypatch.setattr(C, "build_snapshot", with_lab)


def reg(sport):
    return [{"key": "v10", "role": "production"}, {"key": "logit3", "role": "shadow"}]


def test_snapshot_healthy_has_no_findings():
    snap = C.build_snapshot(NOW, FakeBq(), registry_fn=reg)
    assert [f for f in H.evaluate(snap) if f.severity != "info"] == []
    assert {m.model for m in snap.sports["mlb"].models} == {"v10", "logit3"}


def test_unreadable_source_is_reported_not_raised_and_not_green():
    snap = C.build_snapshot(NOW, FakeBq(fail_for=("game_predictions",)), registry_fn=reg)
    assert snap.sources["bigquery:mlb"].startswith("unavailable")
    assert snap.sports["mlb"].models == []
    assert any(f.code == "source_unavailable" for f in H.evaluate(snap))


def test_models_not_on_the_site_are_not_monitored():
    snap = C.build_snapshot(NOW, FakeBq(), registry_fn=lambda s: [{"key": "v10", "role": "production"}])
    assert [m.model for m in snap.sports["mlb"].models] == ["v10"]


def test_queries_are_parameterised_and_use_fixed_tables():
    bq = FakeBq()
    C.build_snapshot(NOW, bq, registry_fn=reg)
    assert all("@" in q for q in bq.sqls if "WHERE" in q)


def test_work_account_refused():
    with pytest.raises(C.Unavailable):
        C.BqCliRunner(account="someone@blocked.example")


def test_dashboard_escapes_and_labels_status(tmp_path):
    snap = C.build_snapshot(NOW, FakeBq(), registry_fn=reg)
    snap.sports["mlb"].banner = "<script>alert(1)</script>"
    snap.sports["mlb"].models[0].bad_rows = 3
    d = dashboard.build(snap, tmp_path)
    html = (tmp_path / "index.html").read_text()
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "Down" in html or "Critical" in html
    assert json.loads((tmp_path / "health.json").read_text())["findings"] == d["findings"]
    assert (tmp_path / "history.jsonl").read_text().count("\n") == 1


def test_dashboard_all_clear(tmp_path):
    dashboard.build(C.build_snapshot(NOW, FakeBq(), registry_fn=reg), tmp_path)
    assert "Nothing needs attention" in (tmp_path / "index.html").read_text()


# --- remedy ---

class Ex:
    def __init__(self):
        self.calls = []
        self.run_job = lambda s, m: self.calls.append(("job", s, m)) or "ran"
        self.hide = lambda s, m, b: self.calls.append(("hide", s, m, b)) or "hid"
        self.prune = lambda d: self.calls.append(("prune", d)) or "pruned"


def gap():
    return H.Finding("coverage_gap", "critical", "gap", "mlb", remedy=H.Remedy("trigger_function", {"sport": "mlb", "mode": "predict_today"}))


ON = {"coverage_gap": {"enabled": True, "max_per_day": 2, "cooldown_minutes": 60}}


def test_default_policy_does_nothing(tmp_path):
    ex = Ex()
    assert remedy.run([gap()], remedy.Policy(), ex, now=NOW, path=tmp_path / "l") == []
    assert ex.calls == []


def test_dry_run_logs_but_does_not_act(tmp_path):
    ex = Ex()
    out = remedy.run([gap()], remedy.Policy(dry_run=True, rules=ON), ex, now=NOW, path=tmp_path / "l")
    assert ex.calls == [] and out[0]["outcome"].startswith("dry-run")


def test_live_acts_then_cooldown_then_daily_limit(tmp_path):
    ex, p, pol = Ex(), tmp_path / "l", remedy.Policy(dry_run=False, rules=ON)
    remedy.run([gap()], pol, ex, now=NOW, path=p)
    assert ex.calls == [("job", "mlb", "predict_today")]
    assert "cooldown" in remedy.run([gap()], pol, ex, now=NOW + timedelta(minutes=10), path=p)[0]["outcome"]
    remedy.run([gap()], pol, ex, now=NOW + timedelta(minutes=90), path=p)
    assert "daily limit" in remedy.run([gap()], pol, ex, now=NOW + timedelta(minutes=200), path=p)[0]["outcome"]
    assert len(ex.calls) == 2


def test_failed_action_counts_and_is_logged(tmp_path):
    ex = Ex()
    ex.run_job = lambda s, m: (_ for _ in ()).throw(RuntimeError("boom"))
    p = tmp_path / "l"
    out = remedy.run([gap()], remedy.Policy(dry_run=False, rules=ON), ex, now=NOW, path=p)
    assert out[0]["outcome"].startswith("failed")
    assert "cooldown" in remedy.run([gap()], remedy.Policy(dry_run=False, rules=ON), ex, now=NOW + timedelta(minutes=5), path=p)[0]["outcome"]


def test_hide_adds_banner_and_alert_never_executes(tmp_path):
    ex = Ex()
    f = H.Finding("bad_output", "critical", "m", "mlb", "v10", H.Remedy("control_hide", {"sport": "mlb", "model": "v10"}))
    a = H.Finding("probability_shift", "warn", "m", "mlb", "v10", H.Remedy("alert"))
    pol = remedy.Policy(dry_run=False, rules={"bad_output": {"enabled": True}, "probability_shift": {"enabled": True}})
    remedy.run([f, a], pol, ex, now=NOW, path=tmp_path / "l")
    assert ex.calls[0][0] == "hide" and "temporarily unavailable" in ex.calls[0][3] and len(ex.calls) == 1


def test_bad_policy_file_fails_safe(tmp_path):
    p = tmp_path / "p.json"
    p.write_text("{not json")
    assert remedy.load_policy(p).dry_run and remedy.load_policy(p).rules == {}
    assert remedy.load_policy(tmp_path / "missing").dry_run
    p.write_text('{"dry_run": false, "rules": {"coverage_gap": {"enabled": true}}}')
    assert remedy.load_policy(p).dry_run is False


def test_run_job_needs_exactly_one_matching_job():
    class S:
        def list(self):
            return [{"name": "mlb-a", "mode": "predict_today", "body_ok": True}, {"name": "mlb-b", "mode": "predict_today", "body_ok": True}]
    with pytest.raises(remedy.RemedyError):
        remedy.make_executors(None, S()).run_job("mlb", "predict_today")


def test_cli_health_exit_code_and_build(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("mllab.cli._health_snapshot", lambda sports: C.build_snapshot(NOW, FakeBq(), registry_fn=reg))
    assert main(["health", "--build", "--out", str(tmp_path / "h")]) == 0
    assert (tmp_path / "h" / "index.html").exists()
    bad = FakeBq()
    orig = bad.query
    bad.query = lambda sql, params=None: [{**orig(sql, params)[0], "bad_rows": 2}] if "n_recent" in sql else orig(sql, params)
    monkeypatch.setattr("mllab.cli._health_snapshot", lambda sports: C.build_snapshot(NOW, bad, registry_fn=reg))
    assert main(["health"]) == 5


# --- pin mismatch ---

URI = "gs://hanks_tank_data/models/vertex/x/model.pkl"


def _pin_snap(control, mismatches):
    s = C.build_snapshot(NOW, FakeBq(), registry_fn=reg)
    s.sports["mlb"].models[0].control = control
    s.sports["mlb"].pin_mismatches = mismatches
    return s


def _codes(s):
    return [f.code for f in H.evaluate(s)]


def test_pin_mismatch_is_critical_named_and_has_no_remedy():
    s = _pin_snap({"lifecycle": "live", "artifact_uri": URI, "artifact_sha256": "a" * 64}, [{"model": "v10", "uri": URI, "at": NOW - timedelta(hours=2)}])
    f = [x for x in H.evaluate(s) if x.code == "pin_mismatch"]
    assert len(f) == 1 and f[0].severity == "critical" and f[0].remedy is None and URI in f[0].message and f[0].model == "v10"


def test_pin_mismatch_clears_when_the_pin_changes_or_is_removed_or_old():
    log = [{"model": "v10", "uri": URI, "at": NOW - timedelta(hours=2)}]
    assert "pin_mismatch" not in _codes(_pin_snap({"lifecycle": "live", "artifact_uri": "gs://b/other", "artifact_sha256": "a" * 64}, log))
    assert "pin_mismatch" not in _codes(_pin_snap({}, log))
    assert "pin_mismatch" not in _codes(_pin_snap({"artifact_uri": URI}, [{"model": "v10", "uri": URI, "at": NOW - timedelta(hours=30)}]))


def test_pin_mismatch_never_triggers_a_remedy(tmp_path):
    s = _pin_snap({"artifact_uri": URI}, [{"model": "v10", "uri": URI, "at": NOW}])
    pol = remedy.Policy(dry_run=False, rules={"pin_mismatch": {"enabled": True}})
    ex = Ex()
    assert remedy.run(H.evaluate(s), pol, ex, now=NOW, path=tmp_path / "l") == [] and ex.calls == []


class Logs:
    def __init__(self, entries=None, fail=False):
        self.entries, self.fail = entries or [], fail

    def read(self, flt, hours=24, limit=20):
        if self.fail:
            raise C.Unavailable("permission denied")
        return self.entries


def test_collector_parses_the_pipeline_error_line():
    line = f"pinned model {URI} sha256 mismatch: expected aaa, got bbb; refusing to load (no fallback after a hash mismatch)"
    got = C.collect_pin_mismatches(Logs([{"textPayload": line, "timestamp": "2026-09-29T18:00:00Z"}, {"textPayload": "unrelated"}]), "mlb")
    assert got == [{"model": "v10", "uri": URI, "at": datetime(2026, 9, 29, 18, 0, tzinfo=timezone.utc)}]
    assert C.collect_pin_mismatches(Logs([{"textPayload": line}]), "nfl") == []


def test_unreadable_logs_degrade_to_unavailable_not_a_finding():
    snap = C.build_snapshot(NOW, FakeBq(), registry_fn=reg, log_reader=Logs(fail=True))
    assert snap.sources["logs"].startswith("unavailable") and "pin_mismatch" not in _codes(snap)


def test_gcloud_log_reader_refuses_the_work_account():
    with pytest.raises(C.Unavailable):
        C.GcloudLogReader(account="someone@blocked.example")


# --- server-side (REST) collectors ---

class FakeResp:
    def __init__(self, code, body):
        self.status_code, self._b, self.text = code, body, str(body)

    def json(self):
        return self._b


class FakeSession:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        return self.resp


def test_api_scheduler_decodes_jobs_like_the_cli_version():
    import base64
    body = base64.b64encode(json.dumps({"mode": "daily"}).encode()).decode()
    sess = FakeSession(FakeResp(200, {"jobs": [{"name": "projects/p/locations/l/jobs/mlb-2026-daily", "state": "ENABLED", "schedule": "0 4 * * *",
                                                "httpTarget": {"body": body}, "status": {}, "lastAttemptTime": "2026-09-29T08:00:00Z"}]}))
    jobs = C.ApiScheduler(api=C.ApiSession(sess)).list()
    assert jobs[0]["name"] == "mlb-2026-daily" and jobs[0]["mode"] == "daily" and jobs[0]["body_ok"] and jobs[0]["last_attempt_ok"] is True
    assert sess.calls[0][0] == "GET" and "cloudscheduler.googleapis.com" in sess.calls[0][1]


def test_api_calls_are_read_only_scoped_and_errors_become_unavailable():
    with pytest.raises(C.Unavailable, match="403"):
        C.ApiScheduler(api=C.ApiSession(FakeSession(FakeResp(403, {"error": "denied"})))).list()


def test_api_log_reader_posts_a_filtered_query_and_feeds_the_pin_collector():
    line = f"pinned model {URI} sha256 mismatch: expected a, got b; refusing to load"
    sess = FakeSession(FakeResp(200, {"entries": [{"textPayload": line, "timestamp": "2026-09-29T18:00:00Z"}]}))
    got = C.collect_pin_mismatches(C.ApiLogReader(api=C.ApiSession(sess)), "mlb")
    assert got[0]["uri"] == URI
    method, url, kw = sess.calls[0]
    assert method == "POST" and "logging.googleapis.com" in url and "sha256 mismatch" in kw["json"]["filter"] and "timestamp>=" in kw["json"]["filter"]


def test_control_from_view_drops_nulls_and_keys_by_sport_and_target():
    class R:
        def query(self, sql, params=None):
            assert "model_control_current" in sql
            return [{"sport": "mlb", "target": "v10", "site_visible": "false", "lifecycle": None, "banner": None},
                    {"sport": "mlb", "target": "*", "banner": "hello"}]
    assert C.control_from_view(R()) == {("mlb", "v10"): {"site_visible": "false"}, ("mlb", "*"): {"banner": "hello"}}


def test_missing_credentials_degrade_instead_of_raising(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "google.auth", None)
    with pytest.raises(C.Unavailable):
        C.ApiSession().call("GET", "https://example.invalid/")
