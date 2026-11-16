import importlib.machinery
import importlib.util
import json
import os
import time
from pathlib import Path

HEALTH = Path(__file__).resolve().parents[1] / "host" / "mllab-health"


def load(monkeypatch, tmp_path, key_age_days=None, limit="90"):
    key = tmp_path / "key.json"
    if key_age_days is not None:
        key.write_text("{}")
        t = time.time() - key_age_days * 86400
        os.utime(key, (t, t))
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key))
    monkeypatch.setenv("MLLAB_MAX_KEY_AGE_DAYS", limit)
    monkeypatch.setenv("MLLAB_ROOT", str(tmp_path))
    loader = importlib.machinery.SourceFileLoader("mllab_health", str(HEALTH))
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("mllab_health", loader))
    loader.exec_module(mod)
    monkeypatch.setattr(mod, "status", lambda url, headers=None: 401 if url.endswith("/auth") else 200)  # services "healthy"
    # mod.subprocess IS the shared subprocess module: patch through monkeypatch so it is restored after the test
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0})())
    # a healthy lab also has a fresh pipeline snapshot and a control app that answers with a clean data map
    (tmp_path / "pipeline-status").mkdir(exist_ok=True)
    (tmp_path / "pipeline-status" / "state.json").write_text(json.dumps({"taken_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), "jobs": []}))
    monkeypatch.setattr(mod, "app_get", lambda path, timeout=30: {"available": True, "sports": {}})
    return mod


def test_old_key_is_reported(monkeypatch, tmp_path):
    probs = load(monkeypatch, tmp_path, key_age_days=120).problems()
    assert len(probs) == 1 and "120 days old" in probs[0] and "rotate-key" in probs[0]


def test_fresh_key_is_fine(monkeypatch, tmp_path):
    assert load(monkeypatch, tmp_path, key_age_days=5).problems() == []


def test_missing_key_file_is_not_a_false_alarm(monkeypatch, tmp_path):
    assert load(monkeypatch, tmp_path, key_age_days=None).problems() == []


def test_limit_is_configurable(monkeypatch, tmp_path):
    assert load(monkeypatch, tmp_path, key_age_days=40, limit="30").problems() != []


def test_auth_not_failing_closed_is_reported(monkeypatch, tmp_path):
    m = load(monkeypatch, tmp_path, key_age_days=1)
    monkeypatch.setattr(m, "status", lambda url, headers=None: 200)  # /auth answering 200 with no headers would mean auth is open
    assert any("auth is not failing closed" in p for p in m.problems())


def weekly_file(tmp_path, ok=True, errors=None, age_days=0):
    d = tmp_path / "audit"; d.mkdir(exist_ok=True)
    p = d / "weekly.json"; p.write_text(json.dumps({"ok": ok, "errors": errors or []}))
    t = time.time() - age_days * 86400
    os.utime(p, (t, t))


def test_failed_weekly_run_is_reported_with_its_reason(monkeypatch, tmp_path):
    weekly_file(tmp_path, ok=False, errors=["data check failed: games (freshness: newest game_date is 9 days old)"])
    probs = load(monkeypatch, tmp_path, key_age_days=1).problems()
    assert len(probs) == 1 and "weekly run failed" in probs[0] and "freshness" in probs[0]


def test_stale_weekly_run_means_the_timer_may_be_dead(monkeypatch, tmp_path):
    weekly_file(tmp_path, ok=True, age_days=15)
    probs = load(monkeypatch, tmp_path, key_age_days=1).problems()
    assert len(probs) == 1 and "15 days old" in probs[0] and "mllab-weekly.timer" in probs[0]


def test_healthy_or_never_run_weekly_is_not_an_incident(monkeypatch, tmp_path):
    assert load(monkeypatch, tmp_path, key_age_days=1).problems() == []  # never ran: the lab is simply not configured for it
    weekly_file(tmp_path, ok=True, age_days=2)
    assert load(monkeypatch, tmp_path, key_age_days=1).problems() == []


def test_unreadable_weekly_file_is_reported(monkeypatch, tmp_path):
    (tmp_path / "audit").mkdir(); (tmp_path / "audit" / "weekly.json").write_text("{not json")
    assert any("unreadable" in p for p in load(monkeypatch, tmp_path, key_age_days=1).problems())
