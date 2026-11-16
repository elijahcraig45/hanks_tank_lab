import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mllab import audit
from mllab.app import app
from mllab.cli import main


def test_forward_auth(monkeypatch):
    monkeypatch.setenv("MLLAB_ALLOWED_LOGIN", "henry@example.com")
    monkeypatch.setenv("MLLAB_SECRET", "s3cret")
    c = TestClient(app)
    login = {"Tailscale-User-Login": "henry@example.com"}
    # browser route: login alone passes (Jupyter's token is the second factor)
    assert c.get("/auth", headers={**login, "X-Forwarded-Uri": "/lab"}).status_code == 200
    assert c.get("/auth", headers={"Tailscale-User-Login": "other@example.com", "X-Forwarded-Uri": "/lab"}).status_code == 401
    assert c.get("/auth").status_code == 401
    # programmatic route: needs login AND secret
    api = {**login, "X-Forwarded-Uri": "/api/deploy"}
    assert c.get("/auth", headers=api).status_code == 401
    assert c.get("/auth", headers={**api, "X-Mllab-Secret": "wrong"}).status_code == 401
    assert c.get("/auth", headers={**api, "X-Mllab-Secret": "s3cret"}).status_code == 200
    assert c.get("/healthz").status_code == 200


def test_auth_closed_when_env_missing(monkeypatch):
    monkeypatch.delenv("MLLAB_ALLOWED_LOGIN", raising=False)
    monkeypatch.delenv("MLLAB_SECRET", raising=False)
    assert TestClient(app).get("/auth", headers={"Tailscale-User-Login": "", "X-Mllab-Secret": ""}).status_code == 401


def test_audit_is_append_only(tmp_path):
    p = tmp_path / "a" / "deploys.log"
    audit.record("one", path=p, x=1)
    audit.record("two", path=p, x=2)
    assert [e["action"] for e in audit.read(p)] == ["one", "two"]


def test_cli_register_promote_audit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MLLAB_REGISTRY", str(tmp_path / "reg"))
    monkeypatch.setenv("MLLAB_AUDIT", str(tmp_path / "audit.log"))
    art = tmp_path / "m.pkl"
    art.write_bytes(b"x")
    assert main(["register", "--dry-run", "p", "m", "v1", str(art)]) == 0
    assert main(["register", "p", "m", "v1", str(art)]) == 0
    assert main(["register", "p", "m", "v1", str(art)]) == 1  # immutable
    assert main(["promote", "p", "m", "v1"]) == 0
    assert [e["action"] for e in audit.read(tmp_path / "audit.log")] == ["register", "promote"]


def test_cli_job_and_confirmation_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MLLAB_ROOT", str(tmp_path / "lab"))
    monkeypatch.setenv("MLLAB_REGISTRY", str(tmp_path / "reg"))
    monkeypatch.setenv("MLLAB_AUDIT", str(tmp_path / "audit.log"))
    spec = tmp_path / "job.yaml"
    (tmp_path / "s.py").write_text("print(1)")
    spec.write_text("name: t\nscript: s.py\ncode_dir: .\n")  # missing timeout_s
    assert main(["job", "submit", str(spec)]) == 1
    assert main(["job", "status", "../../etc"]) == 1
    assert main(["adopt", "--dry-run", "p", "m", "v1", "gs://b/o.pkl"]) == 0
    assert main(["adopt", "--dry-run", "p", "m", "v1", "not-a-uri"]) == 1


def test_cli_bq_check_exit_code_and_bad_spec(tmp_path, monkeypatch, capsys):
    from mllab import schema as S

    good = tmp_path / "s.yaml"; good.write_text("table: d.t\ncolumns: [x]\n")
    monkeypatch.setattr(S, "run_check", lambda spec, **kw: S.Report(spec.table, [S.Finding("error", "column", "missing column x")]))
    assert main(["bq", "check", str(good)]) == 4
    assert "FAILED" in capsys.readouterr().out
    monkeypatch.setattr(S, "run_check", lambda spec, **kw: S.Report(spec.table))
    assert main(["bq", "check", str(good)]) == 0
    bad = tmp_path / "bad.yaml"; bad.write_text("table: 'd.t; drop'\n")
    assert main(["bq", "check", str(bad)]) == 1  # SpecError is reported, not a traceback
