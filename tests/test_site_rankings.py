"""Choosing which ranking columns the site draws: the control fields, the app page behind them, and the CLI.

The columns are a contract with three other repos (hanks_tank_ml writes them, hanks_tank_backend serves the choice, hanks_tank draws it), so the key names are pinned here."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mllab import control as K
from mllab import cli
from mllab.webapp import routes, views
from mllab.webapp.views import Refused

from test_webapp import FakeScheduler, H, W, env, unlock  # noqa: F401  (env is a fixture)

NOW = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _audit_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("MLLAB_AUDIT", str(tmp_path / "audit-default.log"))


def control(events=None):
    """A control whose clock moves forward on every read, as real time does: events written in one test must not share a timestamp."""
    tick = iter(range(10_000))
    return K.Control(K.MemoryStore(events), "henry", now=lambda: NOW + timedelta(seconds=next(tick)))


# ------------------------------------------------------------------------------------------------------------------------------ the contract
def test_the_catalog_keys_are_the_contract_with_the_other_repos():
    cfb = K.RANKINGS["cfb"]
    assert [r["key"] for r in cfb["custom"]] == ["results", "season", "forecast", "resume"]
    assert [r["key"] for r in cfb["media"]] == ["ap", "coaches"]
    assert K.MAX_CUSTOM_RANKINGS == 3
    assert not K.RANKINGS["nfl"]["custom"] and not K.RANKINGS["mlb"]["custom"]


# ------------------------------------------------------------------------------------------------------------------------------ validation
@pytest.mark.parametrize("field,value,want", [
    ("rankings_show", "season,results", "season,results"),
    ("rankings_show", " Season , RESULTS ", "season,results"),
    ("rankings_show", "none", "none"),
    ("rankings_show", "resume,season,results", "resume,season,results"),
    ("rankings_media", "ap", "ap"),
    ("rankings_media", "coaches,ap", "coaches,ap"),
    ("rankings_media", "none", "none"),
    ("rankings_show", K.UNSET, K.UNSET),
])
def test_valid_lists_are_canonicalised(field, value, want):
    assert K.validate("cfb", "*", field, value) == want


@pytest.mark.parametrize("field,value", [
    ("rankings_show", "season,results,forecast,resume"),  # four, the cap is three
    ("rankings_show", "season,season"),
    ("rankings_show", "bogus"),
    ("rankings_show", ""),
    ("rankings_show", ","),
    ("rankings_media", "espn"),
    ("rankings_media", "ap,ap"),
])
def test_invalid_lists_are_rejected(field, value):
    with pytest.raises(K.ControlError):
        K.validate("cfb", "*", field, value)


def test_a_sport_without_rankings_to_choose_refuses():
    with pytest.raises(K.ControlError, match="no custom rankings"):
        K.validate("nfl", "*", "rankings_show", "results")


def test_the_fields_are_sport_wide_only():
    with pytest.raises(K.ControlError, match="sport-wide"):
        K.validate("cfb", "v10", "rankings_show", "season")


def test_any_change_needs_the_typed_confirmation_and_the_state_reads_back():
    c = control()
    with pytest.raises(K.ConfirmationRequired):
        c.set("cfb", "*", "rankings_show", "season,results")
    c.set("cfb", "*", "rankings_show", "season,results", confirm="cfb/*")
    c.set("cfb", "*", "rankings_media", "ap", confirm="cfb/*")
    row = c.state("cfb")[("cfb", "*")]
    assert row["rankings_show"] == "season,results" and row["rankings_media"] == "ap"
    with pytest.raises(K.ConfirmationRequired):
        c.set("cfb", "*", "rankings_show", K.UNSET)  # going back to the default changes the site too
    c.set("cfb", "*", "rankings_show", K.UNSET, confirm="cfb/*")
    assert "rankings_show" not in c.state("cfb")[("cfb", "*")]


def test_the_view_sql_carries_both_columns():
    sql = (K.__file__ and __import__("pathlib").Path(K.__file__).parents[2] / "sql" / "control" / "02_current_view.sql").read_text()
    assert "AS rankings_show" in sql and "AS rankings_media" in sql and "AS rankings_primary" in sql


# ------------------------------------------------------------------------------------------------------------------------------ the app's read
def test_site_rankings_reports_catalog_current_and_whether_the_view_is_ready():
    state = {("cfb", "*"): {"rankings_show": "season,results", "rankings_media": "none"}}
    d = views.site_rankings(lambda: state, lambda: ["sport", "target", "rankings_show", "rankings_media", "rankings_primary"])
    assert d["available"] and d["view_ready"] is True
    assert d["current"]["cfb"] == {"show": ["season", "results"], "media": [], "primary": None}
    assert set(d["catalog"]) == {"cfb"} and d["catalog"]["cfb"]["max_custom"] == 3
    old = views.site_rankings(lambda: {}, lambda: ["sport", "target", "banner"])
    assert old["view_ready"] is False and old["current"]["cfb"] == {"show": None, "media": None, "primary": None}
    # a view with the two older columns but not the new one is not ready: a table-order change would be ignored
    assert views.site_rankings(lambda: {}, lambda: ["sport", "target", "rankings_show", "rankings_media"])["view_ready"] is False


def test_site_rankings_survives_an_unreadable_control_view():
    def boom():
        raise RuntimeError("403 access denied")

    d = views.site_rankings(boom, lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert d["available"] is False and "403" in d["error"] and d["view_ready"] is None and "cfb" in d["catalog"]


# ------------------------------------------------------------------------------------------------------------------------------ the app's write
def test_a_change_is_written_validated_audited_and_undoable(tmp_path):
    store = K.MemoryStore()
    r = views.set_site_rankings(store, "cfb", ["results", "season"], ["ap"], "henry@x", "cfb/*", tmp_path / "audit.log")
    assert r["show"] == ["results", "season"] and r["media"] == ["ap"]
    st = K.current_state(store.all_events(), datetime.now(timezone.utc))[("cfb", "*")]
    assert st["rankings_show"] == "results,season" and st["rankings_media"] == "ap"
    assert all(e.actor == "app:henry@x" for e in store.all_events())
    assert (tmp_path / "audit.log").read_text().count("control_set") == 2
    c = K.Control(store, "henry")
    c.revert("cfb", "*", "rankings_show", confirm="cfb/*")
    assert "rankings_show" not in K.current_state(store.all_events(), datetime.now(timezone.utc))[("cfb", "*")]


def test_reset_and_none_are_distinct():
    store = K.MemoryStore()
    views.set_site_rankings(store, "cfb", [], [], "h", "cfb/*")
    st = K.current_state(store.all_events(), datetime.now(timezone.utc))[("cfb", "*")]
    assert st["rankings_show"] == "none" and st["rankings_media"] == "none"
    views.set_site_rankings(store, "cfb", None, None, "h", "cfb/*")
    assert ("cfb", "*") not in K.current_state(store.all_events(), datetime.now(timezone.utc))


def test_a_field_left_out_is_left_alone():
    store = K.MemoryStore()
    views.set_site_rankings(store, "cfb", ["results"], ["ap"], "h", "cfb/*")
    views.set_site_rankings(store, "cfb", ["season"], ..., "h", "cfb/*")
    st = K.current_state(store.all_events(), datetime.now(timezone.utc))[("cfb", "*")]
    assert st["rankings_show"] == "season" and st["rankings_media"] == "ap"


@pytest.mark.parametrize("sport,show,media,confirm,why", [
    ("cfb", ["results"], ..., None, "confirm with 'cfb/*'"),
    ("cfb", ["results"], ..., "nfl/*", "confirm with 'cfb/*'"),
    ("cfb", ["a", "b", "c", "d"], ..., "cfb/*", "unknown custom"),
    ("cfb", ["results", "season", "forecast", "resume"], ..., "cfb/*", "at most 3"),
    ("cfb", ..., ..., "cfb/*", "nothing to change"),
    ("nfl", ["results"], ..., "nfl/*", "no ranking columns"),
    ("xyz", ["results"], ..., "xyz/*", "no ranking columns"),
])
def test_refusals_say_why_and_write_nothing(sport, show, media, confirm, why):
    store = K.MemoryStore()
    with pytest.raises(Refused) as e:
        views.set_site_rankings(store, sport, show, media, "h", confirm)
    assert why in "; ".join(e.value.reasons)
    assert store.all_events() == []


def test_a_bad_second_field_leaves_the_first_unwritten():
    store = K.MemoryStore()
    with pytest.raises(Refused):
        views.set_site_rankings(store, "cfb", ["results"], ["espn"], "h", "cfb/*")
    assert store.all_events() == []


def test_without_the_write_grant_the_refusal_names_the_command():
    class Denied(K.MemoryStore):
        def append(self, ev):
            raise K.ControlError("bq failed: Access Denied: User does not have bigquery.tables.updateData")

    with pytest.raises(Refused) as e:
        views.set_site_rankings(Denied(), "cfb", ["results", "season"], None, "h", "cfb/*")
    text = "; ".join(e.value.reasons)
    assert "Access Denied" in text and "mllab control rankings cfb --confirm cfb/* --show results,season --media-default" in text


# ------------------------------------------------------------------------------------------------------------------------------ the routes
class FakeBq:
    def __init__(self, rows=None, columns=("sport", "target", "rankings_show", "rankings_media", "rankings_primary")):
        self.rows, self.columns = rows or [], columns

    def query(self, sql, params=None):
        if "INFORMATION_SCHEMA" in sql:
            return [{"column_name": c} for c in self.columns]
        return self.rows


def client(env, store=None, bq=None):
    cfg = routes.Config(root=env, registry=env / "registry", status_dir=env / "status", spool=env / "spool", results=env / "results", audit_path=env / "audit.log")
    app = FastAPI()
    app.include_router(routes.build(cfg, scheduler=FakeScheduler(), bq=bq or FakeBq(), control_store=store or K.MemoryStore()))
    return TestClient(app)


def test_the_page_reads_without_unlocking(env):
    c = client(env, bq=FakeBq([{"sport": "cfb", "target": "*", "rankings_show": "season", "rankings_media": None}]))
    r = c.get("/app-api/site", headers=H)
    assert r.status_code == 200
    assert r.json()["current"]["cfb"] == {"show": ["season"], "media": None, "primary": None}
    assert c.get("/app-api/site").status_code in (401, 403)


def test_changing_needs_the_unlock_and_the_app_header(env):
    store = K.MemoryStore()
    c = client(env, store)
    body = {"sport": "cfb", "show": ["results"], "confirm": "cfb/*"}
    assert c.post("/app-api/site/rankings", json=body, headers=H).status_code in (401, 403)  # not the page: no X-Mllab-App / origin
    unlock(c)
    r = c.post("/app-api/site/rankings", json=body, headers=W)
    assert r.status_code == 200, r.text
    assert [e.value for e in store.all_events()] == ["results"]


def test_a_malformed_list_is_a_400_not_a_write(env):
    store = K.MemoryStore()
    c = client(env, store)
    unlock(c)
    r = c.post("/app-api/site/rankings", json={"sport": "cfb", "show": "results", "confirm": "cfb/*"}, headers=W)
    assert r.status_code == 400 and store.all_events() == []


def test_a_missing_confirmation_comes_back_as_a_refusal_the_page_can_show(env):
    c = client(env)
    unlock(c)
    r = c.post("/app-api/site/rankings", json={"sport": "cfb", "show": ["results"]}, headers=W)
    assert r.status_code == 409 and "confirm with 'cfb/*'" in " ".join(r.json()["reasons"])


def test_the_route_flips_the_table_order_and_rejects_a_non_string(env):
    store = K.MemoryStore()
    c = client(env, store)
    unlock(c)
    r = c.post("/app-api/site/rankings", json={"sport": "cfb", "primary": "rating", "confirm": "cfb/*"}, headers=W)
    assert r.status_code == 200, r.text
    assert [(e.field, e.value) for e in store.all_events()] == [("rankings_primary", "rating")]
    r = c.post("/app-api/site/rankings", json={"sport": "cfb", "primary": ["rating"], "confirm": "cfb/*"}, headers=W)
    assert r.status_code == 400 and len(store.all_events()) == 1
    r = c.post("/app-api/site/rankings", json={"sport": "cfb", "primary": None, "confirm": "cfb/*"}, headers=W)
    assert r.status_code == 200 and store.all_events()[-1].value == K.UNSET


def test_the_page_reports_the_current_table_order(env):
    c = client(env, bq=FakeBq([{"sport": "cfb", "target": "*", "rankings_primary": "rating"}]))
    assert c.get("/app-api/site", headers=H).json()["current"]["cfb"]["primary"] == "rating"


# ------------------------------------------------------------------------------------------------------------------------------ the CLI
def run_cli(monkeypatch, capsys, *argv, store=None):
    store = store or K.MemoryStore()
    monkeypatch.setattr(cli, "_make_control", lambda: K.Control(store, "henry"))
    code = cli.main(["control", *argv])
    return code, capsys.readouterr().out, store


def test_cli_sets_both_and_resets(monkeypatch, capsys):
    code, out, store = run_cli(monkeypatch, capsys, "rankings", "cfb", "--show", "season,results", "--media", "ap", "--confirm", "cfb/*")
    assert code == 0 and "rankings_show = season,results" in out and "rankings_media = ap" in out
    code, _, store = run_cli(monkeypatch, capsys, "rankings", "cfb", "--show-default", "--media-default", "--confirm", "cfb/*", store=store)
    assert ("cfb", "*") not in K.current_state(store.all_events(), datetime.now(timezone.utc))


def test_cli_validates_everything_before_writing_anything(monkeypatch, capsys):
    store = K.MemoryStore()
    monkeypatch.setattr(cli, "_make_control", lambda: K.Control(store, "henry"))
    assert cli.main(["control", "rankings", "cfb", "--show", "results", "--media", "espn", "--confirm", "cfb/*"]) != 0
    assert store.all_events() == []
    assert cli.main(["control", "rankings", "cfb", "--confirm", "cfb/*"]) != 0  # nothing asked for


# ------------------------------------------------------------------------------------------------------------------------------ the server store
class FakeClient:
    def __init__(self, fail=None):
        self.sql, self.fail = [], fail

    def query(self, sql, job_config=None):
        self.sql.append((sql, job_config))
        if self.fail:
            raise RuntimeError(self.fail)

        class Job:
            def result(self_inner):
                return []
        return Job()


def test_the_server_store_appends_with_parameters_and_reports_a_denied_write():
    fc = FakeClient()
    store = K.BqClientStore(client=fc)
    ev = K.Event("id1", NOW, NOW, "cfb", "*", "rankings_show", "season", "app:h", "n")
    store.append(ev)
    sql, cfg = fc.sql[0]
    assert "INSERT INTO `hankstank.control.model_control_events`" in sql and "@value" in sql and "season" not in sql  # the value is a parameter, never in the text
    assert {p.name for p in cfg.query_parameters} >= {"event_id", "event_ts", "sport", "field", "value", "actor"}
    with pytest.raises(K.ControlError, match="bq failed: 403 Access Denied"):
        K.BqClientStore(client=FakeClient("403 Access Denied")).append(ev)


# ------------------------------------------------------------------------------------------------------------------------------ table order (rankings_primary)
@pytest.mark.parametrize("raw,canon", [("resume", "resume"), (" Rating ", "rating"), ("RESUME", "resume"), (K.UNSET, K.UNSET)])
def test_primary_values_are_canonicalised(raw, canon):
    assert K.validate("cfb", "*", "rankings_primary", raw) == canon


@pytest.mark.parametrize("raw", ["", "vibes", "resume,rating", "none"])
def test_primary_rejects_anything_else(raw):
    with pytest.raises(K.ControlError, match="rankings_primary must be one of resume, rating"):
        K.validate("cfb", "*", "rankings_primary", raw)


def test_a_sport_without_a_resume_board_refuses_a_table_order():
    with pytest.raises(K.ControlError, match="no season resume"):
        K.validate("nfl", "*", "rankings_primary", "resume")


def test_primary_is_sport_wide_and_needs_the_typed_phrase():
    with pytest.raises(K.ControlError):
        K.validate("cfb", "v10", "rankings_primary", "rating")
    c = K.Control(K.MemoryStore(), "henry")
    with pytest.raises(K.ConfirmationRequired):
        c.set("cfb", "*", "rankings_primary", "rating")
    c.set("cfb", "*", "rankings_primary", "rating", confirm="cfb/*")
    assert c.state("cfb")[("cfb", "*")]["rankings_primary"] == "rating"
    with pytest.raises(K.ConfirmationRequired):
        c.set("cfb", "*", "rankings_primary", K.UNSET)           # going back to the default also changes the site
    c.set("cfb", "*", "rankings_primary", K.UNSET, confirm="cfb/*")
    assert ("cfb", "*") not in c.state("cfb") or "rankings_primary" not in c.state("cfb")[("cfb", "*")]


def test_the_catalog_lists_the_orders_and_the_default():
    d = views.site_rankings(lambda: {}, None)
    assert [p["key"] for p in d["catalog"]["cfb"]["primary"]] == ["resume", "rating"]
    assert d["catalog"]["cfb"]["primary_default"] == "resume"


def test_the_app_can_flip_the_order_leave_columns_alone_and_go_back(tmp_path):
    store = K.MemoryStore()
    views.set_site_rankings(store, "cfb", ["results"], ["ap"], "h", "cfb/*")
    r = views.set_site_rankings(store, "cfb", ..., ..., "h", "cfb/*", tmp_path / "audit.log", primary="rating")
    assert r["primary"] == "rating" and r["show"] is None
    st = K.current_state(store.all_events(), datetime.now(timezone.utc))[("cfb", "*")]
    assert st["rankings_primary"] == "rating" and st["rankings_show"] == "results" and st["rankings_media"] == "ap"
    assert "--primary rating" in r["commands"][0]
    views.set_site_rankings(store, "cfb", ..., ..., "h", "cfb/*", primary=None)
    st = K.current_state(store.all_events(), datetime.now(timezone.utc))[("cfb", "*")]
    assert "rankings_primary" not in st and st["rankings_show"] == "results"


def test_a_bad_order_is_refused_and_writes_nothing():
    store = K.MemoryStore()
    with pytest.raises(Refused) as e:
        views.set_site_rankings(store, "cfb", ["results"], ..., "h", "cfb/*", primary="vibes")
    assert "rankings_primary must be one of" in "; ".join(e.value.reasons)
    assert store.all_events() == []                                  # the valid --show was not applied either
    with pytest.raises(Refused) as e2:
        views.set_site_rankings(store, "nfl", ..., ..., "h", "nfl/*", primary="resume")
    assert "no ranking columns" in "; ".join(e2.value.reasons)


def test_the_cli_command_for_a_primary_change_is_printed():
    assert views._cli_commands("cfb", ..., ..., "rating") == ["mllab control rankings cfb --confirm cfb/* --primary rating"]
    assert views._cli_commands("cfb", ..., ..., None) == ["mllab control rankings cfb --confirm cfb/* --primary-default"]


def test_cli_sets_and_resets_the_table_order(monkeypatch, capsys):
    code, out, store = run_cli(monkeypatch, capsys, "rankings", "cfb", "--primary", "rating", "--confirm", "cfb/*")
    assert code == 0 and "rankings_primary = rating" in out
    assert K.current_state(store.all_events(), datetime.now(timezone.utc))[("cfb", "*")]["rankings_primary"] == "rating"
    code, _, store = run_cli(monkeypatch, capsys, "rankings", "cfb", "--primary-default", "--confirm", "cfb/*", store=store)
    assert code == 0 and ("cfb", "*") not in K.current_state(store.all_events(), datetime.now(timezone.utc))


def test_cli_refuses_a_bad_order_and_contradictory_flags_before_writing(monkeypatch, capsys):
    store = K.MemoryStore()
    monkeypatch.setattr(cli, "_make_control", lambda: K.Control(store, "henry"))
    assert cli.main(["control", "rankings", "cfb", "--primary", "vibes", "--confirm", "cfb/*"]) != 0
    assert cli.main(["control", "rankings", "cfb", "--primary", "rating", "--primary-default", "--confirm", "cfb/*"]) != 0
    assert cli.main(["control", "rankings", "cfb", "--show", "results", "--primary", "vibes", "--confirm", "cfb/*"]) != 0
    assert store.all_events() == []
