import json
from types import SimpleNamespace

import pytest
import yaml

from mllab import ask as A
from mllab.registry import Registry


class Sent:
    """Fake generate(): records every prompt so tests can inspect exactly what would have left the machine."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, client, model, prompt):
        self.prompts.append((model, prompt))
        return self.replies.pop(0), SimpleNamespace(prompt_token_count=10, candidates_token_count=5)


@pytest.fixture
def up(tmp_path):
    return tmp_path / "usage.jsonl"


# --- what may leave the machine -----------------------------------------------------------------------------------

@pytest.mark.parametrize("secret", [
    "AIzaFakeKeyForTheRedactionTest000000000",
    "GEMINI_API_KEY=abc123secretvalue",
    "MLLAB_SECRET: hunter2hunter2",
    "Authorization: Bearer ya29.a0AfH6SMBxyz",
    "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkq\n-----END PRIVATE KEY-----",
    "5dc224decec9f300ab3e3d1a18dcb2a81e8c9d3543fb4b7a80a3db592c5f5c62",
    "a" * 44,
])
def test_redact_removes_secrets(secret):
    out = A.redact(f"before {secret} after")
    assert secret not in out and "before" in out and "after" in out


def test_redact_leaves_normal_text_alone():
    s = "mllab backtest run specs/mlb_v10_param_sweep.yaml --wait  (n=1957, CI -0.3%..+3.3%)"
    assert A.redact(s) == s


def test_default_context_is_only_the_reference_sheet():
    ctx = A.build_context()
    assert "mllab backtest run SPEC.yaml" in ctx and "(schema only)" not in ctx and "## Registry " not in ctx


def test_table_context_is_schema_only_never_rows():
    schema = lambda t: {"columns": [{"name": "game_pk", "type": "INTEGER"}, {"name": "elo", "type": "FLOAT"}],
                        "num_rows": 3634, "partitioning": "game_date", "clustering": []}
    ctx = A.build_context(tables=["mlb_2026_season.game_v10_features"], schema_fn=schema)
    assert "game_pk INTEGER, elo FLOAT" in ctx and "rows=3634" in ctx


def test_job_log_tail_is_truncated_and_scrubbed():
    log = "\n".join(f"line {i}" for i in range(500)) + "\nGEMINI_API_KEY=supersecretkeyvalue\n"
    runner = SimpleNamespace(logs=lambda j: (log, len(log)), status=lambda j: {"state": "failed"})
    ctx = A.build_context(job="job1", runner=runner)
    assert "line 499" in ctx and "line 100" not in ctx and "supersecretkeyvalue" not in ctx


def test_registry_context_lists_versions_without_pickles(tmp_path):
    reg = Registry(tmp_path / "r")
    a = tmp_path / "m.pkl"; a.write_bytes(b"x")
    reg.register("p", "m", "v1", a, features=["a", "b"])
    ctx = A.build_context(registry=["p/m"], reg=reg)
    assert "v1" in ctx and "features=2" in ctx
    with pytest.raises(A.AskError, match="PROJECT/MODEL"):
        A.build_context(registry=["justone"], reg=reg)


def test_question_is_redacted_before_sending(up):
    gen = Sent("ok")
    A.ask("why does this fail? GEMINI_API_KEY=leakedvalue123", context="CTX", client=object(), generate=gen, usage_path=up)
    assert "leakedvalue123" not in gen.prompts[0][1] and "CTX" in gen.prompts[0][1]


# --- guardrails -----------------------------------------------------------------------------------------

def test_daily_cap_blocks_and_usage_logs_no_question_text(up, monkeypatch):
    monkeypatch.setenv("MLLAB_GEMINI_DAILY_CALLS", "2")
    gen = Sent("a", "b", "c")
    for _ in range(2):
        A.ask("how do I backtest?", context="C", client=object(), generate=gen, usage_path=up)
    with pytest.raises(A.AskError, match="daily cap of 2"):
        A.ask("one more", context="C", client=object(), generate=gen, usage_path=up)
    assert len(gen.prompts) == 2  # the third call never reached the model
    logged = up.read_text()
    assert "how do I backtest" not in logged and json.loads(logged.splitlines()[0])["q_sha"]  # fingerprint only


def test_model_default_and_override(up, monkeypatch):
    gen = Sent("a", "b")
    A.ask("q", context="C", client=object(), generate=gen, usage_path=up)
    monkeypatch.setenv("MLLAB_GEMINI_MODEL", "gemini-9-test")
    A.ask("q", context="C", client=object(), generate=gen, usage_path=up)
    assert [m for m, _ in gen.prompts] == ["gemini-2.5-flash", "gemini-9-test"]


def test_missing_key_gives_actionable_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(A.AskError, match="set-gemini-key.sh"):
        A._client()


def test_empty_question_rejected(up):
    with pytest.raises(A.AskError, match="empty"):
        A.ask("   ", context="C", client=object(), generate=Sent(), usage_path=up)


def test_model_not_found_lists_callable_models():
    class NotFound(Exception):
        code = 404

    class Models:
        def generate_content(self, **kw):
            raise NotFound("404 NOT_FOUND model gone")

        def list(self):
            return [SimpleNamespace(name="models/gemini-x-flash", supported_actions=["generateContent"]),
                    SimpleNamespace(name="models/embed", supported_actions=["embedContent"])]

    with pytest.raises(A.AskError, match="gemini-x-flash") as e:
        A._generate(SimpleNamespace(models=Models()), "old-model", "prompt")
    assert "embed" not in str(e.value)  # only models that can generate are suggested


def test_real_sdk_config_still_builds():
    """Catches SDK API drift (renamed fields) without any network call."""
    from google.genai import types
    cfg = types.GenerateContentConfig(system_instruction=A.SYSTEM, temperature=0.2, max_output_tokens=2048)
    assert cfg.max_output_tokens == 2048


# --- spec drafting ----------------------------------------------------------------------------------------

GOOD = """```yaml
name: quick_check
data: {file: feat.parquet}
columns:
  - {name: prod, column: prod_prob}
```"""
BAD = "```yaml\nname: quick_check\ndata: {}\ncolumns: [{name: prod, column: x}]\n```"


def test_valid_draft_is_written(up, tmp_path):
    out = tmp_path / "s.yaml"
    body, note = A.draft_spec("backtest", "compare prod to elo", out=out, client=object(), generate=Sent(GOOD), usage_path=up)
    assert yaml.safe_load(out.read_text())["name"] == "quick_check" and "written" in note


def test_invalid_draft_gets_one_retry_with_the_error(up, tmp_path):
    gen = Sent(BAD, GOOD)
    out = tmp_path / "s.yaml"
    A.draft_spec("backtest", "x", out=out, client=object(), generate=gen, usage_path=up)
    assert "failed validation" in gen.prompts[1][1] and "exactly one of sql_file or file" in gen.prompts[1][1]
    assert out.exists()


def test_two_invalid_drafts_write_nothing(up, tmp_path):
    out = tmp_path / "s.yaml"
    with pytest.raises(A.AskError, match="did not pass validation twice"):
        A.draft_spec("backtest", "x", out=out, client=object(), generate=Sent(BAD, "no yaml here"), usage_path=up)
    assert not out.exists()


def test_draft_never_overwrites_and_rejects_unknown_kind(up, tmp_path):
    out = tmp_path / "s.yaml"; out.write_text("keep me")
    gen = Sent(GOOD)
    with pytest.raises(A.AskError, match="already exists"):
        A.draft_spec("backtest", "x", out=out, client=object(), generate=gen, usage_path=up)
    assert out.read_text() == "keep me" and gen.prompts == []  # refused before spending a call
    with pytest.raises(A.AskError, match="kind must be"):
        A.draft_spec("deploy", "x", client=object(), generate=gen, usage_path=up)


def test_bq_and_train_kinds_use_their_own_validators(up, tmp_path):
    bq = "```yaml\ntable: mlb_2026_season.games\ncolumns:\n  - {name: game_pk, type: INTEGER, nullable: false}\n```"
    assert "validated" in A.draft_spec("bq", "x", client=object(), generate=Sent(bq), usage_path=up)[1]
    injected = "```yaml\ntable: 'd.t; drop table x'\n```"
    with pytest.raises(A.AskError):  # the SQL-injection guard in the bq validator applies to drafts too
        A.draft_spec("bq", "x", client=object(), generate=Sent(injected, injected), usage_path=up)
    train = "```yaml\nname: t1\ndata: {file: a.parquet}\nfeatures: [f0, f1]\n```"
    assert "validated" in A.draft_spec("train", "x", client=object(), generate=Sent(train), usage_path=up)[1]


def test_show_context_sends_nothing(capsys, monkeypatch):
    from mllab.cli import main
    monkeypatch.setattr(A, "ask", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call the model")))
    assert main(["ask", "--show-context", "anything"]) == 0
    out = capsys.readouterr()
    assert "mllab reference" in out.out and "Nothing was sent" in out.err
