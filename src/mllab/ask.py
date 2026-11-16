"""Ask Gemini how to do things with mllab.

What is sent, and only this: the packaged reference sheet, your question, and context you opt into (`with_tables` schemas,
`with_registry` version names + summary metrics, `with_job` a scrubbed log tail). Never data rows, env files or keys, and
every outgoing byte passes through `redact`. `mllab ask --show-context ...` prints exactly what would be sent, without calling.

Output is advice. Nothing Gemini returns is executed; a drafted spec is validated by the same code that would run it and is
only written if it passes and the target file does not already exist.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

DEFAULT_MODEL = "gemini-2.5-flash"  # Google renames models faster than SDKs move: override with MLLAB_GEMINI_MODEL
LOG_TAIL_LINES = 60
MAX_CONTEXT_CHARS = 40_000

SYSTEM = (
    "You are the built-in assistant for mllab, a private ML lab (registry, BigQuery guards, jobs, backtests, training, deploys). "
    "Answer concisely and practically. Use ONLY the commands, flags and spec keys in the reference; if something is not there, "
    "say you are not sure instead of inventing it. When you give a command, give it whole. When statistics are involved, say "
    "what the sample size can and cannot show. You cannot run anything; the user will run what you suggest."
)

_SECRETS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[private key removed]"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), "[api key removed]"),
    (re.compile(r"(?i)\b(GEMINI_API_KEY|GOOGLE_API_KEY|MLLAB_SECRET|JUPYTER_TOKEN|MLLAB_KUMA_PUSH_TOKEN|password|secret|token)\b\s*[=:]\s*\S+"),
     r"\1=[removed]"),
    (re.compile(r"(?i)(authorization:\s*bearer\s+)\S+"), r"\1[removed]"),
    (re.compile(r"\b[A-Fa-f0-9]{32,}\b"), "[hex removed]"),
    (re.compile(r"\b[A-Za-z0-9_\-]{40,}\b"), "[token removed]"),
]


class AskError(Exception):
    pass


def redact(text: str) -> str:
    for pat, repl in _SECRETS:
        text = pat.sub(repl, text)
    return text


def reference() -> str:
    return resources.files("mllab").joinpath("ask_context.md").read_text()


def build_context(*, tables: list[str] | None = None, registry: list[str] | None = None, job: str | None = None,
                  reg=None, runner=None, schema_fn=None) -> str:
    """Reference sheet plus explicitly requested metadata. Everything is redacted; nothing here reads data rows."""
    parts = [reference()]
    for t in tables or []:
        info = (schema_fn or _schema)(t)
        cols = ", ".join(f"{c['name']} {c['type']}" for c in info["columns"])
        parts.append(f"## Table {t} (schema only)\nrows={info['num_rows']} partition={info['partitioning']} "
                     f"cluster={info['clustering']}\ncolumns: {cols}")
    for pm in registry or []:
        project, _, model = pm.partition("/")
        if not model:
            raise AskError(f"--with-registry wants PROJECT/MODEL, got {pm!r}")
        lines = []
        for v in reg.versions(project, model):
            m = reg.meta(project, model, v)
            hold = (m.metrics or {}).get("holdout", {})
            lines.append(f"- {v}: parent={m.parent} created={m.created_at[:10]} features={len(m.features)} "
                         f"holdout_accuracy={hold.get(v, {}).get('accuracy')}")
        parts.append(f"## Registry {pm}\nchannels: {json.dumps(reg.channels(project, model))}\n" + "\n".join(lines))
    if job:
        text, _ = runner.logs(job)
        tail = "\n".join(text.splitlines()[-LOG_TAIL_LINES:])
        parts.append(f"## Last {LOG_TAIL_LINES} log lines of job {job} (scrubbed)\n{tail}\nstatus: {runner.status(job)}")
    ctx = redact("\n\n".join(parts))
    if len(ctx) > MAX_CONTEXT_CHARS:
        ctx = ctx[:MAX_CONTEXT_CHARS] + "\n[context truncated]"
    return ctx


def _schema(table: str) -> dict:
    from .schema import get_schema

    return get_schema(table)


# --- usage cap -----------------------------------------------------------------------------------

def _usage_path() -> Path:
    return Path(os.environ.get("MLLAB_ROOT", "/mnt/appdata/mllab")) / "audit" / "gemini-usage.jsonl"


def calls_today(path: Path | None = None) -> int:
    p = path or _usage_path()
    if not p.exists():
        return 0
    today = datetime.now(timezone.utc).date().isoformat()
    return sum(1 for line in p.read_text().splitlines() if line and json.loads(line).get("day") == today)


def record_usage(question: str, model: str, usage=None, *, path: Path | None = None) -> None:
    p = path or _usage_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {"day": datetime.now(timezone.utc).date().isoformat(), "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "model": model, "q_sha": hashlib.sha256(question.encode()).hexdigest()[:12],  # a fingerprint, not the question
             "prompt_tokens": getattr(usage, "prompt_token_count", None), "output_tokens": getattr(usage, "candidates_token_count", None)}
    fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
    try:
        os.write(fd, (json.dumps(entry) + "\n").encode())
    finally:
        os.close(fd)


# --- model call ----------------------------------------------------------------------------------

def _client():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise AskError("GEMINI_API_KEY is not set. Run host/set-gemini-key.sh on the server, then restart mllab-jupyter.")
    from google import genai  # google-genai; the older google-generativeai package is deprecated

    return genai.Client(api_key=key)


def _is_not_found(e: Exception) -> bool:
    return getattr(e, "code", None) == 404 or "NOT_FOUND" in str(e) or "is not found" in str(e)


def _generate(client, model: str, prompt: str) -> tuple[str, object]:
    from google.genai import types

    cfg = types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.2,
                                      max_output_tokens=int(os.environ.get("MLLAB_GEMINI_MAX_OUTPUT_TOKENS", "2048")))
    try:
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
    except Exception as e:  # noqa: BLE001 - SDK raises several types; classify by content
        if _is_not_found(e):
            try:
                avail = [m.name for m in client.models.list() if "generateContent" in (getattr(m, "supported_actions", None) or [])]
            except Exception:  # noqa: BLE001
                avail = []
            raise AskError(f"model {model!r} is not available. Set MLLAB_GEMINI_MODEL to one of: "
                           f"{', '.join(avail[:12]) or '(could not list models)'}") from e
        raise AskError(f"Gemini call failed: {str(e)[:300]}") from e
    text = getattr(resp, "text", None)
    if not text:
        raise AskError("Gemini returned no text (it may have been blocked or hit the token limit)")
    return text, getattr(resp, "usage_metadata", None)


def ask(question: str, *, context: str | None = None, client=None, model: str | None = None,
        usage_path: Path | None = None, generate=None) -> str:
    question = question.strip()
    if not question:
        raise AskError("empty question")
    cap = int(os.environ.get("MLLAB_GEMINI_DAILY_CALLS", "50"))
    if calls_today(usage_path) >= cap:
        raise AskError(f"daily cap of {cap} Gemini calls reached (MLLAB_GEMINI_DAILY_CALLS); try again tomorrow or raise it")
    model = model or os.environ.get("MLLAB_GEMINI_MODEL", DEFAULT_MODEL)
    prompt = f"{context if context is not None else build_context()}\n\n---\nQuestion: {redact(question)}"
    client = client or _client()
    text, usage = (generate or _generate)(client, model, prompt)
    record_usage(question, model, usage, path=usage_path)
    return text


# --- spec drafting ---------------------------------------------------------------------------------

def _validators():
    from . import backtest, schema, train

    def bq_spec(raw):
        cols = [schema.ColumnSpec(**({"name": c} if isinstance(c, str) else c)) for c in raw.get("columns", [])]
        schema.validate_spec(schema.TableSpec(**{**{k: v for k, v in raw.items() if k != "columns"}, "columns": cols}))

    return {"backtest": backtest.validate, "train": train.validate, "bq": bq_spec}


def extract_yaml(text: str) -> str | None:
    m = re.search(r"```(?:yaml|yml)?\s*\n(.*?)```", text, re.S)
    return m.group(1).strip() + "\n" if m else None


def draft_spec(kind: str, request: str, *, out: Path | None = None, client=None, model: str | None = None,
               usage_path: Path | None = None, generate=None) -> tuple[str, str]:
    """Ask for a spec, validate it with the real validators (one retry with the error), write it only if valid."""
    import yaml

    validators = _validators()
    if kind not in validators:
        raise AskError(f"kind must be one of {sorted(validators)}")
    if out is not None and out.exists():
        raise AskError(f"{out} already exists; refusing to overwrite (choose another --out)")
    ctx = build_context()
    q = (f"Write a complete {kind} spec YAML for this request, using only keys from the reference. Reply with ONE ```yaml block "
         f"and no other prose.\nRequest: {request}")
    error = None
    for _ in range(2):
        text = ask(q if error is None else f"{q}\n\nYour previous attempt failed validation: {error}\nFix it and reply again.",
                   context=ctx, client=client, model=model, usage_path=usage_path, generate=generate)
        body = extract_yaml(text)
        if body is None:
            error = "no ```yaml block found"
            continue
        try:
            validators[kind](yaml.safe_load(body) or {})
        except Exception as e:  # noqa: BLE001 - any validator failure goes back to the model once
            error = str(e)[:400]
            continue
        if out is not None:
            out.write_text(body)
        return body, (f"validated; written to {out}" if out else "validated (not written; pass --out FILE to save)")
    raise AskError(f"Gemini's draft did not pass validation twice; last error: {error}")
