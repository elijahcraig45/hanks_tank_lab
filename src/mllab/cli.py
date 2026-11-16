"""mllab CLI: registry, adopt, jobs, deploys."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import audit
from .control import ConfirmationRequired
from .deploy.confirm import ConfirmationError
from .jobs import HostCaps, JobError, JobSpec, LocalRunner
from .registry import Registry, RegistryError

ROOT = lambda: os.environ.get("MLLAB_ROOT", "/mnt/appdata/mllab")  # noqa: E731


def _registry() -> Registry:
    return Registry(os.environ.get("MLLAB_REGISTRY", f"{ROOT()}/registry"))


def _runner() -> LocalRunner:
    return LocalRunner(ROOT(), caps=HostCaps(), python_bin=os.environ.get("MLLAB_JOB_PYTHON", "python3"))


def _storage(project: str):
    from google.cloud import storage

    return storage.Client(project=project)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mllab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register", help="register a local artifact as a new immutable version")
    r.add_argument("project"); r.add_argument("model"); r.add_argument("version"); r.add_argument("artifact")
    r.add_argument("--parent"); r.add_argument("--lock"); r.add_argument("--train-script")
    r.add_argument("--features", help="comma-separated feature names")
    r.add_argument("--dry-run", action="store_true")

    a = sub.add_parser("adopt", help="import an existing gs:// model (e.g. deployed V10) as a version")
    a.add_argument("project"); a.add_argument("model"); a.add_argument("version"); a.add_argument("gcs_uri")
    a.add_argument("--lock"); a.add_argument("--features"); a.add_argument("--dry-run", action="store_true")
    a.add_argument("--code", action="append", default=[], help="helper source the pickle needs to load (repeatable)")

    ls = sub.add_parser("list", help="list versions, channels and deployments")
    ls.add_argument("project"); ls.add_argument("model")

    pr = sub.add_parser("promote", help="point a channel at a version")
    pr.add_argument("project"); pr.add_argument("model"); pr.add_argument("version"); pr.add_argument("--channel", default="prod")

    job = sub.add_parser("job", help="run and inspect jobs on this host").add_subparsers(dest="jcmd", required=True)
    js = job.add_parser("submit"); js.add_argument("spec")
    for n in ("status", "cancel", "outputs"):
        job.add_parser(n).add_argument("id")
    jl = job.add_parser("logs"); jl.add_argument("id"); jl.add_argument("--follow", "-f", action="store_true")

    bqp = sub.add_parser("bq", help="inspect and validate BigQuery tables").add_subparsers(dest="bcmd", required=True)
    for n, h in (("describe", "columns, types, size, freshness (free)"), ("infer", "print a spec for a table as it is now")):
        bqp.add_parser(n, help=h).add_argument("table")
    bd = bqp.add_parser("diff", help="schema difference between two tables"); bd.add_argument("a"); bd.add_argument("b")
    bc = bqp.add_parser("check", help="validate a table against a spec file (exit 4 on failure)")
    bc.add_argument("spec"); bc.add_argument("--max-gb", type=float); bc.add_argument("--project", default="hankstank")

    ct = sub.add_parser("control", help="change what Hank's Tank shows and runs, without code or a deploy").add_subparsers(dest="ccmd", required=True)
    sp = lambda p: p.add_argument("sport", choices=["mlb", "nfl", "cfb"])  # noqa: E731
    common = lambda p: (p.add_argument("--confirm", metavar="PHRASE"), p.add_argument("--note", default=""), p.add_argument("--dry-run", action="store_true"))  # noqa: E731
    c = ct.add_parser("show", help="current control state"); c.add_argument("--sport", choices=["mlb", "nfl", "cfb"]); c.add_argument("--json", action="store_true")
    c = ct.add_parser("models", help="the live site's models with their control state"); sp(c)
    c = ct.add_parser("set", help="set any field (advanced)"); sp(c); c.add_argument("target"); c.add_argument("field"); c.add_argument("value")
    c.add_argument("--at", help="take effect at this time (ISO 8601 WITH timezone), a scheduled change"); c.add_argument("--allow-empty", action="store_true"); common(c)
    c = ct.add_parser("hide", help="hide a model from the site (nothing is deleted)"); sp(c); c.add_argument("target"); c.add_argument("--banner", help="text shown to visitors (required for the production model)")
    c.add_argument("--banner-level", default="warn", choices=["info", "warn", "error"]); c.add_argument("--allow-empty", action="store_true"); common(c)
    for name, h in (("unhide", "show a hidden model again"), ("pause", "stop the pipeline producing this model"), ("resume", "let the pipeline produce this model again")):
        c = ct.add_parser(name, help=h); sp(c); c.add_argument("target"); common(c)
    for name, arg in (("label", "text"), ("describe", "text")):
        c = ct.add_parser(name, help="override the label shown on the site" if name == "label" else "override the note shown on the site"); sp(c); c.add_argument("target"); c.add_argument("text"); common(c)
    c = ct.add_parser("order", help="sort position on the Models page (lower first)"); sp(c); c.add_argument("target"); c.add_argument("n", type=int); common(c)
    c = ct.add_parser("tiers", help="confidence-tier cut-offs for the production model"); sp(c); c.add_argument("target"); c.add_argument("high", type=float); c.add_argument("medium", type=float); common(c)
    c = ct.add_parser("pin", help="run exactly this artifact as the production model"); sp(c); c.add_argument("target"); c.add_argument("uri"); c.add_argument("sha256"); common(c)
    c = ct.add_parser("unpin", help="go back to the pipeline's normal model chain"); sp(c); c.add_argument("target"); common(c)
    c = ct.add_parser("banner", help="show a notice on the site for a sport"); sp(c); c.add_argument("text"); c.add_argument("--level", default="info", choices=["info", "warn", "error"]); common(c)
    c = ct.add_parser("banner-clear", help="remove the notice"); sp(c); common(c)
    c = ct.add_parser("rankings", help="which ranking columns the power-rankings table shows (up to 3 of ours beside the media polls)"); sp(c)
    c.add_argument("--show", help="our ranking columns, in order, e.g. season,results (or none)"); c.add_argument("--show-default", action="store_true", help="back to every column that has data")
    c.add_argument("--media", help="media polls to show, e.g. ap,coaches (or none)"); c.add_argument("--media-default", action="store_true", help="back to every poll that has data")
    c.add_argument("--primary", help="what the table is ordered by: resume or rating"); c.add_argument("--primary-default", action="store_true", help="back to the sport's default order (resume for college football)"); common(c)
    c = ct.add_parser("history", help="what changed, when, by whom"); sp(c); c.add_argument("target"); c.add_argument("--field"); c.add_argument("--limit", type=int, default=20)
    c = ct.add_parser("revert", help="undo the latest change to a field"); sp(c); c.add_argument("target"); c.add_argument("field"); common(c)
    c = ct.add_parser("freeze", help="kill switch: hide every model of a sport and show a banner"); sp(c); c.add_argument("message"); common(c)
    c = ct.add_parser("unfreeze", help="undo the latest freeze"); sp(c); common(c)
    sc = ct.add_parser("scheduler", help="Cloud Scheduler jobs").add_subparsers(dest="scmd", required=True)
    sc.add_parser("list", help="jobs, their state and decoded request body")
    for name in ("pause", "resume", "run"):
        c = sc.add_parser(name, help=f"{name} a job (type its name as --confirm)"); c.add_argument("job"); c.add_argument("--confirm"); c.add_argument("--dry-run", action="store_true")

    dt = sub.add_parser("data", help="the lab's data directory").add_subparsers(dest="dcmd2", required=True)
    dt.add_parser("list", help="files backtests and training can read")
    dp = dt.add_parser("promote", help="copy a finished job's output file into the data directory")
    dp.add_argument("job"); dp.add_argument("file", help="path inside the job's outputs dir"); dp.add_argument("name", help="new name, .parquet or .csv")
    dp.add_argument("--replace", action="store_true", help="overwrite an existing data file")

    hl = sub.add_parser("health", help="model health: build the private dashboard, print findings, optionally self-heal (exit 5 if anything is critical)")
    hl.add_argument("--sports", default="mlb", help="comma list of mlb,nfl,cfb"); hl.add_argument("--build", action="store_true", help="write health.json and index.html")
    hl.add_argument("--out", help="output dir (default <root>/health)"); hl.add_argument("--json", action="store_true")
    hl.add_argument("--remedy", action="store_true", help="apply the self-healing policy (dry-run unless the policy says otherwise)")
    hl.add_argument("--policy", default=os.environ.get("MLLAB_REMEDY_POLICY", "/etc/mllab/remedy.json"))
    rm = sub.add_parser("remedy", help="self-healing log and policy").add_subparsers(dest="rmcmd", required=True)
    rm.add_parser("log", help="recent automatic decisions"); rm.add_parser("policy", help="the policy in force")

    wk = sub.add_parser("weekly", help="run the weekly data checks and backtests from a config; records audit/weekly.json")
    wk.add_argument("--config", default="specs/weekly.yaml"); wk.add_argument("--dry-run", action="store_true")

    ev = sub.add_parser("envs", help="cached per-lock environments").add_subparsers(dest="ecmd", required=True)
    ev.add_parser("list", help="size and last use of each cached environment")
    ep = ev.add_parser("prune", help="delete idle environments (dry run unless --yes)")
    ep.add_argument("--older-than", type=float, required=True, metavar="DAYS"); ep.add_argument("--yes", action="store_true")
    ep.add_argument("--include-untracked", action="store_true", help="also delete environments built before use was tracked")

    rn = sub.add_parser("runs", help="list backtest/training runs with a per-run verdict (never ranks across runs)")
    rn.add_argument("--limit", type=int, default=30); rn.add_argument("--json", action="store_true")
    cd = sub.add_parser("card", help="model card for a version: lineage, data, metrics, backtests that included it")
    cd.add_argument("project"); cd.add_argument("model"); cd.add_argument("version"); cd.add_argument("--save", action="store_true")

    ak = sub.add_parser("ask", help="ask Gemini how to do something with mllab (advice only; nothing is executed)")
    ak.add_argument("question", nargs="?", default="")
    ak.add_argument("--with-table", action="append", default=[], metavar="DATASET.TABLE", help="include this table's schema (never rows)")
    ak.add_argument("--with-registry", action="append", default=[], metavar="PROJECT/MODEL", help="include version names and summary metrics")
    ak.add_argument("--with-job", metavar="JOB_ID", help="include the last 60 log lines of a job, scrubbed")
    ak.add_argument("--draft-spec", choices=["backtest", "train", "bq"], help="ask for a spec; it is validated before it is saved")
    ak.add_argument("--out", help="with --draft-spec: file to write (never overwrites)")
    ak.add_argument("--show-context", action="store_true", help="print exactly what would be sent to Gemini, then stop")

    tr = sub.add_parser("train", help="train a model as a job").add_subparsers(dest="rcmd", required=True)
    trr = tr.add_parser("run", help="submit a training job from a spec file"); trr.add_argument("spec")
    trr.add_argument("--wait", action="store_true"); trr.add_argument("--max-gb", type=float)
    rj = sub.add_parser("register-job", help="register a finished training job's model as a new version")
    rj.add_argument("job"); rj.add_argument("project"); rj.add_argument("model"); rj.add_argument("version"); rj.add_argument("--parent")

    bt = sub.add_parser("backtest", help="side-by-side backtests and parameter sweeps").add_subparsers(dest="tcmd", required=True)
    br = bt.add_parser("run", help="submit a backtest job from a spec file"); br.add_argument("spec")
    br.add_argument("--wait", action="store_true", help="wait and print the report")
    br.add_argument("--max-gb", type=float, help="BigQuery scan cap for the data query")
    bt.add_parser("report", help="print the report of a finished backtest job").add_argument("id")

    dep = sub.add_parser("deploy", help="deploy a registered version to GCP").add_subparsers(dest="dcmd", required=True)
    sw = dep.add_parser("cloudfn-swap", help="replace the model object a Cloud Function loads")
    rb = dep.add_parser("cloudfn-rollback", help="restore the object saved by the last swap")
    for p in (sw, rb):
        p.add_argument("project"); p.add_argument("model")
        p.add_argument("--bucket", default="hanks_tank_data"); p.add_argument("--object", required=True, dest="obj")
        p.add_argument("--dry-run", action="store_true"); p.add_argument("--confirm")
        p.add_argument("--channel", default="prod", help="registry channel to move (default prod; use another for tests)")
    sw.add_argument("version")
    cr = dep.add_parser("cloudrun", help="build and deploy a serving container (scale to zero)")
    cr.add_argument("project"); cr.add_argument("model"); cr.add_argument("version"); cr.add_argument("--service", required=True)
    cr.add_argument("--region", default="us-central1"); cr.add_argument("--max-instances", type=int, default=2)
    cr.add_argument("--memory", default="1Gi"); cr.add_argument("--dry-run", action="store_true"); cr.add_argument("--confirm")
    cr.add_argument("--full-lock", action="store_true", help="install the whole registered lock (default: only the runtime libraries)")
    return ap


def _health_snapshot(sports: tuple[str, ...]):
    """Overridable in tests. Each collector degrades on its own; nothing here can raise for a missing permission.
    On the server (MLLAB_BQ_CLIENT=1) everything uses the service account over the APIs; on a laptop it uses the gcloud and bq CLIs."""
    from datetime import datetime, timezone

    from . import site
    from .control import BqCliStore, current_state
    from . import health_collect as HC
    from .scheduler import Scheduler

    now = datetime.now(timezone.utc)
    if os.environ.get("MLLAB_BQ_CLIENT") == "1":
        runner = HC.BqClientRunner()
        return HC.build_snapshot(now, runner, sports=sports, registry_fn=site.cached_registry, control_fn=lambda: HC.control_from_view(runner),
                                 scheduler=HC.ApiScheduler(), log_reader=HC.ApiLogReader())

    def control_fn():
        st = current_state(BqCliStore().all_events(), now)
        return {k: {f: v for f, v in row.items()} for k, row in st.items()}

    return HC.build_snapshot(now, HC.BqCliRunner(), sports=sports, registry_fn=site.cached_registry, control_fn=control_fn, scheduler=Scheduler(),
                             log_reader=HC.GcloudLogReader())


def _health(args) -> int:
    from pathlib import Path

    from . import dashboard, remedy
    from . import health as H
    from .scheduler import Scheduler

    if args.cmd == "remedy":
        if args.rmcmd == "policy":
            pol = remedy.load_policy(os.environ.get("MLLAB_REMEDY_POLICY", "/etc/mllab/remedy.json"))
            print(json.dumps({"dry_run": pol.dry_run, "rules": pol.rules}, indent=2))
        else:
            for e in remedy.read_log()[-30:]:
                print(f"{e['at']}  {e['rule']:22} {e['action']:17} {e['outcome']}")
        return 0
    sports = tuple(x for x in args.sports.split(",") if x)
    snap = _health_snapshot(sports)
    findings = H.evaluate(snap)
    entries = []
    if args.remedy:
        pol = remedy.load_policy(args.policy)
        ex = remedy.make_executors(_make_control(), Scheduler(), _runner())
        entries = remedy.run(findings, pol, ex)
    d = H.to_json(snap, findings)
    if args.build:
        out = Path(args.out) if args.out else Path(ROOT()) / "health"
        dashboard.build(snap, out, remedies=remedy.read_log())
        print(f"wrote {out}/index.html", file=sys.stderr)
    if args.json:
        print(json.dumps(d, indent=1))
    else:
        for f in d["findings"]:
            print(f"[{f['severity']:8}] {f['sport']}/{f['model']}: {f['message']}")
        for e in entries:
            print(f"  remedy {e['rule']}: {e['outcome']}")
        if not d["findings"]:
            print("healthy: nothing needs attention")
    return 5 if any(f["severity"] == "critical" for f in d["findings"]) else 0


def _run(args) -> int:
    if args.cmd in ("health", "remedy"):
        return _health(args)
    reg = _registry()
    if args.cmd == "register":
        if args.dry_run:
            print(f"would register {args.project}/{args.model}/{args.version} from {args.artifact}")
            return 0
        meta = reg.register(args.project, args.model, args.version, args.artifact, parent=args.parent,
                            lockfile=args.lock, train_script=args.train_script,
                            features=args.features.split(",") if args.features else None)
        audit.record("register", project=args.project, model=args.model, version=args.version, sha256=meta.sha256)
        print(json.dumps({"version": meta.version, "sha256": meta.sha256}))
    elif args.cmd == "adopt":
        from .adopt import adopt, parse_gcs_uri

        parse_gcs_uri(args.gcs_uri)
        if args.dry_run:
            print(f"would adopt {args.gcs_uri} as {args.project}/{args.model}/{args.version}")
            return 0
        meta = adopt(reg, args.project, args.model, args.version, args.gcs_uri, client=_storage(args.project),
                     lockfile=args.lock, features=args.features.split(",") if args.features else None,
                     code_snapshot=args.code or None)
        audit.record("adopt", project=args.project, model=args.model, version=args.version, source=meta.source, sha256=meta.sha256)
        print(json.dumps({"version": meta.version, "sha256": meta.sha256, "source": meta.source}))
    elif args.cmd == "list":
        print(json.dumps({"versions": reg.versions(args.project, args.model), "channels": reg.channels(args.project, args.model),
                          "deployments": reg.deployments(args.project, args.model)}, indent=2))
    elif args.cmd == "promote":
        prev = reg.promote(args.project, args.model, args.version, args.channel)
        audit.record("promote", project=args.project, model=args.model, version=args.version, channel=args.channel, previous=prev)
        print(f"{args.channel}: {prev} -> {args.version}")
    elif args.cmd == "job":
        return _job(args)
    elif args.cmd == "bq":
        return _bq(args)
    elif args.cmd == "backtest":
        return _backtest(args, reg)
    elif args.cmd == "train":
        return _train(args, reg)
    elif args.cmd == "ask":
        return _ask(args, reg)
    elif args.cmd == "data":
        from . import data as D

        r = _runner()
        if args.dcmd2 == "list":
            for name, size in D.list_files(r.data_root):
                print(f"{size:>12,}  {name}")
        else:
            audit.record("data_promote", job=args.job, file=args.file, name=args.name)
            print(D.promote(r, args.job, args.file, args.name, replace=args.replace))
    elif args.cmd == "control":
        return _control(args)
    elif args.cmd == "envs":
        from . import envs as E

        r = _runner()
        if args.ecmd == "list":
            print(E.render(E.list_envs(r)))
        else:
            gone = E.prune(r, args.older_than, dry_run=not args.yes, include_untracked=args.include_untracked)
            for e in gone:
                print(f"{'deleted' if args.yes else 'would delete'} {e['name']} ({e['size_bytes'] / 1024**3:.2f} GiB)")
            if args.yes:
                audit.record("envs_prune", removed=[e["name"] for e in gone])
            print(f"{len(gone)} environment(s) {'removed' if args.yes else 'would be removed; pass --yes to delete'}")
    elif args.cmd == "weekly":
        return _weekly(args, reg)
    elif args.cmd == "runs":
        from . import runs as R

        rows = R.list_runs(_runner(), limit=args.limit)
        print(json.dumps(rows, indent=2) if args.json else R.render_runs(rows))
    elif args.cmd == "card":
        from . import runs as R

        if args.save:
            print(R.save_card(reg, _runner(), args.project, args.model, args.version))
        else:
            print(R.card(reg, _runner(), args.project, args.model, args.version), end="")
    elif args.cmd == "register-job":
        from .train import register_from_job

        meta = register_from_job(reg, _runner(), args.job, args.project, args.model, args.version, parent=args.parent)
        audit.record("register_job", job=args.job, project=args.project, model=args.model, version=args.version, sha256=meta.sha256)
        print(json.dumps({"version": meta.version, "sha256": meta.sha256, "parent": meta.parent, "data_hash": meta.data_hash}))
    elif args.cmd == "deploy":
        return _deploy(args, reg)
    return 0


def _bq(args) -> int:
    from . import schema as S

    if args.bcmd == "describe":
        info = S.get_schema(args.table)
        print(f"rows={info['num_rows']} bytes={info['num_bytes']} modified={info['modified']} "
              f"partition={info['partitioning']} cluster={info['clustering']}")
        for c in info["columns"]:
            print(f"  {c['name']:40} {c['type']:10} {c['mode']}")
    elif args.bcmd == "infer":
        print(S.dump_spec(S.infer_spec(args.table)), end="")
    elif args.bcmd == "diff":
        d = S.diff_schemas(S.get_schema(args.a), S.get_schema(args.b))
        print(json.dumps(d, indent=2))
    elif args.bcmd == "check":
        spec = S.load_spec(args.spec)
        kw = {"max_bytes": int(args.max_gb * 1024**3)} if args.max_gb else {}
        report = S.run_check(spec, project=args.project, bq_kwargs=kw)
        print(report.render())
        return 0 if report.ok else 4
    return 0


def _make_control():
    """Overridable in tests. Writes with YOUR gcloud identity through the bq CLI; the registry comes from the live site."""
    import getpass

    from . import site
    from .control import BqCliStore, Control

    return Control(BqCliStore(), getpass.getuser(), known_models=site.known_models, production_keys=site.production_keys)


def _control(args) -> int:
    from . import control as K
    from . import scheduler as SCH
    from .control import UNSET

    if args.ccmd == "scheduler":
        sch = SCH.Scheduler()
        if args.scmd == "list":
            print(SCH.render(sch.list()))
            return 0
        res = getattr(sch, {"run": "run_now"}.get(args.scmd, args.scmd))(args.job, confirm=args.confirm, dry_run=args.dry_run)
        print(res["detail"] + ("  (dry run)" if res.get("dry_run") else ""))
        return 0
    c = _make_control()
    if args.ccmd == "show":
        st = c.state(args.sport)
        if args.json:
            print(json.dumps({f"{k[0]}/{k[1]}": {f: (v.isoformat() if hasattr(v, "isoformat") else v) for f, v in row.items()} for k, row in st.items()}, indent=2))
        else:
            print(_render_state(st))
        return 0
    if args.ccmd == "models":
        from . import site
        reg_models = site.registry(args.sport)
        st = c.state(args.sport)
        print("| model | role | on site now | visible | run state | label override |\n|---|---|---|---|---|---|")
        for m in reg_models:
            row = st.get((args.sport, m["key"]), {})
            print(f"| {m['key']} | {m['role']} | {'yes' if m['available'] else 'no data'} | {K.effective(row, 'site_visible')} | {K.effective(row, 'run_state')} | {row.get('display_label', '-')} |")
        return 0
    if args.ccmd == "history":
        for e in c.history(args.sport, args.target, args.field, args.limit):
            print(f"{e.event_ts:%Y-%m-%d %H:%M:%S}Z  {e.field:16} {e.value[:60]:60} by {e.actor}  {e.note}")
        return 0
    if args.ccmd in ("unfreeze",):
        evs = c.unfreeze(args.sport)
    elif args.ccmd == "freeze":
        evs = c.freeze(args.sport, args.message, confirm=args.confirm)
    else:
        _require_known_target(c, args)
        kw = dict(confirm=getattr(args, "confirm", None), note=getattr(args, "note", ""), dry_run=getattr(args, "dry_run", False))
        t = getattr(args, "target", "*")
        if args.ccmd == "set":
            from datetime import datetime
            at = datetime.fromisoformat(args.at) if args.at else None
            evs = [c.set(args.sport, t, args.field, args.value, at=at, allow_empty=args.allow_empty, **kw)]
        elif args.ccmd == "hide":
            evs = c.hide(args.sport, t, banner=args.banner, banner_level=args.banner_level, allow_empty=args.allow_empty, **kw)
        elif args.ccmd == "unhide":
            evs = [c.set(args.sport, t, "site_visible", "true", **kw)]
        elif args.ccmd == "pause":
            evs = [c.set(args.sport, t, "run_state", "paused", **kw)]
        elif args.ccmd == "resume":
            evs = [c.set(args.sport, t, "run_state", "active", **kw)]
        elif args.ccmd == "label":
            evs = [c.set(args.sport, t, "display_label", args.text, **kw)]
        elif args.ccmd == "describe":
            evs = [c.set(args.sport, t, "public_note", args.text, **kw)]
        elif args.ccmd == "order":
            evs = [c.set(args.sport, t, "sort_order", str(args.n), **kw)]
        elif args.ccmd == "tiers":
            evs = [c.set(args.sport, t, "tier_high", str(args.high), **kw), c.set(args.sport, t, "tier_medium", str(args.medium), **kw)]
        elif args.ccmd == "pin":
            evs = c.pin(args.sport, t, args.uri, args.sha256, confirm=args.confirm, note=args.note, dry_run=args.dry_run)
        elif args.ccmd == "unpin":
            evs = [c.set(args.sport, t, "lifecycle", UNSET, **kw), c.set(args.sport, t, "artifact_uri", UNSET, **kw), c.set(args.sport, t, "artifact_sha256", UNSET, **kw)]
        elif args.ccmd == "banner":
            evs = [c.set(args.sport, "*", "banner", args.text, **kw), c.set(args.sport, "*", "banner_level", args.level, note=kw["note"], dry_run=kw["dry_run"])]
        elif args.ccmd == "banner-clear":
            evs = [c.set(args.sport, "*", "banner", UNSET, **{**kw, "confirm": kw["confirm"]}), c.set(args.sport, "*", "banner_level", UNSET, note=kw["note"], dry_run=kw["dry_run"])]
        elif args.ccmd == "rankings":
            picks = [("rankings_media", UNSET if args.media_default else args.media), ("rankings_show", UNSET if args.show_default else args.show),
                     ("rankings_primary", UNSET if args.primary_default else args.primary)]
            picks = [(f, v) for f, v in picks if v is not None]
            if (not picks or (args.show and args.show_default) or (args.media and args.media_default)
                    or (args.primary and args.primary_default)):
                raise K.ControlError("give --show/--show-default, --media/--media-default and/or --primary/--primary-default (one of each at most)")
            for f, v in picks:  # check every value before writing any
                K.validate(args.sport, "*", f, v)
            evs = [c.set(args.sport, "*", f, v, **kw) for f, v in picks]
        elif args.ccmd == "revert":
            evs = [c.revert(args.sport, t, args.field, confirm=args.confirm, dry_run=args.dry_run)]
        else:  # pragma: no cover
            raise K.ControlError(f"unhandled command {args.ccmd}")
    for e in evs:
        print(f"{'would set' if getattr(args, 'dry_run', False) else 'set'} {e.sport}/{e.target} {e.field} = {e.value}" + (f"  (effective {e.event_ts:%Y-%m-%d %H:%M}Z)" if e.event_ts > e.recorded_at else ""))
    return 0


def _require_known_target(c, args) -> None:
    """A typo must not silently create a control row for a model that does not exist."""
    from . import control as K

    t = getattr(args, "target", None)
    if t in (None, "*") or c.known_models is None:
        return
    try:
        known = c.known_models(args.sport)
    except Exception as e:  # noqa: BLE001 - registry unreachable: refuse rather than guess
        raise K.ControlError(f"cannot check {t!r} against the live model list ({e}); try again when the site API is reachable") from e
    if t not in known:
        raise K.ControlError(f"{t!r} is not a {args.sport} model on the site; known: {', '.join(known)}")


def _render_state(st: dict) -> str:
    if not st:
        return "no control settings: the site and pipeline behave as they always have"
    cols = ["site_visible", "run_state", "lifecycle", "display_label", "public_note", "sort_order", "tier_high", "tier_medium", "banner", "banner_level", "rankings_show", "rankings_media", "rankings_primary"]
    L = ["| sport | target | " + " | ".join(cols) + " | updated |", "|---|---|" + "---|" * (len(cols) + 1)]
    for (sport, target), row in sorted(st.items()):
        L.append(f"| {sport} | {target} | " + " | ".join(str(row.get(c, "")) for c in cols) + f" | {row['updated_at']:%Y-%m-%d %H:%M}Z |")
    return "\n".join(L)


def _ask(args, reg: Registry) -> int:
    from pathlib import Path

    from . import ask as A

    ctx = A.build_context(tables=args.with_table, registry=args.with_registry, job=args.with_job, reg=reg,
                          runner=_runner() if args.with_job else None)
    if args.show_context:
        print(ctx)
        print(f"\n[{len(ctx)} characters would be sent, plus your question. Nothing was sent.]", file=sys.stderr)
        return 0
    if args.draft_spec:
        body, note = A.draft_spec(args.draft_spec, args.question, out=Path(args.out) if args.out else None)
        print(body, end="")
        print(f"[{note}]", file=sys.stderr)
        return 0
    print(A.ask(args.question, context=ctx))
    return 0


def _weekly(args, reg: Registry) -> int:
    from pathlib import Path

    from . import backtest as B
    from . import runs as R
    from . import schema as S
    from . import weekly as W

    cfg_path = Path(args.config)
    cfg = W.load_config(cfg_path)
    base = cfg_path.resolve().parent.parent  # specs/weekly.yaml -> repo root, so entries read like `specs/bq/games.yaml`
    if args.dry_run:
        for k in ("bq_checks", "backtests"):
            for x in cfg.get(k, []):
                print(f"would run {k[:-1]}: {x}")
        return 0
    r = _runner()

    def verdict(job):
        row = next((x for x in R.list_runs(r, limit=10_000) if x["job"] == job), None)
        return row["verdict"] if row else "no verdict"

    res = W.run_weekly(cfg, root=Path(ROOT()), base=base, check_fn=lambda p: S.run_check(S.load_spec(p)),
                       submit_fn=lambda p: B.submit(B.load(p), reg, r, base=p.resolve().parent),
                       status_fn=lambda j: r.status(j)["state"], verdict_fn=verdict)
    print(W.render(res))
    audit.record("weekly", ok=res["ok"], errors=len(res["errors"]))
    return 0 if res["ok"] else 1


def _train(args, reg: Registry) -> int:
    from pathlib import Path

    from . import train as T

    r = _runner()
    raw = T.load(args.spec)
    kw = {"max_bytes": int(args.max_gb * 1024**3)} if args.max_gb else {}
    jid = T.submit(raw, reg, r, bq_kwargs=kw, base=Path(args.spec).resolve().parent)
    audit.record("train_submit", job=jid, name=raw["name"])
    print(jid)
    if args.wait:
        while r.status(jid)["state"] in ("queued", "running"):
            time.sleep(2)
        print(r.status(jid))
        rp = next((p for p in r.outputs(jid) if p.name == "report.md"), None)
        print(rp.read_text() if rp else r.logs(jid)[0][-3000:])
    return 0


def _backtest(args, reg: Registry) -> int:
    from pathlib import Path

    from . import backtest as B

    r = _runner()
    if args.tcmd == "run":
        raw = B.load(args.spec)
        kw = {"max_bytes": int(args.max_gb * 1024**3)} if args.max_gb else {}
        jid = B.submit(raw, reg, r, bq_kwargs=kw, base=Path(args.spec).resolve().parent)
        audit.record("backtest_submit", job=jid, name=raw["name"])
        print(jid)
        if args.wait:
            while r.status(jid)["state"] in ("queued", "running"):
                time.sleep(2)
            print(r.status(jid))
            rp = B.report_path(r, jid)
            print(rp.read_text() if rp else r.logs(jid)[0][-3000:])
    elif args.tcmd == "report":
        rp = B.report_path(r, args.id)
        if rp is None:
            print(f"no report; job is {r.status(args.id)['state']}. Logs: mllab job logs {args.id}", file=sys.stderr)
            return 1
        print(rp.read_text())
    return 0


def _job(args) -> int:
    r = _runner()
    if args.jcmd == "submit":
        jid = r.submit(JobSpec.from_yaml(args.spec))
        audit.record("job_submit", job=jid)
        print(jid)
    elif args.jcmd == "status":
        print(json.dumps(r.status(args.id)))
    elif args.jcmd == "cancel":
        r.cancel(args.id); audit.record("job_cancel", job=args.id); print("cancel requested")
    elif args.jcmd == "outputs":
        for p in r.outputs(args.id):
            print(p)
    elif args.jcmd == "logs":
        off = 0
        while True:
            text, off = r.logs(args.id, off)
            sys.stdout.write(text); sys.stdout.flush()
            if not args.follow or r.status(args.id)["state"] not in ("queued", "running"):
                if args.follow:  # drain what was written just before the job ended
                    text, off = r.logs(args.id, off); sys.stdout.write(text)
                break
            time.sleep(1)
    return 0


def _deploy(args, reg: Registry) -> int:
    if args.dcmd == "cloudfn-swap":
        from .deploy import cloudfn

        client = _storage(args.project)
        plan = cloudfn.plan_swap(reg, args.project, args.model, args.version, bucket=args.bucket, object_name=args.obj, client=client)
        print(plan.diff())
        res = cloudfn.apply_swap(plan, reg, client=client, confirm=args.confirm, dry_run=args.dry_run, channel=args.channel)
        if not res["changed"] and not plan.noop and args.dry_run:
            print(f"(dry run) to apply, re-run with --confirm '{plan.confirm_phrase}'")
        print(json.dumps({k: v for k, v in res.items() if k != "detail"}))
    elif args.dcmd == "cloudfn-rollback":
        from .deploy import cloudfn

        res = cloudfn.rollback_swap(reg, args.project, args.model, bucket=args.bucket, object_name=args.obj,
                                    client=_storage(args.project), confirm=args.confirm, dry_run=args.dry_run, channel=args.channel)
        print(json.dumps(res))
    elif args.dcmd == "cloudrun":
        from .deploy import cloudrun

        res = cloudrun.deploy(reg, args.project, args.model, args.version, service=args.service, region=args.region,
                              work_dir=f"{ROOT()}/cache/cloudrun/{args.service}", confirm=args.confirm, dry_run=args.dry_run,
                              max_instances=args.max_instances, memory=args.memory, full_lock=args.full_lock)
        print(json.dumps(res, indent=2))
        if res.get("dry_run"):
            print(f"(dry run) to apply, re-run with --confirm 'deploy {args.service}'")
    return 0


def S_ERRORS():
    from .ask import AskError
    from .backtest import BacktestError
    from .control import ControlError
    from .data import DataError
    from .envs import EnvError
    from .scheduler import SchedulerError
    from .site import SiteError
    from .schema import SpecError
    from .weekly import WeeklyError

    return (RegistryError, JobError, SpecError, BacktestError, AskError, DataError, WeeklyError, EnvError, ControlError, SchedulerError, SiteError)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except ConfirmationRequired as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3
    except ConfirmationError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3
    except S_ERRORS() as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
