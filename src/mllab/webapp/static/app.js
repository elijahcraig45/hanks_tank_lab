"use strict";
/* mllab control app. No framework, no build step, no inline code (the page's CSP allows only this file). Every element is built with textContent, never
   innerHTML, so nothing the server sends can become markup. */

// ------------------------------------------------------------------------------------------------------------------------------ helpers
function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "text") el.textContent = v;
    else if (k in el && k !== "list") el[k] = v;
    else el.setAttribute(k, v);
  }
  for (const c of kids.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return el;
}
const $ = (s, el = document) => el.querySelector(s);
// Replace an element's children from nested arrays of nodes and text. The native replaceChildren does not flatten, and prints an array as "[object ...]".
function fill(el, ...kids) {
  el.replaceChildren(...kids.flat(Infinity).filter((c) => c != null && c !== false).map((c) => (c.nodeType ? c : document.createTextNode(String(c)))));
}

const MODES = ["off", "dry", "live"];
const MODE_MARK = { off: ["○", "Off"], dry: ["◐", "Dry run"], live: ["●", "Live"] };
const STATUS_MARK = { ok: ["✓", "Healthy"], warn: ["▲", "Warning"], down: ["✖", "Down"], critical: ["✖", "Critical"], paused: ["⏸", "Paused"],
  hidden: ["◌", "Hidden"], info: ["ℹ", "Info"] };

function badge(kind, label, icon) {
  const m = STATUS_MARK[kind] || MODE_MARK[kind] || ["•", kind];
  return h("span", { class: `badge ${kind}` }, h("i", { "aria-hidden": "true", text: icon || m[0] }), label || m[1]);
}
const modeBadge = (mode) => badge(mode, MODE_MARK[mode] ? MODE_MARK[mode][1] : mode);

function rel(iso) {
  const t = Date.parse(iso);
  if (!iso || Number.isNaN(t)) return iso || "—";
  const s = Math.round((Date.now() - t) / 1000), a = Math.abs(s);
  const unit = a < 90 ? `${a}s` : a < 5400 ? `${Math.round(a / 60)}m` : a < 129600 ? `${Math.round(a / 3600)}h` : `${Math.round(a / 86400)}d`;
  return h("span", { title: new Date(t).toLocaleString() }, s >= 0 ? `${unit} ago` : `in ${unit}`);
}
const gcpBadge = (state) => (state === "ENABLED" ? badge("warn", "Enabled") : state === "PAUSED" ? badge("paused", "Paused") : badge("info", state || "?"));
const clip = (s, n) => (s && s.length > n ? s.slice(0, n - 1) + "…" : s || "");

function toast(msg, kind = "ok", ms = 6000) {
  const t = h("div", { class: `toast ${kind}`, role: kind === "err" ? "alert" : null }, msg);
  $("#toasts").append(t);
  setTimeout(() => t.remove(), ms);
}

async function copy(text, btn) {
  try { await navigator.clipboard.writeText(text); btn.textContent = "Copied"; } catch { btn.textContent = "Select + copy"; }
  setTimeout(() => (btn.textContent = "Copy"), 1500);
}
const codeline = (cmd) => { const b = h("button", { class: "btn small", type: "button", text: "Copy" }); b.addEventListener("click", () => copy(cmd, b));
  return h("div", { class: "codeline" }, h("code", { text: cmd }), b); };

// ------------------------------------------------------------------------------------------------------------------------------------ api
async function api(path, opts = {}) {
  const post = opts.method === "POST";
  const res = await fetch("/app-api" + path, {
    method: opts.method || "GET", credentials: "same-origin",
    headers: post ? { "Content-Type": "application/json", "X-Mllab-App": "1" } : {},
    body: post ? JSON.stringify(opts.body || {}) : undefined,
  });
  let data = {};
  try { data = await res.json(); } catch { /* not JSON: leave empty */ }
  if (!res.ok) {
    const reasons = data.reasons || [res.statusText || `HTTP ${res.status}`];
    throw Object.assign(new Error(reasons.join("; ")), { status: res.status, code: data.error, reasons });
  }
  return data;
}

// -------------------------------------------------------------------------------------------------------------------------------- dialogs
const dlg = $("#dlg");
let gen = 0;  // one number per dialog shown: lets a late "close" event of the previous dialog be told apart from the current one's
function openDialog({ title, body, actions = [], onClose }) {
  const mine = ++gen;
  const foot = h("div", { class: "dlg-foot" });
  const api_ = {
    close: () => { if (dlg.open) dlg.close(); },
    setActions(list) {
      foot.replaceChildren(...list.map((a) => h("button", { class: `btn ${a.kind || ""}`, type: "button", disabled: a.disabled, onclick: a.onClick, text: a.label })));
    },
  };
  api_.setActions(actions);
  dlg.replaceChildren(h("div", { class: "dlg-head", text: title }), h("div", { class: "dlg-body" }, body), foot);
  const closed = () => {
    if (mine === gen && dlg.open) return;  // the close of an earlier dialog, arriving after this one opened: not ours
    dlg.removeEventListener("close", closed);
    if (onClose) onClose();
    if (mine === gen) render();
  };
  dlg.addEventListener("close", closed);
  if (!dlg.open) dlg.showModal();
  return api_;
}
const cancelBtn = () => ({ label: "Cancel", onClick: () => dlg.close() });

// ---------------------------------------------------------------------------------------------------------------------------------- state
const S = { session: null, data: {}, error: {}, pending: {}, siteEdit: {}, route: "overview", logTimer: null };
const ROUTES = [["overview", "Overview"], ["pipeline", "Pipeline"], ["data", "Data"], ["models", "Models"], ["jobs", "Jobs"], ["gcp", "GCP"], ["site", "Site"], ["health", "Health"], ["audit", "Audit"],
  ["notebooks", "Notebooks"]];
const NEEDS = { overview: ["pipeline", "models", "jobs", "health", "gcp", "data"], pipeline: ["pipeline"], data: ["pipeline", "data"], models: ["pipeline", "models"], jobs: ["pipeline", "jobs"],
  gcp: ["gcp", "pipeline"], site: ["site", "pipeline"], health: ["pipeline", "health"], audit: ["pipeline", "audit"], notebooks: ["pipeline", "notebooks"] };  // the header chips always need the pipeline

async function load(name, query = "") {
  try { S.data[name] = await api(`/${name}${query}`); delete S.error[name]; } catch (e) { S.error[name] = e; }
}
async function loadSession() {
  try { S.session = await api("/session"); } catch (e) { S.session = { login: null, unlocked: false, error: e }; }
}
async function refresh() {
  await Promise.all([loadSession(), ...(NEEDS[S.route] || []).map((n) => load(n))]);
  if (!dlg.open) render();
}

function unlocked() { return !!(S.session && S.session.unlocked); }
async function ensureUnlocked() {
  if (unlocked()) return true;
  return new Promise((resolve) => {
    const input = h("input", { type: "password", autocomplete: "current-password", id: "pw", "aria-describedby": "pwhelp" });
    const msg = h("div", { class: "callout critical", hidden: true });
    let done = false;
    const finish = (v) => { done = true; resolve(v); dlg.close(); };
    const submit = async () => {
      msg.hidden = true;
      try { await api("/unlock", { method: "POST", body: { password: input.value } }); await loadSession(); renderChrome(); finish(true); }
      catch (e) { msg.textContent = e.message; msg.hidden = false; input.select(); }
    };
    openDialog({
      title: "Unlock changes",
      body: [h("p", {}, "Looking is open to your Tailscale login. Changing anything needs the lab password too, once per device."),
        h("form", { onsubmit: (e) => { e.preventDefault(); submit(); } },
          h("input", { type: "text", name: "username", value: "mllab", autocomplete: "username", hidden: true }),
          h("div", { class: "field" }, h("label", { for: "pw", text: "Lab password" }), input)),
        h("p", { class: "muted", id: "pwhelp" }, "It is MLLAB_SECRET in /etc/mllab/env on the homelab. Fetch it with:"),
        codeline("ssh -t homelab 'sudo grep MLLAB_SECRET /etc/mllab/env'"), msg],
      actions: [{ label: "Cancel", onClick: () => dlg.close() }, { label: "Unlock", kind: "primary", onClick: submit }],
      onClose: () => { if (!done) resolve(false); },
    });
    setTimeout(() => input.focus(), 50);
  });
}

// Run a change. A locked session unlocks first and retries; the broker's answer is awaited so the toast says what actually happened.
async function change(path, body, { label }) {
  if (!(await ensureUnlocked())) return null;
  let res;
  try { res = await api(path, { method: "POST", body }); }
  catch (e) {
    if (e.code === "locked") { await loadSession(); render(); return change(path, body, { label }); }
    throw e;
  }
  if (!res.request) { toast(`${label}: done`); return res; }
  for (let i = 0; i < 45; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    let st;
    try { st = await api(`/pipeline/request/${res.request}`); } catch { continue; }
    if (st.status === "done") { toast(`${label}: ${st.detail}`); return { ...res, detail: st.detail, done: true }; }
    if (st.status === "rejected") { toast(`${label} was rejected: ${st.detail}`, "err", 12000); return { ...res, rejected: true }; }
  }
  toast(`${label}: no answer from the broker after 45 s. Is pipeline-broker-apply.path running?`, "err", 12000);
  return res;
}

// ---------------------------------------------------------------------------------------------------------------------------------- header
function renderChrome() {
  $("#nav").replaceChildren(...ROUTES.map(([id, label]) => h("a", { href: `#/${id}`, "aria-current": S.route === id ? "page" : null, text: label })));
  const p = S.data.pipeline, chips = [];
  if (p) {
    chips.push(p.broker.fresh ? badge("ok", `Broker ${p.broker.age_s}s`) : badge("warn", "Broker stale"));
    chips.push(p.gcp.available ? badge("ok", "GCP") : badge("warn", "GCP unreachable"));
  }
  $("#chips").replaceChildren(...chips);
  const lb = $("#lockbtn");
  const open = !!(S.session && S.session.open);
  lb.textContent = open ? "🔓 No password" : unlocked() ? "🔓 Unlocked" : "🔒 Locked";
  lb.title = open ? "The lab password is switched off (MLLAB_APP_NO_UNLOCK=1 in /etc/mllab/env)" : unlocked() ? "Click to lock this browser again" : "Unlock to make changes";
  lb.disabled = open;
  lb.onclick = async () => { if (open) return; if (unlocked()) { await api("/lock", { method: "POST" }); await loadSession(); render(); } else { await ensureUnlocked(); render(); } };
  const issues = attention();
  const ban = $("#banner"), crit = issues.filter((i) => i.banner);
  ban.hidden = !crit.length;
  ban.className = "banner critical";
  ban.replaceChildren(...crit.slice(0, 2).map((i) => h("div", {}, h("b", { text: "✖ Needs attention" }), i.text)));
}

// ------------------------------------------------------------------------------------------------------------------------ attention list
function attention() {
  const out = [], p = S.data.pipeline, hl = S.data.health;
  if (S.error.pipeline) out.push({ sev: "warn", text: `Pipeline status unavailable: ${S.error.pipeline.message}` });
  if (p) {
    for (const j of [...p.jobs, ...(p.pregame ? [{ ...p.pregame, name: "pregame" }] : [])]) {
      if (j.conflict) out.push({ sev: "critical", banner: true, text: `${j.name} is LIVE on the homelab and still ENABLED in Cloud Scheduler: two writers. Pause it in GCP or switch it off here.` });
    }
    if (!p.broker.fresh) out.push({ sev: "warn", text: p.broker.hint });
    for (const j of p.jobs) if (j.mode !== "off" && j.last && j.last.ok === false) out.push({ sev: "warn", text: `${j.name}: last run FAILED (${clip(j.last.error, 80)})` });
  }
  const dm = S.data.data;
  if (dm && dm.available) {
    for (const sp of Object.values(dm.sports)) {
      const bad = sp.tables.filter((t) => ["stale", "missing", "unchanged", "late"].includes(t.status));
      if (bad.length) out.push({ sev: bad.some((t) => ["stale", "missing"].includes(t.status)) ? "critical" : "warn",
        text: `Data (${sp.sport.toUpperCase()}): ${bad.slice(0, 3).map((t) => `${t.label} is ${DATA_STATUS[t.status][1].toLowerCase()}`).join("; ")}${bad.length > 3 ? ` and ${bad.length - 3} more` : ""}` });
    }
  } else if (dm && !dm.available) out.push({ sev: "warn", text: `Data map unavailable: ${dm.error}` });
  if (hl && hl.findings) for (const f of hl.findings) if (f.severity === "critical") out.push({ sev: "critical", text: `Health: ${f.message}` });
  return out;
}

// ---------------------------------------------------------------------------------------------------------------------------- overview
function viewOverview() {
  const p = S.data.pipeline, m = S.data.models, jb = S.data.jobs, hl = S.data.health, g = S.data.gcp;
  const issues = attention();
  const cards = [];

  const rows = p ? [...p.jobs.filter((j) => !j.paused), ...(p.pregame ? [{ ...p.pregame, name: "pregame" }] : [])] : [];
  const by = (mode) => rows.filter((j) => j.mode === mode).length;
  cards.push(h("section", { class: "card" }, h("h3", {}, "Pipeline", h("a", { href: "#/pipeline", text: "Open" })),
    p ? [h("div", { class: "row gap" }, h("div", {}, h("div", { class: "big", text: by("live") }), h("div", { class: "muted", text: "live here" })),
      h("div", {}, h("div", { class: "big", text: by("dry") }), h("div", { class: "muted", text: "dry run" })),
      h("div", {}, h("div", { class: "big", text: by("off") }), h("div", { class: "muted", text: "off" }))),
    h("p", { class: "muted" }, p.release ? `Release ${p.release.ref || "?"}, staged ` : "No release staged. ", p.release ? rel(p.release.staged_at) : null)]
      : h("p", { class: "muted", text: S.error.pipeline ? S.error.pipeline.message : "Loading…" })));

  cards.push(h("section", { class: "card" }, h("h3", {}, "Models", h("a", { href: "#/models", text: "Open" })),
    m ? (m.projects.flatMap((pr) => pr.models).length
      ? h("ul", { class: "list" }, m.projects.flatMap((pr) => pr.models.map((mo) => h("li", { class: "spread" }, h("span", { class: "mono", text: mo.model }),
        h("span", {}, Object.entries(mo.channels).map(([c, v]) => h("span", { class: "badge", text: `${c} ${v}` })), " ")))))
      : h("p", { class: "muted", text: "Nothing registered yet." })) : h("p", { class: "muted", text: "Loading…" })));

  const active = jb ? jb.jobs.filter((j) => ["queued", "running"].includes(j.state)) : [];
  cards.push(h("section", { class: "card" }, h("h3", {}, "Lab jobs", h("a", { href: "#/jobs", text: "Open" })),
    jb ? [h("div", { class: "big", text: active.length }), h("div", { class: "muted", text: "running or queued" }),
      h("p", { class: "muted" }, jb.jobs[0] ? ["Latest: ", jb.jobs[0].name, " ", rel(jb.jobs[0].submitted_at)] : "No jobs yet.")] : h("p", { class: "muted", text: "Loading…" })));

  const crit = hl && hl.findings ? hl.findings.filter((f) => f.severity === "critical").length : 0, warn = hl && hl.findings ? hl.findings.filter((f) => f.severity === "warn").length : 0;
  cards.push(h("section", { class: "card" }, h("h3", {}, "Model health", h("a", { href: "#/health", text: "Open" })),
    hl && !hl.missing ? [h("div", { class: "row" }, crit ? badge("critical", `${crit} critical`) : badge("ok", "No critical findings"), warn ? badge("warn", `${warn} warning`) : null),
      h("p", { class: "muted" }, "Checked ", rel(hl.taken_at))] : h("p", { class: "muted", text: hl ? "No health snapshot yet." : "Loading…" })));

  const enabled = g && g.jobs ? g.jobs.filter((j) => j.state === "ENABLED").length : 0;
  cards.push(h("section", { class: "card" }, h("h3", {}, "Cloud Scheduler", h("a", { href: "#/gcp", text: "Open" })),
    g ? (g.available ? [h("div", { class: "big", text: enabled }), h("div", { class: "muted", text: "jobs still enabled in GCP" })] : h("p", { class: "muted", text: `Unreachable: ${g.error}` })) : h("p", { class: "muted", text: "Loading…" })));

  return h("div", {}, h("h1", {}, "Overview"),
    issues.length ? h("section", { class: "card stack" }, h("h3", {}, "Needs attention"), h("ul", { class: "list" }, issues.map((i) => h("li", {}, badge(i.sev === "critical" ? "critical" : "warn"), " ", i.text))))
      : h("div", { class: "callout ok" }, badge("ok", "All clear"), " Nothing needs attention."),
    h("h2", {}, "At a glance"), h("div", { class: "grid" }, cards),
    h("h2", {}, "Where things run"),
    h("p", { class: "muted" }, "Hosting stays on App Engine and storage in BigQuery and Cloud Storage. The prediction jobs are moving from Cloud Functions to this homelab, one job at a time: ",
      "each starts Off, goes to Dry run (runs, writes nothing), and only then Live, after its Cloud Scheduler twin is paused."));
}

// ------------------------------------------------------------------------------------------------------------------------------ pipeline
const BUNDLE_TITLE = { mlb: "MLB", nfl: "NFL", cfb: "CFB" };

function lastRun(j) {
  const l = j.last;
  if (!l) return h("span", { class: "muted", text: "never" });
  if (l.skipped) return h("span", {}, badge("info", "Skipped", "⏭"), " ", rel(l.finished));
  return h("span", { title: l.error || "" }, l.ok ? badge("ok", "OK") : badge("down", "Failed"), " ", rel(l.finished), l.seconds != null ? h("span", { class: "muted", text: ` · ${l.seconds}s` }) : null);
}

function effectiveMode(j) {
  const p = S.pending[j.name];
  return p && Date.now() < p.until ? p.mode : j.mode;
}

function modeControl(j) {
  const cur = effectiveMode(j), pending = !!(S.pending[j.name] && Date.now() < S.pending[j.name].until && S.pending[j.name].mode !== j.mode);
  const seg = h("div", { class: `seg${pending ? " pending" : ""}`, role: "group", "aria-label": `Mode of ${j.name}` });
  for (const mode of MODES) {
    const off = j.paused && mode !== "off";
    seg.append(h("button", { type: "button", class: mode, "aria-pressed": String(cur === mode), disabled: off,
      title: off ? "Paused in Cloud Scheduler on purpose; not installed here" : cur === mode ? `Currently ${MODE_MARK[mode][1]}` : `Set to ${MODE_MARK[mode][1]}`,
      onclick: () => { if (cur !== mode) changeMode(j, mode); } }, `${MODE_MARK[mode][0]} ${MODE_MARK[mode][1]}`));
  }
  return seg;
}

async function changeMode(j, mode) {
  if (!(await ensureUnlocked())) return;  // ask for the password before asking anything else
  if (mode === "live") return gateDialog(j, "live");
  const dry = mode === "dry";
  const body = h("div", {}, h("p", {}, dry ? (j.dry_ok === false ? "This job refuses a dry run, so Dry means a preflight only: it checks that everything it needs is reachable, then stops."
    : "It will run on its schedule, do all its work and write nothing.") : "The timer still fires, but the job does nothing until it is set to Dry run or Live."),
  j.mode === "live" ? h("div", { class: "callout warn" }, "It is Live now. This stops it writing to production from here at its next run. Cloud Scheduler is separate: nothing runs it there unless you resume it.") : null);
  openDialog({ title: `Set ${j.name} to ${MODE_MARK[mode][1]}?`, body, actions: [cancelBtn(), { label: `Set to ${MODE_MARK[mode][1]}`, kind: "primary", onClick: async () => {
    dlg.close();
    S.pending[j.name] = { mode, until: Date.now() + 45000 }; render();
    try {
      const r = await change("/pipeline/mode", { job: j.name, mode }, { label: `${j.name} → ${mode}` });
      if (!r || r.rejected) delete S.pending[j.name];
    } catch (e) { delete S.pending[j.name]; toast(e.message, "err", 10000); }
    await refresh();
  } }] });
}

// The gate dialog is used to go live and to run for real: both write production from here, so both show what GCP says right now and demand the job's name.
async function gateDialog(j, kind) {
  if (!(await ensureUnlocked())) return;
  const isPregame = j.name === "pregame";
  const title = kind === "live" ? `Switch ${j.name} to LIVE` : `Run ${j.name} for real, once`;
  const box = h("div", {}, h("p", { class: "muted", text: "Checking Cloud Scheduler…" }));
  const typed = h("input", { type: "text", id: "typed", autocomplete: "off", spellcheck: false, placeholder: j.name });
  let gate = null, acked = new Set(), busy = false, ctl = null;

  const confirmable = () => gate && gate.allowed && typed.value === j.name && gate.acks.every((a) => acked.has(a.key)) && !busy;
  const sync = () => ctl && ctl.setActions([cancelBtn(), { label: "Re-check GCP", onClick: check },
    { label: kind === "live" ? "Go live" : "Run for real", kind: "danger fill", disabled: !confirmable(), onClick: go }]);
  async function check() {
    fill(box, h("p", { class: "muted", text: "Checking Cloud Scheduler…" }));
    try {
      const st = await api("/pipeline?fresh=true"); S.data.pipeline = st;
      const row = isPregame ? st.pregame : st.jobs.find((x) => x.name === j.name);
      gate = row.gate;
    } catch (e) { gate = { allowed: false, reasons: [e.message], commands: [], acks: [] }; }
    fill(box,
      h("p", {}, kind === "live" ? "This job will write to production BigQuery from the homelab at its next scheduled run." : "This runs the job once, now, writing to production BigQuery. The mode file is not changed."),
      gate.allowed ? h("div", { class: "callout ok" }, badge("ok", `${gate.scheduler_job} is paused in Cloud Scheduler`))
        : h("div", { class: "callout critical" }, h("div", {}, badge("critical", "Blocked")), h("ul", {}, gate.reasons.map((r) => h("li", { text: r })))),
      gate.commands.length ? [h("p", { class: "muted" }, "Run these from your laptop (they are audited there):"), gate.commands.map(codeline)] : null,
      gate.acks.map((a) => h("label", { class: "check" }, h("input", { type: "checkbox", checked: acked.has(a.key), onchange: (e) => { e.target.checked ? acked.add(a.key) : acked.delete(a.key); sync(); } }), h("span", { text: a.text }))),
      h("div", { class: "field" }, h("label", { for: "typed" }, "Type ", h("b", { class: "mono", text: j.name }), " to confirm"), typed));
    sync();
  }
  async function go() {
    busy = true; sync();
    const path = kind === "live" ? "/pipeline/mode" : "/pipeline/run";
    const body = kind === "live" ? { job: j.name, mode: "live", confirm: typed.value, acks: [...acked] } : { job: j.name, dry: false, confirm: typed.value, acks: [...acked] };
    if (kind === "live") S.pending[j.name] = { mode: "live", until: Date.now() + 45000 };
    try {
      dlg.close();
      const before = j.last && j.last.finished;
      const r = await change(path, body, { label: kind === "live" ? `${j.name} → live` : `run ${j.name}` });
      if (!r || r.rejected) delete S.pending[j.name];
      else if (kind !== "live" && r.done) watchRun(j.name, before);
    } catch (e) { delete S.pending[j.name]; toast(e.message, "err", 14000); }
    await refresh();
  }
  typed.addEventListener("input", sync);
  ctl = openDialog({ title, body: box, actions: [cancelBtn()] });
  await check();
}

// After a manual run is started, follow the job until its "last run" changes, then say how it went. (The snapshot refreshes every 30 s, so this can lag.)
async function watchRun(name, before) {
  const t0 = Date.now();
  while (Date.now() - t0 < 10 * 60 * 1000) {
    await new Promise((r) => setTimeout(r, 4000));
    await load("pipeline");
    if (!dlg.open) render();
    const row = ((S.data.pipeline || {}).jobs || []).find((x) => x.name === name);
    const l = row && row.last;
    if (l && l.finished && l.finished !== before) {
      toast(`${name}: the run ${l.ok ? "finished OK" : "FAILED"} (${l.seconds}s)${l.error ? `: ${clip(String(l.error), 140)}` : ""}`, l.ok ? "ok" : "err", 20000);
      return;
    }
  }
  toast(`${name}: no result after 10 minutes. Open the job to read its log.`, "err", 12000);
}

async function runDry(j) {
  if (!(await ensureUnlocked())) return;
  const before = j.last && j.last.finished;
  try {
    const r = await change("/pipeline/run", { job: j.name, dry: true }, { label: `dry run ${j.name}` });
    if (r && r.done) watchRun(j.name, before);
  } catch (e) { toast(e.message, "err", 10000); }
  await refresh();
}

function runDialog(j) {
  openDialog({ title: `Run ${j.name} once`, body: [
    h("p", {}, "Runs it now, as the pipeline user, under the same limits as its timer. The job's mode does not change."),
    h("div", { class: "callout" }, h("b", {}, "Dry run: "), j.dry_ok === false ? "this job refuses a dry run, so this is a preflight only: it checks that everything it needs is reachable." : "does all its work and writes nothing."),
    h("div", { class: "callout warn" }, h("b", {}, "Real run: "), "writes to production BigQuery from the homelab. It asks for the same confirmation as going Live.")],
  actions: [cancelBtn(), { label: "Dry run now", kind: "primary", onClick: () => { dlg.close(); runDry(j); } },
    { label: "Real run…", kind: "danger", onClick: () => { dlg.close(); gateDialog(j, "run"); } }] });
}

// Roll a job back to the mode it had before its latest change. This changes the homelab only: Cloud Scheduler is not touched, and the dialog says what that means.
async function rollbackDialog(j) {
  if (!(await ensureUnlocked())) return;
  const rb = j.rollback, cur = effectiveMode(j);
  if (!rb) return toast(`${j.name}: nothing to roll back to. No earlier mode is on record.`, "err");
  if (rb.to === "live") return gateDialog(j, "live");  // going back to Live is going Live: same gate, same typed name
  const twin = j.gate && j.gate.scheduler_job, twinState = j.gcp && j.gcp.state;
  const resume = `mllab control scheduler resume ${twin} --confirm ${twin}`;
  const body = [h("p", {}, `Set ${j.name} from ${MODE_MARK[cur] ? MODE_MARK[cur][1] : cur} back to `, h("b", { text: MODE_MARK[rb.to][1] }), ". This undoes the change made ", rel(rb.at), rb.by ? ` by ${rb.by}` : "",
    ". It takes effect at the job's next run; a run already in progress finishes."),
  cur === "live" && twinState === "PAUSED" ? h("div", { class: "callout warn" }, h("b", {}, "Nothing will run this job in production after this. "),
    `Its Cloud Scheduler twin (${twin}) is paused, and this rollback changes the homelab only. To hand it back to Cloud Functions, run this from your laptop:`, codeline(resume)) : null,
  cur === "live" && twinState === "ENABLED" ? h("div", { class: "callout critical" }, badge("critical", "Two writers"), ` ${twin} is enabled in Cloud Scheduler, so Cloud Functions are already running this job as well.`) : null];
  openDialog({ title: `Roll back ${j.name}`, body, actions: [cancelBtn(), { label: `Roll back to ${MODE_MARK[rb.to][1]}`, kind: "primary", onClick: async () => {
    dlg.close();
    S.pending[j.name] = { mode: rb.to, until: Date.now() + 45000 }; render();
    try { const r = await change("/pipeline/rollback", { job: j.name }, { label: `roll back ${j.name} → ${rb.to}` }); if (!r || r.rejected) delete S.pending[j.name]; }
    catch (e) { delete S.pending[j.name]; toast(e.message, "err", 10000); }
    await refresh();
  } }] });
}

async function jobDialog(j) {
  const isPregame = j.name === "pregame";
  const logBox = h("pre", { class: "log", text: unlocked() ? "Loading log…" : "Unlock to read this job's latest log." });
  const loadLog = async () => {
    if (!unlocked()) return;
    try { const l = await api(`/pipeline/log/${encodeURIComponent(j.name)}`); logBox.textContent = l.text || "No run has been logged yet."; }
    catch (e) { logBox.textContent = e.message; }
  };
  const l = j.last;
  const info = h("dl", { class: "kv" },
    h("dt", {}, "Mode"), h("dd", {}, modeBadge(effectiveMode(j))),
    isPregame ? null : [h("dt", {}, "Bundle"), h("dd", { class: "mono", text: `${j.bundle} · mode "${j.body_mode}"` }),
      h("dt", {}, "Schedule"), h("dd", { class: "mono", text: `${j.cron} (America/New_York)` }), h("dt", {}, "Next run"), h("dd", { class: "mono", text: j.next_run || "—" }),
      h("dt", {}, "Dry run"), h("dd", {}, j.dry_ok === false ? "refused by the job: Dry means preflight only" : "supported: runs, writes nothing")],
    h("dt", {}, "Last run"), h("dd", {}, lastRun(j), l && l.error ? h("div", { class: "muted mono", text: clip(String(l.error), 300) }) : null),
    j.gcp ? [h("dt", {}, "Cloud Scheduler"), h("dd", {}, gcpBadge(j.gcp.state), " ", h("span", { class: "muted", text: j.gate.scheduler_job }))] : null,
    isPregame ? [h("dt", {}, "Stores"), h("dd", { class: "mono", text: Object.keys(j.stores || {}).length ? JSON.stringify(j.stores) : "none yet" })] : null);
  openDialog({ title: j.name, body: [info, h("h3", { class: "muted" }, "Latest log"), logBox],
    actions: [{ label: "Close", onClick: () => dlg.close() }, ...(isPregame || j.paused ? [] : [{ label: "Run dry now", onClick: () => { dlg.close(); runDry(j); } },
      { label: "Run for real…", kind: "danger", onClick: () => { dlg.close(); gateDialog(j, "run"); } }])] });
  loadLog();
}

function viewPipeline() {
  const p = S.data.pipeline;
  if (!p) return S.error.pipeline ? h("div", { class: "empty", text: S.error.pipeline.message }) : h("p", { class: "muted pad", text: "Loading…" });
  const head = h("tr", {}, ["Job", "Mode", "Schedule", "Next run", "Last run", "Cloud Scheduler", "Actions"].map((t) => h("th", { text: t })));
  const rowFor = (j) => {
    const isPg = j.name === "pregame";
    const m = effectiveMode(j);
    return h("tr", { class: `click${j.conflict ? " conflict" : ""}`, tabIndex: 0, onclick: (e) => { if (!e.target.closest(".seg, .actions")) jobDialog(j); },
      onkeydown: (e) => { if (e.key === "Enter" && !e.target.closest(".seg, .actions")) jobDialog(j); } },
    h("td", {}, h("div", { class: "mono", text: isPg ? "pregame" : j.name }), h("div", { class: "sub", text: isPg ? "per-game MLB tasks (plan + tick)" : j.body_mode })),
    h("td", { "data-label": "Mode" }, modeControl(j)),
    h("td", { class: "sub", "data-label": "Schedule" }, isPg ? "plan every 30 min, tick every minute, Mar–Nov" : h("span", { class: "mono", text: j.cron }), j.paused ? h("div", {}, badge("paused", "Paused in GCP too")) : null),
    h("td", { class: "sub mono", "data-label": "Next run", text: m === "off" || j.paused ? "—" : isPg ? "every minute" : j.next_run || "—" }),
    h("td", { "data-label": "Last run" }, lastRun(j)),
    h("td", { "data-label": "Cloud Scheduler" }, j.conflict ? badge("critical", "Enabled: 2 writers") : j.gcp ? gcpBadge(j.gcp.state) : h("span", { class: "muted", text: "—" })),
    h("td", { class: "actions", "data-label": "Actions" },
      isPg || j.paused ? null : h("button", { class: "btn small", type: "button", text: "▶ Run…", title: "Run this job once, now", onclick: () => runDialog(j) }), " ",
      h("button", { class: "btn small", type: "button", text: "↶ Roll back", disabled: !j.rollback || j.paused,
        title: j.rollback ? `Back to ${MODE_MARK[j.rollback.to][1]} (the mode before the change ${j.rollback.by ? "by " + j.rollback.by : "on record"})` : "No earlier mode is on record",
        onclick: () => rollbackDialog(j) })));
  };
  const groups = ["mlb", "nfl", "cfb"].map((b) => {
    const jobs = p.jobs.filter((j) => j.bundle === b);
    const rows = [...jobs, ...(b === "mlb" && p.pregame ? [{ ...p.pregame, name: "pregame" }] : [])];
    return [h("div", { class: "group" }, h("h2", { text: BUNDLE_TITLE[b] })),
      h("div", { class: "scroll" }, h("table", { class: "table pipe" }, h("thead", {}, head.cloneNode(true)), h("tbody", {}, rows.map(rowFor))))];
  });
  return h("div", {}, h("div", { class: "spread" }, h("h1", {}, "Pipeline"), !unlocked() ? h("span", { class: "muted", text: "Locked: you can look, and changes ask for the lab password." }) : null),
    !p.broker.fresh ? h("div", { class: "callout warn" }, badge("warn", "Broker stale"), " ", p.broker.hint) : null,
    h("div", { class: "callout" }, h("b", {}, "How a job moves: "), "Off → Dry run → Live. Dry run exercises the job and writes nothing. Going Live checks that its Cloud Scheduler twin is paused and asks you to type the job's name. Click a row for its log and to run it once."),
    groups);
}

// -------------------------------------------------------------------------------------------------------------------------------- models
function metricsText(mx) {
  const keys = Object.keys(mx || {}).filter((k) => typeof mx[k] === "number").slice(0, 3);
  return keys.length ? keys.map((k) => `${k} ${Number(mx[k]).toFixed(3).replace(/\.?0+$/, "")}`).join(" · ") : "—";
}
async function rollbackChannelDialog(project, model, channel, rb) {
  if (!(await ensureUnlocked())) return;
  const prod = ["prod", "production"].includes(channel), phrase = `${model}@${channel}`;
  const typed = h("input", { type: "text", id: "rbty", autocomplete: "off", spellcheck: false, placeholder: phrase });
  let ctl;
  const sync = () => ctl.setActions([cancelBtn(), { label: `Point ${channel} back at ${rb.to}`, kind: "primary", disabled: prod && typed.value !== phrase, onClick: go }]);
  async function go() {
    dlg.close();
    try {
      const r = await change("/models/rollback", { project, model, channel, confirm: typed.value }, { label: `${model} ${channel} → ${rb.to}` });
      if (r) toast(`${model}: ${channel} is back at ${r.now}${r.previous ? ` (was ${r.previous})` : ""}`);
    } catch (e) { toast(e.message, "err", 10000); }
    await refresh();
  }
  typed.addEventListener("input", sync);
  ctl = openDialog({ title: `Roll back ${model} ${channel}`, body: [h("p", {}, `Point ${channel} back at `, h("b", { class: "mono", text: rb.to }), `, where it was before it moved to ${rb.from} `, rel(rb.at), rb.by ? ` (by ${rb.by})` : "", "."),
    h("div", { class: "callout" }, "A channel is only a pointer: this deploys nothing. If this version is already deployed, rolling back the deployment itself is ", h("code", { text: "mllab deploy cloudfn-rollback" }), " from your laptop."),
    prod ? h("div", { class: "field" }, h("label", { for: "rbty" }, "Type ", h("b", { class: "mono", text: phrase }), " to confirm"), typed) : null], actions: [cancelBtn()] });
  sync();
}

async function promoteDialog(project, model, version) {
  if (!(await ensureUnlocked())) return;
  const ch = h("input", { type: "text", id: "ch", value: "staging", list: "chs", autocomplete: "off" });
  const typed = h("input", { type: "text", id: "ty", autocomplete: "off", spellcheck: false });
  const help = h("p", { class: "muted" });
  let ctl;
  const sync = () => {
    const prod = ["prod", "production"].includes(ch.value.trim());
    help.textContent = prod ? `This moves the ${ch.value.trim()} pointer. Type ${model}@${ch.value.trim()} to confirm.` : "A non-production channel is only a pointer; nothing is deployed.";
    typed.parentElement.hidden = !prod;
    ctl.setActions([cancelBtn(), { label: "Move pointer", kind: "primary", disabled: !ch.value.trim() || (prod && typed.value !== `${model}@${ch.value.trim()}`), onClick: go }]);
  };
  async function go() {
    dlg.close();
    try {
      const r = await change("/models/promote", { project, model, version, channel: ch.value.trim(), confirm: typed.value }, { label: `${model} ${ch.value.trim()} → ${version}` });
      if (r) toast(`${model}: ${ch.value.trim()} now points at ${version}${r.previous ? ` (was ${r.previous})` : ""}`);
    } catch (e) { toast(e.message, "err", 10000); }
    await refresh();
  }
  ch.addEventListener("input", sync); typed.addEventListener("input", sync);
  ctl = openDialog({ title: `Point a channel at ${model} ${version}`, body: [h("div", { class: "field" }, h("label", { for: "ch", text: "Channel" }), ch,
    h("datalist", { id: "chs" }, ["staging", "prod"].map((v) => h("option", { value: v })))), help,
    h("div", { class: "field" }, h("label", { for: "ty", text: "Confirmation" }), typed)], actions: [cancelBtn()] });
  sync();
}
function viewModels() {
  const m = S.data.models;
  if (!m) return S.error.models ? h("div", { class: "empty", text: S.error.models.message }) : h("p", { class: "muted pad", text: "Loading…" });
  if (!m.projects.length) return h("div", {}, h("h1", { text: "Models" }), h("div", { class: "empty", text: "Nothing is registered yet. Register one from a notebook or with `mllab register`." }));
  return h("div", {}, h("h1", { text: "Models" }), h("div", { class: "callout" }, "Versions are immutable and hash-verified. A channel (prod, staging) is a pointer to one version; moving it records the change and does not deploy anything. Deploying is a separate, confirmed step: ", h("code", { text: "mllab deploy" }), "."),
    m.projects.map((pr) => pr.models.map((mo) => h("section", { class: "stack" }, h("div", { class: "group" }, h("h2", { class: "mono", text: `${pr.project} / ${mo.model}` }),
      Object.entries(mo.channels).map(([c, v]) => [h("span", { class: "badge live" }, h("i", { text: "▸" }), `${c} ${v}`),
        mo.rollback && mo.rollback[c] ? h("button", { class: "btn small", type: "button", text: `↶ Roll back to ${mo.rollback[c].to}`, title: `Point ${c} back at ${mo.rollback[c].to}`,
          onclick: () => rollbackChannelDialog(pr.project, mo.model, c, mo.rollback[c]) }) : null])),
    mo.error ? h("div", { class: "callout critical", text: mo.error }) : null,
    h("div", { class: "scroll" }, h("table", { class: "table" }, h("thead", {}, h("tr", {}, ["Version", "Created", "Parent", "Features", "Metrics", "Channels", ""].map((t) => h("th", { text: t })))),
      h("tbody", {}, mo.versions.map((v) => h("tr", {}, h("td", { class: "mono", text: v.version }), h("td", {}, rel(v.created_at)), h("td", { class: "mono", text: v.parent || "—" }),
        h("td", { text: v.features || "—" }), h("td", { class: "sub", text: metricsText(v.metrics) }),
        h("td", {}, v.channels.map((c) => h("span", { class: "badge live" }, h("i", { text: "▸" }), c)), " "),
        h("td", { class: "right" }, h("button", { class: "btn small", type: "button", text: "Point channel…", onclick: () => promoteDialog(pr.project, mo.model, v.version) }))))))),
    mo.deployments.length ? h("p", { class: "muted" }, `Last deploys: ${mo.deployments.map((d) => d.action || d.kind || "deploy").join(", ")}`) : null))));
}

// ----------------------------------------------------------------------------------------------------------------------------------- jobs
// the states jobs.py actually writes: queued, running, finished, failed, terminated (timeout or cancel), lost
const JOB_STATE = { running: "info", queued: "info", finished: "ok", failed: "down", terminated: "warn", lost: "down" };
function jobLogDialog(job) {
  const box = h("pre", { class: "log", text: "Loading…" });
  let offset = 0, text = "", alive = true;
  const tick = async () => {
    if (!alive) return;
    try { const r = await api(`/jobs/${job.id}/log?offset=${offset}`); offset = r.offset; text = (text + r.text).slice(-60000); box.textContent = text || "No output yet."; box.scrollTop = box.scrollHeight; }
    catch (e) { box.textContent = e.message; }
    if (alive) S.logTimer = setTimeout(tick, 2500);
  };
  const stop = h("button", { class: "btn danger", type: "button", text: "Cancel job", hidden: !["queued", "running"].includes(job.state), onclick: async () => {
    try { const r = await change(`/jobs/${job.id}/cancel`, {}, { label: `cancel ${job.name}` }); if (r) { toast("Cancelling…"); dlg.close(); } } catch (e) { toast(e.message, "err", 8000); } } });
  openDialog({ title: `${job.name} (${job.id})`, body: [h("div", {}, badge(JOB_STATE[job.state] || "info", job.state), " ", rel(job.submitted_at), job.detail ? h("div", { class: "muted", text: job.detail }) : null), box],
    actions: [{ label: "Close", onClick: () => dlg.close() }], onClose: () => { alive = false; clearTimeout(S.logTimer); } });
  $(".dlg-foot").prepend(stop);
  if (!unlocked()) { box.textContent = "Unlock to read job logs."; alive = false; ensureUnlocked().then((ok) => { if (ok) { alive = true; tick(); } }); } else tick();
}
function viewJobs() {
  const j = S.data.jobs;
  if (!j) return S.error.jobs ? h("div", { class: "empty", text: S.error.jobs.message }) : h("p", { class: "muted pad", text: "Loading…" });
  return h("div", {}, h("h1", { text: "Jobs" }),
    h("div", { class: "callout" }, "Backtests, training runs and other work launched from the lab, each under hard memory, CPU and time limits. Start them from a notebook or with ", h("code", { text: "mllab job submit" }), "."),
    h("h2", { text: "Recent jobs" }), j.jobs.length ? h("div", { class: "scroll" }, h("table", { class: "table" }, h("thead", {}, h("tr", {}, ["Name", "State", "Submitted", "Limits", ""].map((t) => h("th", { text: t })))),
      h("tbody", {}, j.jobs.map((x) => h("tr", { class: "click", tabIndex: 0, onclick: () => jobLogDialog(x), onkeydown: (e) => { if (e.key === "Enter") jobLogDialog(x); } },
        h("td", { class: "mono", text: x.name }), h("td", {}, badge(JOB_STATE[x.state] || "info", x.state)), h("td", {}, rel(x.submitted_at)),
        h("td", { class: "sub mono", text: Object.entries(x.caps).map(([k, v]) => `${k} ${v}`).join(" · ") || "—" }), h("td", { class: "right sub", text: "log ›" })))))) : h("div", { class: "empty", text: "No jobs yet." }),
    h("h2", { text: "Experiments" }), j.experiments.length ? h("div", { class: "scroll" }, h("table", { class: "table" }, h("thead", {}, h("tr", {}, ["Name", "Kind", "Games", "Range", "Verdict"].map((t) => h("th", { text: t })))),
      h("tbody", {}, j.experiments.map((x) => h("tr", {}, h("td", { class: "mono", text: x.name }), h("td", { text: x.kind }), h("td", { text: x.games ?? "—" }), h("td", { class: "sub", text: x.range ? [].concat(x.range).join(" → ") : "—" }),
        h("td", { class: "sub", text: clip(x.verdict, 140) })))))) : h("div", { class: "empty", text: "No experiments with results yet." }));
}


// ------------------------------------------------------------------------------------------------------------------------------------ data
const DATA_STATUS = { ok: ["ok", "Fresh"], idle: ["paused", "Idle"], late: ["warn", "Late"], stale: ["down", "Stale"], missing: ["down", "Missing"], unchanged: ["warn", "Not updated"], unknown: ["info", "Unknown"] };
const HOW_TEXT = { append: "appends rows: running it twice adds duplicates unless the job removes them", upsert: "replaces rows by key: safe to run again",
  replace_partition: "replaces one partition (a date or week): safe to run again for the same one", truncate: "replaces the whole table", ddl: "created by a deploy script, not by a job",
  external: "loaded from outside the pipeline" };
const STAGE_TITLE = { source: "Sources", raw: "Raw data", features: "Features", predictions: "Predictions", shadows: "Shadow models", sims: "Simulations", rankings: "Rankings", site: "Site" };
const dstatus = (st) => badge(DATA_STATUS[st][0], DATA_STATUS[st][1]);
const bytesText = (n) => (n == null ? "—" : n > 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`);

function tableDialog(t) {
  const kv = h("dl", { class: "kv" },
    h("dt", {}, "Table"), h("dd", { class: "mono", text: t.id }), h("dt", {}, "Stage"), h("dd", { text: STAGE_TITLE[t.stage] || t.stage }),
    h("dt", {}, "Status"), h("dd", {}, dstatus(t.status)), t.reasons.length ? [h("dt", {}, "Why"), h("dd", {}, t.reasons.map((r) => h("div", { text: r })))] : null,
    h("dt", {}, "Rows"), h("dd", { text: t.rows == null ? "—" : t.rows.toLocaleString() }), h("dt", {}, "Size"), h("dd", { text: bytesText(t.bytes) }),
    h("dt", {}, "Last change"), h("dd", {}, t.modified ? [new Date(t.modified).toLocaleString(), " (", rel(t.modified), ")"] : "—"),
    h("dt", {}, "Expected"), h("dd", { text: t.cadence.kind === "static" ? "does not change on a schedule" : `${t.cadence.kind.replace("_", " ")}, within ${t.cadence.max_age_hours} h${t.cadence.note ? `. ${t.cadence.note}` : ""}` }),
    t.key.length ? [h("dt", {}, "Key"), h("dd", { class: "mono", text: t.key.join(", ") })] : null, t.date_column ? [h("dt", {}, "Date column"), h("dd", { class: "mono", text: t.date_column })] : null,
    t.how ? [h("dt", {}, "Written"), h("dd", { text: HOW_TEXT[t.how] || t.how })] : null,
    h("dt", {}, "Written by"), h("dd", {}, t.written_by.length ? t.written_by.map((j) => h("a", { class: "badge", href: "#/pipeline", onclick: () => dlg.close(), text: j })) : h("span", { class: "muted", text: "nothing in the pipeline: loaded by hand or by another system" })),
    t.read_by.length ? [h("dt", {}, "Read by"), h("dd", { text: t.read_by.join(", ") })] : null, t.notes ? [h("dt", {}, "Notes"), h("dd", { text: t.notes })] : null);
  openDialog({ title: t.label, body: kv, actions: [{ label: "Close", onClick: () => dlg.close() }] });
}

function viewData() {
  const d = S.data.data;
  if (!d) return S.error.data ? h("div", { class: "empty", text: S.error.data.message }) : h("p", { class: "muted pad", text: "Loading…" });
  if (!d.available) return h("div", {}, h("h1", { text: "Data" }), h("div", { class: "callout warn" }, badge("warn", "Unavailable"), ` ${d.error}`));
  const head = h("div", { class: "spread" }, h("h1", { text: "Data" }), h("span", { class: "muted" }, "Checked ", rel(d.generated_at), ". Table metadata only: free to read, and it changes nothing."));
  const sports = Object.values(d.sports).map((sp) => {
    const byStage = {};
    for (const t of sp.tables) (byStage[t.stage] = byStage[t.stage] || []).push(t);
    const strip = h("div", { class: "row" }, d.stages.filter((st) => byStage[st]).map((st) => {
      const tabs = byStage[st], worst = tabs.reduce((w, t) => (["stale", "missing"].includes(t.status) ? "down" : ["late", "unchanged"].includes(t.status) && w !== "down" ? "warn" : w), "ok");
      return h("span", { class: `badge ${worst}` }, h("i", { text: worst === "ok" ? "✓" : worst === "warn" ? "▲" : "✖" }), `${STAGE_TITLE[st]} ${tabs.length}`);
    }));
    const rows = sp.tables.map((t) => h("tr", { class: "click", tabIndex: 0, onclick: () => tableDialog(t), onkeydown: (e) => { if (e.key === "Enter") tableDialog(t); } },
      h("td", {}, h("div", { text: t.label }), h("div", { class: "sub mono", text: t.id })),
      h("td", { class: "sub", text: STAGE_TITLE[t.stage] || t.stage }),
      h("td", { class: "right", text: t.rows == null ? "—" : t.rows.toLocaleString() }),
      h("td", {}, t.modified ? rel(t.modified) : h("span", { class: "muted", text: "—" })),
      h("td", {}, dstatus(t.status), t.reasons.length && t.status !== "idle" ? h("div", { class: "sub", text: clip(t.reasons[0], 120) }) : null),
      h("td", { class: "sub" }, t.written_by.length ? t.written_by.join(", ") : "—")));
    return h("section", { class: "stack" }, h("div", { class: "group" }, h("h2", { text: sp.sport.toUpperCase() }), dstatus(sp.status)), strip,
      h("div", { class: "scroll" }, h("table", { class: "table" }, h("thead", {}, h("tr", {}, ["Table", "Stage", "Rows", "Last change", "Status", "Written by"].map((x) => h("th", { text: x })))), h("tbody", {}, rows))));
  });
  return h("div", {}, head, h("div", { class: "callout" }, "Each table is judged against how often it should change, and only while its sport is in season. ",
    "A job that is Live here and finished OK, but whose table has not changed, is flagged: a clean exit is not proof that data moved."), sports);
}

// ------------------------------------------------------------------------------------------------------------------------------------ gcp
function viewGcp() {
  const g = S.data.gcp;
  if (!g) return S.error.gcp ? h("div", { class: "empty", text: S.error.gcp.message }) : h("p", { class: "muted pad", text: "Loading…" });
  if (!g.available) return h("div", {}, h("h1", { text: "Cloud Scheduler" }), h("div", { class: "callout warn" }, badge("warn", "Unreachable"), ` ${g.error}. Without it, nothing can go Live from here: the gate refuses rather than guess.`));
  const cons = g.jobs.filter((j) => j.conflict);
  return h("div", {}, h("h1", { text: "Cloud Scheduler" }),
    cons.length ? h("div", { class: "callout critical" }, badge("critical", "Two writers"), ` ${cons.map((c) => c.name).join(", ")} ${cons.length > 1 ? "are" : "is"} Live on the homelab and still enabled here.`) : h("div", { class: "callout ok" }, badge("ok", "No job is written from both places")),
    h("p", { class: "muted" }, "Read-only. Changes to GCP stay audited laptop commands: this page shows the exact command for each job."),
    h("div", { class: "scroll" }, h("table", { class: "table" }, h("thead", {}, h("tr", {}, ["Job", "State", "Schedule (ET)", "Here", "Last attempt", "Command"].map((t) => h("th", { text: t })))),
      h("tbody", {}, g.jobs.map((j) => h("tr", { class: j.conflict ? "conflict" : "" }, h("td", { class: "mono", text: j.name }),
        h("td", {}, gcpBadge(j.state), j.body_ok === false ? badge("critical", "Bad body") : null),
        h("td", { class: "sub mono", text: j.schedule || "—" }), h("td", {}, j.here ? modeBadge(j.here) : h("span", { class: "muted", text: "—" })),
        h("td", {}, j.last_attempt_at ? [j.last_attempt_ok === false ? badge("down", "Failed") : badge("ok", "OK"), " ", rel(j.last_attempt_at)] : h("span", { class: "muted", text: "never" })),
        h("td", {}, codeline(j.commands[0]))))))));
}


// -------------------------------------------------------------------------------------------------------------------------------- site
const SPORT_NAMES = { cfb: "College football", nfl: "NFL", mlb: "MLB" };
// What each sport's draft is, kept while the page re-renders. show: our columns in order. media: the polls to show.
// primary: null = the sport's default order, or "resume" / "rating". hasPrimary: this sport has a resume board to order by at all.
const siteDraft = (sport, cat, cur) => (S.siteEdit[sport] ||= { show: [...(cur.show || [])], media: cur.media ? [...cur.media] : cat.media.map((m) => m.key),
  primary: cur.primary ?? null, hasPrimary: (cat.primary || []).length > 0 });
function siteApplyDialog(sport, draft, reset) {
  const phrase = `${sport}/*`;
  const typed = h("input", { type: "text", id: "sty", autocomplete: "off", spellcheck: false });
  let ctl;
  const sync = () => ctl.setActions([cancelBtn(), { label: reset ? "Back to default" : "Apply to the site", kind: "primary", disabled: typed.value !== phrase, onClick: go }]);
  async function go() {
    dlg.close();
    try {
      const body = reset ? { sport, show: null, media: null, confirm: typed.value } : { sport, show: draft.show, media: draft.media, confirm: typed.value };
      if (draft.hasPrimary) body.primary = reset ? null : draft.primary;
      const r = await change("/site/rankings", body, { label: `${sport} rankings` });
      if (r) { delete S.siteEdit[sport]; toast(`${SPORT_NAMES[sport]}: ${reset ? "back to default" : "saved"}. ${r.note || ""}`); }
    } catch (e) { toast(e.message, "err", 15000); }
    await refresh();
  }
  typed.addEventListener("input", sync);
  const order = draft.hasPrimary ? (reset || draft.primary == null ? " Ordered by the sport's default (the season resume for college football)." : ` Ordered by the ${draft.primary === "resume" ? "season resume" : "next-week rating"}.`) : "";
  const summary = reset ? `Every ranking column that has data will show again (the default).${order}`
    : `The table will show ${draft.show.length ? draft.show.join(", ") : "no custom columns"}, and ${draft.media.length ? draft.media.join(" and ") + " from the media" : "no media polls"}.${order}`;
  ctl = openDialog({ title: reset ? `Reset ${SPORT_NAMES[sport]} rankings` : `Change the ${SPORT_NAMES[sport]} rankings table`, body: [h("p", {}, summary),
    h("div", { class: "callout" }, "This changes what visitors to hankstank.com see, within about a minute. It is recorded and can be undone the same way."),
    h("div", { class: "field" }, h("label", { for: "sty" }, "Type ", h("b", { class: "mono", text: phrase }), " to confirm"), typed)], actions: [cancelBtn()] });
  sync();
}
function siteCard(sport, cat, cur, ready) {
  const draft = siteDraft(sport, cat, cur);
  const label = (list, key) => (list.find((x) => x.key === key) || {}).label || key;
  const move = (i, d) => { const j = i + d; if (j < 0 || j >= draft.show.length) return; [draft.show[i], draft.show[j]] = [draft.show[j], draft.show[i]]; render(); };
  const toggle = (list, key, on) => { const at = list.indexOf(key); if (on && at < 0) list.push(key); if (!on && at >= 0) list.splice(at, 1); render(); };
  const choices = cat.primary || [], dflt = cat.primary_default;
  const effPrimary = draft.primary ?? dflt;
  const liveOrder = choices.length ? ` · ordered by ${label(choices, cur.primary ?? dflt)}${cur.primary == null ? " (default)" : ""}` : "";
  const allDefault = cur.show == null && cur.media == null && cur.primary == null;
  const live = (cur.show == null && cur.media == null ? "Default: every ranking column that has data" :
    `Custom: ${cur.show == null ? "all of ours" : cur.show.length ? cur.show.map((k) => label(cat.custom, k)).join(", ") : "none of ours"} · media: ${cur.media == null ? "all" : cur.media.length ? cur.media.map((k) => label(cat.media, k)).join(", ") : "none"}`) + liveOrder;
  const dirty = JSON.stringify(draft.show) !== JSON.stringify(cur.show || []) || JSON.stringify([...draft.media].sort()) !== JSON.stringify([...(cur.media || cat.media.map((m) => m.key))].sort())
    || (draft.primary ?? null) !== (cur.primary ?? null);
  const orderRows = choices.map((p) => h("div", { class: "site-row" },
    h("input", { type: "radio", name: `p-${sport}`, id: `p-${sport}-${p.key}`, checked: effPrimary === p.key, onchange: () => { draft.primary = p.key === dflt ? null : p.key; render(); } }),
    h("label", { for: `p-${sport}-${p.key}` }, h("b", { text: p.label }), " ", h("span", { class: "muted", text: p.about }))));
  const customRows = [...draft.show, ...cat.custom.map((c) => c.key).filter((k) => !draft.show.includes(k))].map((key) => {
    const c = cat.custom.find((x) => x.key === key), on = draft.show.includes(key), i = draft.show.indexOf(key);
    const box = h("input", { type: "checkbox", id: `c-${sport}-${key}`, checked: on, disabled: !on && draft.show.length >= cat.max_custom, onchange: (e) => toggle(draft.show, key, e.target.checked) });
    return h("div", { class: "site-row" }, box, h("label", { for: `c-${sport}-${key}` }, h("b", { text: c.label }), " ", h("span", { class: "muted", text: c.about })),
      on ? h("span", { class: "site-move" }, h("button", { class: "btn small", type: "button", text: "↑", title: "Move left", disabled: i === 0, onclick: () => move(i, -1) }),
        h("button", { class: "btn small", type: "button", text: "↓", title: "Move right", disabled: i === draft.show.length - 1, onclick: () => move(i, 1) })) : null);
  });
  const mediaRows = cat.media.map((m) => h("div", { class: "site-row" }, h("input", { type: "checkbox", id: `m-${sport}-${m.key}`, checked: draft.media.includes(m.key), onchange: (e) => toggle(draft.media, m.key, e.target.checked) }),
    h("label", { for: `m-${sport}-${m.key}` }, h("b", { text: m.label }), " ", h("span", { class: "muted", text: m.about }))));
  const resumeOrder = choices.length > 0 && effPrimary === "resume";
  const lead = resumeOrder ? ["#", "Team", "Rec", "Resume", "Range", "Next-week rating"] : ["#", "Team", "Rec", "Rating", "Rank range"];
  const cols = [...lead, ...draft.show.filter((k) => !(resumeOrder && k === "resume")).map((k) => label(cat.custom, k)), ...draft.media.map((k) => label(cat.media, k))];
  return h("section", { class: "card stack" }, h("h2", { text: SPORT_NAMES[sport] || sport }),
    h("p", {}, "On the site now: ", h("b", { text: live })),
    ready === false ? h("div", { class: "callout warn" }, badge("warn", "View not updated"), " The control view does not have all the rankings columns yet (the table order is new), so a change is recorded but the site ignores it. Run once from your laptop: ",
      codeline("bq query --project_id=hankstank --nouse_legacy_sql < sql/control/02_current_view.sql")) : null,
    choices.length ? [h("h3", { text: "Table order" }), ...orderRows] : null,
    h("h3", { text: `Our rankings (up to ${cat.max_custom}, beside the ${resumeOrder ? "resume" : "rating"})` }), customRows,
    h("h3", { text: "Media polls" }), mediaRows,
    h("h3", { text: "Preview of the table's columns" }), h("div", { class: "site-preview" }, cols.map((c, i) => h("span", { class: `badge ${i >= lead.length ? "live" : ""}`, text: c }))),
    h("div", { class: "spread" }, h("button", { class: "btn", type: "button", text: "Back to default…", disabled: allDefault, onclick: () => siteApplyDialog(sport, draft, true) }),
      h("button", { class: "btn primary", type: "button", text: "Apply to the site…", disabled: !dirty, onclick: () => siteApplyDialog(sport, draft, false) })));
}
function viewSite() {
  const d = S.data.site;
  if (!d) return S.error.site ? h("div", { class: "empty", text: S.error.site.message }) : h("p", { class: "muted pad", text: "Loading…" });
  return h("div", {}, h("h1", { text: "Site" }),
    h("div", { class: "callout" }, "Choose how hankstank.com orders its power-rankings table (college football: the season resume, or the next-week rating) and which ranking columns it shows: up to three of our own beside the media polls. The other order stays on the table as a column. ",
      "Nothing is deleted, every change is recorded, and Back to default restores today's table."),
    !d.available ? h("div", { class: "callout warn" }, badge("warn", "Current setting unknown"), ` ${d.error}. You can still prepare a change; saving needs the same access.`) : null,
    Object.entries(d.catalog).map(([sport, cat]) => siteCard(sport, cat, (d.current || {})[sport] || { show: null, media: null, primary: null }, d.view_ready)));
}

// -------------------------------------------------------------------------------------------------------------------------------- health
function viewHealth() {
  const hl = S.data.health;
  if (!hl) return S.error.health ? h("div", { class: "empty", text: S.error.health.message }) : h("p", { class: "muted pad", text: "Loading…" });
  if (hl.missing) return h("div", {}, h("h1", { text: "Model health" }), h("div", { class: "empty", text: "No health snapshot yet. It is built every ten minutes by mllab-modelhealth.timer." }));
  const sev = (s) => (s === "critical" ? "critical" : s === "warn" ? "warn" : "info");
  return h("div", {}, h("div", { class: "spread" }, h("h1", { text: "Model health" }), h("a", { class: "btn", href: "/dash/", target: "_blank", rel: "noopener", text: "Static dashboard ↗" })),
    h("p", { class: "muted" }, "Checked ", rel(hl.taken_at), ". Sources: ", Object.entries(hl.sources || {}).map(([k, v]) => h("span", { class: "badge", title: v }, h("i", { text: v === "ok" ? "✓" : "▲" }), k)), " "),
    h("h2", { text: "Findings" }), hl.findings.length ? h("ul", { class: "list card" }, hl.findings.map((f) => h("li", {}, badge(sev(f.severity), f.severity), " ", h("b", { class: "mono", text: `${f.sport}/${f.model}` }), " ", f.message)))
      : h("div", { class: "callout ok" }, badge("ok", "Nothing found")),
    Object.values(hl.sports || {}).map((s) => [h("h2", { text: `${s.sport.toUpperCase()} models` }), h("div", { class: "scroll" }, h("table", { class: "table" },
      h("thead", {}, h("tr", {}, ["Model", "Status", "Last prediction", "Recent rows", "Mean p", "Shadow"].map((t) => h("th", { text: t })))),
      h("tbody", {}, (s.models || []).map((m) => h("tr", {}, h("td", { class: "mono", text: m.model }), h("td", {}, badge(m.status)), h("td", {}, m.last_prediction_at ? rel(m.last_prediction_at) : "—"),
        h("td", { text: m.n_recent }), h("td", { text: m.prob_mean == null ? "—" : m.prob_mean.toFixed(3) }), h("td", { text: m.is_shadow ? "yes" : "" }))))))]));
}

// -------------------------------------------------------------------------------------------------------------------------- audit, notebooks
function viewAudit() {
  const a = S.data.audit;
  if (!a) return S.error.audit ? h("div", { class: "empty", text: S.error.audit.message }) : h("p", { class: "muted pad", text: "Loading…" });
  const detail = (e) => Object.entries(e).filter(([k]) => !["ts", "actor", "action"].includes(k)).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join("  ");
  return h("div", {}, h("h1", { text: "Audit log" }), h("p", { class: "muted", text: "Every change made through this app, the CLI or the broker. Append-only." }),
    a.entries.length ? h("div", { class: "scroll" }, h("table", { class: "table" }, h("thead", {}, h("tr", {}, ["When", "Action", "By", "Detail"].map((t) => h("th", { text: t })))),
      h("tbody", {}, a.entries.map((e) => h("tr", {}, h("td", { class: "nowrap" }, rel(e.ts)), h("td", { class: "mono", text: e.action }), h("td", { class: "sub", text: e.by || e.actor || "" }), h("td", { class: "sub mono", text: clip(detail(e), 220) })))))) : h("div", { class: "empty", text: "Nothing recorded yet." }));
}
function viewNotebooks() {
  const n = S.data.notebooks;
  if (!n) return S.error.notebooks ? h("div", { class: "empty", text: S.error.notebooks.message }) : h("p", { class: "muted pad", text: "Loading…" });
  return h("div", {}, h("div", { class: "spread" }, h("h1", { text: "Notebooks" }), h("a", { class: "btn primary", href: n.lab, target: "_blank", rel: "noopener", text: "Open JupyterLab ↗" })),
    n.notebooks.length ? h("div", { class: "scroll" }, h("table", { class: "table" }, h("thead", {}, h("tr", {}, ["Notebook", "Modified", ""].map((t) => h("th", { text: t })))),
      h("tbody", {}, n.notebooks.map((x) => h("tr", {}, h("td", { class: "mono", text: x.path }), h("td", {}, rel(x.modified)), h("td", { class: "right" }, h("a", { class: "btn small", href: x.link, target: "_blank", rel: "noopener", text: "Open ↗" }))))))) :
      h("div", { class: "empty", text: "No notebooks in the lab's work folder yet." }));
}

// ------------------------------------------------------------------------------------------------------------------------------- routing
const VIEWS = { overview: viewOverview, pipeline: viewPipeline, data: viewData, models: viewModels, jobs: viewJobs, gcp: viewGcp, site: viewSite, health: viewHealth, audit: viewAudit, notebooks: viewNotebooks };
function render() {
  renderChrome();
  const y = window.scrollY;
  $("#view").replaceChildren((VIEWS[S.route] || viewOverview)());
  window.scrollTo(0, y);
  document.title = `${(ROUTES.find((r) => r[0] === S.route) || ["", "mllab"])[1]} · mllab`;
}
function route() {
  const id = (location.hash.match(/^#\/(\w+)/) || [])[1];
  S.route = VIEWS[id] ? id : "overview";
  render();
  refresh();
}
window.addEventListener("hashchange", route);
setInterval(() => { if (!document.hidden && !dlg.open) refresh(); }, 15000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
route();
