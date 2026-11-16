# Spec: model control, remaining work

Vocabulary follows `GLOSSARY.md`. The base control plane, health dashboard and self-healing engine are built, tested and (for the site, the MLB pipeline and the control dataset) deployed. This spec covers what is left, decided in the grilling session of 2026-09-29.

## Problem Statement

Henry can now hide, pause, pin and relabel models on Hank's Tank without a deploy, but four gaps remain that make the control plane unsafe or incomplete in real use:

1. When the production model (V10 for MLB) is hidden or paused, the site has no headline model. Visitors see the games but no main pick, and Henry has to decide in the middle of an incident what should take its place.
2. If the control read fails, the site fails open, so a model that was deliberately hidden reappears for as long as the outage lasts.
3. The control field named `role` means something different from the site registry's `role`, which invites mistakes when reading, writing and documenting control settings.
4. A pinned artifact that fails its fingerprint check stops the pipeline, correctly, but nothing tells Henry: the health dashboard has no finding for it.

The football functions also still run code from before the control plane, and the lab has not been installed with the new health timer, so the two-week self-healing dry-run has not started.

## Solution

From Henry's point of view:

- Hiding or pausing the production model makes the site promote a **stand-in model** automatically. The stand-in is the eligible model with the best log loss so far this season. The site always shows a notice saying a stand-in is in use and why. Nothing is promoted unless Henry hid or paused the production model on purpose.
- A short control-read failure no longer un-hides anything. The site keeps applying the last control state it read successfully for up to six hours, then fails open as before.
- The control field is renamed **lifecycle** (live, shadow, archived), informational only, so that "role" refers to one thing everywhere: the registry's card type.
- A pin fingerprint mismatch shows up as a critical finding on the health dashboard, and the run keeps failing loudly. It never falls back silently.
- The football functions get the pause and tier controls through a code-only redeploy that leaves their environment settings and scheduler jobs alone. The lab gets installed, with read-only access first, and the self-healing dry-run starts. After two weeks of reading the log, one rule (`coverage_gap`) may be switched live. Auto-hide stays off.

## User Stories

1. As the site owner, I want the site to promote a stand-in model when I hide the production model, so that visitors still see a main pick.
2. As the site owner, I want the same to happen when I pause the production model on purpose, so that a deliberate pause is handled the same way as a hide.
3. As the site owner, I want the stand-in chosen by best season-to-date log loss, so that I do not have to pick a replacement mid-incident.
4. As the site owner, I want the stand-in chosen only among visible models that have predictions for today's games, so that a model with no data is never promoted.
5. As the site owner, I want the derived models (Elo and the betting market) excluded from stand-in selection, so that a baseline or the betting line is never presented as our own pick.
6. As the site owner, I want at least 50 graded games before log loss decides the stand-in, so that an early-season fluke does not choose it.
7. As the site owner, I want a fixed order (sim_blend, then logit3) used when there are too few graded games, so that a stand-in is still chosen predictably.
8. As the site owner, I want a stand-in never to be promoted unless I hid or paused the production model, so that nothing changes without a decision of mine.
9. As a site visitor, I want a notice saying which stand-in is in use and that the usual model is offline, so that I am not misled about whose prediction I am reading.
10. As the site owner, I want that notice to be automatic and impossible to turn off, so that a stand-in can never go live unexplained.
11. As the site owner, I want my own banner text to still appear alongside the automatic notice, so that I can add context.
12. As a site visitor, I want the games list to keep working while the production model is hidden, so that the site is still useful.
13. As the site owner, I want the default model view to fall to the stand-in when the production model is hidden, so that the page does not open on an empty column.
14. As the site owner, I want the stand-in re-chosen when the control state changes, so that a change takes effect within about a minute, like every other control change.
15. As the site owner, I want the stand-in choice to be stable between control changes and scoreboard refreshes, so that the headline does not flip from one page load to the next.
16. As the site owner, I want to see which model is the current stand-in in the lab, so that I know what visitors see without opening the site.
17. As the site owner, I want unhiding the production model to remove the stand-in immediately, so that recovery is one action.
18. As the site owner, I want a hidden model to stay hidden through a short control outage, so that a failed read does not reveal something I removed.
19. As the site owner, I want the last successfully read control state remembered for up to six hours after a failed read, so that brief BigQuery problems have no visible effect.
20. As the site owner, I want the site to fail open after six hours without a good read, so that a long outage cannot leave stale controls in force indefinitely.
21. As the site owner, I want a warning logged once per failure window and again when the remembered state expires, so that I can see an outage happened.
22. As the site owner, I want a change I make during an outage to be visible once reads recover, so that nothing I did is lost.
23. As the site owner, I want the control field renamed from `role` to `lifecycle`, so that "role" means only the registry's card type.
24. As the site owner, I want `lifecycle` documented as informational, so that nobody expects it to change behaviour by itself.
25. As the site owner, I want the rename applied to the control table's current-state view, the lab, the site backend and the ML pipeline together, so that no consumer reads a field that no longer exists.
26. As the site owner, I want the rename done while the control table is still empty, so that no existing setting needs migrating.
27. As the site owner, I want the lab to reject the old field name with a clear message after the rename, so that a stale script fails visibly instead of silently doing nothing.
28. As the site owner, I want a pin to keep requiring a fingerprint, so that an unverified file can never run in production.
29. As the site owner, I want a pinned artifact that fails its fingerprint check to stop the run and write nothing, so that the pipeline never guesses.
30. As the site owner, I want a critical "pin mismatch" finding on the dashboard when that happens, so that I learn about it without reading logs.
31. As the site owner, I want that finding to say which model and which artifact failed, so that I can act on it directly.
32. As the site owner, I want the finding never to trigger an automatic action, so that a mismatch is always handled by me.
33. As the site owner, I want the pipeline never to fall back silently after a mismatch, so that production never runs a model I did not choose.
34. As the site owner, I want the NFL and CFB functions redeployed with code only, so that pause and tier controls reach football.
35. As the site owner, I want that redeploy to leave the functions' environment settings untouched, so that the shadow writers keep running.
36. As the site owner, I want that redeploy to leave the scheduler jobs untouched, so that no job is deleted, recreated or unpaused by accident.
37. As the site owner, I want to verify after the redeploy that each function is active with unchanged settings, so that I know nothing drifted.
38. As the site owner, I want the first scheduled football run after the redeploy checked for control-related errors, so that a problem is caught within a day.
39. As the site owner, I want the lab's health timer installed on the home server, so that the dashboard and the self-healing log build every ten minutes.
40. As the site owner, I want read-only access granted to the lab's service account first, so that the dashboard shows real data without any ability to change anything.
41. As the site owner, I want the write-capable access held back until I switch a rule on, so that the lab cannot act before I decide it may.
42. As the site owner, I want every self-healing rule to start in dry-run, so that I can read what it would have done for two weeks.
43. As the site owner, I want only `coverage_gap` considered for going live afterwards, so that the first automatic action is the safest one.
44. As the site owner, I want auto-hide to stay off, so that a false alarm can never take a working model off the public site.
45. As the site owner, I want the dashboard to show the exact control command for each finding, so that fixing something is copy and paste.
46. As the site owner, I want the four branches pushed with one pull request per repository against the live deploy line, so that production code is not only on my laptop.
47. As the site owner, I want the health dashboard reachable from the home-server launcher, so that I can find it without remembering the address.
48. As a maintainer, I want the control contract and the glossary to describe the stand-in, the last-good state and `lifecycle`, so that a future reader gets the same vocabulary.

## Implementation Decisions

- **Stand-in selection.** The site backend chooses the stand-in at read time from the visible models. It reuses the live per-model scoreboard that already reports log loss with intervals. The choice uses the point estimate on games all candidate models were scored on, measured season to date, and requires at least 50 games; below that it uses the fixed order sim_blend then logit3. Only visible models with predictions for today's games are eligible, and the derived models are never eligible. Confidence intervals are not used to break ties.
- **When it applies.** Only when the production model has been hidden or paused by a deliberate control setting. The absence of production rows on its own does not trigger it.
- **What it changes.** The stand-in becomes the headline model and the default view on the compare and slate pages. The response carries an explicit stand-in marker naming the model and the reason, and the frontend renders an automatic notice from it. The notice is separate from, and shown alongside, any owner banner. It cannot be disabled by any control setting.
- **Caching.** The stand-in is derived from data that is already cached; it is recomputed when the control version changes or the scoreboard refreshes, and is otherwise stable. The existing control-version cache key and 60 second browser cap apply.
- **Last known good control state.** The backend's control reader keeps the last successful state and keeps applying it for up to six hours after failed reads, then reverts to failing open. It logs one warning per failure window and one when the remembered state expires. A successful read replaces it immediately. The ML pipeline is unchanged: it reads once per run and fails open.
- **Rename `role` to `lifecycle`.** The control field, its validation, the current-state view, the lab commands and documentation, the backend parser and the ML parser all change together in one coordinated release while the control table is empty. `lifecycle` accepts live, shadow and archived and is informational. The lab rejects the old field name with a message pointing to the new one. The registry's role is untouched. Whether a pin still depends on `lifecycle` is an open question below.
- **Pin mismatch finding.** A new critical health rule reports a pin mismatch, naming the model and artifact, with no remedy attached. The pipeline's behaviour stays as built: it fails the run and writes nothing on a mismatch, and never falls back after one. How the lab learns of the mismatch is an open question below.
- **Football redeploy.** Code-only redeploys of the NFL and CFB functions, keeping the live environment settings and leaving scheduler jobs alone, checked afterwards for active state and unchanged settings.
- **Lab install and access.** Install the new timer and dashboard route on the home server. Grant the lab's service account read-only access first; the write-capable tier is granted only when a rule is switched live. Self-healing starts in dry-run for every rule and runs for two weeks before any rule is considered.
- **Git.** Push the four branches and open one pull request per repository against the current deploy line.
- **Not to change.** Everything fails open. The lab remains the only writer. No public write endpoint is added. Plain text only for labels, notes and banners. Production changes need Henry's approval.

## Testing Decisions

- **What makes a good test.** It asserts external behaviour only: what a request returns, what a control command accepts or refuses, what a health rule reports for a given snapshot. It does not assert internal structure. Every new behaviour is tested both with control state present and with it empty or unavailable, because the empty case must behave exactly like before.
- **Seam 1: backend HTTP contract** for the model compare route and the slate routes, with the BigQuery read mocked and an injectable clock. Covers: stand-in chosen by log loss; fixed order below 50 games; eligibility rules; no stand-in without a deliberate hide or pause; the automatic notice present alongside an owner banner; stand-in removed when the model is unhidden; last-known-good behaviour across a failed read, at five hours and at seven hours; and the one-warning-per-window logging.
- **Seam 2: lab Control API with the DuckDB parity test** that runs the real current-state view SQL against random event streams and compares with the Python evaluation. Covers the rename to `lifecycle`, rejection of the old name, and that the view and the Python agree after the rename.
- **Seam 3: pure health rules over a snapshot.** Covers the pin-mismatch finding: critical, names model and artifact, no remedy, and no effect on other findings.
- **Football and install checks** are operational verifications, not automated tests: function state and settings after redeploy, first scheduled run's logs, dashboard rendering real data after the read grant.
- **Prior art.** The existing backend control-plane contract tests, the lab's control parity tests and the health-rule tests are the models to follow.

## Out of Scope

- A browser page that writes control settings. The CLI and notebooks are the only write path.
- Automatic hiding of any model, and the write-capable access tier, until Henry decides to go live on a rule.
- Any self-healing rule other than `coverage_gap` going live in this spec.
- Automatic promotion when the production model merely has no rows today.
- The features usage report (skipped by the user).
- Merging the deploy line into the main branch.
- Architecture decision records. None were requested.

## Open Questions

Six questions surfaced in the conversation and are unresolved; Henry chose not to dig into them before this spec was written.

1. **How does the lab learn about a pin mismatch?** The check runs inside the Cloud Function and raises. The lab needs a signal: a status the function writes, a log query, or the run's result. Answerable by Henry, with input from the ML pipeline code.
2. **Should a stand-in take the headline when the production model is paused (not hidden)?** A paused model keeps showing what it already has, so it is still visible. Answerable by Henry.
3. **Does the stand-in rule apply to football (`xgb` in NFL and CFB)?** The decision was stated for V10 only, and it is not confirmed that a live log loss exists for the football models. Answerable by Henry and a check of the football scoreboard.
4. **Do the legacy prediction endpoints also serve stand-in predictions,** or only the compare and slate pages? Today they return the games with predictions nulled and marked hidden. Answerable by Henry.
5. **Should a pin still depend on `lifecycle` being live?** Today it does, under the old name. The deployed backend and ML read the old field, so the rename needs a coordinated release. Answerable by Henry.
6. **What evidence ends the two-week dry-run?** No criterion for switching `coverage_gap` live has been set. Answerable by Henry after reading the first week of the log.

## Further Notes

- **State at writing (2026-09-29, evening).** Deployed: the control dataset and `model_sha256` column, backend version control-2 at full traffic (rollback: the previous live version), the frontend and the MLB pipeline function. Football redeploy commands were prepared for the same night; the result was not known when this was written.
- **Uncommitted.** Two flags were added to the football deploy scripts to support code-only redeploys (function only, keep environment). They are uncommitted in the ML worktree. Also unpushed: all four branches.
- **Local-only.** The launcher card for the dashboard exists locally in a directory that is not a git repository and is not deployed.
- **Owner actions.** Granting the read-only access and running the home-server installer are Henry's to run. The dry-run clock starts at install.
- **Numbers to remember.** Tests at the last full run: lab 473, backend 320, frontend 258, ML 411.
