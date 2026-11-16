# Model control

How Henry changes what Hank's Tank shows and runs, and how model health is watched, without editing code or redeploying.

## Language

**Model control plane**:
The set of settings that decide which models the site shows and which the pipeline runs. Only the lab writes them; the site and the pipeline only read them.
_Avoid_: admin panel, feature flags

**Target**:
The specific model, or the whole sport, that a control setting applies to.
_Avoid_: model version, model name

**Hide**:
Remove a model from everything the site shows, while it keeps running and writing predictions.
_Avoid_: disable, delete, retire

**Pause**:
Stop the pipeline from producing predictions for a model, while the site keeps showing what it already has.
_Avoid_: stop, disable, kill

**Pin**:
Force production to run one specific saved copy of a model, identified by its content fingerprint, instead of the default.
_Avoid_: lock, freeze, fix

**Banner**:
A plain-text notice shown to site visitors for one sport, explaining a change.
_Avoid_: alert, announcement

**Headline model**:
The model whose predictions the site presents first and treats as its main pick.
_Avoid_: default model, featured model, primary

**Stand-in model**:
The model automatically promoted to headline while the production model is hidden or paused, chosen for the best log loss so far this season among the eligible models. The site always says when one is in use.
_Avoid_: fallback model, backup model, replacement

**Lifecycle**:
A label for where a model is in its life: live, shadow or archived. It is informational and changes nothing by itself. Distinct from a model's role on the site (production, shadow, reference, benchmark).
_Avoid_: role, status, stage

**Dry-run**:
A self-healing mode that records the action a rule would have taken without taking it.
_Avoid_: simulation, test mode

**Self-healing rule**:
A named health check that, once switched on, may take one specific bounded action, and nothing else, when it fires.
_Avoid_: auto-fix, automation
