# Policy-Agreement Adaptation: Adapt Only When Hidden Physics Changes the Action

## Core principle

A hidden physical parameter matters only when it changes the action that should be taken for the current task. If two physical contexts produce the same epsilon-optimal action set, distinguishing them is unnecessary.

The method therefore does not estimate friction, payload, or a numerically correct T2-z latent. It maintains a credible set of task policies induced by frozen latent-world-model hypotheses.

## Problem formulation

Let z be a frozen T2-z environment context, and let A(o,g) be the native candidate action bank. The frozen LeWM/T2-z rollout and a deterministic task-progress evaluator define:

\[
J_z(a)=P(\text{task success}\mid o,g,a,z).
\]

The epsilon-optimal action set for context z is:

\[
\mathcal A_\epsilon(z)=
\{a\in\mathcal A:
J_z(a)\geq \max_{a'\in\mathcal A}J_z(a')-\epsilon\}.
\]

Define goal-conditioned policy equivalence:

\[
z\sim_g z'
\Longleftrightarrow
\mathcal A_\epsilon(z)=\mathcal A_\epsilon(z').
\]

The controller only needs the quotient classes induced by this relation. Contexts that differ physically but induce the same action set are merged permanently for the current task.

## Point 1: Policy-Agreement Quotient

Use the frozen T2-z environment latent table as a set of valid context hypotheses. For each hypothesis, evaluate the native action bank and compute its epsilon-optimal action set.

Maintain a credible hypothesis set H_t, or equivalently the collection of distinct action sets:

\[
\Pi_t=\{\mathcal A_\epsilon(z):z\in H_t\}.
\]

The agreement set is:

\[
\mathcal A_{\mathrm{agree}}(H_t)
=
\bigcap_{z\in H_t}\mathcal A_\epsilon(z).
\]

If this intersection is non-empty, execute an action in the intersection. No latent update, probe, physical identification, or extra interaction is needed.

This is the first contribution:

> Hidden dynamics are compressed into policy disagreement, the only part of physical uncertainty that can change control.

## Point 2: Disagreement-Triggered Intervention

Adaptation is event-triggered rather than support-budget-triggered.

If:

\[
\mathcal A_{\mathrm{agree}}(H_t)\neq\varnothing,
\]

execute a common action and continue.

If:

\[
\mathcal A_{\mathrm{agree}}(H_t)=\varnothing,
\]

the current physical uncertainty is control-relevant. Select an intervention action that separates competing policy classes while remaining viable:

\[
a_{\mathrm{int}}
=
\arg\max_{a\in\mathcal A}
\operatorname{Split}(a,H_t)
\]

subject to:

\[
\min_{z\in H_t}J_z(a)\geq \tau_{\mathrm{progress}},
\]

\[
\min_{z\in H_t}S_z(a)\geq \tau_{\mathrm{safe}}.
\]

The split score measures how differently the competing contexts predict the binary task event:

\[
\operatorname{Split}(a,H_t)
=
\sum_{z,z'\in H_t}
w_zw_{z'}
\left|
J_z(a)-J_{z'}(a)
\right|.
\]

After observing the event y, remove hypotheses inconsistent with the observed task outcome:

\[
H_{t+1}
=
\left\{
z\in H_t:
\left|J_z(a_{\mathrm{int}})-y\right|\leq \epsilon_{\mathrm{obs}}
\right\}.
\]

For noisy events, replace hard removal with a clipped likelihood update. The main method should use set-membership filtering; probabilities are optional robustness variants.

After the disagreement disappears, return to ordinary robust control:

\[
a_{\mathrm{robust}}
=
\arg\max_{a\in\mathcal A}
\min_{z\in H_t}J_z(a).
\]

The method has no fixed K. It adapts only when adaptation is necessary.

## Why this is more fundamental

The original SCALE-AP asks whether a latent hypothesis is likely. Q-Adapt asks whether two latent hypotheses have similar action-outcome signatures. Policy-Agreement Adaptation asks a more fundamental question:

\[
\text{Does the hidden uncertainty change the action?}
\]

This is exactly aligned with the task-success objective.

It removes:

- raw latent Euclidean geometry;
- arbitrary latent perturbation radius;
- generic entropy weighting;
- a fixed number of support interactions;
- the requirement to identify physical parameters;
- the need to make the full physical belief converge.

The belief can remain broad forever if all surviving hypotheses agree on the action.

## Minimal decisive experiment

Use frozen T2-z + LeWM on one hidden-friction/payload task.

### Phase 1: policy-disagreement audit

For every training latent z and every native planner candidate:

1. compute J_z(a);
2. construct A_epsilon(z);
3. measure pairwise Jaccard overlap of action sets;
4. compare raw latent distance with policy-set disagreement.

This tests whether latent uncertainty is actually control-relevant.

### Phase 2: four methods

1. Frozen native planner.
2. Mean-z T2-z test-time adaptation.
3. Q-Adapt robust quotient without event-triggered intervention.
4. Full Policy-Agreement Adaptation.

The full method must use the same frozen model, action bank, event detector, and query episodes.

### Phase 3: primary result

Measure:

- task success rate;
- number of interventions;
- number of actions before agreement;
- catastrophic contact failures;
- success per physical interaction.

MSE is diagnostic only.

The target result is not merely higher success. It is:

\[
\text{same or higher success}
+
\text{fewer unnecessary interactions}
+
\text{no adaptation when policies already agree}.
\]

## Research claim

The paper should claim:

> Test-time physical adaptation should be triggered by disagreement among task-optimal action sets, rather than by latent prediction error or uncertainty alone.

This creates a new distinction between:

- **physical uncertainty**, which may be irrelevant;
- **control-relevant uncertainty**, which changes the action;
- **disagreement-triggered adaptation**, which spends real interactions only on the latter.

