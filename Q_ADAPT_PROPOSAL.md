# Q-Adapt: Task-Conditioned Quotient Adaptation for Frozen Latent World Models

## One-sentence idea

The robot does not need to identify friction, mass, or the true T2-z coordinate. It only needs to identify which hidden contexts produce different task consequences for the actions it can execute.

## Why SCALE-AP is not the final idea

A Euclidean particle cloud around z0 assumes that nearby latent coordinates have comparable physical meaning. T2-z latents are learned coordinates and have no guaranteed gauge or local metric. A generic entropy bonus also makes the method look like Bayesian active exploration. Q-Adapt removes both assumptions.

## Point 1: Task-Conditioned Context Quotient

Given a frozen world model, the current observation, goal, and a candidate action bank A, define the behavioral signature of a context latent:

\[
\Phi_{o,g}(z)=
[p_1(z),p_2(z),\ldots,p_M(z)],
\]

\[
p_m(z)=P(\text{task progress}\mid F_\theta(o,a_m,z),g).
\]

Two latent contexts are equivalent for the current task if their signatures are close:

\[
z\sim_{o,g}z'
\Longleftrightarrow
\|\Phi_{o,g}(z)-\Phi_{o,g}(z')\|_\infty\leq\epsilon_\Phi.
\]

This is a finite-horizon, task-conditioned behavioral quotient. It is deliberately weaker than recovering physical parameters and weaker than exact MDP bisimulation. It keeps only distinctions that change the outcome of executable actions.

Candidate latents are taken from the frozen T2-z training environment table, then merged in signature space. No new encoder, predictor, or adaptation network is trained. The quotient is recomputed for the current observation and goal, so the same physical context may be equivalent for one task and distinguishable for another.

The posterior is maintained over quotient classes C_k rather than raw coordinates:

\[
w_k^{t+1}\propto
w_k^t
\bar p_k(a_t)^{y_t}
[1-\bar p_k(a_t)]^{1-y_t}.
\]

The key conceptual change is:

\[
\text{physical parameter estimation}
\quad\rightarrow\quad
\text{task-conditioned behavioral equivalence}.
\]

## Point 2: Progress-Preserving Quotient Refinement

A support action should split the remaining quotient classes, but only if it remains a useful and safe task action.

For action a, define class disagreement:

\[
D(a)=
\sum_{k,k'}w_kw_{k'}
\left|\bar p_k(a)-\bar p_{k'}(a)\right|.
\]

Let the credible set C_(1-alpha) contain the quotient classes with posterior mass 1-alpha. Select the next probe by:

\[
a_{\mathrm{probe}}
=
\arg\max_{a\in\mathcal A_t}D(a)
\]

subject to:

\[
\sum_kw_k\bar p_k(a)\geq\tau_{\mathrm{progress}},
\]

\[
\min_{k\in C_{1-\alpha}}s_k(a)\geq\tau_{\mathrm{safe}}.
\]

This is not an information-gain bonus. It is a constrained partition-refinement problem: maximize the separation of remaining behavioral explanations while preserving a task margin.

After the support phase, exploit with a credible-set robust action:

\[
a_{\mathrm{robust}}
=
\arg\max_{a\in\mathcal A_t}
\min_{k\in C_{1-\alpha}}\bar p_k(a).
\]

Thus the same quotient supports both adaptation and control:

- support: split the surviving explanations;
- exploitation: choose an action that works across the surviving explanations.

## Main algorithm

1. Load the frozen T2-z environment latent table and LeWM/T2-z predictor.
2. Generate the native candidate action bank from the existing planner.
3. For every stored latent and every candidate action, render the frozen predicted rollout.
4. Convert rollouts to a deterministic task-progress probability using geometry and the goal.
5. Cluster latent rows by their action-outcome signatures.
6. Initialize a uniform posterior over quotient classes.
7. For K support interactions:
   - select a progress-preserving refinement action;
   - execute it;
   - obtain a binary task event;
   - update quotient-class weights.
8. During query control, use the credible-set robust action.
9. Never update world-model parameters and never read query outcomes before adaptation ends.

## Minimal decisive experiment

Use the existing frozen T2-z + LeWM PushT/FetchSlide interface.

Methods:

1. Native frozen planner.
2. Original Mean-z/T2-z adaptation.
3. SCALE-AP baseline.
4. Q-Adapt without refinement.
5. Full Q-Adapt.

Run K=0,1,2 on one unseen friction-payload combination first. Use identical action candidates, seeds, chunk length, frameskip, event detector, and query episodes.

The decisive result is:

- Q-Adapt > Q-Adapt without refinement: Point 2 works.
- Q-Adapt without refinement > Mean-z: Point 1 works.
- Q-Adapt > SCALE-AP: quotient representation is better than Euclidean particles.
- success improves at K=1/2 while MSE is not used as the primary metric.

## What would count as a real contribution

The paper should claim that hidden physical adaptation can be performed in a task-conditioned behavioral quotient of a frozen world model. It should not claim generic information gain, generic Bayesian filtering, or recovery of physical parameters.

A convincing result requires:

- raw latent distance is a poor predictor of action-outcome similarity;
- signature-space classes are more stable across tasks;
- Q-Adapt reduces interaction count on unseen combinations;
- the progress constraint prevents the information-only failure mode;
- the gain is measured by task success and survives a new episode order or collection day.

