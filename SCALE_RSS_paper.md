# SCALE-AP: Success-Conditioned Latent Filtering with Active Probing for Training-Free Robot Adaptation

## Abstract

Robots can fail when friction, payload, or contact dynamics change while the scene and task instruction remain visually unchanged. A frozen latent world model already contains multiple plausible explanations of the observation, but standard deployment treats its environment latent as a point estimate. We propose SCALE-AP, a training-free adaptation method with two tightly coupled ideas. First, SCALE represents the opaque T2-z context as a posterior over a small deterministic ensemble of latent hypotheses and updates this posterior from binary task events observed after executed actions. The update is the closed-form solution of a KL-constrained mirror-descent problem. Second, Active Probing selects support actions by maximizing expected task progress and posterior information gain, penalized by event risk and constrained by a geometric safety margin. The method changes neither the LeWM/T2-z parameters nor the action policy weights. Its primary claim is task success under hidden dynamics shifts, with prediction MSE treated only as a diagnostic. The central experiment tests whether task-event filtering and task-compatible information seeking reduce the number of real interactions needed to adapt to unseen friction-payload combinations.

## 1. Problem

Let a frozen latent world model be

\[
\hat{o}_{t+1:t+H}=F_\theta(o_{t-L:t},a_{t:t+H-1},z),
\]

where \(z\) is an opaque environment context. The deployment environment has an unknown context \(z^\star\), but no physical parameter labels are available. Given a candidate action block \(a\), a fixed task-event function \(\mathcal E\) maps a predicted rollout to the probability that the action makes useful progress:

\[
p_i(a)=\mathcal E(F_\theta(o,a,z_i),g).
\]

The proposed method does not try to recover a globally correct physical parameter. It maintains only the uncertainty that changes the current task outcome. This is a deliberately narrower target than minimizing a world-model prediction loss.

## 2. Two-point method

### 2.1 Point 1: SCALE task-event latent filtering

Start from the existing T2-z latent \(z_0\). Create a center particle and deterministic antipodal probes:

\[
z_i=\Pi_{[0,1]^d}(z_0+\rho P_ru_i),
\]

where \(P_r\) is a fixed low-rank basis computed before evaluation and \(u_i\) is a fixed spherical code. The default is \(N=17\), \(r=8\). No proposal network is trained.

After executing an action, the robot obtains a binary event \(y\): useful progress or no progress/failure. The particle likelihood is

\[
\ell_i(y|a)=-y\log p_i(a)-(1-y)\log(1-p_i(a)).
\]

The next posterior is the solution to

\[
w^{t+1}=\arg\min_{w\in\Delta^N}
\sum_iw_i\ell_i(y_t|a_t)+\frac{1}{\beta}D_{\mathrm{KL}}(w\|w^t),
\]

which gives

\[
w_i^{t+1}=
\frac{w_i^t\exp[-\beta\ell_i(y_t|a_t)]}
{\sum_jw_j^t\exp[-\beta\ell_j(y_t|a_t)]}.
\]

The next exploitation action maximizes posterior expected task progress with a risk penalty:

\[
a_{\mathrm{exploit}}=\arg\max_a
\left[\mu(a)-\lambda\sigma^2(a)\right],
\]

\[
\mu(a)=\sum_iw_ip_i(a),\qquad
\sigma^2(a)=\sum_iw_i[p_i(a)-\mu(a)]^2.
\]

This is the first contribution: an environment latent is used as a task-conditioned hypothesis distribution, and real task events filter it without backpropagation.

### 2.2 Point 2: Active Probing

During the first \(K\) support interactions, action selection adds the expected information gain of the same binary event:

\[
\mathcal I(a)=H(w)-\sum_{y\in\{0,1\}}\bar p_y(a)H(w^{a,y}),
\]

where \(w^{a,y}\) is the posterior that would follow event \(y\), and \(\bar p_y(a)\) is its posterior predictive probability. The probe is selected by

\[
a_{\mathrm{probe}}=\arg\max_{a\in\mathcal A}
\mu(a)-\lambda\sigma^2(a)+\eta\mathcal I(a),
\]

subject to

\[
\sum_iw_is_i(a)\ge s_{\min}.
\]

Here \(s_i(a)\) is a deterministic geometric safety probability or a frozen model estimate. Information gain alone is intentionally a negative control: it may select a highly diagnostic but task-destructive contact. SCALE-AP asks for a useful, low-risk action whose outcome also separates the hypotheses.

The two points are inseparable at the conceptual level but independently testable:

- Passive SCALE versus Frozen/Mean-z tests whether task-event posterior filtering is useful.
- SCALE-AP versus Passive SCALE tests whether choosing the information-bearing action reduces the interaction budget.
- Information-only versus SCALE-AP tests whether the success and safety terms are necessary.

## 3. Relation to prior work

The method is positioned as a zero-training posterior-and-action-selection layer over an existing frozen world model.

- T2-z optimizes a context latent at test time; SCALE-AP does not run latent-gradient TTT. It keeps a small posterior and updates only probability weights from task events.
- Few-Shot Residual Adaptation learns self-supervised latent prediction residuals inside a pretrained world model. SCALE-AP leaves the predictor and its parameters unchanged.
- JEPA-TTT updates a latent dynamics predictor online using replay and prediction supervision. SCALE-AP has no online optimizer and uses only the event generated by the task.
- Feedback World Model maintains a learned feedback state and action-aware guidance. SCALE-AP instead exposes an explicit posterior over context hypotheses and adds a closed-form information-gain action rule.
- ASID and ensemble-based active exploration select informative motions for system identification or model learning. SCALE-AP does not identify mass/friction, retrain a simulator, or learn an exploration policy; its probe is task-compatible and operates on a frozen T2-z/LeWM action bank.
- Online latent physical-property adaptation updates a learned latent from prediction discrepancies and conditions a policy on that estimate. SCALE-AP uses no object-specific latent table at deployment and updates from task events, which makes the target weaker but the deployment procedure training-free.

The novelty claim therefore cannot be “information gain for robots” or “online latent adaptation” in isolation. The precise claim is the combination of (i) task-event Bayesian/mirror filtering over an opaque physical-context latent and (ii) a safety-constrained probe objective that selects actions for both task progress and posterior contraction, without updating the model.

## 4. Experimental design

Use planar pushing with a fixed RGB camera, randomized start and target, and independent friction and payload changes. The appearance must remain constant. Use deterministic geometric events from object-goal distance and terminal success; do not train an event head during evaluation.

Compare Frozen, the existing Mean-z T2-z adaptation, Random latent, Passive SCALE, Information-only, SCALE-AP, and an Oracle-z diagnostic. Evaluate \(K\in\{0,1,2,5\}\) support interactions on seen-ID, unseen-combination, and extrapolation conditions. Keep action candidates, episode order, control frequency, action chunk length, seeds, and event thresholds identical across methods.

The primary metric is paired task success. Report per-condition success, paired bootstrap confidence intervals, catastrophic contact failures, action blocks per success, and wall-clock cost. Latent MSE and prediction MSE can be plotted to diagnose failure modes, but cannot be used as the improvement claim.

The minimum result for an RSS submission is:

1. SCALE-AP exceeds Mean-z and Passive SCALE at \(K=1\) or \(K=2\) on an unseen friction-payload combination.
2. The paired 95% bootstrap interval has a positive lower bound and the direction repeats with a new episode order or collection day.
3. Catastrophic contact failures do not increase.
4. Information-only is weaker than SCALE-AP, establishing that the second point is not just generic exploration.
5. The full result is obtained with no parameter update and no query outcome access.

If these gates fail, the correct paper is a limitation study showing when task-event filtering cannot identify the relevant latent directions.

## 5. Reproducibility

The repository contains a NumPy implementation of the posterior filter and active-probe score, a frozen-model bridge, unit tests, a toy hidden-friction smoke test, an 8-GB GPU smoke protocol, and the full hardware experiment checklist. The external LeWM and T2-z repositories are not vendored into the prototype; the bridge accepts a project-specific callback that renders one frozen context and returns the deterministic task-event probability.

## References

- T2-z, arXiv:2610.22364, official implementation: https://github.com/T2-z-latent-optimization/T2-z
- LeWM, arXiv:2603.19312, official implementation: https://github.com/lucas-maes/le-wm
- Suliga et al., Few-Shot Residual Adaptation of Pretrained Latent World Models, arXiv:2611.13592.
- JEPA-TTT, Persistent Test-Time Training of Latent World Models for Planning under Dynamics Shifts, arXiv:2610.00722.
- An et al., Feedback World Model Enables Precise Guidance of Diffusion Policy, arXiv:2605.15705.
- Memmel et al., ASID: Active Exploration for System Identification in Robotic Manipulation, arXiv:2404.12308.
- Kajitsuka et al., Online Adaptation to Latent Physical Properties for Learning from Demonstration, arXiv:2611.06118.
