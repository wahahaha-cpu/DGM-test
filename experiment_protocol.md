# SCALE hardware protocol

## Core claim
A frozen T2-z + LeWM controller can adapt to hidden dynamics shifts by maintaining a task-success posterior over a small latent ensemble. The method must improve real task success at equal interaction budget, with no parameter updates and no query-target access.

## Method variants
1. Frozen base controller.
2. Mean-z T2-z adaptation.
3. SCALE without risk penalty.
4. Full SCALE.
5. Oracle context (diagnostic upper bound only; never a deployable baseline).

Do not add VLM, residual adapter, JEPA-TTT, DTM, flow steering, or RL in this experiment.

## Task
Use planar pushing with one object, fixed camera, and randomized start/goal. Vary table surface friction and object payload independently. The visual appearance of the object must remain unchanged across payload conditions.

## Data split
- 3 training surface/payload combinations for calibration of the frozen base model.
- 3 ID test combinations.
- 3 unseen friction-payload combinations.
- 3 extrapolation combinations outside the training range.

For each combination collect 15 independent episodes: 5 support-pool episodes, 5 query episodes, 5 repeatability episodes.

## Test-time loop
1. Initialize the fixed latent center from the existing T2-z protocol.
2. Create 17 deterministic particles: center, antipodal coordinate probes, and fixed orthogonal/random directions.
3. For each action chunk, roll out the frozen world model under each particle and compute the predicted binary task event (progress toward goal or success margin).
4. Execute the action selected by posterior expected event probability minus risk variance.
5. Observe only the real binary event from the executed chunk; update particle weights by the KL-constrained exponential update.
6. Never read query trajectories, query success, or query loss before adaptation ends.

## Primary metric
Paired task success rate over the same episode order and seeds. Report bootstrap confidence intervals over environments and episodes. MSE is diagnostics only and cannot be used as the main improvement claim.

## Minimum publishable result
The full SCALE method must beat mean-z adaptation at K=1 or K=2 support episodes on an unseen friction-payload combination, with a positive paired 95% CI lower bound and no increase in catastrophic contact failures. Repeat the result with a second episode order or collection day.
