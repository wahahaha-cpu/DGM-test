# Q-Adapt: Task-Conditioned Quotient Adaptation

The current research method is Q-Adapt. SCALE-AP remains implemented as a
baseline, not the final method.

Q-Adapt does not assume that Euclidean distance in the raw T2-z latent space
has physical meaning. It maps each frozen context latent to its task-outcome
signature over the native action bank, merges behaviorally equivalent latents,
and updates a posterior over quotient classes. During support, it selects an
action that separates the remaining classes subject to progress and safety
constraints. During exploitation, it selects the action with the best
worst-case progress over the credible classes.

~~~text
latent context -> task-outcome signature -> quotient classes
                -> progress-preserving refinement -> robust action
~~~

See Q_ADAPT_PROPOSAL.md for the research formulation and
Q_ADAPT_EXECUTION.md for runnable code and T2-z/LeWM integration.

Quick CPU check:

~~~bash
python3 -m pip install -e .
python3 -m unittest discover -s scale_tta/tests -v
python3 scripts/run_qadapt_toy.py --episodes 1000 --seed 7
~~~

## SCALE-AP baseline

SCALE-AP is a training-free test-time adaptation wrapper for a frozen latent world model. It has two points: SCALE filters a small posterior over latent context hypotheses from binary task events; Active Probing selects an early action that is useful and expected to separate those hypotheses. It does not update the world model, fit a residual adapter, or use latent MSE as the deployment objective.

The first point maintains a small deterministic ensemble of environment-context hypotheses z_i. For each candidate action chunk, the frozen model predicts a task event (progress/success) under every z_i. After executing one chunk, the real robot returns a binary task event. The ensemble weights are updated by a KL-constrained exponential-weights step:

~~~text
w_next = argmin_w <w, event_loss> + (1/beta) KL(w || w_prev)
       = normalize(w_prev * exp(-beta * event_loss))
~~~

The second point changes only the first K support actions. It scores each existing candidate action by posterior success, risk, and the expected entropy reduction of the same binary event:

~~~text
probe_score(a) = mean_event(a) - risk * variance(a)
                + information_weight * predictive_information_gain(a)
~~~

An optional geometric safety callback removes probes below a minimum safety probability. The method is training-free after the base world model and deterministic event interface are available.

## Why this is a distinct hypothesis

Recent test-time world-model methods adapt prediction residuals or update a latent predictor online. SCALE-AP instead treats the context latent as a posterior over task-relevant dynamics hypotheses and uses task-compatible action outcomes both to filter the posterior and to choose informative support actions. The primary metric is task success rate; prediction MSE is diagnostic only.

## Local smoke test

The current environment has no CUDA device and no PyTorch installation, so the first verification uses a NumPy toy dynamics model:

~~~bash
python3 -m pip install -e .
python3 scripts/run_toy.py --episodes 2000 --seed 7
python3 -m unittest discover -s scale_tta/tests -v
~~~

The same filter can wrap the existing T2-z/LeWM predictor by implementing the predict_event callback in scale_tta/core.py. The active-probe API is SCALEFilter.choose_probe_action(...); the full hardware sequence and acceptance gates are in experiment_protocol.md.

scale_tta/bridge.py contains the deliberately thin integration wrapper. The project-specific T2-z/Wan pipeline remains external; it supplies a callback that renders a candidate context and returns a deterministic task-event probability from the predicted video.

