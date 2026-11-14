# SCALE: Success-Conditioned Adaptive Latent Ensemble

SCALE is a training-free test-time adaptation wrapper for a frozen latent world model. It does not update the world model, does not fit a residual adapter, and does not use latent MSE as the deployment objective.

The core idea is to maintain a small deterministic ensemble of environment-context hypotheses `z_i`. For each candidate action chunk, the frozen model predicts a task event (progress/success) under every `z_i`. After executing one chunk, the real robot returns a binary task event. The ensemble weights are updated by a KL-constrained exponential-weights step:

```text
w_next = argmin_w <w, event_loss> + (1/beta) KL(w || w_prev)
       = normalize(w_prev * exp(-beta * event_loss))
```

The next action is selected using posterior expected task success with a risk penalty. The method is training-free after the base world model and task event head are available.

## Why this is a distinct hypothesis

Recent test-time world-model methods adapt prediction residuals or update a latent predictor online. SCALE instead treats the context latent as a posterior over task-relevant dynamics hypotheses and uses action outcomes as event likelihoods. The primary metric is task success rate; prediction MSE is optional diagnostics only.

## Local smoke test

The current environment has no CUDA device and no PyTorch installation, so the first verification uses a NumPy toy dynamics model:

```bash
python3 -m pip install -e .
python3 scripts/run_toy.py --episodes 2000 --seed 7
python3 -m unittest discover -s scale_tta/tests -v
```

The same filter can wrap the existing T2-z/LeWM predictor by implementing the `predict_event` callback in `scale_tta/core.py`.
