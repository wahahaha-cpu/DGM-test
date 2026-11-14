#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path
import numpy as np
from scale_tta import SCALEConfig, SCALEFilter
from scale_tta.toy import ToyPush, model_event

ACTIONS = (-0.15, 0.05, 0.15, 0.25, 0.35, 0.45)

def run_episode(rng, method: str, friction: float, horizon: int = 8):
    env = ToyPush(friction=friction, horizon=horizon)
    state = env.reset(rng)
    if method == "scale":
        cfg = SCALEConfig(latent_dim=8, n_particles=17, radius=0.22,
                          beta=5.0, risk_aversion=0.20)
        filt = SCALEFilter(np.full(8, 0.55), cfg, seed=int(rng.integers(2**31 - 1)))
    for _ in range(horizon):
        if method == "scale":
            action, _ = filt.choose_action(ACTIONS, lambda z, a: model_event(z, state, a))
        else:
            z = np.full(8, 0.55)
            action = max(ACTIONS, key=lambda a: model_event(z, state, a))
        new_state, event = env.step(action)
        if method == "scale":
            filt.update(event, action, lambda z, a: model_event(z, state, a))
        state = new_state
        if env.success():
            return 1
    return env.success()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="runs/toy_results.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    results = {}
    for method in ("mean", "scale"):
        vals = []
        for _ in range(args.episodes):
            friction = float(rng.choice([0.20, 0.45, 0.70, 0.95]))
            vals.append(run_episode(rng, method, friction))
        results[method] = {"success_rate": float(np.mean(vals)), "n": len(vals)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
