#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scale_tta import QAdaptConfig, QAdaptFilter
from scale_tta.toy import ToyPush, model_event

ACTIONS = (-0.15, 0.05, 0.15, 0.25, 0.35, 0.45)


def latent_table() -> np.ndarray:
    values = [0.20, 0.45, 0.70, 0.95]
    return np.asarray([[value] + [0.55] * 7 for value in values], dtype=np.float64)


def run_episode(rng: np.random.Generator, friction: float, horizon: int = 8, support_steps: int = 2) -> int:
    env = ToyPush(friction=friction, horizon=horizon)
    state = env.reset(rng)
    qadapt = None

    for step in range(horizon):
        predict = lambda z, action: model_event(z, state, action)
        actions = ACTIONS
        if qadapt is None:
            qadapt = QAdaptFilter.from_latent_table(
                latent_table(),
                actions,
                predict,
                QAdaptConfig(
                    signature_threshold=0.03,
                    progress_threshold=0.15,
                    credible_mass=0.90,
                    beta=4.0,
                ),
            )

        if step < support_steps:
            action, _ = qadapt.choose_probe_action(actions, predict)
        else:
            action, _ = qadapt.choose_robust_action(actions, predict)

        new_state, event = env.step(action)
        qadapt.update(event, action, predict)
        state = new_state
        if env.success():
            return 1

    return env.success()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="runs/qadapt_toy_results.json")
    args = parser.parse_args()

    values = []
    for episode in range(args.episodes):
        friction = float(np.random.default_rng(args.seed + episode).choice(
            [0.20, 0.45, 0.70, 0.95]
        ))
        episode_rng = np.random.default_rng(args.seed + 100000 * (episode + 1))
        values.append(run_episode(episode_rng, friction))

    result = {"qadapt": {"success_rate": float(np.mean(values)), "n": len(values)}}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

