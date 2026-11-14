"""A small hidden-friction pushing task for CPU verification."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class ToyPush:
    friction: float
    noise: float = 0.015
    horizon: int = 8
    goal: float = 1.0
    success_radius: float = 0.10

    def reset(self, rng: np.random.Generator) -> float:
        self.x = 0.0
        self.rng = rng
        return self.x

    def step(self, action: float) -> Tuple[float, int]:
        # Higher friction makes the same command less effective.
        displacement = (1.15 - self.friction) * float(action)
        displacement += float(self.rng.normal(0.0, self.noise))
        old_err = abs(self.goal - self.x)
        self.x = float(np.clip(self.x + displacement, -0.5, 1.5))
        progress = int(abs(self.goal - self.x) < old_err)
        return self.x, progress

    def success(self) -> int:
        return int(abs(self.goal - self.x) <= self.success_radius)


def model_event(z: np.ndarray, state: float, action: float, goal: float = 1.0) -> float:
    """Frozen event model. z[0] encodes effective friction, other dims are nuisance."""
    friction = float(np.clip(z[0], 0.05, 1.05))
    pred_x = state + (1.15 - friction) * float(action)
    old_err = abs(goal - state)
    new_err = abs(goal - pred_x)
    # Smooth Bernoulli approximation of progress.
    margin = old_err - new_err
    return float(1.0 / (1.0 + np.exp(-25.0 * margin)))
