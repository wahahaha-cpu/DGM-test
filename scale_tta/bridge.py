"""Minimal bridge from a frozen T2-z/LeWM predictor to SCALE.

The project-specific predictor is intentionally injected as a callback because
T2-z checkpoints differ in pipeline wrappers. A callback receives a candidate
context and an action chunk and must return a task-event probability. This keeps
SCALE training-free and avoids copying the 5B pipeline into this repository.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple
import numpy as np

from .core import SCALEFilter


class SCALEPlanner:
    def __init__(self, scale_filter: SCALEFilter,
                 predict_event: Callable[[np.ndarray, object], float]):
        self.filter = scale_filter
        self.predict_event = predict_event

    def select_and_update(self, actions: Sequence[object], observed_event: int) -> Tuple[object, dict]:
        action, selection = self.filter.choose_action(actions, self.predict_event)
        update = self.filter.update(observed_event, action, self.predict_event)
        return action, {"selection": selection, "update": update}

    def select_probe_and_update(
        self,
        actions: Sequence[object],
        observed_event: int,
        predict_safe: Callable[[np.ndarray, object], float] | None = None,
        information_weight: float = 0.25,
        min_safety: float = 0.0,
    ) -> Tuple[object, dict]:
        action, selection = self.filter.choose_probe_action(
            actions,
            self.predict_event,
            predict_safe=predict_safe,
            information_weight=information_weight,
            min_safety=min_safety,
        )
        update = self.filter.update(observed_event, action, self.predict_event)
        return action, {"selection": selection, "update": update}

    def context(self) -> np.ndarray:
        return self.filter.posterior_mean.copy()
