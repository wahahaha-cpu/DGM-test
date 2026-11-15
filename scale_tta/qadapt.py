"""Task-conditioned quotient adaptation for a frozen latent world model.

This is a small NumPy reference implementation. It deliberately operates on
behavioral classes of context latents rather than assuming that Euclidean
distance in the raw latent coordinate system is meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class QAdaptConfig:
    signature_threshold: float = 0.10
    progress_threshold: float = 0.0
    safety_threshold: float = 0.0
    credible_mass: float = 0.90
    beta: float = 4.0
    epsilon_event: float = 0.05


def cluster_signature_rows(signatures: Array, threshold: float) -> Array:
    """Greedy deterministic clustering in task-outcome signature space."""
    signatures = np.asarray(signatures, dtype=np.float64)
    if signatures.ndim != 2 or signatures.shape[0] == 0:
        raise ValueError("signatures must have shape [n_latents, n_actions]")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    centroids = []
    labels = []
    counts = []
    for row in signatures:
        if not centroids:
            centroids.append(row.copy())
            counts.append(1)
            labels.append(0)
            continue
        distances = np.max(np.abs(np.asarray(centroids) - row[None, :]), axis=1)
        idx = int(np.argmin(distances))
        if distances[idx] <= threshold:
            labels.append(idx)
            counts[idx] += 1
            centroids[idx] += (row - centroids[idx]) / counts[idx]
        else:
            labels.append(len(centroids))
            centroids.append(row.copy())
            counts.append(1)
    return np.asarray(labels, dtype=np.int64)


class QAdaptFilter:
    """Posterior over task-conditioned behavioral quotient classes."""

    def __init__(
        self,
        class_latents: Array,
        class_signatures: Optional[Array] = None,
        config: QAdaptConfig = QAdaptConfig(),
    ):
        latents = np.asarray(class_latents, dtype=np.float64)
        if latents.ndim != 2 or latents.shape[0] == 0:
            raise ValueError("class_latents must have shape [n_classes, latent_dim]")
        self.class_latents = latents
        self.config = config
        self.weights = np.full(latents.shape[0], 1.0 / latents.shape[0])
        self.class_signatures = None if class_signatures is None else np.asarray(
            class_signatures, dtype=np.float64
        )
        if self.class_signatures is not None and self.class_signatures.shape[0] != latents.shape[0]:
            raise ValueError("class_signatures and class_latents must have equal rows")

    @classmethod
    def from_latent_table(
        cls,
        latent_table: Array,
        actions: Sequence[object],
        predict_event: Callable[[Array, object], float],
        config: QAdaptConfig = QAdaptConfig(),
    ) -> "QAdaptFilter":
        """Build quotient classes from frozen T2-z environment latents."""
        table = np.asarray(latent_table, dtype=np.float64)
        signatures = np.asarray(
            [[predict_event(z, action) for action in actions] for z in table],
            dtype=np.float64,
        )
        labels = cluster_signature_rows(signatures, config.signature_threshold)
        classes = []
        class_sigs = []
        for label in range(int(labels.max()) + 1):
            mask = labels == label
            classes.append(np.mean(table[mask], axis=0))
            class_sigs.append(np.mean(signatures[mask], axis=0))
        return cls(np.asarray(classes), np.asarray(class_sigs), config)

    def _probabilities(
        self,
        action,
        predict_event: Callable[[Array, object], float],
    ) -> Array:
        probs = np.asarray(
            [predict_event(z, action) for z in self.class_latents],
            dtype=np.float64,
        )
        return np.clip(probs, 0.0, 1.0)

    def _credible_indices(self) -> Array:
        order = np.argsort(-self.weights)
        cumulative = np.cumsum(self.weights[order])
        n = int(np.searchsorted(cumulative, self.config.credible_mass) + 1)
        return order[:n]

    def choose_probe_action(
        self,
        actions: Sequence[object],
        predict_event: Callable[[Array, object], float],
        predict_safe: Optional[Callable[[Array, object], float]] = None,
    ) -> Tuple[object, dict]:
        """Maximize class separation subject to progress and safety."""
        scores = []
        details = []
        credible = self._credible_indices()
        for action in actions:
            p = self._probabilities(action, predict_event)
            mean = float(self.weights @ p)
            disagreement = float(
                np.sum(
                    self.weights[:, None]
                    * self.weights[None, :]
                    * np.abs(p[:, None] - p[None, :])
                )
            )
            safe_probability = 1.0
            if predict_safe is not None:
                safe = np.clip(
                    np.asarray(
                        [predict_safe(z, action) for z in self.class_latents],
                        dtype=np.float64,
                    ),
                    0.0,
                    1.0,
                )
                safe_probability = float(self.weights @ safe)
            feasible = mean >= self.config.progress_threshold
            if predict_safe is not None:
                feasible = feasible and float(np.min(
                    [safe_probability, *[
                        float(predict_safe(self.class_latents[i], action))
                        for i in credible
                    ]]
                )) >= self.config.safety_threshold
            score = disagreement if feasible else -np.inf
            scores.append(score)
            details.append({
                "action": action,
                "mean_progress": mean,
                "disagreement": disagreement,
                "safe_probability": safe_probability,
                "feasible": feasible,
                "score": float(score),
                "class_progress": p.tolist(),
            })
        if not np.isfinite(scores).any():
            raise ValueError("no probe action satisfies progress/safety constraints")
        index = int(np.argmax(scores))
        return actions[index], {"selected_index": index, "candidates": details}

    def choose_robust_action(
        self,
        actions: Sequence[object],
        predict_event: Callable[[Array, object], float],
    ) -> Tuple[object, dict]:
        """Choose the action with the best worst-case credible-class progress."""
        credible = self._credible_indices()
        scores = []
        details = []
        for action in actions:
            p = self._probabilities(action, predict_event)
            score = float(np.min(p[credible]))
            scores.append(score)
            details.append({
                "action": action,
                "credible_min_progress": score,
                "class_progress": p.tolist(),
            })
        index = int(np.argmax(scores))
        return actions[index], {"selected_index": index, "candidates": details}

    def update(
        self,
        observed_event: int,
        action,
        predict_event: Callable[[Array, object], float],
    ) -> dict:
        y = int(observed_event)
        if y not in (0, 1):
            raise ValueError("observed_event must be binary")
        p = np.clip(
            self._probabilities(action, predict_event),
            self.config.epsilon_event,
            1.0 - self.config.epsilon_event,
        )
        likelihood = p if y else (1.0 - p)
        new_weights = self.weights * np.power(likelihood, self.config.beta)
        new_weights /= max(float(np.sum(new_weights)), 1e-300)
        self.weights = new_weights
        return {
            "observed_event": y,
            "weights": self.weights.tolist(),
            "credible_classes": self._credible_indices().tolist(),
            "effective_classes": float(1.0 / np.sum(self.weights ** 2)),
        }

