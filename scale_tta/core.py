"""Training-free Success-Conditioned Adaptive Latent Ensemble (SCALE)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple
import numpy as np

Array = np.ndarray

@dataclass(frozen=True)
class SCALEConfig:
    latent_dim: int
    n_particles: int = 17
    radius: float = 0.20
    beta: float = 4.0
    epsilon_event: float = 0.05
    risk_aversion: float = 0.15
    radius_growth_on_failure: float = 1.15
    radius_decay_on_success: float = 0.92
    min_radius: float = 0.02
    max_radius: float = 0.60

class SCALEFilter:
    """Particle posterior over an opaque context latent.

    `predict_event(z, action)` must return a Bernoulli probability or a binary
    event prediction in [0, 1]. It is called only on frozen model outputs.
    `action_score` uses the same callback and should return per-particle event
    probabilities for a candidate action.
    """
    def __init__(self, z0: Array, config: SCALEConfig, seed: int = 0,
                 basis: Optional[Array] = None):
        z0 = np.asarray(z0, dtype=np.float64)
        if z0.shape != (config.latent_dim,):
            raise ValueError(f"z0 must have shape ({config.latent_dim},)")
        if config.n_particles < 3:
            raise ValueError("n_particles must be at least 3")
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.z0 = z0.copy()
        if basis is None:
            self.basis = np.eye(config.latent_dim, dtype=np.float64)
        else:
            basis = np.asarray(basis, dtype=np.float64)
            if basis.ndim != 2 or basis.shape[0] != config.latent_dim:
                raise ValueError("basis must have shape [latent_dim, rank]")
            # QR makes the probe directions orthonormal without learning.
            self.basis, _ = np.linalg.qr(basis)
        self.radius = float(config.radius)
        self.particles = self._make_particles(z0, self.radius)
        self.weights = np.full(config.n_particles, 1.0 / config.n_particles)
        self.events_seen = 0

    def _make_particles(self, center: Array, radius: float) -> Array:
        """Create a center plus a deterministic spherical code.

        The code has no learned parameters. Directions are orthogonal when the
        particle budget permits; remaining directions are deterministic random
        unit vectors generated from the local seed.
        """
        n, d = self.cfg.n_particles, self.basis.shape[1]
        dirs = np.zeros((n, d), dtype=np.float64)
        dirs[0] = 0.0
        basis_count = min(n - 1, d)
        dirs[1:1 + basis_count, :basis_count] = np.eye(basis_count)
        for k in range(1 + basis_count, n):
            v = self.rng.normal(size=d)
            v /= max(np.linalg.norm(v), 1e-12)
            dirs[k] = v
        # Add antipodal directions when there is room, which makes the probe
        # locally symmetric around the current posterior center.
        for k in range(1, min(n, 1 + 2 * d), 2):
            if k + 1 < n:
                dirs[k + 1] = -dirs[k]
        return np.clip(center[None, :] + radius * (dirs @ self.basis.T), 0.0, 1.0)

    def event_probabilities(self, action, predict_event: Callable[[Array, object], float]) -> Array:
        probs = np.asarray([predict_event(z, action) for z in self.particles], dtype=np.float64)
        return np.clip(probs, 0.0, 1.0)

    def choose_action(
        self,
        actions: Sequence[object],
        predict_event: Callable[[Array, object], float],
    ) -> Tuple[object, dict]:
        """Choose action by posterior event probability minus weighted risk."""
        scores = []
        details = []
        for action in actions:
            p = self.event_probabilities(action, predict_event)
            mean = float(self.weights @ p)
            variance = float(self.weights @ (p - mean) ** 2)
            score = mean - self.cfg.risk_aversion * variance
            scores.append(score)
            details.append({"action": action, "mean_event": mean,
                            "variance": variance, "score": score,
                            "particle_event": p.tolist()})
        idx = int(np.argmax(scores))
        return actions[idx], {"selected_index": idx, "candidates": details}

    def update(self, observed_event: int, action, predict_event: Callable[[Array, object], float]) -> dict:
        """Bayesian/exponentiated update from one binary task event.

        This is the closed-form solution of a KL-constrained mirror-descent
        step on the particle simplex. No model parameters are updated.
        """
        y = int(observed_event)
        if y not in (0, 1):
            raise ValueError("observed_event must be binary")
        p = np.clip(self.event_probabilities(action, predict_event),
                    self.cfg.epsilon_event, 1.0 - self.cfg.epsilon_event)
        losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        logw = np.log(np.maximum(self.weights, 1e-300)) - self.cfg.beta * losses
        logw -= np.max(logw)
        new_weights = np.exp(logw)
        new_weights /= np.sum(new_weights)
        old_entropy = -float(np.sum(self.weights * np.log(np.maximum(self.weights, 1e-300))))
        self.weights = new_weights
        self.events_seen += 1
        if y:
            self.radius = max(self.cfg.min_radius, self.radius * self.cfg.radius_decay_on_success)
        else:
            self.radius = min(self.cfg.max_radius, self.radius * self.cfg.radius_growth_on_failure)
        posterior_z = self.weights @ self.particles
        return {
            "observed_event": y,
            "event_loss": losses.tolist(),
            "weights": self.weights.tolist(),
            "posterior_z": posterior_z.tolist(),
            "effective_particles": float(1.0 / np.sum(self.weights ** 2)),
            "entropy_before": old_entropy,
            "radius": self.radius,
        }

    @property
    def posterior_mean(self) -> Array:
        return self.weights @ self.particles
