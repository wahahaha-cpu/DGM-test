import unittest
import numpy as np
from scale_tta import SCALEConfig, SCALEFilter

class TestSCALE(unittest.TestCase):
    def test_weights_normalize_and_move(self):
        cfg = SCALEConfig(latent_dim=4, n_particles=9)
        f = SCALEFilter(np.full(4, 0.5), cfg, seed=0)
        before = f.posterior_mean.copy()
        p = lambda z, a: float(np.clip(z[0], 0.01, 0.99))
        out = f.update(1, None, p)
        self.assertAlmostEqual(float(np.sum(f.weights)), 1.0, places=9)
        self.assertEqual(len(out["weights"]), 9)
        self.assertGreaterEqual(out["effective_particles"], 1.0)
        self.assertFalse(np.allclose(before, f.posterior_mean))

    def test_action_selection_prefers_event(self):
        cfg = SCALEConfig(latent_dim=2, n_particles=5)
        f = SCALEFilter(np.array([0.5, 0.5]), cfg, seed=1)
        action, info = f.choose_action([0, 1], lambda z, a: 0.9 if a == 1 else 0.1)
        self.assertEqual(action, 1)
        self.assertEqual(info["selected_index"], 1)

    def test_probe_prefers_disagreement(self):
        cfg = SCALEConfig(latent_dim=2, n_particles=5, risk_aversion=0.0)
        f = SCALEFilter(np.array([0.5, 0.5]), cfg, seed=1)

        def event(z, action):
            if action == "uninformative":
                return 0.5
            return 0.05 if z[0] < 0.5 else 0.95

        gain_flat, _ = f.predictive_information_gain("uninformative", event)
        gain_split, _ = f.predictive_information_gain("split", event)
        self.assertLess(gain_flat, 1e-8)
        self.assertGreater(gain_split, gain_flat)
        action, info = f.choose_probe_action(
            ["uninformative", "split"], event, information_weight=1.0
        )
        self.assertEqual(action, "split")
        self.assertEqual(info["selected_index"], 1)

    def test_probe_respects_safety_gate(self):
        cfg = SCALEConfig(latent_dim=2, n_particles=5, risk_aversion=0.0)
        f = SCALEFilter(np.array([0.5, 0.5]), cfg, seed=2)
        event = lambda z, a: 0.5
        safe = lambda z, a: 0.1 if a == "danger" else 1.0
        action, info = f.choose_probe_action(
            ["danger", "safe"], event, predict_safe=safe,
            information_weight=1.0, min_safety=0.9
        )
        self.assertEqual(action, "safe")
        self.assertFalse(info["candidates"][0]["feasible"])

if __name__ == "__main__":
    unittest.main()
