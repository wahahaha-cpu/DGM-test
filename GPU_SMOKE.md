# 8-GB GPU smoke test

The current executor has no `/dev/nvidia*`, `nvidia-smi`, or PyTorch installation, so it cannot execute this stage here. Run the following on the RTX 5070/8-GB host after the T2-z environment and checkpoint are mounted.

## Stage A: SCALE implementation check

```bash
python3 -m unittest discover -s scale_tta/tests -v
python3 scripts/run_toy.py --episodes 2000 --seed 7
```

## Stage B: one frozen T2-z rollout

Use one `sim_pushbox_friction` initial frame, one action chunk, and three latent hypotheses: center plus two antipodal probes in the first PCA direction of the training context table. Keep the world model frozen. Render only the predicted task event from each candidate video.

Recommended memory controls:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Start with the existing inference configuration and reduce only the inference step count for the smoke test. Record peak allocated memory, wall time, and the three event probabilities. Do not claim a success improvement from this smoke test.

## Stage C: minimal task-success comparison

Run the frozen baseline, mean-context T2-z, and SCALE on 5 episodes per held-out friction condition. Use identical candidate action banks, seeds, and control frequency. The gate to expand is:

- no OOM;
- finite generated videos;
- SCALE event posterior changes after an observed support event;
- at least one held-out condition has a nonzero success count.

Only after this gate should the full K=0/1/2/5 protocol be launched.
