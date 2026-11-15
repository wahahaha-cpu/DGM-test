# 8-GB GPU smoke test for SCALE-AP

The current executor has no /dev/nvidia*, nvidia-smi, or PyTorch installation, so it cannot execute this stage here. Run the following on the RTX 5070/8-GB host after the T2-z environment and checkpoint are mounted.

## Stage A: SCALE-AP implementation check

~~~bash
python3 -m unittest discover -s scale_tta/tests -v
python3 scripts/run_toy.py --episodes 2000 --seed 7
~~~

## Stage B: one frozen T2-z rollout

Use one sim_pushbox_friction initial frame, one action chunk, and three latent hypotheses: center plus two antipodal probes in the first PCA direction of the training context table. Keep the world model frozen. Render only the predicted task event from each candidate video, then compute predictive information gain for the candidate action bank.

Recommended memory controls:

~~~bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
~~~

Start with the existing inference configuration and reduce only the inference step count for the smoke test. Record peak allocated memory, wall time, the three event probabilities, information gain for each candidate action, and the selected probe. Do not claim a success improvement from this smoke test.

## Stage C: minimal task-success comparison

Run the frozen baseline, mean-context T2-z, Passive SCALE, and SCALE-AP on 5 episodes per held-out friction condition. Use identical candidate action banks, seeds, and control frequency. The gate to expand is:

- no OOM;
- finite generated videos;
- SCALE event posterior changes after an observed support event;
- the active-probe score returns a finite action and is higher for a separating action than for an uninformative action;
- at least one held-out condition has a nonzero success count.

Only after this gate should the full K=0/1/2/5 protocol in experiment_protocol.md be launched.
