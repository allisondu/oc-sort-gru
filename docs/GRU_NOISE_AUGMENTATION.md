# Detector-noise-augmented GRU training

**Hypothesis.** The clean-trained GRU is trained on DanceTrack ground-truth boxes. Inside OC-SORT it receives YOLOX detections instead. A GRU trained to predict the *clean* next box from *noisy* histories should be more reliable inside the tracker.

## What changes compared with the clean-trained ("standard") GRU

| | Standard GRU | Noise-augmented GRU |
|---|---|---|
| History input | GT boxes t−10 … t−1 | GT boxes with jitter added to each observation, resampled every epoch |
| Target | GT box at t, encoded relative to the last history box | GT box at t (clean), encoded relative to the **noisy** last history box |
| Missing frames | none (standard mode) | none (masks unchanged) |
| Normalization statistics | clean training windows | the same clean training windows (identical values) |
| Holdout losses | clean | clean **and** fixed-noise |
| `best.pt` selection | clean holdout loss | fixed-noise holdout loss |

Everything else is identical:

- architecture, feature version, history length
- train/holdout split (v2), epochs, batch size, optimizer, learning rate, Smooth-L1 β
- seed, which also gives the same initialization and the same batch order
- inference eligibility (full 10-frame history, previous frame observed)
- tracker integration

## Noise model (`gru_motion/augment.py`, preset `jitter_v1`)

For every observed history slot, independently:

```
cx' = cx + e_x * w      e_x ~ N(0, 0.025^2), truncated at ±3σ
cy' = cy + e_y * h      e_y ~ N(0, 0.025^2), truncated at ±3σ
w'  = w * exp(e_w)      e_w ~ N(0, 0.025^2), truncated at ±3σ
h'  = h * exp(e_h)      e_h ~ N(0, 0.025^2), truncated at ±3σ
```

- `w` and `h` are the clean sizes.
- Every training window is perturbed (`apply_prob = 1.0`), because at inference every history box is a detection.
- Truncation clips the standard-normal draw.

**These parameters are a modeling assumption, not fitted detector statistics.** No train-split YOLOX detections exist in this project; only the train ground truth was downloaded, and producing detections would require the full train image archive. Detections on the official validation split exist, but that split is reserved for tracking evaluation and is not used to fit anything. The value 0.025 is the midpoint of an a priori 2–3% range. It is fixed for this experiment and must not be tuned on validation HOTA or IDF1.

## Reproducibility

- **Training noise:** `numpy.random.default_rng([train_noise_seed, 1, epoch])`. A fresh draw is made each epoch from the same clean windows, so the clean target stays paired with the same history.
- **Noisy holdout:** drawn once from `default_rng([holdout_noise_seed, 2])`, with `holdout_noise_seed = 542` by default and independent of the training seed. Every holdout window is perturbed. The SHA-256 of the noisy holdout features is stored in the checkpoint.
- **Metadata:** `training_conditions.augmentation` and `train_info.augmentation` record the configuration, seeds and noise realized in each epoch. `train_info.checkpoint.selection_metric` records the selection criterion.

## Command

```
python -m gru_motion.train --split_config configs/gru/dancetrack_split_v2.json \
    --training_mode standard --box_noise jitter_v1 --epochs 30 --batch_size 512 \
    --lr 1e-3 --seed 0 --device cuda --out_dir gru_runs/noise_augmented/standard_v1_seed0_a40
```

With no `--box_noise`, the training script reproduces the clean-trained pipeline bit for bit. This was verified against commit `5fae5e6`.
