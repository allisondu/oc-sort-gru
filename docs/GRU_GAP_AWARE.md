# Gap-aware, noise-augmented GRU

**Hypothesis.** A GRU trained with explicit trailing observation gaps, masks and detector-like noise can predict across short missed-detection intervals better than OC-SORT's Kalman filter. That should help re-association after 1–3 missed frames.

## Temporal indexing (tracker and training)

OC-SORT calls `predict()` for every track before association, and that call increments `trk.age` and `trk.time_since_update`. Observations are stored under the age at which they were matched. While associating frame t:

- **Window:** the GRU window covers slots t−10 … t−1 (`history.track_window`). The target is the box at t.
- **Missing slots:** the trailing missing slots are the consecutive missed frames before t; this is the gap g.
- **Frames since the last observation:** `time_since_update = g + 1`. This is the prediction horizon.
- **Valid observations:** the number of genuine detections in the window.

| missed slots g | last observation | target | horizon = `time_since_update` |
|---|---|---|---|
| 0 | t−1 | t | 1 |
| 1 | t−2 | t | 2 |
| 2 | t−3 | t | 3 |
| 3 | t−4 | t | 4 |

**What counts as an observation:** only matched detections, plus the birth detection. Kalman predictions, ORU's virtual boxes and unmatched steps never enter the window.

## Features (unchanged, `FEATURE_VERSION` 1)

- **Reference box:** the last *valid* slot. The features already defined it this way for any trailing gap.
- **Missing slots:** encoded as zeros with mask 0. Slot positions are fixed, so there is no temporal compression.
- **Horizon:** the model can infer it from the number of trailing zero-mask slots. A static object differs from a missing one through the mask channel.
- **Target:** `[(cx′−cx_r)/h_r, (cy′−cy_r)/h_r, log(w′/w_r), log(h′/h_r)]`, computed relative to the last observed (noisy) box.
- **Model:** unchanged (GRU 1×64, 5 inputs, 4 outputs), with direct prediction and no autoregression.

## Training (`--training_mode gap_aware --max_trailing_gap 3 --box_noise jitter_v1`)

- **Source windows:** the same standard windows as before, from split v2 (target plus 10 observed history frames).
- **Gap assignment:** each epoch, every source window gets one gap g ∈ {0, 1, 2, 3}, exactly balanced. Assignment comes from `default_rng([seed, 3, epoch])`.
  - The last g slots are masked; every earlier slot stays observed (the "full prefix" rule).
  - Noise (unchanged `jitter_v1`) is applied to the observed slots only. The target stays clean.
- **Same per-epoch budget as before:** one example per source window, so 297,599 examples per epoch.
- **Normalization and the clean `eval_train` loss:** both use one fixed balanced gap assignment (`default_rng([seed, 4])`) of the clean training windows. This is a necessary difference from the earlier models, because the target distribution now includes horizons 2–4.
- **Holdout:** every holdout window is evaluated at every gap, both clean and with the fixed noise realization (seed 542). The noise is drawn once on the full windows and then masked, so gap 0 is exactly the earlier noisy holdout.
- **`best.pt` selection:** the unweighted mean of the four per-gap noisy holdout losses.

## Inference support and tracker

| Support field | Value |
|---|---|
| `max_trailing_gap` | 3 |
| `prediction_horizons` | [1, 2, 3, 4] |
| `min_valid_obs` | 7 |
| `require_full_prefix` | True (optional key; absent means False) |

**Tracker eligibility** (`OCSortGRU`):

| Condition | Result |
|---|---|
| gap ≤ 3, valid observations = 10 − gap | GRU |
| gap ≥ 4 | `fallback_unsupported_gap` |
| a hole before the gap | `fallback_history_holes` (new) |
| fewer than 7 valid observations | `fallback_insufficient_history` |
| invalid output | `fallback_invalid_output` |

- **Runtime overrides** may only narrow the support (e.g. `--gru_max_trailing_gap 1`); `require_full_prefix` is always kept.
- **Existing checkpoints** have no `require_full_prefix` key, so their policy and outputs are unchanged. This was checked byte-for-byte on all 25 val sequences for the baseline, standard and noise trackers.

## Command

```
python -m gru_motion.train --split_config configs/gru/dancetrack_split_v2.json \
    --training_mode gap_aware --max_trailing_gap 3 --box_noise jitter_v1 --epochs 30 \
    --batch_size 512 --lr 1e-3 --seed 0 --device cuda --out_dir gru_runs/gap_aware/noise_v1_seed0_a40
```
