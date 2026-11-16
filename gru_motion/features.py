"""
Trajectory-window features for the GRU motion model.

This module is the single source of truth for coordinate conventions and
normalisation. The tracker (trackers/ocsort_gru) and the future training
pipeline must both go through these functions so the two can never drift apart.

Coordinate conventions
----------------------
* xyxy   : [x1, y1, x2, y2] in original image pixels (what OC-SORT stores in
           track.observations / last_observation, and what it associates on).
* cxcywh : [cx, cy, w, h] in original image pixels. Internal to this module.
* The Kalman state [cx, cy, s=w*h, r=w/h, ...] is never used here; GRU
  features are derived from observed detections only.

Window layout (no temporal compression)
----------------------------------------
A window has L slots for the L frames immediately preceding the target frame:
slot i (0-based) holds frame (target - L + i). A frame with no observation is a
*missing* slot: features are zero and the mask channel is 0. Temporal spacing
is therefore always preserved; observations from non-consecutive frames are
never packed next to each other.

Normalisation (per window, relative to a reference box)
-------------------------------------------------------
The reference box (cx_r, cy_r, w_r, h_r) is the LAST VALID observation in the
window. For every valid slot with box (cx, cy, w, h):

    f0 = (cx - cx_r) / h_r          position relative to reference, in units
    f1 = (cy - cy_r) / h_r          of reference height (isotropic scale, so
                                    motion direction/angles are preserved)
    f2 = log(w / w_r)               log size ratios (scale-invariant,
    f3 = log(h / h_r)               always yield positive sizes on decoding)
    f4 = 1                          validity mask (0 for missing slots)

Height is used for both position axes because person boxes' widths vary with
pose (arms out) much more than heights, and a single isotropic scale keeps
direction changes undistorted.

Output / target (4-d), for the target frame box (cx', cy', w', h'):

    o = [(cx' - cx_r) / h_r, (cy' - cy_r) / h_r, log(w' / w_r), log(h' / h_r)]

i.e. the displacement of the next box from the last observed box, in the same
normalised units as the inputs. Inverse transform (decode_outputs):

    cx' = cx_r + o0 * h_r
    cy' = cy_r + o1 * h_r
    w'  = w_r * exp(clip(o2, -MAX_LOG_SCALE, MAX_LOG_SCALE))
    h'  = h_r * exp(clip(o3, -MAX_LOG_SCALE, MAX_LOG_SCALE))

Any further standardisation (mean/std of features and targets) lives inside the
model as buffers (gru_motion/model.py), so a checkpoint carries its own stats.
"""
import numpy as np

# Bump whenever the feature definition above changes; checkpoints record it and
# loading a checkpoint built for another feature version fails.
FEATURE_VERSION = 1
INPUT_DIM = 5          # f0..f3 + mask
OUTPUT_DIM = 4
MAX_LOG_SCALE = 1.0    # decoded size can change by at most e^1 vs. the reference box
INPUT_LOG_CLIP = 3.0   # clip on input log-size features (numerical safety only)
MIN_SIZE = 1e-3        # boxes with w or h below this (pixels) are treated as missing


def xyxy_to_cxcywh(b):
    b = np.asarray(b, dtype=np.float64)
    w = b[..., 2] - b[..., 0]
    h = b[..., 3] - b[..., 1]
    return np.stack([b[..., 0] + w / 2.0, b[..., 1] + h / 2.0, w, h], axis=-1)


def cxcywh_to_xyxy(b):
    b = np.asarray(b, dtype=np.float64)
    hw, hh = b[..., 2] / 2.0, b[..., 3] / 2.0
    return np.stack([b[..., 0] - hw, b[..., 1] - hh, b[..., 0] + hw, b[..., 1] + hh], axis=-1)


def build_window(obs_by_frame, target_frame, history_len):
    """
    Gather the observations strictly before `target_frame`.

    obs_by_frame : mapping frame_index -> xyxy box (extra trailing entries such
                   as a score are ignored). Frame indices are integers on a
                   uniform time grid (video frames for GT, track age in the tracker).
    Returns (boxes, mask): boxes (L, 4) cxcywh with zeros in missing slots,
    mask (L,) float {0, 1}. Only frames target-L .. target-1 are read, so the
    target frame and anything after it can never leak into the input.
    """
    boxes = np.zeros((history_len, 4), dtype=np.float64)
    mask = np.zeros(history_len, dtype=np.float64)
    start = target_frame - history_len
    for i in range(history_len):
        b = obs_by_frame.get(start + i)
        if b is None:
            continue
        c = xyxy_to_cxcywh(np.asarray(b, dtype=np.float64)[:4])
        if not np.all(np.isfinite(c)) or c[2] < MIN_SIZE or c[3] < MIN_SIZE:
            continue
        boxes[i] = c
        mask[i] = 1.0
    return boxes, mask


def window_stats(mask):
    """(num_valid, trailing_missing) for a (L,) or (B, L) mask."""
    mask = np.asarray(mask)
    num_valid = mask.sum(axis=-1)
    # trailing_missing = number of missing slots after the last valid one
    L = mask.shape[-1]
    rev = mask[..., ::-1] > 0
    has_any = rev.any(axis=-1)
    trailing = np.where(has_any, rev.argmax(axis=-1), L)
    return num_valid, trailing


def encode_windows(boxes, mask):
    """
    boxes (B, L, 4) cxcywh, mask (B, L). Every window must contain at least one
    valid slot. Returns (features (B, L, 5) float32, ref (B, 4) cxcywh float64).
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    B, L, _ = boxes.shape
    num_valid, trailing = window_stats(mask)
    if np.any(num_valid < 1):
        raise ValueError("encode_windows: every window needs at least one valid observation")
    last_idx = L - 1 - trailing                      # index of the last valid slot
    ref = boxes[np.arange(B), last_idx]              # (B, 4)
    cx_r, cy_r, w_r, h_r = (ref[:, k:k + 1] for k in range(4))
    feats = np.zeros((B, L, INPUT_DIM), dtype=np.float64)
    valid = mask > 0
    safe_w = np.where(valid, boxes[..., 2], 1.0)
    safe_h = np.where(valid, boxes[..., 3], 1.0)
    feats[..., 0] = (boxes[..., 0] - cx_r) / h_r
    feats[..., 1] = (boxes[..., 1] - cy_r) / h_r
    feats[..., 2] = np.clip(np.log(safe_w / w_r), -INPUT_LOG_CLIP, INPUT_LOG_CLIP)
    feats[..., 3] = np.clip(np.log(safe_h / h_r), -INPUT_LOG_CLIP, INPUT_LOG_CLIP)
    feats[..., :4] *= mask[..., None]                # missing slots are exactly zero
    feats[..., 4] = mask
    return feats.astype(np.float32), ref


def encode_targets(target_cxcywh, ref):
    """Training targets (B, 4) for target boxes (B, 4) cxcywh given refs (B, 4)."""
    t = np.asarray(target_cxcywh, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    out = np.stack([(t[:, 0] - ref[:, 0]) / ref[:, 3],
                    (t[:, 1] - ref[:, 1]) / ref[:, 3],
                    np.log(t[:, 2] / ref[:, 2]),
                    np.log(t[:, 3] / ref[:, 3])], axis=-1)
    return out.astype(np.float32)


def decode_outputs(outputs, ref):
    """Inverse of encode_targets: outputs (B, 4) -> boxes (B, 4) cxcywh, w/h > 0."""
    o = np.asarray(outputs, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    cx = ref[:, 0] + o[:, 0] * ref[:, 3]
    cy = ref[:, 1] + o[:, 1] * ref[:, 3]
    w = ref[:, 2] * np.exp(np.clip(o[:, 2], -MAX_LOG_SCALE, MAX_LOG_SCALE))
    h = ref[:, 3] * np.exp(np.clip(o[:, 3], -MAX_LOG_SCALE, MAX_LOG_SCALE))
    return np.stack([cx, cy, w, h], axis=-1)
