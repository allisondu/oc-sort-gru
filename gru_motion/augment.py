"""
Detector-like bounding-box localisation noise for GRU training windows.

Applied to the raw cxcywh HISTORY boxes (B, L, 4) before the shared feature
encoder (features.encode_windows), so the feature definition is unchanged.
Targets are never perturbed: the training target is the clean GT box of the
target frame, encoded (features.encode_targets) relative to the possibly-noisy
last observed box, exactly as inference decodes the GRU output relative to the
last (noisy) detection.

Noise model (per valid observation slot, independently across slots/windows):

    cx' = cx + e_x * w            e_x ~ TruncN(0, center_x_std^2, +-clip_sigma std)
    cy' = cy + e_y * h            e_y ~ TruncN(0, center_y_std^2, ...)
    w'  = w * exp(e_w)            e_w ~ TruncN(0, log_w_std^2, ...)
    h'  = h * exp(e_h)            e_h ~ TruncN(0, log_h_std^2, ...)

(w, h on the right-hand side are the clean sizes.) Truncation is implemented by
clipping the standard-normal draw to [-clip_sigma, clip_sigma] before scaling
(clip_sigma = 0 disables it). A window is perturbed with probability
apply_prob; otherwise it is left exactly clean. Missing slots (mask 0) and the
mask itself are never changed. w', h' > 0 always (exp); all values finite.

Determinism: perturb_windows() consumes a fixed number of draws from the
supplied numpy Generator (B*L*4 normals + B uniforms) regardless of the
configuration, so identical (seed, inputs, config) give identical outputs.
"""
from dataclasses import asdict, dataclass

import numpy as np

NOISE_MODEL_VERSION = 1


@dataclass(frozen=True)
class BoxNoiseConfig:
    center_x_std: float = 0.0     # std of cx offset, as a fraction of box width
    center_y_std: float = 0.0     # std of cy offset, as a fraction of box height
    log_w_std: float = 0.0        # std of log-width perturbation
    log_h_std: float = 0.0        # std of log-height perturbation
    apply_prob: float = 1.0       # probability that a window is perturbed at all
    clip_sigma: float = 3.0       # truncate draws at +-clip_sigma std (0 = no truncation)

    def __post_init__(self):
        for k in ("center_x_std", "center_y_std", "log_w_std", "log_h_std", "clip_sigma"):
            v = getattr(self, k)
            if not (np.isfinite(v) and v >= 0):
                raise ValueError("{} must be finite and >= 0 (got {})".format(k, v))
        if not (0.0 <= self.apply_prob <= 1.0):
            raise ValueError("apply_prob must be in [0, 1] (got {})".format(self.apply_prob))
        if self.log_w_std > 0.5 or self.log_h_std > 0.5 or self.center_x_std > 0.5 or self.center_y_std > 0.5:
            raise ValueError("noise std > 0.5 of box size is not a localisation-noise model")

    @property
    def enabled(self):
        return self.apply_prob > 0 and any(
            s > 0 for s in (self.center_x_std, self.center_y_std, self.log_w_std, self.log_h_std))

    def to_dict(self):
        d = asdict(self)
        d.update({"noise_model": "per-observation iid truncated Gaussian, size-relative (cxcywh)",
                  "noise_model_version": NOISE_MODEL_VERSION})
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in ("center_x_std", "center_y_std", "log_w_std", "log_h_std",
                                        "apply_prob", "clip_sigma")})


def perturb_windows(boxes, mask, cfg, rng):
    """
    boxes (B, L, 4) cxcywh, mask (B, L) -> NEW perturbed boxes (B, L, 4).
    Inputs are not modified; mask is returned unchanged by the caller's choice
    (this function never touches it). rng: numpy.random.Generator.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    mask = np.asarray(mask)
    B, L, _ = boxes.shape
    z = rng.standard_normal((B, L, 4))
    apply = rng.random(B) < cfg.apply_prob
    out = boxes.copy()
    if not cfg.enabled or B == 0:
        return out
    if cfg.clip_sigma > 0:
        np.clip(z, -cfg.clip_sigma, cfg.clip_sigma, out=z)
    sel = (mask > 0) & apply[:, None]                     # (B, L) slots to perturb
    w, h = boxes[..., 2], boxes[..., 3]
    cx = boxes[..., 0] + z[..., 0] * cfg.center_x_std * w
    cy = boxes[..., 1] + z[..., 1] * cfg.center_y_std * h
    nw = w * np.exp(z[..., 2] * cfg.log_w_std)
    nh = h * np.exp(z[..., 3] * cfg.log_h_std)
    noisy = np.stack([cx, cy, nw, nh], axis=-1)
    out[sel] = noisy[sel]
    if not np.all(np.isfinite(out)) or np.any(out[sel][:, 2:] <= 0):
        raise FloatingPointError("perturb_windows produced invalid boxes")
    return out


def epoch_rng(seed, epoch):
    """Generator for the training noise of one epoch (independent stream per (seed, epoch))."""
    return np.random.default_rng([int(seed), 1, int(epoch)])


def holdout_rng(seed):
    """Generator for the fixed noisy-holdout set (stream disjoint from all training epochs)."""
    return np.random.default_rng([int(seed), 2])


# ----------------------------------------------------------------------------- observation gaps

def balanced_gaps(n, max_gap, rng):
    """
    Deterministic, exactly balanced assignment of trailing-gap lengths 0..max_gap to
    n windows (counts differ by at most 1), in random order drawn from `rng`.
    """
    g = np.arange(n) % (max_gap + 1)
    return g[rng.permutation(n)]


def apply_trailing_gaps(boxes, mask, gaps):
    """
    Mark the last gaps[i] slots of window i as MISSING: mask 0 and box zeros (the
    canonical missing-slot encoding of features.build_window). Slot positions are
    kept (no temporal compression). Returns NEW (boxes, mask); inputs are not modified.
    Windows must keep at least one observed slot.
    """
    boxes = np.array(boxes, dtype=np.float64, copy=True)
    mask = np.array(mask, dtype=np.float64, copy=True)
    gaps = np.asarray(gaps, dtype=int)
    L = mask.shape[1]
    if np.any(gaps < 0) or np.any(gaps >= L):
        raise ValueError("gap lengths must be in [0, history_len - 1]")
    drop = np.arange(L)[None, :] >= (L - gaps)[:, None]          # (B, L) trailing slots to drop
    mask[drop] = 0.0
    boxes[drop] = 0.0
    if np.any(mask.sum(1) < 1):
        raise ValueError("every window needs at least one observed slot")
    return boxes, mask


def gap_rng(seed, epoch):
    """Generator for the training gap assignment of one epoch (stream 3, disjoint from noise streams 1/2)."""
    return np.random.default_rng([int(seed), 3, int(epoch)])


def fixed_gap_rng(seed):
    """Generator for the FIXED gap assignment used for normalisation / clean eval-train (stream 4)."""
    return np.random.default_rng([int(seed), 4])
