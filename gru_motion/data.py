"""
DanceTrack ground truth -> GRU training windows.

All window construction and encoding goes through gru_motion.features
(build_window / encode_windows / encode_targets), i.e. exactly the code the
tracker uses at inference, so train and test features cannot diverge.

Pipeline
--------
1. load_gt(path): parse MOT-format gt.txt
   (frame, id, x_left, y_top, w, h, conf, class, visibility; frames 1-based,
   DanceTrack ids 0-based). Malformed rows are dropped: non-finite values,
   w <= 0 or h <= 0, conf == 0 (MOT "ignore"), and any (frame, id) pair that
   appears more than once (ambiguous).
2. Tracks are grouped per (sequence, id): {frame -> xyxy}. Windows never span
   two identities or two sequences, because each window is built from one
   track's dict only.
3. enumerate_samples(): candidate (seq, id, target_frame) triples in sorted
   order, filtered by a SamplingSpec. The window reads frames
   target-L .. target-1 only (build_window), the target box is the GT at
   target_frame; nothing at or after the target enters the input.
4. Deterministic subsampling with a seeded RNG (max_windows).
5. build_arrays(): (features (N, L, 5), targets (N, 4)) via the shared encoder.

Training modes (SamplingSpec.mode)
----------------------------------
Window slots always cover frames target-L .. target-1 of ONE track; the target
frame must itself be annotated (the target is always a real GT box).
  * "standard": all L slots observed (=> trailing gap 0, horizon 1).
  * "partial":  >= min_valid_obs observed slots anywhere in the window, older
                slots may be missing (masked, temporal position preserved), but
                slot L-1 (frame target-1) must be observed (trailing gap 0,
                horizon 1). Includes full windows as a special case.
  * future "gap_aware": trailing gap > 0, i.e. predicting more than one frame
                past the last observation. Expressible with the same spec
                (max_trailing_gap > 0) but deliberately rejected for now.

Extension point: `augment` callable in build_arrays(boxes, mask, rng) ->
(boxes, mask), applied to the raw cxcywh window before encoding (jitter,
observation drop-out). Default is the identity; not used yet.
"""
import configparser
import glob
import os
from dataclasses import asdict, dataclass

import numpy as np

from .features import build_window, encode_targets, encode_windows, window_stats


# Official DanceTrack validation sequences. They must never be used for GRU
# training or holdout; they are reserved for tracking evaluation.
DANCETRACK_VAL_SEQUENCES = frozenset("dancetrack{:04d}".format(i) for i in (
    4, 5, 7, 10, 14, 18, 19, 25, 26, 30, 34, 35, 41, 43, 47, 58, 63, 65, 73, 77, 79, 81, 90, 94, 97))

TRAINING_MODES = ("standard", "partial")


@dataclass(frozen=True)
class SamplingSpec:
    mode: str = "standard"
    history_len: int = 10
    min_valid_obs: int = 10        # standard: forced to history_len
    max_trailing_gap: int = 0      # missing frames between last observation and target

    @classmethod
    def make(cls, mode, history_len=10, min_valid_obs=None, max_trailing_gap=0):
        """Validated constructor. min_valid_obs=None -> mode default (L for standard, 3 for partial)."""
        if mode not in TRAINING_MODES:
            raise ValueError("training mode must be one of {} (gap-aware is not implemented yet)".format(
                TRAINING_MODES))
        if history_len < 2:
            raise ValueError("history_len must be >= 2")
        if max_trailing_gap != 0:
            raise ValueError("max_trailing_gap > 0 means gap-aware (multi-frame-horizon) training, "
                             "which is not implemented yet; use 0")
        if mode == "standard":
            if min_valid_obs not in (None, history_len):
                raise ValueError("standard mode requires all {} slots observed; do not set "
                                 "min_valid_obs={}".format(history_len, min_valid_obs))
            min_valid_obs = history_len
        else:
            min_valid_obs = 3 if min_valid_obs is None else min_valid_obs
            if not (1 <= min_valid_obs <= history_len):
                raise ValueError("min_valid_obs must be in [1, history_len]")
        return cls(mode, int(history_len), int(min_valid_obs), int(max_trailing_gap))

    @property
    def require_full_history(self):
        return self.min_valid_obs == self.history_len

    def accepts(self, num_valid, trailing_missing):
        """Window predicate shared by training sampling and (via checkpoint metadata) inference."""
        return (trailing_missing <= self.max_trailing_gap) and (num_valid >= self.min_valid_obs)

    def training_conditions(self):
        """Plain-dict description of what the model saw (stored in checkpoints)."""
        return {"training_mode": self.mode, "history_len": self.history_len,
                "min_valid_obs": self.min_valid_obs, "require_full_history": self.require_full_history,
                "max_trailing_gap": self.max_trailing_gap,
                "prediction_horizons": list(range(1, self.max_trailing_gap + 2)),
                "target": "real GT box at the frame immediately after the window",
                "augmentation": None}

    def to_dict(self):
        d = asdict(self)
        d["require_full_history"] = self.require_full_history
        return d


def load_gt(path):
    """Return (tracks {id: {frame: xyxy (4,)}}, stats dict) for one gt.txt."""
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    stats = {"rows": int(len(raw))}
    if raw.shape[1] < 6:
        raise ValueError("{}: expected >= 6 columns, got {}".format(path, raw.shape[1]))
    ok = np.all(np.isfinite(raw[:, :6]), axis=1) & (raw[:, 4] > 0) & (raw[:, 5] > 0)
    if raw.shape[1] > 6:
        ok &= raw[:, 6] != 0
    stats["dropped_malformed"] = int((~ok).sum())
    raw = raw[ok]
    key = raw[:, 0].astype(np.int64) * 100000 + raw[:, 1].astype(np.int64)
    uniq, counts = np.unique(key, return_counts=True)
    dup = np.isin(key, uniq[counts > 1])
    stats["dropped_duplicate"] = int(dup.sum())
    raw = raw[~dup]
    tracks = {}
    for f, i, x, y, w, h in raw[:, :6]:
        tracks.setdefault(int(i), {})[int(f)] = np.array([x, y, x + w, y + h], dtype=np.float64)
    stats["ids"] = len(tracks)
    stats["boxes"] = int(len(raw))
    return tracks, stats


def list_sequences(root, split="train", seqs=None, max_seqs=None):
    dirs = sorted(d for d in glob.glob(os.path.join(root, split, "*")) if os.path.isfile(
        os.path.join(d, "gt", "gt.txt")))
    if seqs:
        dirs = [d for d in dirs if os.path.basename(d) in set(seqs)]
    if max_seqs is not None:
        dirs = dirs[:max_seqs]
    return dirs


def read_seqinfo(seq_dir):
    p = os.path.join(seq_dir, "seqinfo.ini")
    if not os.path.isfile(p):
        return {}
    cp = configparser.ConfigParser()
    cp.read(p)
    return dict(cp["Sequence"]) if "Sequence" in cp else {}


def load_sequences(seq_dirs):
    """{seq_name: tracks}, per-sequence info list."""
    data, info = {}, []
    for d in seq_dirs:
        name = os.path.basename(d)
        tracks, st = load_gt(os.path.join(d, "gt", "gt.txt"))
        data[name] = tracks
        si = read_seqinfo(d)
        st.update({"sequence": name, "frame_rate": si.get("framerate"),
                   "im_w": si.get("imwidth"), "im_h": si.get("imheight")})
        info.append(st)
    return data, info


def enumerate_samples(data, spec):
    """
    Sorted list of (seq, id, target_frame) whose window satisfies `spec`.
    The target frame must be annotated; only frames < target are inspected for
    the window (same build_window the tracker uses), so nothing at or after the
    target enters the input, and each window comes from a single track.
    """
    L = spec.history_len
    out = []
    for seq in sorted(data):
        for tid in sorted(data[seq]):
            frames = data[seq][tid]
            for t in sorted(frames):
                _, mask = build_window(frames, t, L)
                nv, tr = window_stats(mask)
                if spec.accepts(nv, tr):
                    out.append((seq, tid, t))
    return out


def subsample(samples, max_windows, seed):
    """Deterministic subset (sorted order preserved)."""
    if max_windows is None or len(samples) <= max_windows:
        return list(samples)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(samples), size=max_windows, replace=False))
    return [samples[i] for i in idx]


def build_arrays(data, samples, spec, augment=None, rng=None):
    """Encode samples -> (feats (N, L, 5) float32, targets (N, 4) float32, refs (N, 4))."""
    L = spec.history_len
    n = len(samples)
    boxes = np.zeros((n, L, 4))
    masks = np.zeros((n, L))
    tgt = np.zeros((n, 4))
    for j, (seq, tid, t) in enumerate(samples):
        frames = data[seq][tid]
        b, m = build_window(frames, t, L)
        if augment is not None:
            b, m = augment(b, m, rng)
        boxes[j], masks[j] = b, m
        x1, y1, x2, y2 = frames[t]
        tgt[j] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]
    feats, ref = encode_windows(boxes, masks)
    targets = encode_targets(tgt, ref)
    return feats, targets, ref


def normalization_stats(feats, targets, eps=1e-4):
    """
    Mean/std of the 4 feature channels over VALID, NON-REFERENCE slots only
    (missing slots are zeros by construction and the reference slot is always
    exactly zero, so including either would only bias the stats), and of the
    targets. Must be computed on training windows only.
    """
    mask = feats[..., 4] > 0
    _, trailing = window_stats(mask.astype(np.float64))
    ref_idx = feats.shape[1] - 1 - trailing
    use = mask.copy()
    use[np.arange(len(feats)), ref_idx] = False
    vals = feats[..., :4][use].astype(np.float64)
    if len(vals) < 2:
        raise ValueError("not enough valid history slots to compute normalisation stats")
    in_mean, in_std = vals.mean(0), np.maximum(vals.std(0), eps)
    t = targets.astype(np.float64)
    out_mean, out_std = t.mean(0), np.maximum(t.std(0), eps)
    return in_mean, in_std, out_mean, out_std


def load_split_config(path, data_root, split="train"):
    """
    Load a fixed sequence-level split (configs/gru/dancetrack_split.json) and
    verify it: sequences exist with gt.txt, partitions are disjoint, and no
    official validation sequence is present. Returns (cfg, train_dirs,
    holdout_dirs, sha256 of the file).
    """
    import hashlib
    import json
    raw = open(path, "rb").read()
    cfg = json.loads(raw)
    train, hold = list(cfg["train_sequences"]), list(cfg["holdout_sequences"])
    problems = []
    if len(set(train)) != len(train) or len(set(hold)) != len(hold):
        problems.append("duplicate names inside a partition")
    overlap = sorted(set(train) & set(hold))
    if overlap:
        problems.append("sequences in both partitions: {}".format(overlap))
    val = sorted((set(train) | set(hold)) & DANCETRACK_VAL_SEQUENCES)
    if val:
        problems.append("official validation sequences present: {}".format(val))
    def _dir(name):
        return os.path.join(data_root, split, name)
    missing = [n for n in train + hold if not os.path.isfile(os.path.join(_dir(n), "gt", "gt.txt"))]
    if missing:
        problems.append("missing gt.txt for: {}".format(missing))
    if problems:
        raise ValueError("invalid split config {}: {}".format(path, "; ".join(problems)))
    return cfg, [_dir(n) for n in train], [_dir(n) for n in hold], hashlib.sha256(raw).hexdigest()
