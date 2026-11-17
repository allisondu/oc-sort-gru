"""
Sanity tests for the direct-GRU branch. No dataset or trained model needed.

    python tests/test_gru_direct.py        (or: pytest tests/test_gru_direct.py)

The random-weight models built here are test fixtures for plumbing only; they
are never written to a path the runner would use.
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gru_motion.data import SamplingSpec
from gru_motion import (GRUBoxPredictor, GRUCheckpointError, GRUMotionModel, build_window,
                        decode_outputs, encode_targets, encode_windows, load_checkpoint, save_checkpoint)
from gru_motion.features import cxcywh_to_xyxy, window_stats, xyxy_to_cxcywh
from trackers.ocsort_gru import OCSortGRU
from trackers.ocsort_gru.history import track_window
from trackers.ocsort_tracker.ocsort import KalmanBoxTracker, OCSort


def test_coordinate_roundtrip():
    b = np.array([[100.0, 50.0, 160.0, 250.0]])
    assert np.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(b)), b)


def test_window_layout_and_no_leakage():
    obs = {3: [0, 0, 10, 20, .9], 5: [2, 0, 12, 20, .9], 6: [4, 0, 14, 20, .9],
           7: [999, 999, 1000, 1000, .9]}                     # frame 7 = target; must not be read
    boxes, mask = build_window(obs, target_frame=7, history_len=5)  # slots = frames 2..6
    assert mask.tolist() == [0, 1, 0, 1, 1]                  # gap at frame 4 preserved, not compressed
    assert np.allclose(boxes[4], [9, 10, 10, 20])
    assert np.all(boxes[mask == 0] == 0)
    nv, tr = window_stats(mask)
    assert nv == 3 and tr == 0
    _, tr2 = window_stats(build_window(obs, 9, 5)[1])        # last obs at 7 -> 1 trailing missing (8)
    assert tr2 == 1


def test_encode_decode_inverse():
    rng = np.random.default_rng(0)
    B, L = 6, 10
    boxes = np.zeros((B, L, 4))
    boxes[..., 0] = 500 + rng.normal(0, 20, (B, L))
    boxes[..., 1] = 400 + rng.normal(0, 20, (B, L))
    boxes[..., 2] = 120 * np.exp(rng.normal(0, .05, (B, L)))
    boxes[..., 3] = 300 * np.exp(rng.normal(0, .05, (B, L)))
    mask = (rng.random((B, L)) > .3).astype(float)
    mask[:, -1] = 1
    feats, ref = encode_windows(boxes, mask)
    assert feats.shape == (B, L, 5) and np.all(np.isfinite(feats))
    assert np.all(feats[..., :4][mask == 0] == 0)
    assert np.allclose(feats[:, -1, :4], 0)                  # last valid slot is the reference
    tgt = boxes[:, -1] + np.array([7.0, -3.0, 4.0, -6.0])
    assert np.allclose(decode_outputs(encode_targets(tgt, ref), ref), tgt)
    # decoded sizes stay positive even for absurd outputs
    d = decode_outputs(np.full((B, 4), -50.0), ref)
    assert np.all(d[:, 2:] > 0)


def _fixture_model(seed=0):
    torch.manual_seed(seed)
    return GRUMotionModel(hidden_size=64, num_layers=1)


def _save_fixture(path, mode, epochs=1, windows=1, model=None, **support_override):
    spec = SamplingSpec.make(mode, 10)
    support = None
    if support_override:
        from gru_motion.model import support_from_training
        support = dict(support_from_training(spec.training_conditions()), **support_override)
    save_checkpoint(path, model or _fixture_model(), spec.training_conditions(),
                    {"epochs": epochs, "num_train_windows": windows, "note": "test fixture"}, support)
    return path


def _expect(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("expected {}".format(exc.__name__))


# ----------------------------------------------------------------------------- checkpoints

def test_checkpoint_validation():
    with tempfile.TemporaryDirectory() as d:
        for bad in (None, os.path.join(d, "missing.pt")):
            _expect(GRUCheckpointError, load_checkpoint, bad)
        untrained = _save_fixture(os.path.join(d, "untrained.pt"), "standard", epochs=0, windows=0)
        assert "no record of training" in str(_expect(GRUCheckpointError, load_checkpoint, untrained))
        foreign = os.path.join(d, "foreign.pt")
        torch.save({"model": {}}, foreign)
        _expect(GRUCheckpointError, load_checkpoint, foreign)
        m = _fixture_model()
        ok = _save_fixture(os.path.join(d, "ok.pt"), "partial", model=m)
        # load_checkpoint uses torch.load(weights_only=True)
        m2, meta = load_checkpoint(ok)
        x = torch.randn(3, 10, 5)
        x[..., 4] = 1
        assert torch.allclose(m.predict(x), m2.predict(x))
        assert meta["format"].endswith("/v2") and not meta["legacy_checkpoint"]
        assert meta["training_conditions"]["training_mode"] == "partial"
        assert meta["inference_support"] == {"history_len": 10, "min_valid_obs": 3, "require_full_history": False,
                                             "max_trailing_gap": 0, "prediction_horizons": [1]}


def test_support_cannot_exceed_training():
    with tempfile.TemporaryDirectory() as d:
        # broader-than-training support is refused at save time
        _expect(ValueError, _save_fixture, os.path.join(d, "a.pt"), "standard",
                min_valid_obs=3, require_full_history=False)
        _expect(ValueError, _save_fixture, os.path.join(d, "b.pt"), "partial",
                max_trailing_gap=2, prediction_horizons=[1, 2, 3])
        # narrower support is allowed
        _save_fixture(os.path.join(d, "c.pt"), "partial", min_valid_obs=5)
        # a tampered file claiming broader support is refused at load time
        p = _save_fixture(os.path.join(d, "t.pt"), "standard")
        ck = torch.load(p, weights_only=True)
        ck["inference_support"].update(min_valid_obs=3, require_full_history=False)
        torch.save(ck, p)
        assert "below what training saw" in str(_expect(GRUCheckpointError, load_checkpoint, p))


def test_legacy_checkpoint_is_conservative():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "v1.pt")
        m = _fixture_model()
        torch.save({"format": "oc-sort-gru/motion-checkpoint/v1", "feature_version": 1, "history_len": 10,
                    "model_config": m.config, "state_dict": m.state_dict(),
                    "train_info": {"epochs": 3, "num_train_windows": 2000}}, p)
        _, meta = load_checkpoint(p)
        assert meta["legacy_checkpoint"] and meta["training_conditions"] is None
        assert meta["inference_support"]["min_valid_obs"] == 10
        assert meta["inference_support"]["max_trailing_gap"] == 0
        # and the tracker refuses to widen it
        pred = GRUBoxPredictor.from_checkpoint(p)
        _expect(ValueError, OCSortGRU, det_thresh=0.6, predictor=pred, min_obs=3)


def test_runtime_policy_only_narrows():
    from trackers.ocsort_gru.tracker import effective_policy
    std = GRUBoxPredictor(_fixture_model(), 10)   # no support given -> conservative (full-history) support
    par_support = {"history_len": 10, "min_valid_obs": 3, "require_full_history": False,
                   "max_trailing_gap": 0, "prediction_horizons": [1]}
    assert effective_policy(std.support)["min_valid_obs"] == 10
    _expect(ValueError, effective_policy, std.support, min_obs=3)            # broader -> rejected
    _expect(ValueError, effective_policy, par_support, min_obs=2)
    _expect(ValueError, effective_policy, par_support, max_trailing_gap=1)
    assert effective_policy(par_support, min_obs=6)["min_valid_obs"] == 6   # stricter -> fine
    assert effective_policy(par_support)["min_valid_obs"] == 3


# ----------------------------------------------------------------------------- tracker integration

def _synthetic_frames(n_frames=80, n_obj=4, seed=0, miss=0.15):
    """Objects on curved paths with missed detections -> list of (N,5) arrays."""
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n_frames):
        dets = []
        for k in range(n_obj):
            if rng.random() < miss:
                continue                                   # missed detection
            cx = 200 + 300 * k + 40 * np.sin(t / 6.0 + k)
            cy = 400 + 30 * np.cos(t / 5.0 + k)
            dets.append([cx - 50, cy - 120, cx + 50, cy + 120, 0.95])
        frames.append(np.array(dets) if dets else np.empty((0, 5)))
    return frames


class _SentinelPredictor(object):
    """Returns a recognisable box so we can tell GRU rows from Kalman rows."""
    def __init__(self, support):
        self.history_len, self.support, self.calls = 10, support, []

    def predict_xyxy(self, boxes, mask):
        self.calls.append(mask.copy())
        return np.tile([1.0, 2.0, 3.0, 4.0], (len(boxes), 1))


def _run_policy(support, check_rows=True):
    pred = _SentinelPredictor(support)
    trk = OCSortGRU(det_thresh=0.6, predictor=pred, log_decisions=True)
    orig = OCSortGRU._predict_boxes

    def checked():
        trks = orig(trk)
        start = len(trk.decision_log) - len(trk.trackers)
        for i, t in enumerate(trk.trackers):
            dec = trk.decision_log[start + i][-1]
            assert all(k < t.age for k in t.observations)            # no current/future observations
            if dec == "gru_used":
                assert np.allclose(trks[i, :4], [1, 2, 3, 4])
            else:                                                    # Kalman fallback row untouched
                assert np.allclose(trks[i, :4], t.history[-1][0][:4])
        return trks
    if check_rows:
        trk._predict_boxes = checked
    for dets in _synthetic_frames():
        trk.update(dets.copy(), (1080, 1920), (1080, 1920))
    return trk, pred


def test_standard_checkpoint_policy_in_tracker():
    support = {"history_len": 10, "min_valid_obs": 10, "require_full_history": True,
               "max_trailing_gap": 0, "prediction_horizons": [1]}
    trk, pred = _run_policy(support)
    s = trk.stats
    assert s["track_predictions"] == sum(v for k, v in s.items() if k != "track_predictions")
    assert s["gru_used"] > 0 and s["fallback_insufficient_history"] > 0
    assert s["fallback_missing_recent_observation"] > 0 and s["fallback_unsupported_gap"] == 0
    for _, _, age, total, nv, since, dec in trk.decision_log:
        assert nv <= min(total, 10) and nv <= age                    # age / total / window counts differ
        if dec == "gru_used":
            assert nv == 10 and since == 1                           # full history, last frame observed
    assert all(m.all() for m in pred.calls)                          # model only ever saw full windows


def test_partial_checkpoint_policy_in_tracker():
    support = {"history_len": 10, "min_valid_obs": 3, "require_full_history": False,
               "max_trailing_gap": 0, "prediction_horizons": [1]}
    trk, pred = _run_policy(support)
    used = [r for r in trk.decision_log if r[-1] == "gru_used"]
    assert any(r[4] < 10 for r in used)                              # incomplete histories accepted
    assert all(r[4] >= 3 and r[5] == 1 for r in used)                # min obs + recent obs enforced
    assert trk.stats["fallback_missing_recent_observation"] > 0
    for m in pred.calls:
        assert np.all(m[:, -1] == 1) and np.all(m.sum(1) >= 3)


def test_kalman_box_equivalence_when_gru_returns_kalman():
    """If the 'GRU' echoes the Kalman boxes, OCSortGRU must reproduce OCSort exactly.
    Exercises row alignment between eligible tracks, predictions and `trks`."""
    class KalmanEcho(object):
        history_len = 10
        support = {"history_len": 10, "min_valid_obs": 3, "require_full_history": False,
                   "max_trailing_gap": 0, "prediction_horizons": [1]}
        tracker = None

        def predict_xyxy(self, boxes, mask):
            trs = self.tracker.trackers
            nv, tr = window_stats(np.array([track_window(t, 10)[1] for t in trs]))
            elig = np.flatnonzero((tr <= 0) & (nv >= 3))
            assert len(elig) == len(boxes)
            return np.array([trs[i].history[-1][0] for i in elig])   # Kalman box from predict()

    frames = _synthetic_frames(seed=1)
    # Run sequentially: KalmanBoxTracker.count is a class-level global ID counter
    # shared by all tracker instances (reset in OCSort.__init__).
    base = OCSort(det_thresh=0.6)
    out_base = [base.update(d.copy(), (1080, 1920), (1080, 1920)) for d in frames]
    echo = KalmanEcho()
    gru = OCSortGRU(det_thresh=0.6, predictor=echo)
    echo.tracker = gru
    out_gru = [gru.update(d.copy(), (1080, 1920), (1080, 1920)) for d in frames]
    assert all(np.array_equal(a, b) for a, b in zip(out_base, out_gru))
    assert gru.stats["gru_used"] > 0


# ----------------------------------------------------------------------------- data / training modes

def _write_gt(d, rows):
    p = os.path.join(d, "gt.txt")
    open(p, "w").write("\n".join(rows) + "\n")
    return p


def test_gt_loader_and_window_sampling():
    """Parser edge cases + standard windows never cross gaps; targets align with GT."""
    from gru_motion.data import build_arrays, enumerate_samples, load_gt, subsample
    rows = []
    for f in range(1, 16):                        # id 0: frames 1..15, gap at 8
        if f != 8:
            rows.append("{},0,{},{},10,20,1,1,1".format(f, 100 + 2 * f, 50))
    for f in range(1, 13):                        # id 1: frames 1..12, consecutive
        rows.append("{},1,{},{},12,24,1,1,1".format(f, 300, 60 + 3 * f))
    rows += ["3,2,10,10,0,20,1,1,1",              # zero width -> dropped
             "4,2,10,10,5,nan,1,1,1",             # non-finite -> dropped
             "5,3,10,10,5,5,1,1,1", "5,3,11,11,5,5,1,1,1"]   # duplicate (frame,id) -> both dropped
    with tempfile.TemporaryDirectory() as d:
        tracks, st = load_gt(_write_gt(d, rows))
    assert st["dropped_malformed"] == 2 and st["dropped_duplicate"] == 2
    assert sorted(tracks) == [0, 1]
    spec = SamplingSpec.make("standard", history_len=5)
    data = {"seqA": tracks}
    s = enumerate_samples(data, spec)
    # id 0 (gap at 8): target t needs frames t-5..t all annotated -> t in {6, 7, 14, 15}
    assert [x[2] for x in s if x[1] == 0] == [6, 7, 14, 15]
    assert [x[2] for x in s if x[1] == 1] == list(range(6, 13))
    assert subsample(s, 4, 0) == subsample(s, 4, 0) and len(subsample(s, 4, 0)) == 4
    feats, targets, ref = build_arrays(data, s, spec)
    assert feats.shape == (len(s), 5, 5) and np.all(feats[..., 4] == 1)
    dec = decode_outputs(targets, ref)
    for j, (_, tid, t) in enumerate(s):
        assert np.allclose(cxcywh_to_xyxy(dec[j]), tracks[tid][t])          # target = GT at t
        assert np.allclose(cxcywh_to_xyxy(ref[j]), tracks[tid][t - 1])      # reference = GT at t-1


def test_training_modes():
    from gru_motion.data import build_arrays, enumerate_samples
    # one track, frames 1..20 with frames 5 and 12 missing
    frames = {f: np.array([10.0 * f, 0, 10.0 * f + 10, 20]) for f in range(1, 21) if f not in (5, 12)}
    data = {"s": {7: frames}}
    std = SamplingSpec.make("standard", 5)
    par = SamplingSpec.make("partial", 5, min_valid_obs=3)
    ts = [t for _, _, t in enumerate_samples(data, std)]
    tp = [t for _, _, t in enumerate_samples(data, par)]
    # standard: t and t-5..t-1 all present
    assert ts == [t for t in sorted(frames) if all(t - k in frames for k in range(1, 6))]
    # partial: t present, t-1 present, >= 3 of t-5..t-1 present
    exp = [t for t in sorted(frames) if (t - 1) in frames and sum((t - k) in frames for k in range(1, 6)) >= 3]
    assert tp == exp and set(ts) < set(tp)
    assert 4 in tp and 4 not in ts                 # short history (3 obs: frames 1..3) accepted by partial only
    assert 6 not in tp and 13 not in tp            # most recent frame (5 / 12) missing -> rejected
    assert 3 not in tp                             # only 2 obs -> below min_valid_obs
    feats, _, _ = build_arrays(data, [("s", 7, 9)], par)       # window = frames 4..8, frame 5 missing
    assert feats[0, :, 4].tolist() == [1, 0, 1, 1, 1] and np.all(feats[0, 1, :4] == 0)
    # partial min_valid_obs enforced: requiring all 5 reproduces standard
    t5 = [t for _, _, t in enumerate_samples(data, SamplingSpec.make("partial", 5, min_valid_obs=5))]
    assert t5 == ts
    # invalid specs
    _expect(ValueError, SamplingSpec.make, "partial", 5, None, 1)            # gap-aware not implemented
    _expect(ValueError, SamplingSpec.make, "standard", 10, 3)                # standard is always full
    _expect(ValueError, SamplingSpec.make, "gap_aware", 10)


def test_no_leakage_and_no_identity_or_video_crossing():
    from gru_motion.data import build_arrays, enumerate_samples
    def track(x0, n=30):
        return {f: np.array([x0 + f, 0.0, x0 + f + 10, 20.0]) for f in range(1, n + 1)}
    data = {"seqA": {0: track(0), 1: track(1000)}, "seqB": {0: track(5000)}}
    for spec in (SamplingSpec.make("standard", 10), SamplingSpec.make("partial", 10)):
        s = enumerate_samples(data, spec)
        f1, t1, r1 = build_arrays(data, s, spec)
        # every window's boxes come from its own track: positions are within ~10 frames of the reference
        assert np.all(np.abs(f1[..., 0]) * 20 < 15)
        # poison every frame >= target: inputs must not change
        for j, (seq, tid, t) in enumerate(s[:40]):
            pois = {k: dict(v) for k, v in data[seq].items()}
            for f in pois[tid]:
                if f >= t:
                    pois[tid][f] = pois[tid][f] + 777.0
            f2, _, _ = build_arrays({seq: pois}, [(seq, tid, t)], spec)
            assert np.array_equal(f1[j], f2[0])


# ----------------------------------------------------------------------------- dataset / split

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "datasets_gru", "dancetrack_gt")
_SPLIT = os.path.join(_ROOT, "configs", "gru", "dancetrack_split.json")


def test_dataset_and_split():
    if not os.path.isdir(os.path.join(_DATA, "train")):
        print("  (skipped: datasets_gru not present)")
        return
    import json
    import zlib
    from gru_motion.data import DANCETRACK_VAL_SEQUENCES, list_sequences, load_gt, load_split_config
    names = [os.path.basename(d) for d in list_sequences(_DATA, "train")]
    assert len(names) == 40 and not (set(names) & DANCETRACK_VAL_SEQUENCES)
    man = json.load(open(os.path.join(_DATA, "train", "FETCH_MANIFEST.json")))
    for n in names:
        for suf in ("gt/gt.txt", "seqinfo.ini"):
            e = man["{}/{}".format(n, suf)]
            raw = open(os.path.join(_DATA, "train", n, suf), "rb").read()
            assert len(raw) == e["bytes"] and "{:08x}".format(zlib.crc32(raw) & 0xFFFFFFFF) == e["crc32"]
        tracks, st = load_gt(os.path.join(_DATA, "train", n, "gt", "gt.txt"))
        assert st["boxes"] > 0 and st["dropped_malformed"] == 0 and st["dropped_duplicate"] == 0
    cfg, tr, ho, sha = load_split_config(_SPLIT, _DATA)
    cfg2, tr2, ho2, sha2 = load_split_config(_SPLIT, _DATA)
    assert sha == sha2 and tr == tr2 and ho == ho2                   # deterministic reuse
    assert not (set(cfg["train_sequences"]) & set(cfg["holdout_sequences"]))
    assert sorted(cfg["train_sequences"] + cfg["holdout_sequences"]) == sorted(names)
    # regenerating with the recorded seed/method reproduces the same holdout
    perm = np.random.default_rng(cfg["seed"]).permutation(len(names))
    assert sorted(sorted(names)[i] for i in perm[:cfg["num_holdout_sequences"]]) == cfg["holdout_sequences"]
    # invalid split files are rejected
    with tempfile.TemporaryDirectory() as d:
        for bad in ({"train_sequences": names[:3], "holdout_sequences": names[2:4]},
                    {"train_sequences": names[:3] + ["dancetrack0004"], "holdout_sequences": []},
                    {"train_sequences": ["dancetrack9999"], "holdout_sequences": []}):
            p = os.path.join(d, "s.json")
            json.dump(bad, open(p, "w"))
            _expect(ValueError, load_split_config, p, _DATA)


# ----------------------------------------------------------------------------- best / last checkpoints

def _tiny_train(out_dir, epochs, holdout_script=None, extra=()):
    """Tiny real-data run (2 train seqs, 1 holdout seq from the TRAIN split). holdout_script
    replaces the holdout loss values per epoch (index 0 = before training)."""
    import gru_motion.train as T
    real = T.eval_loss
    calls = {"n": 0}

    def scripted(model, x, y, bs):
        if holdout_script is not None and x is not None and len(x) == scripted.hold_n:
            v = holdout_script[calls["n"]]
            calls["n"] += 1
            real(model, x, y, bs)                      # still exercise the real eval path
            return v
        return real(model, x, y, bs)
    scripted.hold_n = None

    T.eval_loss = scripted
    try:
        argv = ["--data_root", _DATA, "--seqs", "dancetrack0001", "dancetrack0002", "dancetrack0006",
                "--holdout_seqs", "dancetrack0006", "--training_mode", "standard", "--max_windows", "300",
                "--epochs", str(epochs), "--batch_size", "64", "--seed", "0", "--out_dir", out_dir] + list(extra)
        # holdout size = number of standard windows of dancetrack0006
        from gru_motion.data import enumerate_samples, load_sequences
        hd, _ = load_sequences([os.path.join(_DATA, "train", "dancetrack0006")])
        scripted.hold_n = len(enumerate_samples(hd, SamplingSpec.make("standard", 10)))
        return T.main(argv)
    finally:
        T.eval_loss = real


def _state(path):
    return torch.load(path, weights_only=True)["state_dict"]


def test_best_and_last_checkpoint_selection():
    if not os.path.isdir(os.path.join(_DATA, "train")):
        print("  (skipped: datasets_gru not present)")
        return
    import hashlib
    split_hash = hashlib.sha256(open(_SPLIT, "rb").read()).hexdigest()
    with tempfile.TemporaryDirectory() as d:
        # holdout improves until epoch 2, worsens at epoch 3
        a = os.path.join(d, "a")
        _tiny_train(a, 3, holdout_script=[0.50, 0.40, 0.30, 0.35])
        best, last = torch.load(os.path.join(a, "best.pt"), weights_only=True), \
            torch.load(os.path.join(a, "last.pt"), weights_only=True)
        bi, li = best["train_info"], last["train_info"]
        assert bi["checkpoint"]["kind"] == "best" and bi["checkpoint"]["weights_epoch"] == 2
        assert bi["checkpoint"]["holdout_loss"] == 0.30 and bi["checkpoint"]["status"] == "final"
        assert bi["epochs"] == 2 and bi["epochs_completed"] == 3 and len(bi["loss_history"]) == 4
        assert li["checkpoint"]["kind"] == "last" and li["checkpoint"]["weights_epoch"] == 3 and li["epochs"] == 3
        assert bi["normalization"] == li["normalization"] and bi["config"] == li["config"]
        # best.pt weights are exactly the epoch-2 weights: a deterministic 2-epoch rerun reproduces them
        b = os.path.join(d, "b")
        _tiny_train(b, 2)
        s2, sb, sl = _state(os.path.join(b, "last.pt")), best["state_dict"], last["state_dict"]
        assert all(torch.equal(s2[k], sb[k]) for k in sb)
        assert not all(torch.equal(sl[k], sb[k]) for k in sb)
        # both load through the normal validation path and the tracker enforces full history
        for name in ("best.pt", "last.pt"):
            pred = GRUBoxPredictor.from_checkpoint(os.path.join(a, name))
            assert pred.support["min_valid_obs"] == 10 and pred.support["require_full_history"]
            _expect(ValueError, OCSortGRU, det_thresh=0.6, predictor=pred, min_obs=3)
        # ties keep the earliest epoch
        c = os.path.join(d, "c")
        _tiny_train(c, 3, holdout_script=[0.50, 0.30, 0.30, 0.40])
        assert torch.load(os.path.join(c, "best.pt"), weights_only=True)["train_info"]["checkpoint"]["weights_epoch"] == 1
        # epoch 0 (untrained) is never eligible, even if it has the lowest holdout loss
        e0 = os.path.join(d, "e0")
        _tiny_train(e0, 2, holdout_script=[0.10, 0.40, 0.50])
        assert torch.load(os.path.join(e0, "best.pt"), weights_only=True)["train_info"]["checkpoint"]["weights_epoch"] == 1
        # optional early stopping
        e = os.path.join(d, "e")
        _tiny_train(e, 10, holdout_script=[0.5, 0.4, 0.45, 0.46, 0.47], extra=["--early_stop_patience", "2"])
        li = torch.load(os.path.join(e, "last.pt"), weights_only=True)["train_info"]
        assert li["epochs_completed"] == 3 and li["early_stopped"]
        # no holdout -> refused, unless explicitly a smoke run (then last.pt only)
        import gru_motion.train as T
        base = ["--data_root", _DATA, "--seqs", "dancetrack0001", "--training_mode", "standard",
                "--max_windows", "100", "--epochs", "1"]
        _expect(SystemExit, T.main, base + ["--out_dir", os.path.join(d, "n")])
        T.main(base + ["--out_dir", os.path.join(d, "s"), "--allow_no_holdout"])
        assert os.listdir(os.path.join(d, "s")) and not os.path.exists(os.path.join(d, "s", "best.pt"))
        assert os.path.exists(os.path.join(d, "s", "last.pt"))
        # refuses to overwrite
        _expect(SystemExit, T.main, base + ["--out_dir", os.path.join(d, "s"), "--allow_no_holdout"])
    assert hashlib.sha256(open(_SPLIT, "rb").read()).hexdigest() == split_hash   # split untouched


def test_split_v2_valid_and_reproducible():
    p2 = os.path.join(_ROOT, "configs", "gru", "dancetrack_split_v2.json")
    if not (os.path.exists(p2) and os.path.isdir(os.path.join(_DATA, "train"))):
        print("  (skipped)")
        return
    import json
    from gru_motion.data import load_split_config
    cfg, tr, ho, _ = load_split_config(p2, _DATA)
    assert cfg["split_version"] == 2 and len(ho) == 5 and len(tr) == 35
    sys.path.insert(0, os.path.join(_ROOT, "tools"))
    from make_gru_split import stratified_holdout
    stats = json.load(open(os.path.join(_ROOT, "configs", "gru", "dancetrack_train_sequence_stats.json")))["sequences"]
    hold, info = stratified_holdout(stats, 5, cfg["seed"])
    assert hold == cfg["holdout_sequences"]
    bins = info["crowding_bins"]
    assert sorted(sum(1 for b in bins if n in b) for n in hold) == [1] * 5   # one per crowding bin
    assert all(len({i for i, b in enumerate(bins) if n in b}) == 1 for n in hold)
    assert len({next(i for i, b in enumerate(bins) if n in b) for n in hold}) == 5


# ----------------------------------------------------------------------------- offline evaluation

def _curved_track(n, gap=()):
    return {f: np.array([300 + 4 * f + 15 * np.sin(f / 4.0), 200 + 3 * np.cos(f / 3.0) * 10,
                         300 + 4 * f + 15 * np.sin(f / 4.0) + 80 + 0.5 * f, 200 + 3 * np.cos(f / 3.0) * 10 + 200])
            for f in range(1, n + 1) if f not in gap}


def test_eval_const_vel_manual():
    from gru_motion.eval_offline import pred_const_vel, pred_last_obs
    hist = np.zeros((1, 10, 4))
    hist[0, :, :] = [[100 + 3 * k, 50 - 2 * k, 40 + k, 80 + 2 * k] for k in range(10)]
    hist[0, -2] = [124, 34, 48, 96]          # b(t-2)
    hist[0, -1] = [130, 30, 49, 99]          # b(t-1)
    cv, clipped = pred_const_vel(hist.copy())
    assert np.allclose(cv[0], [136, 26, 50, 102]) and clipped == 0
    assert np.allclose(pred_last_obs(hist)[0], [130, 30, 49, 99])
    h2 = hist.copy(); h2[0, -1, 2] = 2; h2[0, -2, 2] = 10
    cv2, clipped2 = pred_const_vel(h2)
    assert cv2[0, 2] == 1.0 and clipped2 == 1


class _RecordingOCSort(OCSort):
    def _predict_boxes(self):
        trks = super()._predict_boxes()
        self.rec.append(trks.copy())
        return trks


def test_eval_kalman_window_matches_ocsort_tracker():
    from gru_motion.eval_offline import kalman_window_predict
    tr = _curved_track(11)
    hist = np.array([tr[f] for f in range(1, 11)])
    trk = _RecordingOCSort(det_thresh=0.6)
    trk.rec = []
    for f in range(1, 12):                                    # frame 11 = target: tracker predicts before matching
        trk.update(np.r_[tr[f], 0.95][None], (1080, 1920), (1080, 1920))
    assert len(trk.trackers) == 1 and trk.trackers[0].hits == 10   # one track, matched every frame
    assert np.allclose(trk.rec[-1][0, :4], kalman_window_predict(hist), atol=1e-9)


def test_eval_kalman_track_matches_ocsort_tracker_with_gap():
    from gru_motion.eval_offline import kalman_track_predictions
    tr = _curved_track(40, gap=(15, 16, 17))
    preds = kalman_track_predictions(tr)
    trk = _RecordingOCSort(det_thresh=0.6)
    trk.rec = []
    for f in range(1, 41):
        d = np.r_[tr[f], 0.95][None] if f in tr else np.empty((0, 5))
        trk.update(d, (1080, 1920), (1080, 1920))
    assert len({t.id for t in trk.trackers}) == 1               # re-associated after the gap (ORU path used)
    for f in range(2, 41):
        assert np.allclose(trk.rec[f - 1][0, :4], preds[f], atol=1e-9), f
    # no leakage: corrupting frames >= t never changes the prediction for t
    for t in (10, 18, 30):
        pois = {f: (b + 500.0 if f >= t else b) for f, b in tr.items()}
        assert np.allclose(kalman_track_predictions(pois)[t], preds[t])


def test_eval_gru_matches_predictor_and_model():
    from gru_motion.features import decode_outputs, encode_windows
    m = _fixture_model()
    pred = GRUBoxPredictor(m, 10)
    rng = np.random.default_rng(0)
    hist = np.stack([np.c_[500 + np.cumsum(rng.normal(0, 3, 10)), 400 + np.cumsum(rng.normal(0, 3, 10)),
                           np.full(10, 100.0), np.full(10, 250.0)] for _ in range(7)])
    mask = np.ones((7, 10))
    feats, ref = encode_windows(hist, mask)
    manual = cxcywh_to_xyxy(decode_outputs(m.predict(torch.from_numpy(feats)).numpy(), ref))
    assert np.allclose(pred.predict_xyxy(hist, mask), manual)



# ----------------------------------------------------------------------------- inference optimisation equivalence

class _RefTracker(OCSortGRU):
    """Pre-optimisation _predict_boxes path: per-track track_window + model.predict under no_grad."""
    def _predict_boxes(self):
        import trackers.ocsort_gru.tracker as TG
        orig = TG.track_windows

        def per_track(trackers, L):
            boxes, masks = np.zeros((len(trackers), L, 4)), np.zeros((len(trackers), L))
            for i, t in enumerate(trackers):
                boxes[i], masks[i] = track_window(t, L)
            return boxes, masks
        TG.track_windows = per_track
        try:
            return super()._predict_boxes()
        finally:
            TG.track_windows = orig


class _RefPredictor(object):
    """Pre-optimisation GRUBoxPredictor.predict_xyxy (torch.no_grad + model.predict)."""
    def __init__(self, p):
        self.p, self.support, self.history_len, self.meta = p, p.support, p.history_len, p.meta

    def predict_xyxy(self, boxes, mask):
        if len(boxes) == 0:
            return np.zeros((0, 4))
        feats, ref = encode_windows(boxes, mask)
        with torch.no_grad():
            out = self.p.model.predict(torch.from_numpy(feats))
        return cxcywh_to_xyxy(decode_outputs(out.cpu().numpy().astype(np.float64), ref))


def _random_tracks(rng, n):
    trks = []
    for _ in range(n):
        t = KalmanBoxTracker(np.r_[rng.uniform(0, 500, 2), rng.uniform(600, 900, 2), 0.9])
        t.gru_birth_obs = np.r_[t.get_state()[0], 0.9] if rng.random() < 0.8 else None
        if t.gru_birth_obs is None:
            del t.gru_birth_obs
        age = int(rng.integers(0, 30))
        t.age = age
        for k in range(1, age):
            if rng.random() < 0.7:
                b = np.r_[rng.uniform(0, 500, 2), rng.uniform(600, 900, 2), 0.9].astype(
                    np.float32 if rng.random() < 0.5 else np.float64)
                r = rng.random()
                if r < 0.03:
                    b[2] = b[0]                 # zero width -> invalid slot
                elif r < 0.05:
                    b[1] = np.nan               # non-finite -> invalid slot
                t.observations[k] = b
        trks.append(t)
    return trks


def test_track_windows_matches_per_track():
    from trackers.ocsort_gru.history import track_windows
    rng = np.random.default_rng(0)
    for trial in range(300):
        trks = _random_tracks(rng, int(rng.integers(0, 12)))
        for L in (10, 5):
            b, m = track_windows(trks, L)
            ref = [track_window(t, L) for t in trks]
            rb = np.stack([r[0] for r in ref]) if ref else np.zeros((0, L, 4))
            rm = np.stack([r[1] for r in ref]) if ref else np.zeros((0, L))
            assert b.dtype == rb.dtype == np.float64 and b.shape == rb.shape
            assert np.array_equal(b, rb) and np.array_equal(m, rm), trial   # bit-identical


def test_predictor_inference_mode_identical():
    p = GRUBoxPredictor(_fixture_model(3), 10)
    ref = _RefPredictor(p)
    rng = np.random.default_rng(1)
    for n in (0, 1, 7, 33):
        boxes = np.c_[rng.uniform(100, 900, (n * 10, 2)), rng.uniform(40, 300, (n * 10, 2))].reshape(n, 10, 4)
        mask = (rng.random((n, 10)) < 0.8).astype(float)
        if n:
            mask[:, -1] = 1
        boxes[mask == 0] = 0
        assert np.array_equal(p.predict_xyxy(boxes, mask), ref.predict_xyxy(boxes, mask))


def _multi_object_dets(seed, n_frames=120):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(100, 800, (8, 2)); vel = rng.normal(0, 4, (8, 2))
    dets = {}
    for f in range(1, n_frames + 1):
        pos += vel + rng.normal(0, 1.5, pos.shape)
        b = np.c_[pos - [30, 70], pos + [30, 70]]
        keep = rng.random(8) > (0.35 if f % 17 < 4 else 0.08)        # bursts of misses -> gaps
        d = b[keep] + rng.normal(0, 2, (keep.sum(), 4))
        dets[f] = np.c_[d, rng.uniform(0.62, 0.99, keep.sum())] if keep.any() else np.empty((0, 5))
    return dets


def _run_tracker(cls, predictor, dets, **kw):
    trk = cls(det_thresh=0.6, predictor=predictor, log_decisions=True, **kw)
    outs, trks_seen = [], []
    orig = trk._predict_boxes

    def spy():
        r = orig()
        trks_seen.append(r.copy())
        return r
    trk._predict_boxes = spy
    for f in sorted(dets):
        outs.append(trk.update(dets[f].copy(), (1000, 1000), (1000, 1000)))
    return outs, trks_seen, dict(trk.stats), list(trk.decision_log)


def test_optimised_tracker_identical_to_reference():
    """Same first-round boxes, decisions, stats and outputs as the pre-optimisation path.
    Covers frames with zero / one / many eligible tracks, gaps, births, deaths, Kalman fallback."""
    predictors = {"fixture": GRUBoxPredictor(_fixture_model(4), 10)}
    for name, ck in (("std", "gru_runs/standard_v1/seed0_a40/best.pt"),
                     ("noise", "gru_runs/noise_augmented/standard_v1_seed0_a40/best.pt"),
                     ("gap", "gru_runs/gap_aware/noise_v1_seed0_a40/best.pt")):
        path = os.path.join(_ROOT, ck)
        if os.path.isfile(path):
            predictors[name] = GRUBoxPredictor.from_checkpoint(path)
    for name, p in predictors.items():
        for seed in (0, 1, 2):
            dets = _multi_object_dets(seed)
            # run sequentially: KalmanBoxTracker.count is a class-global id counter
            ref = _run_tracker(_RefTracker, _RefPredictor(p), dets)
            new = _run_tracker(OCSortGRU, p, dets)
            assert len(ref[1]) == len(new[1])
            assert all(np.array_equal(a, b) for a, b in zip(ref[1], new[1])), (name, seed)   # first-round boxes
            assert all(np.array_equal(a, b) for a, b in zip(ref[0], new[0])), (name, seed)   # outputs + ids
            assert ref[2] == new[2] and ref[3] == new[3], (name, seed)                        # stats, decisions
            used = [r for r in new[3] if r[6] == "gru_used"]
            per_frame = {}
            for r in used:
                per_frame[r[0]] = per_frame.get(r[0], 0) + 1
            assert used and max(per_frame.values()) > 1 and len(new[3]) > len(used)       # mixed eligibility
            if name == "gap":
                assert any(r[6] == "gru_used" and r[5] > 1 for r in new[3])               # gap predictions used
    # always-invalid predictor: identical fallback path
    class _Nan(object):
        support, history_len, meta = predictors["fixture"].support, 10, {}

        def predict_xyxy(self, boxes, mask):
            return np.full((len(boxes), 4), np.nan)
    dets = _multi_object_dets(5)
    ref = _run_tracker(_RefTracker, _Nan(), dets)
    new = _run_tracker(OCSortGRU, _Nan(), dets)
    assert all(np.array_equal(a, b) for a, b in zip(ref[0], new[0])) and ref[2] == new[2]
    assert new[2]["fallback_invalid_output"] > 0 and new[2]["gru_used"] == 0
    # zero tracks / empty frames
    t = OCSortGRU(det_thresh=0.6, predictor=predictors["fixture"])
    assert t.update(np.empty((0, 5)), (1000, 1000), (1000, 1000)).shape == (0, 5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
