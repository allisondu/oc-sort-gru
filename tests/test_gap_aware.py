"""
Tests for gap-aware GRU training and short-gap inference support.

    python tests/test_gap_aware.py
"""
import hashlib
import os
import sys
import tempfile

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from gru_motion import GRUBoxPredictor, GRUMotionModel, build_window, load_checkpoint, save_checkpoint  # noqa: E402
from gru_motion.augment import (BoxNoiseConfig, apply_trailing_gaps, balanced_gaps, gap_rng,          # noqa: E402
                                holdout_rng, perturb_windows)
from gru_motion.data import SamplingSpec, build_raw_windows, encode_raw, enumerate_samples, load_sequences  # noqa: E402
from gru_motion.features import cxcywh_to_xyxy, decode_outputs, window_stats, xyxy_to_cxcywh            # noqa: E402
from gru_motion.model import GRUCheckpointError, validate_support                                       # noqa: E402
from trackers.ocsort_gru import OCSortGRU                                                               # noqa: E402
from trackers.ocsort_gru.history import track_window                                                    # noqa: E402
from trackers.ocsort_gru.tracker import effective_policy                                                # noqa: E402
from trackers.ocsort_tracker.ocsort import KalmanBoxTracker, OCSort                                     # noqa: E402

_DATA = os.path.join(_ROOT, "datasets_gru", "dancetrack_gt")
GAP_SUPPORT = {"history_len": 10, "min_valid_obs": 7, "require_full_history": False, "max_trailing_gap": 3,
               "prediction_horizons": [1, 2, 3, 4], "require_full_prefix": True}
JITTER = BoxNoiseConfig(0.025, 0.025, 0.025, 0.025, 1.0, 3.0)


def _expect(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError("expected {}".format(exc.__name__))


def _box(f):                                 # known synthetic trajectory: x moves 5 px / frame
    return np.array([100. + 5 * f, 50., 160. + 5 * f, 250.])


# ----------------------------------------------------------------------------- indexing / masking

def test_tracker_temporal_indexing_and_horizon():
    """Track observed at frames 0..11, missed at 12..14, re-observed at 15 (age == frame here)."""
    trk = KalmanBoxTracker(np.r_[_box(0), .9])
    trk.gru_birth_obs = np.r_[_box(0), .9]
    observed = {f: f <= 11 or f >= 15 for f in range(1, 17)}
    for f in range(1, 17):
        trk.predict()                                       # association for frame f happens now
        boxes, mask = track_window(trk, 10)                 # slots = frames f-10 .. f-1
        nv, tr = window_stats(mask)
        last_obs = max(k for k in range(0, f) if k == 0 or observed[k])
        assert trk.age == f
        assert tr == f - 1 - last_obs                       # trailing missing slots = consecutive misses
        assert trk.time_since_update == f - last_obs        # = frames since last observation = horizon
        assert int(tr) + 1 == trk.time_since_update
        assert np.allclose(boxes[9 - tr], xyxy_to_cxcywh(_box(last_obs)))   # reference = last genuine obs
        assert np.all(boxes[mask == 0] == 0)
        if f == 15:                                          # reappearance frame: 3 missed slots, horizon 4
            assert tr == 3 and trk.time_since_update == 4 and nv == 7
            assert not np.any(np.all(np.isclose(boxes, xyxy_to_cxcywh(_box(15))), axis=1))
        trk.update(np.r_[_box(f), .9] if observed[f] else None)


def test_trailing_gap_masking_and_balance():
    rng = np.random.default_rng(0)
    boxes = rng.uniform(50, 300, (40, 10, 4)); mask = np.ones((40, 10))
    b0, m0 = boxes.copy(), mask.copy()
    gaps = np.arange(40) % 4
    nb, nm = apply_trailing_gaps(boxes, mask, gaps)
    assert np.array_equal(boxes, b0) and np.array_equal(mask, m0)            # no mutation
    for i, g in enumerate(gaps):
        assert nm[i].tolist() == [1.0] * (10 - g) + [0.0] * g                # positions kept, no compression
        assert np.all(nb[i, 10 - g:] == 0) and np.array_equal(nb[i, :10 - g], boxes[i, :10 - g])
    z, zm = apply_trailing_gaps(boxes, mask, np.zeros(40, int))
    assert np.array_equal(z, boxes) and np.array_equal(zm, mask)            # gap 0 = identity
    _expect(ValueError, apply_trailing_gaps, boxes, mask, np.full(40, 10))
    a = balanced_gaps(1001, 3, gap_rng(0, 1)); b = balanced_gaps(1001, 3, gap_rng(0, 1))
    c = balanced_gaps(1001, 3, gap_rng(0, 2)); d = balanced_gaps(1001, 3, gap_rng(1, 1))
    assert np.array_equal(a, b) and not np.array_equal(a, c) and not np.array_equal(a, d)
    counts = np.bincount(a, minlength=4)
    assert counts.max() - counts.min() <= 1 and counts.sum() == 1001


def test_noise_only_on_observed_slots():
    rng = np.random.default_rng(1)
    boxes = np.c_[rng.uniform(100, 500, (200, 10, 2)), rng.uniform(40, 200, (200, 10, 2))].reshape(200, 10, 4)
    nb, nm = apply_trailing_gaps(boxes, np.ones((200, 10)), balanced_gaps(200, 3, gap_rng(0, 1)))
    noisy = perturb_windows(nb, nm, JITTER, holdout_rng(542))
    assert np.all(noisy[nm == 0] == 0)                                       # missing stays missing
    assert np.all(np.any(noisy[nm == 1] != nb[nm == 1], axis=-1))            # observed slots perturbed
    assert np.all(noisy[nm == 1][:, 2:] > 0)


def test_reference_target_encoding_and_decoding_with_gaps():
    boxes = np.stack([np.stack([xyxy_to_cxcywh(_box(f)) for f in range(10)])] * 4)       # frames 0..9
    tgt = np.stack([xyxy_to_cxcywh(_box(10))] * 4)                                         # target frame 10
    nb, nm = apply_trailing_gaps(boxes, np.ones((4, 10)), np.arange(4))
    feats, targets, ref = encode_raw(nb, nm, tgt)
    for g in range(4):
        assert np.allclose(ref[g], xyxy_to_cxcywh(_box(9 - g)))         # last genuine observation
        horizon = g + 1
        assert np.isclose(targets[g, 0], 5.0 * horizon / 200.0)         # dx = 5 px * horizon, in units of h=200
        assert np.allclose(decode_outputs(targets[g:g + 1].astype(np.float64), ref[g:g + 1])[0], tgt[g], rtol=1e-5)
        assert np.all(feats[g, 10 - g:] == 0) and np.all(feats[g, :10 - g, 4] == 1)
    # a static object (all boxes equal) is distinguishable from missing slots via the mask channel
    st, sm = apply_trailing_gaps(np.repeat(boxes[:1, :1], 10, axis=1), np.ones((1, 10)), np.array([2]))
    f, _, _ = encode_raw(st, sm, tgt[:1])
    assert np.all(f[0, :8, :4] == 0) and f[0, :8, 4].tolist() == [1] * 8 and f[0, 8:, 4].tolist() == [0, 0]


def test_no_future_leakage_or_identity_crossing():
    spec = SamplingSpec.make("gap_aware", 10, max_trailing_gap=3)
    frames = {f: _box(f) for f in range(1, 40)}
    other = {f: _box(f) + 400 for f in range(1, 40)}
    data = {"a": {1: frames, 2: other}, "b": {1: {f: _box(f) + 900 for f in range(1, 40)}}}
    samples = enumerate_samples(data, spec.source_spec())
    assert samples == enumerate_samples(data, SamplingSpec.make("standard", 10))
    b, m, t = build_raw_windows(data, samples, spec.source_spec())
    j = samples.index(("a", 1, 20))
    pois = {"a": {1: {f: (v + 777 if f >= 20 else v) for f, v in frames.items()}, 2: other}, "b": data["b"]}
    b2, _, _ = build_raw_windows(pois, [("a", 1, 20)], spec.source_spec())
    assert np.array_equal(b[j], b2[0])                                      # target & later frames never read
    assert np.allclose(t[j], xyxy_to_cxcywh(frames[20]))
    for (s, i, tf), bw in zip(samples, b):                                  # windows from one identity/sequence
        src = data[s][i]
        assert np.allclose(bw, np.stack([xyxy_to_cxcywh(src[f]) for f in range(tf - 10, tf)]))


# ----------------------------------------------------------------------------- spec / support / checkpoints

def test_spec_and_support_enforcement():
    s = SamplingSpec.make("gap_aware", 10, max_trailing_gap=3)
    assert (s.min_valid_obs, s.max_trailing_gap, s.require_full_prefix) == (7, 3, True)
    assert s.accepts(10, 0) and s.accepts(7, 3) and not s.accepts(6, 3) and not s.accepts(6, 4)
    assert not s.accepts(8, 1)                                              # hole before the gap
    tc = s.training_conditions(None, {"gaps": [0, 1, 2, 3]})
    assert tc["prediction_horizons"] == [1, 2, 3, 4] and tc["require_full_prefix"]
    _expect(ValueError, SamplingSpec.make, "gap_aware", 10, max_trailing_gap=0)
    _expect(ValueError, SamplingSpec.make, "gap_aware", 10, 5, 3)
    assert "require_full_prefix" not in SamplingSpec.make("standard", 10).to_dict()
    assert "require_full_prefix" not in SamplingSpec.make("standard", 10).training_conditions()
    validate_support(GAP_SUPPORT, 10, tc)
    wider = dict(GAP_SUPPORT, max_trailing_gap=4, prediction_horizons=[1, 2, 3, 4, 5], min_valid_obs=6)
    _expect(ValueError, validate_support, wider, 10, tc)                    # gap 4 never trained
    _expect(ValueError, validate_support, dict(GAP_SUPPORT, require_full_prefix=False), 10, tc)
    _expect(ValueError, validate_support, dict(GAP_SUPPORT, min_valid_obs=6), 10, tc)
    _expect(ValueError, validate_support, dict(GAP_SUPPORT, prediction_horizons=[1, 2, 3]), 10, tc)
    # runtime may only narrow; prefix requirement is kept
    p = effective_policy(GAP_SUPPORT, max_trailing_gap=1)
    assert p["max_trailing_gap"] == 1 and p["prediction_horizons"] == [1, 2] and p["require_full_prefix"]
    _expect(ValueError, effective_policy, GAP_SUPPORT, max_trailing_gap=4)
    _expect(ValueError, effective_policy, GAP_SUPPORT, min_obs=6)
    std = {"history_len": 10, "min_valid_obs": 10, "require_full_history": True, "max_trailing_gap": 0,
           "prediction_horizons": [1]}
    assert "require_full_prefix" not in effective_policy(std)              # old checkpoints unchanged


def test_checkpoint_roundtrip_and_old_checkpoints():
    spec = SamplingSpec.make("gap_aware", 10, max_trailing_gap=3)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "g.pt")
        save_checkpoint(p, GRUMotionModel(), spec.training_conditions({"x": 1}, {"gaps": [0, 1, 2, 3]}),
                        {"epochs": 1, "num_train_windows": 5})
        _, meta = load_checkpoint(p)
        assert meta["inference_support"] == GAP_SUPPORT
        assert meta["training_conditions"]["gap_sampling"] == {"gaps": [0, 1, 2, 3]}
        raw = torch.load(p, weights_only=True)
        raw["inference_support"]["max_trailing_gap"] = 4
        raw["inference_support"]["prediction_horizons"] = [1, 2, 3, 4, 5]
        torch.save(raw, p)
        _expect(GRUCheckpointError, load_checkpoint, p)                    # tampered support rejected
    for ck, sha in (("gru_runs/standard_v1/seed0_a40/best.pt", "9f062a5d"),
                    ("gru_runs/noise_augmented/standard_v1_seed0_a40/best.pt", "af0ae943")):
        path = os.path.join(_ROOT, ck)
        if os.path.isfile(path):
            pr = GRUBoxPredictor.from_checkpoint(path)
            assert pr.meta["sha256"].startswith(sha)
            pol = effective_policy(pr.support)
            assert pol["max_trailing_gap"] == 0 and pol["min_valid_obs"] == 10 and "require_full_prefix" not in pol


# ----------------------------------------------------------------------------- tracker integration

class _Recorder(object):
    """Stub predictor: records every window it receives, predicts the last observed box."""
    def __init__(self, support):
        self.support, self.history_len, self.meta, self.calls = support, 10, {}, []

    def predict_xyxy(self, boxes, mask):
        self.calls.append((boxes.copy(), mask.copy()))
        nv, tr = window_stats(mask)
        return cxcywh_to_xyxy(boxes[np.arange(len(boxes)), 9 - tr])


def _single_track(miss, n=40):
    dets = {}
    for f in range(1, n + 1):
        dets[f] = np.empty((0, 5)) if f in miss else np.r_[_box(f), .95][None]
    return dets


def test_tracker_eligibility_and_fallback():
    for gap_len, expect in ((1, "gru_used"), (2, "gru_used"), (3, "gru_used"), (4, "fallback_unsupported_gap")):
        miss = set(range(15, 15 + gap_len))
        rec = _Recorder(GAP_SUPPORT)
        trk = OCSortGRU(det_thresh=0.6, predictor=rec, log_decisions=True)
        dets = _single_track(miss)
        for f in sorted(dets):
            trk.update(dets[f].copy(), (1000, 1000), (1000, 1000))
        log = {r[0]: r for r in trk.decision_log}
        reap = 15 + gap_len                                                # reappearance frame
        assert log[reap][5] == gap_len + 1                                 # frames_since_last_obs = horizon
        assert log[reap][4] == 10 - gap_len                                # valid obs in window
        assert log[reap][6] == expect, (gap_len, log[reap])
        assert trk.stats["fallback_invalid_output"] == 0
    # windows the GRU received at the reappearance frame of a 3-frame gap: 7 genuine obs, 3 masked slots
    rec = _Recorder(GAP_SUPPORT)
    trk = OCSortGRU(det_thresh=0.6, predictor=rec)
    dets = _single_track({15, 16, 17})
    calls_before = None
    for f in sorted(dets):
        if f == 18:
            calls_before = len(rec.calls)
        trk.update(dets[f].copy(), (1000, 1000), (1000, 1000))
    boxes, mask = rec.calls[calls_before]
    assert mask[0].tolist() == [1] * 7 + [0] * 3
    assert np.allclose(boxes[0, 6], xyxy_to_cxcywh(_box(14)))             # last genuine obs = frame 14
    assert not np.any(np.all(np.isclose(boxes[0], xyxy_to_cxcywh(_box(18))), axis=1))
    # standard-support checkpoints: same gaps fall back exactly as before
    rec = _Recorder({"history_len": 10, "min_valid_obs": 10, "require_full_history": True,
                     "max_trailing_gap": 0, "prediction_horizons": [1]})
    trk = OCSortGRU(det_thresh=0.6, predictor=rec, log_decisions=True)
    for f, d in sorted(_single_track({15}).items()):
        trk.update(d.copy(), (1000, 1000), (1000, 1000))
    assert {r[0]: r for r in trk.decision_log}[16][6] == "fallback_missing_recent_observation"


def test_history_holes_fall_back():
    rec = _Recorder(GAP_SUPPORT)
    trk = OCSortGRU(det_thresh=0.6, predictor=rec, log_decisions=True)
    dets = _single_track({12, 16})         # hole at 12, then gap of 1 at 16
    for f in sorted(dets):
        trk.update(dets[f].copy(), (1000, 1000), (1000, 1000))
    log = {r[0]: r for r in trk.decision_log}
    assert log[17][6] == "fallback_history_holes"                          # 8 valid + 1 trailing != 10
    assert log[14][6] == "fallback_history_holes"                          # hole at 12, gap 0, 9 valid
    assert log[26][6] == "fallback_history_holes"                          # window 16..25: hole at 16
    assert log[27][6] == "gru_used"                                        # window 17..26 complete again
    assert trk.stats["fallback_history_holes"] >= 2


def test_invalid_output_falls_back_and_reproduces_ocsort():
    """A gap-aware predictor that always fails -> every row uses Kalman -> outputs identical to OC-SORT."""
    class _Broken(_Recorder):
        def predict_xyxy(self, boxes, mask):
            return np.full((len(boxes), 4), np.nan)
    rng = np.random.default_rng(3)
    dets = {}
    for f in range(1, 80):
        b = np.array([[100 + 4 * f, 100, 160 + 4 * f, 220], [400, 300 + 3 * f, 470, 450 + 3 * f],
                      [130 + 4 * f, 110, 190 + 4 * f, 230]], float)
        keep = rng.random(3) > 0.25
        dets[f] = np.c_[b[keep] + rng.normal(0, 1, (keep.sum(), 4)), np.full(keep.sum(), 0.9)]
    # run sequentially: KalmanBoxTracker.count is a class-global id counter reset by each constructor
    ref = OCSort(det_thresh=0.6)
    out_ref = [ref.update(dets[f].copy(), (1000, 1000), (1000, 1000)) for f in sorted(dets)]
    gru = OCSortGRU(det_thresh=0.6, predictor=_Broken(GAP_SUPPORT))
    out_gru = [gru.update(dets[f].copy(), (1000, 1000), (1000, 1000)) for f in sorted(dets)]
    assert all(np.array_equal(a, b) for a, b in zip(out_ref, out_gru))
    assert gru.stats["gru_used"] == 0 and gru.stats["fallback_invalid_output"] > 0
    for ta, tb in zip(ref.trackers, gru.trackers):
        assert np.array_equal(ta.kf.x, tb.kf.x)                        # Kalman states untouched


def test_tiny_gap_aware_training_run():
    if not os.path.isdir(os.path.join(_DATA, "train", "dancetrack0006")):
        print("  (skipped: datasets_gru not present)")
        return
    import gru_motion.train as T
    real_eval = T.eval_loss
    seen = []

    def ev(model, x, y, bs):
        seen.append(hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest())
        return real_eval(model, x, y, bs)
    T.eval_loss = ev
    try:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "g")
            T.main(["--data_root", _DATA, "--seqs", "dancetrack0001", "dancetrack0002", "dancetrack0006",
                    "--holdout_seqs", "dancetrack0006", "--training_mode", "gap_aware", "--max_trailing_gap", "3",
                    "--box_noise", "jitter_v1", "--max_windows", "400", "--epochs", "3", "--batch_size", "64",
                    "--seed", "0", "--out_dir", out])
            best = torch.load(os.path.join(out, "best.pt"), weights_only=True)
            ti, tc = best["train_info"], best["training_conditions"]
            assert tc["training_mode"] == "gap_aware" and tc["prediction_horizons"] == [1, 2, 3, 4]
            assert best["inference_support"] == GAP_SUPPORT
            assert tc["augmentation"]["config"]["center_x_std"] == 0.025 and tc["gap_sampling"]["gaps"] == [0, 1, 2, 3]
            ck = ti["checkpoint"]
            assert ck["selection_metric"] == "holdout_noisy" and "UNWEIGHTED MEAN" in ck["selection"]["metric"]
            hist = ti["loss_history"]
            for r in hist:
                bg = r["holdout_by_gap"]
                assert set(bg) == {"0", "1", "2", "3"}
                assert np.isclose(r["holdout_noisy_loss"], np.mean([bg[k]["noisy"] for k in bg]))
                assert np.isclose(r["holdout_loss"], np.mean([bg[k]["clean"] for k in bg]))
                assert all(np.isfinite([bg[k][c] for k in bg for c in ("clean", "noisy")]))
            counts = [r["train_gap_counts"] for r in hist[1:]]
            assert all(sorted(c.values()) == [100, 100, 100, 100] for c in counts)   # exactly balanced
            assert all("train_noise" in r for r in hist[1:])
            # holdout tensors identical across epochs (8 sets: 4 gaps x clean/noisy); train-eval set fixed
            per_epoch = len(seen) // len(hist)
            assert per_epoch == 9
            for e in range(1, len(hist)):
                assert seen[e * 9:(e + 1) * 9] == seen[:9]
            # noisy g=0 holdout = exactly the noise-augmented model's fixed noisy holdout realisation
            hd, _ = load_sequences([os.path.join(_DATA, "train", "dancetrack0006")])
            sp = SamplingSpec.make("standard", 10)
            hb, hm, ht = build_raw_windows(hd, enumerate_samples(hd, sp), sp)
            nf, _, _ = encode_raw(perturb_windows(hb, hm, JITTER, holdout_rng(542)), hm, ht)
            assert tc["augmentation"]["noisy_holdout"]["features_sha256"] == hashlib.sha256(nf.tobytes()).hexdigest()
            # loads into the predictor and the tracker with gap support; predictions valid for all gaps
            pr = GRUBoxPredictor.from_checkpoint(os.path.join(out, "best.pt"))
            for g in range(4):
                b, m = apply_trailing_gaps(hb[:50], hm[:50], np.full(50, g))
                p = pr.predict_xyxy(b, m)
                assert np.all(np.isfinite(p)) and np.all(p[:, 2] > p[:, 0]) and np.all(p[:, 3] > p[:, 1])
            t = OCSortGRU(det_thresh=0.6, predictor=pr)
            assert t.policy["max_trailing_gap"] == 3 and t.policy["require_full_prefix"]
    finally:
        T.eval_loss = real_eval


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
