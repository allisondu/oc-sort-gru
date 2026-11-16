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

from gru_motion import (GRUBoxPredictor, GRUCheckpointError, GRUMotionModel, build_window,
                        decode_outputs, encode_targets, encode_windows, load_checkpoint, save_checkpoint)
from gru_motion.features import cxcywh_to_xyxy, window_stats, xyxy_to_cxcywh
from trackers.ocsort_gru import OCSortGRU
from trackers.ocsort_gru.history import track_window
from trackers.ocsort_tracker.ocsort import OCSort


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


def test_checkpoint_validation():
    with tempfile.TemporaryDirectory() as d:
        for bad in (None, os.path.join(d, "missing.pt")):
            try:
                load_checkpoint(bad)
                raise AssertionError("accepted missing checkpoint")
            except GRUCheckpointError:
                pass
        untrained = os.path.join(d, "untrained.pt")
        save_checkpoint(untrained, _fixture_model(), 10, {"epochs": 0, "num_train_windows": 0})
        try:
            load_checkpoint(untrained)
            raise AssertionError("accepted checkpoint without training record")
        except GRUCheckpointError as e:
            assert "no record of training" in str(e)
        foreign = os.path.join(d, "foreign.pt")
        torch.save({"model": {}}, foreign)
        try:
            load_checkpoint(foreign)
            raise AssertionError("accepted foreign checkpoint")
        except GRUCheckpointError:
            pass
        ok = os.path.join(d, "ok.pt")
        m = _fixture_model()
        save_checkpoint(ok, m, 10, {"epochs": 1, "num_train_windows": 1, "note": "test fixture"})
        m2, meta = load_checkpoint(ok)
        x = torch.randn(3, 10, 5)
        x[..., 4] = 1
        assert torch.allclose(m.predict(x), m2.predict(x))
        assert meta["history_len"] == 10 and len(meta["sha256"]) == 64


def _synthetic_frames(n_frames=60, n_obj=4, seed=0):
    """Objects on curved paths with occasional missed detections -> list of (N,5) arrays."""
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n_frames):
        dets = []
        for k in range(n_obj):
            if rng.random() < 0.1:
                continue                                   # missed detection
            cx = 200 + 300 * k + 40 * np.sin(t / 6.0 + k)
            cy = 400 + 30 * np.cos(t / 5.0 + k)
            dets.append([cx - 50, cy - 120, cx + 50, cy + 120, 0.95])
        frames.append(np.array(dets) if dets else np.empty((0, 5)))
    return frames


def test_tracker_window_timing():
    model = _fixture_model()
    pred = GRUBoxPredictor(model, history_len=10)
    trk_gru = OCSortGRU(det_thresh=0.6, predictor=pred, max_gap=1, min_obs=3)
    checks = {"windows": 0}
    orig = trk_gru._predict_boxes

    def checked_predict_boxes():
        # Kalman-only boxes computed the upstream way must equal what the GRU
        # tracker's Kalman filters predicted (GRU must not touch KF state).
        trks = orig()
        for i, t in enumerate(trk_gru.trackers):
            kf_box = t.history[-1][0]                      # Kalman box appended by predict()
            assert np.all(np.isfinite(kf_box))
            assert all(k < t.age for k in t.observations)  # no current/future observations
            boxes, mask = track_window(t, 10)
            nv, tr = window_stats(mask)
            if t.last_observation.sum() >= 0 and t.time_since_update <= 10:
                assert tr + 1 == t.time_since_update        # window gap == frames since last obs
                assert np.allclose(cxcywh_to_xyxy(boxes[9 - tr]), t.last_observation[:4])
            checks["windows"] += 1
        return trks
    trk_gru._predict_boxes = checked_predict_boxes

    for dets in _synthetic_frames():
        x = trk_gru.update(dets.copy(), (1080, 1920), (1080, 1920))
        assert x.shape[1] == 5
    s = trk_gru.stats
    assert s["gru"] > 0 and s["track_predictions"] == s["gru"] + s["fallback_gap"] + \
        s["fallback_short_history"] + s["fallback_invalid_output"]
    assert checks["windows"] > 0


def test_kalman_box_equivalence_when_gru_returns_kalman():
    """If the 'GRU' echoes the Kalman boxes, OCSortGRU must reproduce OCSort exactly.
    Exercises row alignment between eligible tracks, predictions and `trks`."""
    class KalmanEcho(object):
        history_len = 10
        tracker = None

        def predict_xyxy(self, boxes, mask):
            trs = self.tracker.trackers
            nv, tr = window_stats(np.array([track_window(t, 10)[1] for t in trs]))
            elig = np.flatnonzero((tr + 1 <= self.tracker.max_gap) & (nv >= self.tracker.min_obs))
            assert len(elig) == len(boxes)
            return np.array([trs[i].history[-1][0] for i in elig])   # Kalman box from predict()

    frames = _synthetic_frames(seed=1)
    # Run sequentially: KalmanBoxTracker.count is a class-level global ID counter
    # shared by all tracker instances (reset in OCSort.__init__).
    base = OCSort(det_thresh=0.6)
    out_base = [base.update(d.copy(), (1080, 1920), (1080, 1920)) for d in frames]
    echo = KalmanEcho()
    gru = OCSortGRU(det_thresh=0.6, predictor=echo, max_gap=1, min_obs=3)
    echo.tracker = gru
    out_gru = [gru.update(d.copy(), (1080, 1920), (1080, 1920)) for d in frames]
    assert all(np.array_equal(a, b) for a, b in zip(out_base, out_gru))
    assert gru.stats["gru"] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
