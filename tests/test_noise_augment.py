"""
Tests for detector-noise augmentation (gru_motion/augment.py) and its use in
gru_motion/train.py. Real-data training tests use the DanceTrack TRAIN split
annotations under datasets_gru/ and are skipped if absent.

    python tests/test_noise_augment.py      (or: pytest tests/test_noise_augment.py)
"""
import hashlib
import os
import sys
import tempfile

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from gru_motion import GRUBoxPredictor, build_window, decode_outputs                       # noqa: E402
from gru_motion.augment import BoxNoiseConfig, epoch_rng, holdout_rng, perturb_windows      # noqa: E402
from gru_motion.data import (SamplingSpec, build_arrays, build_raw_windows, encode_raw,      # noqa: E402
                             enumerate_samples, load_sequences, normalization_stats)
from trackers.ocsort_gru import OCSortGRU                                                   # noqa: E402

_DATA = os.path.join(_ROOT, "datasets_gru", "dancetrack_gt")
JITTER = BoxNoiseConfig(0.025, 0.025, 0.025, 0.025, 1.0, 3.0)


def _windows(n=64, L=10, seed=0, missing=False):
    rng = np.random.default_rng(seed)
    cx = 600 + np.cumsum(rng.normal(0, 4, (n, L)), 1)
    cy = 400 + np.cumsum(rng.normal(0, 4, (n, L)), 1)
    w = rng.uniform(40, 200, (n, 1)) * np.exp(np.cumsum(rng.normal(0, .01, (n, L)), 1))
    h = w * rng.uniform(1.8, 3.0, (n, 1))
    boxes = np.stack([cx, cy, w, h], -1)
    mask = np.ones((n, L))
    if missing:
        mask[:, :3] = rng.random((n, 3)) < 0.5
        boxes[mask == 0] = 0.0
    tgt = boxes[:, -1] + np.array([3.0, -2.0, 1.0, 2.0])
    return boxes, mask, tgt


def _expect(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError("expected {}".format(exc.__name__))


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ----------------------------------------------------------------------------- unit tests

def test_zero_noise_reproduces_standard_inputs():
    boxes, mask, tgt = _windows()
    for cfg in (BoxNoiseConfig(), BoxNoiseConfig(0, 0, 0, 0, 1.0), JITTER.__class__(**dict(
            JITTER.__dict__, apply_prob=0.0))):
        assert not cfg.enabled
        out = perturb_windows(boxes, mask, cfg, np.random.default_rng(0))
        assert np.array_equal(out, boxes)
        f0, t0, _ = encode_raw(boxes, mask, tgt)
        f1, t1, _ = encode_raw(out, mask, tgt)
        assert np.array_equal(f0, f1) and np.array_equal(t0, t1)


def test_noise_changes_observations_not_targets_or_inputs():
    boxes, mask, tgt = _windows()
    b0, m0, t0 = boxes.copy(), mask.copy(), tgt.copy()
    out = perturb_windows(boxes, mask, JITTER, np.random.default_rng(1))
    assert np.array_equal(boxes, b0) and np.array_equal(mask, m0) and np.array_equal(tgt, t0)  # no mutation
    assert np.all(np.any(out != boxes, axis=-1))                     # every valid slot perturbed (p = 1)
    # relative magnitudes follow the configured stds (n = 640 slots per coordinate)
    rel = np.c_[((out - boxes)[..., 0] / boxes[..., 2]).ravel(), ((out - boxes)[..., 1] / boxes[..., 3]).ravel(),
                np.log(out[..., 2] / boxes[..., 2]).ravel(), np.log(out[..., 3] / boxes[..., 3]).ravel()]
    assert np.all(np.abs(rel.std(0) - 0.025) < 0.004), rel.std(0)
    assert np.all(np.abs(rel) <= 3 * 0.025 + 1e-12)                  # truncation at 3 sigma


def test_outputs_finite_valid_and_masks_unchanged():
    boxes, mask, tgt = _windows(n=500, missing=True)
    strong = BoxNoiseConfig(0.5, 0.5, 0.5, 0.5, 1.0, 0.0)            # extreme, untruncated
    for cfg in (JITTER, strong):
        out = perturb_windows(boxes, mask, cfg, np.random.default_rng(2))
        assert np.all(np.isfinite(out))
        valid = mask > 0
        assert np.all(out[valid][:, 2:] > 0)
        assert np.all(out[~valid] == 0)                              # missing slots untouched
        f, t, _ = encode_raw(out, mask, tgt)
        assert np.array_equal(f[..., 4], mask.astype(np.float32))    # mask channel unchanged
        assert np.all(np.isfinite(f)) and np.all(np.isfinite(t))
    _expect(ValueError, BoxNoiseConfig, -0.1)
    _expect(ValueError, BoxNoiseConfig, 0.1, apply_prob=1.5)
    _expect(ValueError, BoxNoiseConfig, float("nan"))


def test_apply_prob_perturbs_a_fraction_of_windows():
    boxes, mask, _ = _windows(n=4000)
    out = perturb_windows(boxes, mask, BoxNoiseConfig(0.02, 0.02, 0.02, 0.02, apply_prob=0.3),
                          np.random.default_rng(3))
    changed = np.any(out != boxes, axis=(1, 2))
    assert abs(changed.mean() - 0.3) < 0.03
    assert np.array_equal(out[~changed], boxes[~changed])            # untouched windows exactly clean


def test_input_target_alignment():
    """Target = clean GT box encoded against the NOISY reference: decoding with that
    reference returns the clean box exactly, and the reference is the last noisy slot."""
    boxes, mask, tgt = _windows(missing=True)
    out = perturb_windows(boxes, mask, JITTER, np.random.default_rng(4))
    f, t, ref = encode_raw(out, mask, tgt)
    assert np.allclose(ref, out[:, -1])
    assert np.allclose(decode_outputs(t.astype(np.float64), ref), tgt, rtol=1e-5)
    _, tc, refc = encode_raw(boxes, mask, tgt)
    assert not np.allclose(t, tc)          # encoded target changes only through the noisy reference
    assert np.allclose(decode_outputs(tc.astype(np.float64), refc), tgt, rtol=1e-5)


def test_seed_determinism_and_epoch_resampling():
    boxes, mask, _ = _windows()
    a = perturb_windows(boxes, mask, JITTER, epoch_rng(0, 1))
    b = perturb_windows(boxes, mask, JITTER, epoch_rng(0, 1))
    c = perturb_windows(boxes, mask, JITTER, epoch_rng(1, 1))       # other seed
    d = perturb_windows(boxes, mask, JITTER, epoch_rng(0, 2))       # other epoch
    e = perturb_windows(boxes, mask, JITTER, holdout_rng(0))        # holdout stream
    assert np.array_equal(a, b)
    for o in (c, d, e):
        assert not np.allclose(a, o)
    assert np.array_equal(perturb_windows(boxes, mask, JITTER, holdout_rng(542)),
                          perturb_windows(boxes, mask, JITTER, holdout_rng(542)))


def test_build_arrays_unchanged_by_refactor():
    spec = SamplingSpec.make("standard", 10)
    rng = np.random.default_rng(0)
    frames = {f: np.array([100 + 3 * f, 50, 160 + 3 * f, 250]) + rng.normal(0, 1, 4) for f in range(1, 40)}
    data = {"s": {1: frames}}
    samples = enumerate_samples(data, spec)
    f, t, r = build_arrays(data, samples, spec)
    b, m, tg = build_raw_windows(data, samples, spec)
    f2, t2, r2 = encode_raw(b, m, tg)
    assert np.array_equal(f, f2) and np.array_equal(t, t2) and np.array_equal(r, r2)
    bw, _ = build_window(frames, samples[0][2], 10)
    assert np.array_equal(b[0], bw)


# ----------------------------------------------------------------------------- training integration

def _train(out_dir, extra=(), script=None):
    """Tiny real-data run (2 train seqs + 1 holdout seq, all from the TRAIN split).
    Records every eval_loss / perturb_windows call. script: {"clean": [...], "noisy": [...]}
    replaces holdout losses (index 0 = before training)."""
    import gru_motion.train as T
    real_eval, real_perturb = T.eval_loss, T.perturb_windows
    log = {"eval": [], "perturb": []}
    hd, _ = load_sequences([os.path.join(_DATA, "train", "dancetrack0006")])
    hold_n = len(enumerate_samples(hd, SamplingSpec.make("standard", 10)))
    count = {"n": 0}

    def ev(model, x, y, bs):
        v = real_eval(model, x, y, bs)
        kind = "train"
        if len(x) == hold_n:
            kind = "clean" if count["n"] % 2 == 0 or "--box_noise" not in extra else "noisy"
            idx = count["n"] // (2 if "--box_noise" in extra else 1)
            count["n"] += 1
            if script is not None:
                v = script[kind][idx]
        log["eval"].append((kind, _sha(x.cpu().numpy()), _sha(y.cpu().numpy()), v))
        return v

    def pt(boxes, mask, cfg, rng):
        out = real_perturb(boxes, mask, cfg, rng)
        log["perturb"].append((len(boxes), _sha(out), _sha(boxes)))
        return out

    T.eval_loss, T.perturb_windows = ev, pt
    try:
        argv = ["--data_root", _DATA, "--seqs", "dancetrack0001", "dancetrack0002", "dancetrack0006",
                "--holdout_seqs", "dancetrack0006", "--training_mode", "standard", "--max_windows", "400",
                "--epochs", "3", "--batch_size", "64", "--seed", "0", "--out_dir", out_dir] + list(extra)
        T.main(argv)
    finally:
        T.eval_loss, T.perturb_windows = real_eval, real_perturb
    return log, hold_n


def _have_data():
    if not os.path.isdir(os.path.join(_DATA, "train", "dancetrack0006")):
        print("  (skipped: datasets_gru not present)")
        return False
    return True


def test_training_noise_resampled_and_holdout_fixed():
    if not _have_data():
        return
    with tempfile.TemporaryDirectory() as d:
        log, hold_n = _train(os.path.join(d, "n"), ["--box_noise", "jitter_v1"])
        train_calls = [p for p in log["perturb"] if p[0] != hold_n]
        hold_calls = [p for p in log["perturb"] if p[0] == hold_n]
        assert len(train_calls) == 3 and len(hold_calls) == 1          # one draw per epoch, holdout once
        assert len({p[1] for p in train_calls}) == 3                    # different noise every epoch
        assert len({p[2] for p in train_calls}) == 1                    # always from the same clean boxes
        clean = [e for e in log["eval"] if e[0] == "clean"]
        noisy = [e for e in log["eval"] if e[0] == "noisy"]
        assert len(clean) == 4 and len(noisy) == 4                      # epochs 0..3
        assert len({(e[1], e[2]) for e in noisy}) == 1                  # noisy holdout fixed across epochs
        assert len({(e[1], e[2]) for e in clean}) == 1
        assert clean[0][1] != noisy[0][1]
        # clean holdout = holdout built with no augmentation at all
        hd, _ = load_sequences([os.path.join(_DATA, "train", "dancetrack0006")])
        spec = SamplingSpec.make("standard", 10)
        hf, _, _ = build_arrays(hd, enumerate_samples(hd, spec), spec)
        assert clean[0][1] == _sha(hf)
        # "eval train" is on the clean training windows, identical in every epoch
        assert len({e[1] for e in log["eval"] if e[0] == "train"}) == 1
        # noisy holdout is reproducible: a second run gives the same fixed set and the same weights
        log2, _ = _train(os.path.join(d, "n2"), ["--box_noise", "jitter_v1"])
        assert [e[1] for e in log2["eval"] if e[0] == "noisy"] == [e[1] for e in noisy]
        a = torch.load(os.path.join(d, "n", "last.pt"), weights_only=True)["state_dict"]
        b = torch.load(os.path.join(d, "n2", "last.pt"), weights_only=True)["state_dict"]
        assert all(torch.equal(a[k], b[k]) for k in a)
        # another training-noise seed changes training noise but NOT the noisy holdout
        log3, _ = _train(os.path.join(d, "n3"), ["--box_noise", "jitter_v1", "--train_noise_seed", "7"])
        assert [p[1] for p in log3["perturb"] if p[0] != hold_n] != [p[1] for p in train_calls]
        assert [e[1] for e in log3["eval"] if e[0] == "noisy"] == [e[1] for e in noisy]


def test_noisy_selection_and_checkpoint_metadata():
    if not _have_data():
        return
    with tempfile.TemporaryDirectory() as d:
        # clean holdout is best at epoch 1, noisy holdout at epoch 2 -> best.pt must be epoch 2
        out = os.path.join(d, "s")
        _train(out, ["--box_noise", "jitter_v1"],
               script={"clean": [0.5, 0.2, 0.3, 0.4], "noisy": [0.9, 0.6, 0.5, 0.7]})
        best = torch.load(os.path.join(out, "best.pt"), weights_only=True)
        last = torch.load(os.path.join(out, "last.pt"), weights_only=True)
        ck = best["train_info"]["checkpoint"]
        assert ck["weights_epoch"] == 2 and ck["selection_metric"] == "holdout_noisy"
        assert ck["selection_loss"] == 0.5 and ck["holdout_noisy_loss"] == 0.5 and ck["holdout_loss"] == 0.3
        assert ck["selection"]["loss_key"] == "holdout_noisy_loss"
        assert last["train_info"]["checkpoint"]["weights_epoch"] == 3
        # augmentation recorded in training_conditions and train_info
        for c in (best, last):
            aug = c["training_conditions"]["augmentation"]
            assert aug["config"]["center_x_std"] == 0.025 and aug["config"]["apply_prob"] == 1.0
            assert aug["preset"] == "jitter_v1" and aug["train_noise_seed"] == 0
            assert aug["noisy_holdout"]["seed"] == 542 and len(aug["noisy_holdout"]["features_sha256"]) == 64
            assert c["train_info"]["augmentation"] == aug
            assert all("train_noise" in r for r in c["train_info"]["loss_history"][1:])
        # normalisation = clean TRAINING windows only (holdout excluded), same as a clean run
        tr, _ = load_sequences([os.path.join(_DATA, "train", s) for s in ("dancetrack0001", "dancetrack0002")])
        spec = SamplingSpec.make("standard", 10)
        from gru_motion.data import subsample
        f, t, _ = build_arrays(tr, subsample(enumerate_samples(tr, spec), 400, 0), spec)
        st = normalization_stats(f, t)
        nm = best["train_info"]["normalization"]
        for k, v in zip(("in_mean", "in_std", "out_mean", "out_std"), st):
            assert np.allclose(nm[k], v)
        _train(os.path.join(d, "c"))
        clean_ck = torch.load(os.path.join(d, "c", "best.pt"), weights_only=True)
        assert clean_ck["train_info"]["normalization"]["in_mean"] == nm["in_mean"]
        assert clean_ck["training_conditions"]["augmentation"] is None
        assert clean_ck["train_info"]["checkpoint"]["selection_metric"] == "holdout_clean"
        # loads into the existing predictor with unchanged (full-history) inference support
        for name in ("best.pt", "last.pt"):
            p = GRUBoxPredictor.from_checkpoint(os.path.join(out, name))
            assert p.support == clean_ck["inference_support"]
            assert p.support["min_valid_obs"] == 10 and p.support["require_full_history"]
            assert p.support["max_trailing_gap"] == 0
            _expect(ValueError, OCSortGRU, det_thresh=0.6, predictor=p, min_obs=3)
            boxes, mask, _ = _windows(n=5)
            assert np.all(np.isfinite(p.predict_xyxy(boxes, mask)))


def test_selection_metric_guards():
    if not _have_data():
        return
    import gru_motion.train as T
    with tempfile.TemporaryDirectory() as d:
        _expect(SystemExit, T.main, ["--data_root", _DATA, "--seqs", "dancetrack0001", "dancetrack0006",
                                     "--holdout_seqs", "dancetrack0006", "--training_mode", "standard",
                                     "--selection_metric", "holdout_noisy", "--out_dir", os.path.join(d, "x")])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
