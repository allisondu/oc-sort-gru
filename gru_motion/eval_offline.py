"""
Offline one-frame-ahead motion-prediction evaluation on DanceTrack TRAIN-split
holdout sequences (never the official validation split).

    python -m gru_motion.eval_offline --ckpt gru_runs/standard_v1/seed0_a40/best.pt \
        --split_config configs/gru/dancetrack_split_v2.json \
        --out_dir gru_runs/evaluations/standard_v1_holdout

Samples
-------
Every (sequence, id, t) of the holdout sequences whose GT boxes at t-10..t-1
AND t all exist (identical to gru_motion.data standard sampling). History =
GT boxes t-10..t-1 of that one identity; target = GT box at t. All methods get
the same history; the target and anything later are never passed to them.

Methods (boxes as cx, cy, w, h; b_k = observed box at frame k)
---------------------------------------------------------------
last_obs     pred = b_{t-1}
const_vel    pred = b_{t-1} + (b_{t-1} - b_{t-2}) for all four of cx, cy, w, h;
             w, h floored at 1 px (count reported).
kalman_win   OC-SORT's own KalmanBoxTracker (trackers/ocsort_tracker/ocsort.py,
             unmodified: 7-d state [cx, cy, s=w*h, r=w/h, vcx, vcy, vs], same F,
             H, Q, R, P, conversions). Born from b_{t-10} exactly as a new track,
             then predict(); update(b_k) for k = t-9..t-1, then predict() ->
             prediction for t. Continuous observations, so ORU never triggers.
             Same information as the GRU (10 frames).
kalman_track DIAGNOSTIC, closer to real tracking: the same KalmanBoxTracker run
             along the identity's whole GT track from its first frame,
             predict() every frame, update(box) when annotated and update(None)
             otherwise (OC-SORT's unmatched path, so ORU re-update fires on
             re-observation); deleted after > 30 missed frames (max_age) and
             re-born at the next observation. Its prediction for t uses all
             history <= t-1, i.e. MORE information than the GRU.
gru          trained checkpoint through GRUBoxPredictor (same encoder/decoder
             as the tracker).

Metrics: IoU with the target, centre error (px and / target height), relative
width / height error, and an association-oriented check against the OTHER GT
boxes in frame t. Results per sequence, micro (over windows) and macro (mean of
per-sequence values). Windows overlap heavily, so the five holdout sequences
are the independent units; uncertainty is reported by sequence-level resampling.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

from .data import SamplingSpec, enumerate_samples, load_sequences, load_split_config
from .features import build_window, cxcywh_to_xyxy, xyxy_to_cxcywh
from .predictor import GRUBoxPredictor

METHODS = ("last_obs", "const_vel", "kalman_win", "kalman_track", "gru")
L = 10
MAX_AGE = 30          # OC-SORT default (utils/args.py track_buffer / OCSort max_age)
TURN_BINS = (20.0, 60.0)     # degrees: straight < 20 <= moderate < 60 <= sharp


# ----------------------------------------------------------------------------- predictors

def pred_last_obs(hist):                     # hist (N, L, 4) cxcywh
    return hist[:, -1].copy()


def pred_const_vel(hist):
    p = 2 * hist[:, -1] - hist[:, -2]
    clipped = (p[:, 2] < 1) | (p[:, 3] < 1)
    p[:, 2:] = np.maximum(p[:, 2:], 1.0)
    return p, int(clipped.sum())


def _kbt():
    from trackers.ocsort_tracker.ocsort import KalmanBoxTracker
    return KalmanBoxTracker


def kalman_window_predict(hist_xyxy):
    """OC-SORT KalmanBoxTracker: born at hist[0], predict/update hist[1:], predict next. -> xyxy (4,)"""
    KBT = _kbt()
    trk = KBT(np.r_[hist_xyxy[0], 1.0], delta_t=3)
    for b in hist_xyxy[1:]:
        trk.predict()
        trk.update(np.r_[b, 1.0])
    return trk.predict()[0]


def kalman_track_predictions(frames):
    """{frame: xyxy prediction made before seeing that frame} along one GT track (see module doc)."""
    KBT = _kbt()
    fs = sorted(frames)
    trk, preds = None, {}
    for f in range(fs[0], fs[-1] + 1):
        if trk is not None:
            preds[f] = trk.predict()[0]
            trk.update(np.r_[frames[f], 1.0] if f in frames else None)
            if trk.time_since_update > MAX_AGE:
                trk = None                     # OC-SORT deletes the track
        elif f in frames:
            trk = KBT(np.r_[frames[f], 1.0], delta_t=3)
    return preds


# ----------------------------------------------------------------------------- metrics

def iou_xyxy(a, b):
    ix = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    iy = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = ix * iy
    ar = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])
    return inter / (ar(a) + ar(b) - inter)


def window_metrics(pred_c, gt_c):
    pred_x, gt_x = cxcywh_to_xyxy(pred_c), cxcywh_to_xyxy(gt_c)
    ce = np.linalg.norm(pred_c[:, :2] - gt_c[:, :2], axis=1)
    return {"iou": iou_xyxy(pred_x, gt_x), "center_px": ce, "center_norm": ce / gt_c[:, 3],
            "rel_w": np.abs(pred_c[:, 2] - gt_c[:, 2]) / gt_c[:, 2],
            "rel_h": np.abs(pred_c[:, 3] - gt_c[:, 3]) / gt_c[:, 3]}


def summarize(m, idx=None):
    s = (lambda v: v) if idx is None else (lambda v: v[idx])
    iou = s(m["iou"])
    return {"n": int(len(iou)), "iou_mean": float(iou.mean()), "iou_median": float(np.median(iou)),
            "iou_std": float(iou.std()), "iou_ge_0.5": float((iou >= 0.5).mean()),
            "iou_ge_0.7": float((iou >= 0.7).mean()),
            "center_px_mean": float(s(m["center_px"]).mean()), "center_px_median": float(np.median(s(m["center_px"]))),
            "center_norm_mean": float(s(m["center_norm"]).mean()),
            "center_norm_median": float(np.median(s(m["center_norm"]))),
            "rel_w_mean": float(s(m["rel_w"]).mean()), "rel_w_median": float(np.median(s(m["rel_w"]))),
            "rel_h_mean": float(s(m["rel_h"]).mean()), "rel_h_median": float(np.median(s(m["rel_h"]))),
            "assoc_own_iou_lt_0.3": float((iou < 0.3).mean()),
            "assoc_prefers_other_object": float(s(m["prefers_other"]).mean())}


# ----------------------------------------------------------------------------- history-only categories

def history_features(hist):
    """Grouping features from frames t-10..t-1 only (never the target)."""
    c, h = hist[..., :2], hist[..., 3]
    hn = h[:, -1]
    d_old, d_new = c[:, -4] - c[:, -7], c[:, -1] - c[:, -4]          # two 3-frame displacements
    n_old, n_new = np.linalg.norm(d_old, axis=1), np.linalg.norm(d_new, axis=1)
    moving = (n_old > 0.03 * hn) & (n_new > 0.03 * hn)
    cosang = np.sum(d_old * d_new, axis=1) / np.maximum(n_old * n_new, 1e-9)
    turn = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    step = np.linalg.norm(c[:, 1:] - c[:, :-1], axis=2) / h[:, 1:]
    speed = step[:, -3:].mean(1)                                       # mean normalised speed, last 3 steps
    acc = np.linalg.norm(c[:, 2:] - (2 * c[:, 1:-1] - c[:, :-2]), axis=2) / h[:, 2:]
    cv_hist = acc.mean(1)                                              # mean CV residual over the window
    return {"turn_deg": turn, "moving": moving, "speed": speed, "cv_err_hist": cv_hist}


def tertile_labels(v, names=("low", "mid", "high")):
    q = np.quantile(v, [1 / 3, 2 / 3])
    lab = np.where(v < q[0], names[0], np.where(v < q[1], names[1], names[2]))
    return lab, [float(q[0]), float(q[1])]


# ----------------------------------------------------------------------------- main

def git_state(root):
    try:
        c = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "-C", root, "diff", "HEAD"], text=True)
        untracked = subprocess.check_output(["git", "-C", root, "ls-files", "--others", "--exclude-standard"],
                                            text=True).split()
        h = hashlib.sha256(diff.encode())
        for u in sorted(untracked):
            h.update(u.encode() + open(os.path.join(root, u), "rb").read())
        return {"commit": c, "dirty": bool(diff.strip() or untracked), "uncommitted_files": untracked,
                "uncommitted_state_sha256": h.hexdigest()}
    except Exception as e:
        return {"commit": "unknown", "error": str(e)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split_config", default="configs/gru/dancetrack_split_v2.json")
    ap.add_argument("--data_root", default="datasets_gru/dancetrack_gt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--diag_ckpt", default=None, help="optional second checkpoint (e.g. last.pt), reported as diagnostic")
    ap.add_argument("--max_windows", type=int, default=None, help="sanity runs only")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(os.path.join(args.out_dir, "summary.json")) and not args.overwrite:
        sys.exit("{} already holds an evaluation; pass --overwrite or choose another --out_dir".format(args.out_dir))
    t0 = time.time()

    # ---- data (holdout only) -------------------------------------------------
    cfg, _, hold_dirs, split_sha = load_split_config(args.split_config, args.data_root)
    spec = SamplingSpec.make("standard", L)
    data, _ = load_sequences(hold_dirs)
    samples = enumerate_samples(data, spec)
    if args.max_windows:
        samples = samples[:args.max_windows]
    N = len(samples)
    hist = np.zeros((N, L, 4)); mask = np.zeros((N, L)); gt = np.zeros((N, 4))
    for j, (s, tid, t) in enumerate(samples):
        hist[j], mask[j] = build_window(data[s][tid], t, L)
        gt[j] = xyxy_to_cxcywh(data[s][tid][t])
    assert mask.all(), "standard windows must be fully observed"
    seqs = np.array([str(s) for s, _, _ in samples])
    hist_xyxy = cxcywh_to_xyxy(hist)

    # ---- predictors ----------------------------------------------------------
    preds, notes = {}, {}
    preds["last_obs"] = pred_last_obs(hist)
    preds["const_vel"], notes["const_vel_size_clipped"] = pred_const_vel(hist)
    tk = time.time()
    preds["kalman_win"] = xyxy_to_cxcywh(np.array([kalman_window_predict(h) for h in hist_xyxy]))
    notes["kalman_win_seconds"] = time.time() - tk
    track_preds = {(s, tid): kalman_track_predictions(data[s][tid]) for s, tid in sorted({(s, i) for s, i, _ in samples})}
    preds["kalman_track"] = xyxy_to_cxcywh(np.array([track_preds[(s, tid)][t] for s, tid, t in samples]))
    predictor = GRUBoxPredictor.from_checkpoint(args.ckpt, device="cpu")
    meta = predictor.meta
    sup = predictor.support
    assert meta["feature_version"] == 1 and meta["training_conditions"]["training_mode"] == "standard"
    assert meta["history_len"] == L and sup["min_valid_obs"] == L and sup["max_trailing_gap"] == 0
    assert not predictor.model.training
    tg = time.time()
    preds["gru"] = xyxy_to_cxcywh(predictor.predict_xyxy(hist, mask))
    notes["gru_seconds"] = time.time() - tg
    methods = list(METHODS)
    if args.diag_ckpt:
        dp = GRUBoxPredictor.from_checkpoint(args.diag_ckpt, device="cpu")
        preds["gru_diag"] = xyxy_to_cxcywh(dp.predict_xyxy(hist, mask))
        methods.append("gru_diag")
    for m in methods:
        p = preds[m]
        assert np.all(np.isfinite(p)) and np.all(p[:, 2:] > 0), "invalid boxes from {}".format(m)

    # ---- metrics -------------------------------------------------------------
    by_frame = {}
    for s in data:
        for tid, fr in data[s].items():
            for f, b in fr.items():
                by_frame.setdefault((s, f), []).append((tid, b))
    others = [np.array([b for i, b in by_frame[(s, t)] if i != tid]) for s, tid, t in samples]
    M = {}
    for m in methods:
        M[m] = window_metrics(preds[m], gt)
        px = cxcywh_to_xyxy(preds[m])
        own = M[m]["iou"]
        M[m]["prefers_other"] = np.array([o.size > 0 and iou_xyxy(px[j][None], o).max() > own[j]
                                          for j, o in enumerate(others)])

    seq_names = [str(s) for s in sorted(set(seqs))]
    # centre-vs-size decomposition: swap in the last observation's size (or centre)
    gt_x = cxcywh_to_xyxy(gt)
    decomp = {}
    for m in methods:
        pc = preds[m].copy(); pc[:, 2:] = preds["last_obs"][:, 2:]
        ps = preds["last_obs"].copy(); ps[:, 2:] = preds[m][:, 2:]
        decomp[m] = {"iou_full": float(M[m]["iou"].mean()),
                     "iou_own_centre_lastobs_size": float(iou_xyxy(cxcywh_to_xyxy(pc), gt_x).mean()),
                     "iou_lastobs_centre_own_size": float(iou_xyxy(cxcywh_to_xyxy(ps), gt_x).mean())}
    feats = history_features(hist)
    turn_lab = np.where(~feats["moving"], "near_static",
                        np.where(feats["turn_deg"] < TURN_BINS[0], "straight",
                                 np.where(feats["turn_deg"] < TURN_BINS[1], "moderate", "sharp")))
    speed_lab, speed_q = tertile_labels(feats["speed"], ("slow", "medium", "fast"))
    cvh_lab, cvh_q = tertile_labels(feats["cv_err_hist"])
    realized = np.linalg.norm(gt[:, :2] - (2 * hist[:, -1, :2] - hist[:, -2, :2]), axis=1) / hist[:, -1, 3]
    real_lab, real_q = tertile_labels(realized)
    crowd = np.array([len(by_frame.get((s, t - 1), [])) for s, _, t in samples])
    crowd_lab = np.where(crowd <= 5, "<=5", np.where(crowd <= 10, "6-10", ">10"))
    groupings = {"turn_angle_hist": turn_lab, "speed_hist": speed_lab, "cv_error_hist": cvh_lab,
                 "crowding_t-1": crowd_lab, "DIAGNOSTIC_realized_cv_error_uses_target": real_lab}

    overall = {m: summarize(M[m]) for m in methods}
    per_seq = {s: {m: summarize(M[m], seqs == s) for m in methods} for s in seq_names}
    macro = {m: {k: float(np.mean([per_seq[s][m][k] for s in seq_names])) for k in overall[m] if k != "n"}
             for m in methods}

    # paired GRU - baseline differences: per sequence, macro, and sequence-level bootstrap
    rng = np.random.default_rng(args.seed)
    diffs = {}
    for base in ("last_obs", "const_vel", "kalman_win", "kalman_track"):
        d = {}
        for k in ("iou_mean", "center_norm_mean", "center_norm_median", "rel_w_mean", "rel_h_mean",
                  "assoc_prefers_other_object", "assoc_own_iou_lt_0.3"):
            v = np.array([per_seq[s]["gru"][k] - per_seq[s][base][k] for s in seq_names])
            boots = v[rng.integers(0, len(v), size=(args.bootstrap, len(v)))].mean(1)
            d[k] = {"per_sequence": dict(zip(seq_names, v.tolist())), "macro": float(v.mean()),
                    "seq_bootstrap_95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    "gru_better_in": int(((v > 0) if k == "iou_mean" else (v < 0)).sum()),
                    "num_sequences": len(v)}
        diffs["gru_minus_" + base] = d

    cat_rows = []
    for gname, lab in groupings.items():
        for c in sorted(set(lab)):
            idx = lab == c
            row = {"grouping": gname, "category": c, "n": int(idx.sum()),
                   "n_sequences": int(len(set(seqs[idx])))}
            for m in methods:
                sm = summarize(M[m], idx)
                row[m + "_iou"] = sm["iou_mean"]
                row[m + "_center_norm"] = sm["center_norm_mean"]
            for base in ("const_vel", "kalman_win", "kalman_track"):
                row["gru_minus_{}_iou".format(base)] = row["gru_iou"] - row[base + "_iou"]
                row["gru_minus_{}_center_norm".format(base)] = row["gru_center_norm"] - row[base + "_center_norm"]
            # per-sequence consistency of the GRU-vs-kalman_win IoU gain inside this category
            per = [summarize(M["gru"], idx & (seqs == s))["iou_mean"] - summarize(M["kalman_win"], idx & (seqs == s))["iou_mean"]
                   for s in seq_names if (idx & (seqs == s)).sum() >= 20]
            row["gru_beats_kalman_win_iou_in_seqs"] = "{}/{}".format(int(sum(v > 0 for v in per)), len(per))
            cat_rows.append(row)

    # association-decision flips between GRU and window Kalman (IoU threshold 0.3)
    flips = {"kalman_win_lt_0.3_gru_ge_0.3": int(((M["kalman_win"]["iou"] < 0.3) & (M["gru"]["iou"] >= 0.3)).sum()),
             "gru_lt_0.3_kalman_win_ge_0.3": int(((M["gru"]["iou"] < 0.3) & (M["kalman_win"]["iou"] >= 0.3)).sum()),
             "kalman_track_lt_0.3_gru_ge_0.3": int(((M["kalman_track"]["iou"] < 0.3) & (M["gru"]["iou"] >= 0.3)).sum()),
             "gru_lt_0.3_kalman_track_ge_0.3": int(((M["gru"]["iou"] < 0.3) & (M["kalman_track"]["iou"] >= 0.3)).sum())}

    # ---- outputs -------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    counts = {str(s): int((seqs == s).sum()) for s in seq_names}
    summary = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, "-m", "gru_motion.eval_offline"] + (argv or sys.argv[1:])),
        "code": git_state(root),
        "checkpoint": {"path": meta["path"], "sha256": meta["sha256"],
                       "weights_epoch": meta["train_info"]["checkpoint"]["weights_epoch"],
                       "kind": meta["train_info"]["checkpoint"]["kind"],
                       "training_git_commit": meta["train_info"]["git_commit"],
                       "inference_support": sup},
        "split": {"path": os.path.abspath(args.split_config), "sha256": split_sha, "split_id": cfg.get("split_id"),
                  "holdout_sequences": seq_names},
        "samples": {"total": N, "per_sequence": counts, "rule": "standard: GT at t-10..t-1 and t all present, one identity"},
        "methods_doc": __doc__,
        "category_definitions": {
            "turn_angle_hist": "angle between c(t-4)-c(t-7) and c(t-1)-c(t-4); straight <{} deg, moderate <{} deg, "
                               "sharp >= ; near_static if either 3-frame displacement <= 0.03 h(t-1)".format(*TURN_BINS),
            "speed_hist": "mean |c(k)-c(k-1)|/h(k), k=t-3..t-1; tertile cut points {}".format(speed_q),
            "cv_error_hist": "mean |c(k)-(2c(k-1)-c(k-2))|/h(k), k=t-8..t-1; tertile cut points {}".format(cvh_q),
            "crowding_t-1": "number of annotated objects in frame t-1",
            "DIAGNOSTIC_realized_cv_error_uses_target": "|c(t)-(2c(t-1)-c(t-2))|/h(t-1) tertiles {} -- uses the target "
                                                        "frame; descriptive only, not a selection criterion".format(real_q)},
        "notes": notes,
        "overall_micro": overall, "overall_macro": macro, "per_sequence": per_seq,
        "paired_differences": diffs, "association_flips_iou_0.3": flips,
        "per_category": cat_rows,
        "centre_size_decomposition": decomp,
        "runtime_seconds": time.time() - t0,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    keys = list(overall["gru"].keys())
    with open(os.path.join(args.out_dir, "per_sequence.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "method"] + keys)
        for s in seq_names + ["MICRO", "MACRO"]:
            for m in methods:
                r = per_seq[s][m] if s in per_seq else (overall[m] if s == "MICRO" else dict(macro[m], n=N))
                w.writerow([s, m] + [r[k] for k in keys])
    with open(os.path.join(args.out_dir, "per_category.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cat_rows[0].keys()))
        w.writeheader()
        w.writerows(cat_rows)
    with open(os.path.join(args.out_dir, "per_window.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "id", "frame", "gt_cx", "gt_cy", "gt_w", "gt_h", "turn_deg", "moving", "speed_hist",
                    "cv_err_hist", "crowd_t-1"] + ["{}_{}".format(m, k) for m in methods
                                                   for k in ("cx", "cy", "w", "h", "iou", "center_norm")])
        for j, (s, tid, t) in enumerate(samples):
            w.writerow([s, tid, t] + ["%.2f" % v for v in gt[j]] +
                       ["%.2f" % feats["turn_deg"][j], int(feats["moving"][j]), "%.5f" % feats["speed"][j],
                        "%.5f" % feats["cv_err_hist"][j], int(crowd[j])] +
                       [x for m in methods for x in ("%.2f" % preds[m][j, 0], "%.2f" % preds[m][j, 1],
                                                     "%.2f" % preds[m][j, 2], "%.2f" % preds[m][j, 3],
                                                     "%.5f" % M[m]["iou"][j], "%.5f" % M[m]["center_norm"][j])])
    write_report(os.path.join(args.out_dir, "report.md"), summary, methods)
    print("evaluated {} windows ({}) in {:.1f}s -> {}".format(N, counts, time.time() - t0, args.out_dir))
    return summary


def write_report(path, S, methods):
    f4 = lambda v: "{:.4f}".format(v)
    lines = ["# Offline motion-prediction evaluation (holdout, standard windows)", "",
             "Checkpoint `{}` (epoch {}, sha256 `{}`); split `{}` (sha256 `{}`); code {}{}.".format(
                 S["checkpoint"]["path"], S["checkpoint"]["weights_epoch"], S["checkpoint"]["sha256"][:16],
                 S["split"]["split_id"], S["split"]["sha256"][:16], S["code"].get("commit", "?")[:10],
                 " + uncommitted changes" if S["code"].get("dirty") else ""),
             "", "Samples: {} windows: {}".format(S["samples"]["total"], S["samples"]["per_sequence"]), "",
             "`kalman_track` is a diagnostic with MORE history than the GRU (whole track, OC-SORT ORU path).", "",
             "## Overall (micro = pooled windows, macro = mean of 5 sequences)", "",
             "| method | IoU mean (micro) | IoU mean (macro) | IoU median | IoU>=0.5 | IoU>=0.7 | ctr px mean | ctr/h mean (micro) | ctr/h mean (macro) | ctr/h median | rel w err | rel h err | prefers other obj |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        o, a = S["overall_micro"][m], S["overall_macro"][m]
        lines.append("| {} | {} | {} | {} | {:.2%} | {:.2%} | {:.2f} | {} | {} | {} | {} | {} | {:.3%} |".format(
            m, f4(o["iou_mean"]), f4(a["iou_mean"]), f4(o["iou_median"]), o["iou_ge_0.5"], o["iou_ge_0.7"],
            o["center_px_mean"], f4(o["center_norm_mean"]), f4(a["center_norm_mean"]), f4(o["center_norm_median"]),
            f4(o["rel_w_mean"]), f4(o["rel_h_mean"]), o["assoc_prefers_other_object"]))
    lines += ["", "## Per sequence: IoU mean / centre error (/h) mean", "",
              "| sequence | n | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
    for s, d in S["per_sequence"].items():
        lines.append("| {} | {} | ".format(s, d["gru"]["n"]) + " | ".join(
            "{} / {}".format(f4(d[m]["iou_mean"]), f4(d[m]["center_norm_mean"])) for m in methods) + " |")
    lines += ["", "## GRU minus baseline (per sequence; macro; sequence-level bootstrap 95% over 5 sequences)", ""]
    for b, d in S["paired_differences"].items():
        for k in ("iou_mean", "center_norm_mean", "rel_w_mean", "rel_h_mean"):
            e = d[k]
            lines.append("- {} `{}`: macro {:+.4f}, CI [{:+.4f}, {:+.4f}], GRU better in {}/{} seqs; per seq {}".format(
                b, k, e["macro"], *e["seq_bootstrap_95"], e["gru_better_in"], e["num_sequences"],
                {s[-4:]: round(v, 4) for s, v in e["per_sequence"].items()}))
    lines += ["", "## Centre vs size (micro IoU mean)", "",
              "| method | full prediction | own centre + last-obs size | last-obs centre + own size |", "|---|---|---|---|"]
    for m, d in S["centre_size_decomposition"].items():
        lines.append("| {} | {} | {} | {} |".format(m, f4(d["iou_full"]), f4(d["iou_own_centre_lastobs_size"]),
                                                   f4(d["iou_lastobs_centre_own_size"])))
    lines += ["", "Association flips at IoU 0.3: {}".format(S["association_flips_iou_0.3"]), "",
              "## Categories (IoU mean; GRU minus baselines)", "",
              "| grouping | category | n | seqs | last | CV | KF-win | KF-track | GRU | GRU-CV | GRU-KFwin | GRU-KFtrack | ctr/h GRU-KFwin | GRU>KFwin in seqs |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in S["per_category"]:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} | {} |".format(
            r["grouping"], r["category"], r["n"], r["n_sequences"], f4(r["last_obs_iou"]), f4(r["const_vel_iou"]),
            f4(r["kalman_win_iou"]), f4(r["kalman_track_iou"]), f4(r["gru_iou"]), r["gru_minus_const_vel_iou"],
            r["gru_minus_kalman_win_iou"], r["gru_minus_kalman_track_iou"], r["gru_minus_kalman_win_center_norm"],
            r["gru_beats_kalman_win_iou_in_seqs"]))
    lines += ["", "## Category definitions", ""] + ["- **{}**: {}".format(k, v) for k, v in S["category_definitions"].items()]
    lines += ["", "Command: `{}`".format(S["command"])]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
