"""
Profile / benchmark tracker-only speed of OC-SORT and OCSortGRU on the cached
DanceTrack val detections (same settings as tools/run_ocsort_dance_local.py).

    python tools/profile_gru_inference.py bench   --out <json> [--root <checkout>] [--reps 3]
    python tools/profile_gru_inference.py timers  --out <json> [--root <checkout>]
    python tools/profile_gru_inference.py cprofile --out <txt> --tracker gap [--seqs ...]

`--root` imports the tracker/GRU code from another checkout (e.g. a worktree of
the pre-optimisation commit), so "before" and "after" share one harness.
Timing = wall time of the tracker.update() calls plus the runner's per-frame
result packing (exactly what the runner's "tracker-only FPS" measures).
All detections are loaded into memory first; one warm-up sequence is run before
timing; torch threads are fixed with --threads.
"""
import argparse
import glob
import json
import os
import platform
import sys
import time

import numpy as np

CKPTS = {"std": "gru_runs/standard_v1/seed0_a40/best.pt",
         "noise": "gru_runs/noise_augmented/standard_v1_seed0_a40/best.pt",
         "gap": "gru_runs/gap_aware/noise_v1_seed0_a40/best.pt"}
TRACKERS = ("ocsort", "std", "noise", "gap")


def setup(root, threads):
    root = os.path.abspath(root)
    sys.path.insert(0, root)
    import torch
    torch.set_num_threads(threads)
    return root, torch


def load_dets(det_cache, seqs=None):
    import torch
    out = {}
    for c in sorted(glob.glob(os.path.join(det_cache, "dancetrack*.pt"))):
        s = os.path.basename(c)[:-3]
        if seqs and s not in seqs:
            continue
        d = torch.load(c, weights_only=False)
        out[s] = (d["dets"], tuple(d["img_hw"]))
    return out


def make(kind, predictors):
    from utils.args import make_parser
    a = make_parser().parse_args([])
    kw = dict(iou_threshold=a.iou_thresh, asso_func=a.asso, delta_t=a.deltat, inertia=a.inertia, use_byte=a.use_byte)
    if kind == "ocsort":
        from trackers.ocsort_tracker.ocsort import OCSort
        return OCSort(det_thresh=a.track_thresh, **kw), a
    from trackers.ocsort_gru import OCSortGRU
    return OCSortGRU(det_thresh=a.track_thresh, predictor=predictors[kind], **kw), a


def run_seq(kind, predictors, dets, img_hw, img_size):
    trk, a = make(kind, predictors)
    t0 = time.perf_counter()
    res = []
    for f in sorted(dets):
        o = dets[f]
        online = trk.update(None if o is None else o.clone(), img_hw, img_size)
        tl, ids = [], []
        for t in online:
            tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
            if tlwh[2] * tlwh[3] > a.min_box_area:
                tl.append(tlwh); ids.append(t[4])
        res.append((f, tl, ids))
    return time.perf_counter() - t0, res, trk


def predictors_for(root):
    from gru_motion import GRUBoxPredictor
    return {k: GRUBoxPredictor.from_checkpoint(os.path.join(root, v) if not os.path.isabs(v) else v)
            for k, v in CKPTS.items()}


def cmd_bench(a, root, torch):
    from yolox.exp import get_exp
    img_size = get_exp(os.path.join(root, "exps/example/mot/yolox_dancetrack_val.py"), None).test_size
    data = load_dets(a.det_cache, a.seqs)
    frames = sum(len(d) for d, _ in data.values())
    P = predictors_for(a.ckpt_root)
    w = sorted(data)[0]
    out = {"root": root, "frames": frames, "sequences": len(data), "threads": torch.get_num_threads(),
           "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
           "machine": platform.platform(), "results": {}}
    order = [k for k in TRACKERS if k in a.trackers]
    for rep in range(a.reps):
        for k in order:                                   # interleave trackers within each repetition
            run_seq(k, P, *data[w], img_size)             # warm-up (one sequence)
            tot = 0.0
            per_seq = {}
            for s in sorted(data):
                el, _, _ = run_seq(k, P, *data[s], img_size)
                tot += el
                per_seq[s] = el
            r = out["results"].setdefault(k, {"seconds": [], "fps": [], "ms_per_frame": []})
            r["seconds"].append(tot); r["fps"].append(frames / tot); r["ms_per_frame"].append(1000 * tot / frames)
            print("rep {} {:6s} {:8.1f} FPS  {:.3f} ms/frame".format(rep, k, frames / tot, 1000 * tot / frames), flush=True)
    for k, r in out["results"].items():
        f = np.array(r["fps"]); m = np.array(r["ms_per_frame"])
        r.update({"fps_mean": float(f.mean()), "fps_std": float(f.std(ddof=1)) if len(f) > 1 else 0.0,
                  "ms_mean": float(m.mean()), "ms_std": float(m.std(ddof=1)) if len(m) > 1 else 0.0,
                  "fps_min": float(f.min()), "fps_max": float(f.max())})
    json.dump(out, open(a.out, "w"), indent=2)


class Acc(object):
    def __init__(self):
        self.t, self.n = {}, {}

    def wrap(self, name, fn):
        acc = self

        def w(*args, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kw)
            finally:
                acc.t[name] = acc.t.get(name, 0.0) + time.perf_counter() - t0
                acc.n[name] = acc.n.get(name, 0) + 1
        return w


def cmd_timers(a, root, torch):
    """Component timers (wall time, inclusive) via thin wrappers around module-level functions."""
    import trackers.ocsort_tracker.ocsort as O
    import trackers.ocsort_gru.tracker as TG
    import gru_motion.predictor as PR
    from yolox.exp import get_exp
    img_size = get_exp(os.path.join(root, "exps/example/mot/yolox_dancetrack_val.py"), None).test_size
    data = load_dets(a.det_cache, a.seqs)
    frames = sum(len(d) for d, _ in data.values())
    P = predictors_for(a.ckpt_root)
    out = {"root": root, "frames": frames, "note": "inclusive wall-time per component; nested components "
                                                   "(e.g. model forward inside predict_xyxy) are also counted "
                                                   "in their parent", "trackers": {}}
    for k in [x for x in TRACKERS if x in a.trackers]:
        acc = Acc()
        saved = []

        def patch(mod, name, label):
            if hasattr(mod, name):
                saved.append((mod, name, getattr(mod, name)))
                setattr(mod, name, acc.wrap(label, getattr(mod, name)))
        patch(O, "associate", "association_first_round")
        patch(O.KalmanBoxTracker, "predict", "kalman_predict")
        patch(O.KalmanBoxTracker, "update", "kalman_update")
        patch(O.OCSort, "update", "update_total")
        if k != "ocsort":
            patch(TG, "track_window", "history_track_window")
            patch(TG, "track_windows", "history_track_windows_batched")
            patch(TG, "window_stats", "eligibility_window_stats")
            patch(TG.OCSortGRU, "_predict_boxes", "predict_boxes_total")
            patch(PR, "encode_windows", "feature_encode_windows")
            patch(PR, "decode_outputs", "decode_outputs")
            patch(PR, "cxcywh_to_xyxy", "decode_to_xyxy")
            for p in P.values():
                saved.append((p.model, "predict", p.model.predict))
                p.model.predict = acc.wrap("model_forward_predict", p.model.predict)
                saved.append((p, "predict_xyxy", p.predict_xyxy))
                p.predict_xyxy = acc.wrap("predictor_predict_xyxy_total", p.predict_xyxy)
        bs, calls = [], [0]
        if k != "ocsort":
            inner = P[k].predict_xyxy

            def counting(boxes, mask, _inner=inner):
                bs.append(len(boxes)); calls[0] += 1
                return _inner(boxes, mask)
            P[k].predict_xyxy = counting
        tot = 0.0
        for s in sorted(data):
            el, _, trk = run_seq(k, P, *data[s], img_size)
            tot += el
        for mod, name, orig in reversed(saved):
            setattr(mod, name, orig)
        d = {"frames": frames, "total_s": tot, "ms_per_frame": 1000 * tot / frames,
             "components_ms_per_frame": {n: 1000 * v / frames for n, v in sorted(acc.t.items())},
             "calls": dict(acc.n)}
        if bs:
            d["gru_forward_calls"] = calls[0]
            d["gru_calls_per_frame"] = calls[0] / frames
            d["gru_batch_size_mean"] = float(np.mean(bs))
            d["gru_batch_size_max"] = int(np.max(bs))
        out["trackers"][k] = d
        print(k, json.dumps(d, indent=1), flush=True)
    json.dump(out, open(a.out, "w"), indent=2)


def cmd_cprofile(a, root, torch):
    import cProfile
    import pstats
    from yolox.exp import get_exp
    img_size = get_exp(os.path.join(root, "exps/example/mot/yolox_dancetrack_val.py"), None).test_size
    data = load_dets(a.det_cache, a.seqs)
    P = predictors_for(a.ckpt_root)
    k = a.trackers[0]
    pr = cProfile.Profile()
    pr.enable()
    for s in sorted(data):
        run_seq(k, P, *data[s], img_size)
    pr.disable()
    with open(a.out, "w") as f:
        st = pstats.Stats(pr, stream=f).sort_stats("cumulative")
        st.print_stats(45)
        st.sort_stats("tottime").print_stats(30)
    print(open(a.out).read()[:6000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "timers", "cprofile"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--ckpt_root", default=os.getcwd(), help="where gru_runs/ lives")
    ap.add_argument("--det_cache", default="dance_detections_cache/val")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--trackers", nargs="*", default=list(TRACKERS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    a.det_cache = os.path.abspath(a.det_cache)
    a.ckpt_root = os.path.abspath(a.ckpt_root)
    a.out = os.path.abspath(a.out)
    root, torch = setup(a.root, a.threads)
    os.chdir(root)
    {"bench": cmd_bench, "timers": cmd_timers, "cprofile": cmd_cprofile}[a.cmd](a, root, torch)


if __name__ == "__main__":
    main()
