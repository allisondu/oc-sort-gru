"""
Per-sequence statistics of DanceTrack TRAIN ground truth, used only to assess
the representativeness of the GRU train/holdout split (never model results).

All metrics are computed from annotations; they are proxies, not measures of
visual difficulty. Definitions (h = box height, c = box centre):

  frames            seqLength from seqinfo.ini
  ids               distinct identities
  objs_mean/max     annotated boxes per frame (frames 1..seqLength, empty frames count as 0)
  overlap_rate      fraction of annotated boxes that have IoU > 0.1 with at least
                    one other box in the same frame (interaction proxy)
  overlap_rate_03   same with IoU > 0.3 (heavy overlap proxy)
  cv_err_median     median over all (id, t) with t-1, t, t+1 annotated of
  cv_err_p90        |c_{t+1} - (2 c_t - c_{t-1})| / h_t, i.e. one-step error of
                    constant-velocity extrapolation in box heights
                    (motion non-linearity proxy, the quantity our GRU targets)
  turn_rate         fraction of (id, t) with c at t-5, t, t+5 annotated and both
                    5-frame displacements > 0.1 h_t, whose direction changes by
                    more than 60 degrees (abrupt direction change proxy)
  speed_median      median |c_{t+1} - c_t| / h_t
  gap_rate          annotation gaps (missing frames between a track's first and
                    last frame) per 1000 annotated boxes
  missing_frac      missing frames inside track lifespans / lifespan frames
  resolution        imWidth x imHeight

    python tools/analyze_dancetrack_sequences.py --out configs/gru/dancetrack_train_sequence_stats.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gru_motion.data import list_sequences, load_gt, read_seqinfo  # noqa: E402

METRICS = ("frames", "ids", "objs_mean", "objs_max", "overlap_rate", "overlap_rate_03", "cv_err_median",
           "cv_err_p90", "turn_rate", "speed_median", "gap_rate", "missing_frac")


def pairwise_iou(b):
    x1 = np.maximum(b[:, None, 0], b[None, :, 0]); y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2]); y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / (a[:, None] + a[None, :] - inter)
    np.fill_diagonal(iou, 0)
    return iou


def sequence_stats(seq_dir):
    tracks, st = load_gt(os.path.join(seq_dir, "gt", "gt.txt"))
    si = read_seqinfo(seq_dir)
    n_frames = int(si["seqlength"])
    by_frame = {}
    for tid, fr in tracks.items():
        for f, b in fr.items():
            by_frame.setdefault(f, []).append(b)
    counts = np.array([len(by_frame.get(f, [])) for f in range(1, n_frames + 1)])
    ov1 = ov3 = 0
    for f, bs in by_frame.items():
        if len(bs) > 1:
            m = pairwise_iou(np.array(bs)).max(1)
            ov1 += int((m > 0.1).sum()); ov3 += int((m > 0.3).sum())
    cv, turns, speed = [], [], []
    gaps = missing = span = 0
    for tid, fr in tracks.items():
        fs = sorted(fr)
        c = {f: np.array([(fr[f][0] + fr[f][2]) / 2, (fr[f][1] + fr[f][3]) / 2]) for f in fs}
        h = {f: fr[f][3] - fr[f][1] for f in fs}
        for f in fs:
            if f - 1 in c and f + 1 in c:
                cv.append(np.linalg.norm(c[f + 1] - (2 * c[f] - c[f - 1])) / h[f])
            if f + 1 in c:
                speed.append(np.linalg.norm(c[f + 1] - c[f]) / h[f])
            if f - 5 in c and f + 5 in c:
                a, b = c[f] - c[f - 5], c[f + 5] - c[f]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 0.1 * h[f] and nb > 0.1 * h[f]:
                    turns.append(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))) > 60)
        d = np.diff(fs)
        gaps += int((d > 1).sum()); missing += int((d - 1).sum()); span += fs[-1] - fs[0] + 1
    return {
        "sequence": os.path.basename(seq_dir), "frames": n_frames, "ids": len(tracks),
        "boxes": st["boxes"], "resolution": "{}x{}".format(si["imwidth"], si["imheight"]),
        "objs_mean": float(counts.mean()), "objs_max": int(counts.max()),
        "overlap_rate": ov1 / st["boxes"], "overlap_rate_03": ov3 / st["boxes"],
        "cv_err_median": float(np.median(cv)), "cv_err_p90": float(np.percentile(cv, 90)),
        "turn_rate": float(np.mean(turns)) if turns else 0.0, "speed_median": float(np.median(speed)),
        "gap_rate": 1000.0 * gaps / st["boxes"], "missing_frac": missing / span,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="datasets_gru/dancetrack_gt")
    ap.add_argument("--out", default="configs/gru/dancetrack_train_sequence_stats.json")
    args = ap.parse_args()
    rows = [sequence_stats(d) for d in list_sequences(args.data_root, "train")]
    with open(args.out, "w") as f:
        json.dump({"definitions": __doc__.split("All metrics")[1].split("    python")[0].strip(),
                   "sequences": rows}, f, indent=2)
        f.write("\n")
    print("wrote {} ({} sequences)".format(args.out, len(rows)))


if __name__ == "__main__":
    main()
