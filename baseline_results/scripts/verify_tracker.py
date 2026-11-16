"""
Feeds cached YOLOX detections (from tools/run_ocsort_dance_local.py) into the unmodified
OC-SORT tracker, checks that it initializes, returns boxes and keeps IDs across frames, and
draws boxes + track IDs onto the frames.

Usage (from the OC_SORT root):
  python baseline_results/scripts/verify_tracker.py --det dance_detections_cache/val/dancetrack0004.pt \
      --seq dancetrack0004 [--max_frames 10] [--out baseline_results/visualizations/verify] [--video]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from trackers.ocsort_tracker.ocsort import OCSort
from utils.args import make_parser

p = argparse.ArgumentParser()
p.add_argument("--det", required=True, help="<seq>.pt or <seq>.partial.pt detection cache")
p.add_argument("--seq", required=True)
p.add_argument("--data_root", default=os.path.join(ROOT, "datasets", "dancetrack", "val"))
p.add_argument("--max_frames", type=int, default=None)
p.add_argument("--out", default=os.path.join(ROOT, "baseline_results", "visualizations", "verify"))
p.add_argument("--save_frames", nargs="*", type=int, default=None, help="frame ids to save as png (default: all)")
p.add_argument("--video", action="store_true", help="also write an mp4 of all processed frames")
a = p.parse_args()

# tracker hyper-parameters come straight from the repo's parser defaults, as in the official script
targs = make_parser().parse_args([])
TEST_SIZE = (800, 1440)  # exps/example/mot/yolox_dancetrack_val.py
c = torch.load(a.det, weights_only=False)
assert tuple(c["test_size"]) == TEST_SIZE, c["test_size"]
dets, img_hw = c["dets"], tuple(c["img_hw"])
frames = sorted(dets)[:a.max_frames] if a.max_frames else sorted(dets)

tracker = OCSort(det_thresh=targs.track_thresh, iou_threshold=targs.iou_thresh, asso_func=targs.asso,
                 delta_t=targs.deltat, inertia=targs.inertia, use_byte=targs.use_byte)
params = {k: getattr(tracker, k) for k in ("det_thresh", "iou_threshold", "max_age", "min_hits", "delta_t",
                                           "inertia", "use_byte")}
params["asso_func"] = tracker.asso_func.__name__
print("tracker params:", params)

os.makedirs(a.out, exist_ok=True)
rng = np.random.RandomState(0)
colors = {}
writer = None
per_frame, id_frames = [], {}
for fid in frames:
    out = dets[fid]
    tracks = tracker.update(None if out is None else out.clone(), img_hw, TEST_SIZE)
    n_det = 0 if out is None else int((out[:, 4] * out[:, 5] > targs.track_thresh).sum())
    ids = [int(t[4]) for t in tracks]
    for i in ids:
        id_frames.setdefault(i, []).append(fid)
    per_frame.append({"frame": fid, "dets_above_thresh": n_det, "track_ids": ids})
    print("frame {:4d}: {} dets>{} -> {} tracks, ids {}".format(fid, n_det, targs.track_thresh, len(tracks), ids))

    save_png = a.save_frames is None or fid in a.save_frames
    if not (save_png or a.video):
        continue
    img = cv2.imread(os.path.join(a.data_root, a.seq, "img1", "{:08d}.jpg".format(fid)))
    for t in tracks:
        x1, y1, x2, y2, tid = [int(round(v)) for v in t[:5]]
        col = colors.setdefault(tid, tuple(int(v) for v in rng.randint(64, 255, 3)))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
        cv2.putText(img, "ID {}".format(tid), (x1, max(y1 - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
    cv2.putText(img, "{} frame {}".format(a.seq, fid), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    if save_png:
        cv2.imwrite(os.path.join(a.out, "{}_{:06d}.jpg".format(a.seq, fid)), img)
    if a.video:
        if writer is None:
            writer = cv2.VideoWriter(os.path.join(a.out, "{}.mp4".format(a.seq)), cv2.VideoWriter_fourcc(*"mp4v"),
                                     20, (img.shape[1] // 2, img.shape[0] // 2))
        writer.write(cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
if writer is not None:
    writer.release()

frames_with_tracks = sum(1 for r in per_frame if r["track_ids"])
summary = {
    "seq": a.seq, "frames": len(frames), "frames_with_tracks": frames_with_tracks,
    "unique_ids": len(id_frames),
    "id_lifetimes": {str(k): [min(v), max(v), len(v)] for k, v in sorted(id_frames.items())},
    "tracker_params": params,
}
assert frames_with_tracks > 0, "tracker returned no boxes"
print("summary:", json.dumps({k: v for k, v in summary.items() if k != "id_lifetimes"}))
print("id lifetimes [first, last, n_frames]:", summary["id_lifetimes"])
with open(os.path.join(a.out, "{}_verify.json".format(a.seq)), "w") as f:
    json.dump({"summary": summary, "per_frame": per_frame}, f, indent=1)
