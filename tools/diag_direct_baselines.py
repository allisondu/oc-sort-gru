"""
DIAGNOSTIC CONTROL (separate from the primary benchmark): run OCSortGRU with a
NON-LEARNED predictor in place of the GRU, under exactly the same eligibility
policy as the standard checkpoint (full 10-frame history, last frame observed).
Answers: is the GRU's tracking change specific to learned motion, or would any
substitution of first-round boxes move the metrics by a similar amount?

Predictors (same window input as the GRU, from gru_motion.features.build_window):
  const_vel  b(t-1) + (b(t-1) - b(t-2)) on cx, cy, w, h (w, h >= 1 px)
  last_obs   b(t-1)

    python tools/diag_direct_baselines.py --predictor const_vel --expn diag_direct_const_vel
then TrackEval on YOLOX_outputs/<expn>/<expn>_val.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gru_motion.features import cxcywh_to_xyxy  # noqa: E402
from gru_motion.model import conservative_support  # noqa: E402
from trackers.ocsort_gru import OCSortGRU  # noqa: E402
from utils.args import make_parser  # noqa: E402
from utils.utils import write_results_no_score  # noqa: E402
from yolox.exp import get_exp  # noqa: E402


class StubPredictor(object):
    def __init__(self, kind, history_len=10):
        self.kind, self.history_len = kind, history_len
        self.support = conservative_support(history_len)        # same policy as the standard GRU checkpoint
        self.meta = {"stub": kind}

    def predict_xyxy(self, boxes, mask):
        if self.kind == "last_obs":
            p = boxes[:, -1].copy()
        else:
            p = 2 * boxes[:, -1] - boxes[:, -2]
            p[:, 2:] = np.maximum(p[:, 2:], 1.0)
        return cxcywh_to_xyxy(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", choices=["const_vel", "last_obs"], required=True)
    ap.add_argument("--expn", required=True)
    ap.add_argument("--exp_file", default="exps/example/mot/yolox_dancetrack_val.py")
    ap.add_argument("--det_cache", default="dance_detections_cache/val")
    a = ap.parse_args()
    out = os.path.join("YOLOX_outputs", a.expn, a.expn + "_val")
    if os.path.exists(out):
        sys.exit("{} exists".format(out))
    os.makedirs(out)
    targs = make_parser().parse_args([])
    img_size = get_exp(a.exp_file, None).test_size
    pred = StubPredictor(a.predictor)
    stats = {}
    import glob
    for cache in sorted(glob.glob(os.path.join(a.det_cache, "dancetrack*.pt"))):
        seq = os.path.basename(cache)[:-3]
        c = torch.load(cache, weights_only=False)
        dets, img_hw = c["dets"], tuple(c["img_hw"])
        trk = OCSortGRU(det_thresh=targs.track_thresh, predictor=pred, iou_threshold=targs.iou_thresh,
                        asso_func=targs.asso, delta_t=targs.deltat, inertia=targs.inertia, use_byte=targs.use_byte)
        results = []
        for f in sorted(dets):
            o = dets[f]
            online = trk.update(None if o is None else o.clone(), img_hw, img_size)
            tl, ids = [], []
            for t in online:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                if tlwh[2] * tlwh[3] > targs.min_box_area:
                    tl.append(tlwh); ids.append(t[4])
            results.append((f, tl, ids))
        write_results_no_score(os.path.join(out, seq + ".txt"), results)
        stats[seq] = dict(trk.stats)
    tot = {k: sum(s[k] for s in stats.values()) for k in next(iter(stats.values()))}
    print(a.predictor, tot)


if __name__ == "__main__":
    main()
