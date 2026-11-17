"""
DIAGNOSTIC (separate from the primary benchmark): does the GRU change OC-SORT's
first-round association decisions, and when it does, which choice is right?

Runs OCSortGRU on cached detections exactly as the runner does. OC-SORT's
first-round `associate` is wrapped: the real call (with the GRU-substituted
boxes) is returned unchanged, and a counterfactual call with the Kalman boxes
of the same frame (trk.history[-1], appended by the unmodified predict()) is
computed and compared. Tracker behaviour is unchanged; the script verifies its
outputs are byte-identical to the primary run's result files.

Correctness of a disagreement uses val GT for ANALYSIS ONLY: each detection is
labelled with the GT id it overlaps with IoU >= 0.5 (greedy), and each track
carries the GT id of its most recent observation. A choice is "correct" if the
chosen detection's GT id equals the track's GT id. Only the single-frame
decision is compared; downstream divergence is not simulated.

    python tools/diag_gru_association.py --ckpt gru_runs/standard_v1/seed0_a40/best.pt \
        --primary YOLOX_outputs/eval_gru_standard_v1/eval_gru_standard_v1_val \
        --out gru_runs/evaluations/tracking_standard_v1_val/assoc_diag
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import trackers.ocsort_tracker.ocsort as O  # noqa: E402
from gru_motion import GRUBoxPredictor  # noqa: E402
from trackers.ocsort_gru import OCSortGRU  # noqa: E402
from utils.args import make_parser  # noqa: E402
from utils.utils import write_results_no_score  # noqa: E402
from yolox.exp import get_exp  # noqa: E402


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter)


def gt_labels(boxes, gt_boxes, gt_ids, thr=0.5):
    """Greedy one-to-one det->GT id labels (IoU >= thr), -1 if none."""
    lab = -np.ones(len(boxes), dtype=int)
    m = iou_mat(boxes, gt_boxes)
    while m.size and m.max() >= thr:
        i, j = np.unravel_index(m.argmax(), m.shape)
        lab[i] = gt_ids[j]
        m[i, :] = -1; m[:, j] = -1
    return lab


class DiagGRU(OCSortGRU):
    def _predict_boxes(self):
        trks = super()._predict_boxes()
        self._kf_trks = np.array([[*t.history[-1][0][:4], 0.0] for t in self.trackers]).reshape(-1, 5)
        self._gru_rows = ~np.all(np.isclose(trks[:, :4], self._kf_trks[:, :4]), axis=1) if len(trks) else np.zeros(0, bool)
        return trks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--primary", required=True, help="primary GRU run result folder to compare outputs against")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exp_file", default="exps/example/mot/yolox_dancetrack_val.py")
    ap.add_argument("--det_cache", default="dance_detections_cache/val")
    ap.add_argument("--gt_root", default="datasets/dancetrack/val")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    targs = make_parser().parse_args([])                       # OC-SORT defaults, same as the runner
    img_size = get_exp(a.exp_file, None).test_size
    pred = GRUBoxPredictor.from_checkpoint(a.ckpt)

    orig_associate = O.associate
    state = {}

    def wrapped(dets, trks, iou_threshold, velocities, previous_obs, vdc_weight):
        res = orig_associate(dets, trks, iou_threshold, velocities, previous_obs, vdc_weight)
        trk = state["tracker"]
        res_kf = orig_associate(dets, trk._kf_trks, iou_threshold, velocities, previous_obs, vdc_weight)
        g = {int(t): int(d) for d, t in res[0]}
        k = {int(t): int(d) for d, t in res_kf[0]}
        st = state["stats"]
        st["frames"] += 1
        st["first_round_track_rows"] += len(trks)
        st["gru_rows"] += int(trk._gru_rows.sum())
        diff_tracks = [t for t in set(g) | set(k) if g.get(t) != k.get(t)]
        if diff_tracks:
            st["frames_with_any_difference"] += 1
        fid = state["frame"]
        gtb, gti = state["gt"].get(fid, (np.zeros((0, 4)), np.zeros(0, int)))
        dlab = gt_labels(dets[:, :4], gtb, gti)
        for t in diff_tracks:
            st["differing_track_decisions"] += 1
            st["differing_on_gru_row" if trk._gru_rows[t] else "differing_on_kalman_row"] += 1
            tid = state["track_gt"].get(trk.trackers[t].id)
            gd, kd = g.get(t), k.get(t)
            kind = ("gru_matched_kf_unmatched" if kd is None else "kf_matched_gru_unmatched" if gd is None
                    else "different_detection")
            st[kind] += 1
            if tid is None:
                st["track_gt_unknown"] += 1
                continue
            gc = gd is not None and dlab[gd] == tid
            kc = kd is not None and dlab[kd] == tid
            key = "gru_correct_kf_not" if gc and not kc else "kf_correct_gru_not" if kc and not gc else \
                "both_correct" if gc and kc else "neither_correct"
            st[key] += 1
            st[kind + "__" + key] = st.get(kind + "__" + key, 0) + 1
        return res

    O.associate = wrapped
    keys = ["frames", "first_round_track_rows", "gru_rows", "frames_with_any_difference", "differing_track_decisions",
            "differing_on_gru_row", "differing_on_kalman_row", "gru_matched_kf_unmatched", "kf_matched_gru_unmatched",
            "different_detection", "track_gt_unknown", "gru_correct_kf_not", "kf_correct_gru_not", "both_correct",
            "neither_correct"]
    per_seq, identical = {}, {}
    for cache in sorted(glob.glob(os.path.join(a.det_cache, "dancetrack*.pt"))):
        seq = os.path.basename(cache)[:-3]
        c = torch.load(cache, weights_only=False)
        dets, img_hw = c["dets"], tuple(c["img_hw"])
        gt = np.loadtxt(os.path.join(a.gt_root, seq, "gt", "gt.txt"), delimiter=",", ndmin=2)
        gtf = {}
        for f in np.unique(gt[:, 0]).astype(int):
            r = gt[gt[:, 0] == f]
            gtf[f] = (np.c_[r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]], r[:, 1].astype(int))
        tracker = DiagGRU(det_thresh=targs.track_thresh, predictor=pred, iou_threshold=targs.iou_thresh,
                          asso_func=targs.asso, delta_t=targs.deltat, inertia=targs.inertia, use_byte=targs.use_byte)
        st = {k: 0 for k in keys}
        state.update(tracker=tracker, stats=st, gt=gtf, track_gt={})
        results = []
        for frame_id in sorted(dets):
            state["frame"] = frame_id
            out = dets[frame_id]
            online = tracker.update(None if out is None else out.clone(), img_hw, img_size)
            # refresh track -> GT identity from this frame's observations
            gtb, gti = gtf.get(frame_id, (np.zeros((0, 4)), np.zeros(0, int)))
            obs = [t for t in tracker.trackers if t.time_since_update == 0]
            if obs:
                lab = gt_labels(np.array([t.last_observation[:4] for t in obs]), gtb, gti)
                for t, l in zip(obs, lab):
                    if l >= 0:
                        state["track_gt"][t.id] = int(l)
            tl, ids = [], []
            for t in online:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                if tlwh[2] * tlwh[3] > targs.min_box_area:
                    tl.append(tlwh); ids.append(t[4])
            results.append((frame_id, tl, ids))
        res_file = os.path.join(a.out, "tracks", seq + ".txt")
        os.makedirs(os.path.dirname(res_file), exist_ok=True)
        write_results_no_score(res_file, results)
        identical[seq] = open(res_file, "rb").read() == open(os.path.join(a.primary, seq + ".txt"), "rb").read()
        per_seq[seq] = st
        print(seq, st["differing_track_decisions"], "differing decisions | identical to primary:", identical[seq])
    O.associate = orig_associate
    all_keys = sorted({k for s in per_seq.values() for k in s})
    tot = {k: int(sum(s.get(k, 0) for s in per_seq.values())) for k in all_keys}
    json.dump({"doc": __doc__, "totals": tot, "per_sequence": per_seq, "outputs_identical_to_primary": identical,
               "all_identical": all(identical.values())}, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps(tot, indent=1), "\nall outputs identical to primary:", all(identical.values()))


if __name__ == "__main__":
    main()
