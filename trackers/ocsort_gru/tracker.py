"""
OCSortGRU: OC-SORT whose first-stage association boxes come from a trained GRU
("direct prediction"), with Kalman prediction as fallback.

What changes vs. OCSort (only _predict_boxes / _new_track are overridden):
  * Every track's Kalman filter is still advanced exactly as in OC-SORT
    (super()._predict_boxes()); its state, ORU freeze/unfreeze, and update
    equations are untouched.
  * For eligible tracks, the row of `trks` that OCSort.update passes to
    associate() (IoU + OCM cost, first round) is replaced by the GRU box.
    With use_byte=True those same rows are also used by the BYTE round.
  * OCM still uses observation-derived velocities and k_previous_obs; OCR still
    uses last_observation; output boxes are still last_observation. None of
    these read the predicted boxes, so they behave exactly as in OC-SORT.

Eligibility for a GRU prediction (otherwise the Kalman box is kept):
  * trailing gap: the last real observation is at most `max_gap` frames before
    the current frame (max_gap=1 -> observed in the previous frame);
  * at least `min_obs` valid observations inside the L-frame window;
  * the decoded GRU box is finite with positive width/height.
"""
import numpy as np

from gru_motion.features import window_stats
from trackers.ocsort_tracker.ocsort import OCSort
from .history import track_window


class OCSortGRU(OCSort):
    def __init__(self, det_thresh, predictor, max_gap=1, min_obs=3, **kwargs):
        super().__init__(det_thresh, **kwargs)
        L = predictor.history_len
        if not (1 <= max_gap <= L - 1):
            raise ValueError("max_gap must be in [1, history_len-1={}], got {}".format(L - 1, max_gap))
        if not (2 <= min_obs <= L):
            raise ValueError("min_obs must be in [2, history_len={}], got {}".format(L, min_obs))
        self.predictor = predictor
        self.max_gap = int(max_gap)
        self.min_obs = int(min_obs)
        self.stats = {"track_predictions": 0, "gru": 0, "fallback_gap": 0,
                      "fallback_short_history": 0, "fallback_invalid_output": 0}

    def _new_track(self, det):
        trk = super()._new_track(det)
        trk.gru_birth_obs = np.array(det, dtype=np.float64)  # copy; see history.py
        return trk

    def _predict_boxes(self):
        trks = super()._predict_boxes()   # Kalman predict for ALL tracks (state unchanged below)
        n = len(self.trackers)
        if n == 0:
            return trks
        L = self.predictor.history_len
        boxes = np.zeros((n, L, 4))
        masks = np.zeros((n, L))
        for i, trk in enumerate(self.trackers):
            boxes[i], masks[i] = track_window(trk, L)
        num_valid, trailing = window_stats(masks)
        gap_ok = trailing + 1 <= self.max_gap          # frames since last observation
        hist_ok = num_valid >= self.min_obs
        eligible = np.flatnonzero(gap_ok & hist_ok)

        self.stats["track_predictions"] += n
        self.stats["fallback_gap"] += int((~gap_ok).sum())
        self.stats["fallback_short_history"] += int((gap_ok & ~hist_ok).sum())
        if len(eligible) == 0:
            return trks

        pred = self.predictor.predict_xyxy(boxes[eligible], masks[eligible])
        ok = np.all(np.isfinite(pred), axis=1) & (pred[:, 2] > pred[:, 0]) & (pred[:, 3] > pred[:, 1])
        self.stats["fallback_invalid_output"] += int((~ok).sum())
        self.stats["gru"] += int(ok.sum())
        trks[eligible[ok], :4] = pred[ok]
        return trks

    def update_public(self, *args, **kwargs):
        raise NotImplementedError("OCSortGRU only supports update() (private-detection DanceTrack path)")
