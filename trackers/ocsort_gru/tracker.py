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
    uses last_observation; output boxes are still last_observation.

Three different per-track quantities (do not conflate them):
  * track age        trk.age: frames since the track was born (grows every frame).
  * num_valid        real observations inside the L-frame window (<= L).
  * frames since last observation  trk.time_since_update after predict()
                     (1 = observed in the previous frame); the window's
                     trailing gap = frames_since_last_obs - 1.

Eligibility = the checkpoint's inference_support, optionally narrowed at runtime:
  1. trailing gap <= max_trailing_gap        (else fallback_missing_recent_observation
                                               if the limit is 0, else fallback_unsupported_gap)
  2. num_valid >= min_valid_obs              (else fallback_insufficient_history)
     (standard checkpoints have min_valid_obs = L, i.e. full history)
  3. decoded box finite with positive size   (else fallback_invalid_output)
Runtime settings may only be MORE restrictive than the checkpoint; anything
broader raises ValueError (never silently widened or clipped).
"""
import numpy as np

from gru_motion.features import window_stats
from trackers.ocsort_tracker.ocsort import OCSort
from .history import track_window

STAT_KEYS = ("track_predictions", "gru_used", "fallback_missing_recent_observation",
             "fallback_unsupported_gap", "fallback_insufficient_history", "fallback_invalid_output")
DECISION_COLUMNS = ("frame", "track_id", "track_age", "total_observations", "num_valid_in_window",
                    "frames_since_last_obs", "decision")


def effective_policy(support, min_obs=None, max_trailing_gap=None):
    """
    Combine checkpoint support with optional runtime overrides.
    None -> use the checkpoint's value. Broader-than-checkpoint -> ValueError.
    """
    L = support["history_len"]
    mv = support["min_valid_obs"] if min_obs is None else int(min_obs)
    gap = support["max_trailing_gap"] if max_trailing_gap is None else int(max_trailing_gap)
    if mv < support["min_valid_obs"] or mv > L:
        raise ValueError("min_obs={} not allowed: checkpoint supports min_valid_obs >= {} (history_len {}){}".format(
            mv, support["min_valid_obs"], L,
            " -- this is a full-history (standard) checkpoint" if support["require_full_history"] else ""))
    if gap < 0 or gap > support["max_trailing_gap"]:
        raise ValueError("max_trailing_gap={} not allowed: checkpoint supports at most {}".format(
            gap, support["max_trailing_gap"]))
    return {"history_len": L, "min_valid_obs": mv, "max_trailing_gap": gap,
            "require_full_history": mv == L, "prediction_horizons": list(range(1, gap + 2))}


class OCSortGRU(OCSort):
    def __init__(self, det_thresh, predictor, min_obs=None, max_trailing_gap=None,
                 log_decisions=False, **kwargs):
        super().__init__(det_thresh, **kwargs)
        self.predictor = predictor
        self.policy = effective_policy(predictor.support, min_obs, max_trailing_gap)
        self.stats = {k: 0 for k in STAT_KEYS}
        self.decision_log = [] if log_decisions else None

    def _new_track(self, det):
        trk = super()._new_track(det)
        trk.gru_birth_obs = np.array(det, dtype=np.float64)  # copy; see history.py
        return trk

    def _predict_boxes(self):
        trks = super()._predict_boxes()   # Kalman predict for ALL tracks (state unchanged below)
        n = len(self.trackers)
        if n == 0:
            return trks
        L = self.policy["history_len"]
        boxes = np.zeros((n, L, 4))
        masks = np.zeros((n, L))
        for i, trk in enumerate(self.trackers):
            boxes[i], masks[i] = track_window(trk, L)
        num_valid, trailing = window_stats(masks)       # trailing == L if no obs in window

        decision = np.full(n, "gru_used", dtype=object)
        gap_bad = trailing > self.policy["max_trailing_gap"]
        decision[gap_bad] = ("fallback_missing_recent_observation" if self.policy["max_trailing_gap"] == 0
                             else "fallback_unsupported_gap")
        hist_bad = ~gap_bad & (num_valid < self.policy["min_valid_obs"])
        decision[hist_bad] = "fallback_insufficient_history"
        eligible = np.flatnonzero(~gap_bad & ~hist_bad)

        if len(eligible):
            pred = self.predictor.predict_xyxy(boxes[eligible], masks[eligible])
            ok = np.all(np.isfinite(pred), axis=1) & (pred[:, 2] > pred[:, 0]) & (pred[:, 3] > pred[:, 1])
            decision[eligible[~ok]] = "fallback_invalid_output"
            trks[eligible[ok], :4] = pred[ok]

        self.stats["track_predictions"] += n
        for d in decision:
            self.stats[d] += 1
        if self.decision_log is not None:
            for i, trk in enumerate(self.trackers):
                birth_extra = int(getattr(trk, "gru_birth_obs", None) is not None and 0 not in trk.observations)
                self.decision_log.append((self.frame_count, trk.id + 1, trk.age,
                                          len(trk.observations) + birth_extra, int(num_valid[i]),
                                          trk.time_since_update, decision[i]))
        return trks

    def update_public(self, *args, **kwargs):
        raise NotImplementedError("OCSortGRU only supports update() (private-detection DanceTrack path)")
