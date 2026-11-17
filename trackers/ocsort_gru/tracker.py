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
  3. if require_full_prefix (gap-aware checkpoints): every slot before the
     trailing gap observed, num_valid == L - trailing gap
                                             (else fallback_history_holes)
  4. decoded box finite with positive size   (else fallback_invalid_output)
Prediction horizon = frames since the last observation = trk.time_since_update
after predict() = trailing gap + 1 (logged as frames_since_last_obs).
Gap-aware checkpoints (max_trailing_gap 3) therefore predict horizons 1..4.
Runtime settings may only be MORE restrictive than the checkpoint; anything
broader raises ValueError (never silently widened or clipped).
"""
import numpy as np

from gru_motion.features import window_stats
from trackers.ocsort_tracker.ocsort import OCSort
from .history import track_window, track_windows  # noqa: F401  (track_window kept for diagnostics)

STAT_KEYS = ("track_predictions", "gru_used", "fallback_missing_recent_observation",
             "fallback_unsupported_gap", "fallback_insufficient_history", "fallback_history_holes",
             "fallback_invalid_output")
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
    pol = {"history_len": L, "min_valid_obs": mv, "max_trailing_gap": gap,
           "require_full_history": mv == L, "prediction_horizons": list(range(1, gap + 2))}
    if support.get("require_full_prefix", False):
        pol["require_full_prefix"] = True        # never dropped at runtime (training had no holes)
    return pol


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
        boxes, masks = track_windows(self.trackers, L)   # batched, bit-identical to per-track track_window
        num_valid, trailing = window_stats(masks)       # trailing == L if no obs in window

        decision = np.full(n, "gru_used", dtype=object)
        gap_bad = trailing > self.policy["max_trailing_gap"]
        decision[gap_bad] = ("fallback_missing_recent_observation" if self.policy["max_trailing_gap"] == 0
                             else "fallback_unsupported_gap")
        hist_bad = ~gap_bad & (num_valid < self.policy["min_valid_obs"])
        decision[hist_bad] = "fallback_insufficient_history"
        holes = np.zeros(n, dtype=bool)
        if self.policy.get("require_full_prefix", False):
            holes = ~gap_bad & ~hist_bad & (num_valid + trailing != L)
            decision[holes] = "fallback_history_holes"
        eligible = np.flatnonzero(~gap_bad & ~hist_bad & ~holes)

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
