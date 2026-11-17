"""
Build GRU input windows from an OC-SORT track's own observation history.

Timing (see KalmanBoxTracker in trackers/ocsort_tracker/ocsort.py):
  * A track is born at age 0 from a detection. Upstream does NOT put that birth
    detection into track.observations, so OCSortGRU._new_track stores it
    separately as track.gru_birth_obs (observations itself is left untouched,
    because OCM reads it).
  * Each frame, OCSort.update first calls predict() on every track, which
    increments track.age. So during association in frame t, track.age is t's
    index on the track's time grid and track.observations holds real detections
    keyed by the age at which they were matched, all < track.age.
  * The window therefore covers ages age-L .. age-1 and predicts the box at
    `age`. The current frame's detections are matched only afterwards, so they
    cannot leak in.
Only real matched detections are used; Kalman predictions and ORU's virtual
boxes (which live only inside the Kalman filter) never enter the window.
"""
import numpy as np

from gru_motion.features import MIN_SIZE, build_window


class _ObsWithBirth(object):
    """Read-only view: track.observations plus the birth detection at age 0."""
    __slots__ = ("obs", "birth")

    def __init__(self, obs, birth):
        self.obs, self.birth = obs, birth

    def get(self, k):
        v = self.obs.get(k)
        if v is None and k == 0:
            return self.birth
        return v


def track_window(trk, history_len):
    """(boxes (L, 4) cxcywh, mask (L,)) for predicting the box at trk.age."""
    birth = getattr(trk, "gru_birth_obs", None)
    obs = trk.observations if birth is None else _ObsWithBirth(trk.observations, birth)
    return build_window(obs, trk.age, history_len)


def track_windows(trackers, history_len):
    """
    Batched equivalent of [track_window(t, L) for t in trackers] -> (boxes (N, L, 4), mask (N, L)).

    Same slots, same observation sources (track.observations + gru_birth_obs at
    age 0), same validity rule and the same float64 cxcywh arithmetic as
    features.build_window, but the per-box conversion and validity checks run
    once, vectorised, over all observed slots of all tracks (the per-box numpy
    calls dominated tracker overhead). Results are bit-identical
    (tests/test_gru_direct.py::test_track_windows_matches_per_track).
    """
    n, L = len(trackers), history_len
    boxes = np.zeros((n, L, 4), dtype=np.float64)
    mask = np.zeros((n, L), dtype=np.float64)
    rows, ii, jj = [], [], []
    for i, trk in enumerate(trackers):
        obs, start = trk.observations, trk.age - L
        birth = getattr(trk, "gru_birth_obs", None)
        for j in range(L):
            k = start + j
            b = obs.get(k)
            if b is None:
                if k != 0 or birth is None:
                    continue
                b = birth
            rows.append(b[:4])
            ii.append(i)
            jj.append(j)
    if not rows:
        return boxes, mask
    b = np.array(rows, dtype=np.float64)
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    c = np.stack([b[:, 0] + w / 2.0, b[:, 1] + h / 2.0, w, h], axis=-1)
    ok = np.all(np.isfinite(c), axis=1) & ~(c[:, 2] < MIN_SIZE) & ~(c[:, 3] < MIN_SIZE)
    ii, jj = np.asarray(ii)[ok], np.asarray(jj)[ok]
    boxes[ii, jj] = c[ok]
    mask[ii, jj] = 1.0
    return boxes, mask
