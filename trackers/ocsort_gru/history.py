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
from gru_motion.features import build_window


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
