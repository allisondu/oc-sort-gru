"""
Numpy-in / numpy-out wrapper around a trained GRUMotionModel, used by the
tracker. One batched forward pass per call; no tracker knowledge here.
"""
import numpy as np
import torch

from .features import cxcywh_to_xyxy, decode_outputs, encode_windows
from .model import conservative_support, load_checkpoint, validate_support


class GRUBoxPredictor(object):
    """
    `support` = the checkpoint's inference_support (which windows the model may
    be given). It is enforced by the tracker (trackers/ocsort_gru); when built
    without a checkpoint (tests), it defaults to the most conservative support.
    """
    def __init__(self, model, history_len, device="cpu", meta=None, support=None):
        self.model = model.to(device).eval()
        self.history_len = int(history_len)
        self.device = torch.device(device)
        self.meta = meta or {}
        self.support = dict(support) if support is not None else conservative_support(self.history_len)
        validate_support(self.support, self.history_len)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        model, meta = load_checkpoint(path, map_location="cpu")
        return cls(model, meta["history_len"], device=device, meta=meta, support=meta["inference_support"])

    def predict_xyxy(self, boxes, mask):
        """
        boxes (B, L, 4) cxcywh, mask (B, L) from features.build_window, each
        window with >= 1 valid slot. Returns (B, 4) xyxy float64 predictions for
        the frame right after the window. Rows may be non-finite only if the
        model itself produced non-finite outputs; callers must check.
        """
        if len(boxes) == 0:
            return np.zeros((0, 4))
        feats, ref = encode_windows(boxes, mask)
        with torch.no_grad():
            out = self.model.predict(torch.from_numpy(feats).to(self.device))
        out = out.cpu().numpy().astype(np.float64)
        return cxcywh_to_xyxy(decode_outputs(out, ref))
