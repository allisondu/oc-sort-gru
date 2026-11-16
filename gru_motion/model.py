"""
GRU motion model and checkpoint format.

Architecture: GRU(input=5, hidden=H, layers=N, batch_first) over the L-slot
window built by gru_motion.features -> last hidden state -> Linear(H, 4).

Standardisation statistics (feature/target mean and std) are registered
buffers, so they are saved in the state_dict and travel with the checkpoint.
The mask channel is never standardised and missing slots stay exactly zero.

Checkpoints are plain dicts written by save_checkpoint(); load_checkpoint()
validates them and refuses anything that does not look like a trained model,
so the tracker can never silently run on random weights.
"""
import hashlib
import math
import os

import torch
import torch.nn as nn

from .features import FEATURE_VERSION, INPUT_DIM, OUTPUT_DIM

CHECKPOINT_FORMAT = "oc-sort-gru/motion-checkpoint/v1"
DEFAULT_MODEL_CONFIG = {"hidden_size": 64, "num_layers": 1, "dropout": 0.0}
DEFAULT_HISTORY_LEN = 10


class GRUCheckpointError(RuntimeError):
    pass


class GRUMotionModel(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.config = {"hidden_size": int(hidden_size), "num_layers": int(num_layers),
                       "dropout": float(dropout)}
        self.gru = nn.GRU(INPUT_DIM, hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, OUTPUT_DIM)
        self.register_buffer("in_mean", torch.zeros(INPUT_DIM - 1))
        self.register_buffer("in_std", torch.ones(INPUT_DIM - 1))
        self.register_buffer("out_mean", torch.zeros(OUTPUT_DIM))
        self.register_buffer("out_std", torch.ones(OUTPUT_DIM))

    def set_normalization(self, in_mean, in_std, out_mean, out_std):
        """Set standardisation stats (computed on training data only)."""
        for name, val in (("in_mean", in_mean), ("in_std", in_std),
                          ("out_mean", out_mean), ("out_std", out_std)):
            t = torch.as_tensor(val, dtype=torch.float32).reshape(getattr(self, name).shape)
            if name.endswith("std") and bool((t <= 0).any()):
                raise ValueError("{} must be positive".format(name))
            getattr(self, name).copy_(t)

    def forward(self, feats):
        """feats (B, L, 5) from features.encode_windows -> standardised output (B, 4)."""
        mask = feats[..., 4:5]
        x = (feats[..., :4] - self.in_mean) / self.in_std * mask
        x = torch.cat([x, mask], dim=-1)
        _, h = self.gru(x)
        return self.head(h[-1])

    def normalize_targets(self, targets):
        return (targets - self.out_mean) / self.out_std

    def denormalize_outputs(self, y):
        return y * self.out_std + self.out_mean

    @torch.no_grad()
    def predict(self, feats):
        """Raw-unit outputs (see features.decode_outputs) for (B, L, 5) features."""
        return self.denormalize_outputs(self.forward(feats))


def save_checkpoint(path, model, history_len, train_info):
    """
    Write a checkpoint. `train_info` must describe real training, e.g.
    {"epochs": 30, "num_train_windows": 812345, "train_sequences": [...],
     "seed": 0, "git_commit": "...", ...}; load_checkpoint() requires
    epochs > 0 and num_train_windows > 0.
    """
    ckpt = {
        "format": CHECKPOINT_FORMAT,
        "feature_version": FEATURE_VERSION,
        "history_len": int(history_len),
        "model_config": dict(model.config),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "train_info": dict(train_info),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(ckpt, path)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path, map_location="cpu"):
    """Load and validate a checkpoint. Returns (model in eval mode, metadata dict)."""
    hint = ("Train a model first (Phase 2) and pass its path via --gru_ckpt. "
            "The GRU tracker never runs without a validated trained checkpoint.")
    if not path:
        raise GRUCheckpointError("No GRU checkpoint given. " + hint)
    if not os.path.isfile(path):
        raise GRUCheckpointError("GRU checkpoint not found: {}. {}".format(path, hint))
    try:
        # weights_only=True: checkpoint is plain tensors/dicts, no pickled code
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise GRUCheckpointError("Could not read GRU checkpoint {}: {}".format(path, e))

    if not isinstance(ckpt, dict) or ckpt.get("format") != CHECKPOINT_FORMAT:
        raise GRUCheckpointError("{} is not a GRU motion checkpoint (expected format '{}'). {}".format(
            path, CHECKPOINT_FORMAT, hint))
    if ckpt.get("feature_version") != FEATURE_VERSION:
        raise GRUCheckpointError(
            "{} was built for feature version {} but the code uses version {}; retrain the model.".format(
                path, ckpt.get("feature_version"), FEATURE_VERSION))
    info = ckpt.get("train_info") or {}
    if not (info.get("epochs", 0) > 0 and info.get("num_train_windows", 0) > 0):
        raise GRUCheckpointError(
            "{} has no record of training (train_info.epochs / num_train_windows missing or 0); "
            "refusing to use a possibly untrained model. {}".format(path, hint))
    history_len = ckpt.get("history_len")
    if not isinstance(history_len, int) or history_len < 2:
        raise GRUCheckpointError("{}: invalid history_len {!r}".format(path, history_len))

    model = GRUMotionModel(**ckpt["model_config"])
    try:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    except Exception as e:
        raise GRUCheckpointError("{}: state_dict does not match model_config: {}".format(path, e))
    for name, t in model.state_dict().items():
        if not torch.isfinite(t).all():
            raise GRUCheckpointError("{}: non-finite values in '{}'".format(path, name))
    if (model.in_std <= 0).any() or (model.out_std <= 0).any():
        raise GRUCheckpointError("{}: normalisation std must be positive".format(path))

    model.to(map_location).eval()
    meta = {"path": os.path.abspath(path), "sha256": file_sha256(path),
            "history_len": history_len, "model_config": dict(ckpt["model_config"]),
            "feature_version": ckpt["feature_version"], "train_info": info,
            "num_params": sum(p.numel() for p in model.parameters())}
    return model, meta
