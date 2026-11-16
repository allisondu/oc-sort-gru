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

Checkpoint schema (format v2; all metadata is plain JSON types so the file
loads with torch.load(weights_only=True)):

    format               "oc-sort-gru/motion-checkpoint/v2"
    feature_version      int, must equal features.FEATURE_VERSION
    history_len          int L (convenience copy)
    model_config         {hidden_size, num_layers, dropout}
    state_dict           weights + normalisation buffers
    training_conditions  what the model SAW: {training_mode, history_len,
                         min_valid_obs, require_full_history, max_trailing_gap,
                         prediction_horizons, target, augmentation}
    inference_support    what the tracker MAY SUBMIT: {history_len,
                         min_valid_obs, require_full_history,
                         max_trailing_gap, prediction_horizons}; must be no
                         broader than training_conditions (validated on save
                         and on load)
    train_info           provenance: epochs, num_train_windows, sequences,
                         split hash, seed, git commit, losses, ...

Legacy v1 checkpoints (no conditions recorded) load with the most
conservative support: full history, trailing gap 0, horizon 1, and
meta["legacy_checkpoint"] = True.
"""
import hashlib
import json
import os

import torch
import torch.nn as nn

from .features import FEATURE_VERSION, INPUT_DIM, OUTPUT_DIM

CHECKPOINT_FORMAT = "oc-sort-gru/motion-checkpoint/v2"
LEGACY_FORMATS = ("oc-sort-gru/motion-checkpoint/v1",)
SUPPORT_KEYS = ("history_len", "min_valid_obs", "require_full_history", "max_trailing_gap",
                "prediction_horizons")
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


def support_from_training(training_conditions):
    """Default inference support = exactly the conditions the model was trained on."""
    return {k: training_conditions[k] for k in SUPPORT_KEYS}


def conservative_support(history_len):
    """Most restrictive support: full history, last frame observed, one-frame horizon."""
    return {"history_len": int(history_len), "min_valid_obs": int(history_len), "require_full_history": True,
            "max_trailing_gap": 0, "prediction_horizons": [1]}


def validate_support(support, history_len, training_conditions=None):
    """Raise ValueError unless `support` is well-formed and no broader than training."""
    if not isinstance(support, dict) or any(k not in support for k in SUPPORT_KEYS):
        raise ValueError("inference_support must contain {}".format(SUPPORT_KEYS))
    L, mv, gap = support["history_len"], support["min_valid_obs"], support["max_trailing_gap"]
    if L != history_len:
        raise ValueError("inference_support.history_len {} != model history_len {}".format(L, history_len))
    if not (isinstance(mv, int) and 1 <= mv <= L):
        raise ValueError("inference_support.min_valid_obs must be an int in [1, {}]".format(L))
    if support["require_full_history"] != (mv == L):
        raise ValueError("require_full_history must equal (min_valid_obs == history_len)")
    if not (isinstance(gap, int) and 0 <= gap <= L - 1):
        raise ValueError("inference_support.max_trailing_gap must be an int in [0, {}]".format(L - 1))
    if list(support["prediction_horizons"]) != list(range(1, gap + 2)):
        raise ValueError("prediction_horizons must be 1..max_trailing_gap+1")
    if training_conditions is not None:
        tc = training_conditions
        if tc.get("history_len") != L:
            raise ValueError("training history_len differs from support history_len")
        if mv < tc.get("min_valid_obs", L):
            raise ValueError("support min_valid_obs {} is below what training saw ({})".format(
                mv, tc.get("min_valid_obs")))
        if gap > tc.get("max_trailing_gap", 0):
            raise ValueError("support max_trailing_gap {} exceeds what training saw ({})".format(
                gap, tc.get("max_trailing_gap")))


def save_checkpoint(path, model, training_conditions, train_info, inference_support=None):
    """
    Write a v2 checkpoint.
    training_conditions: SamplingSpec.training_conditions() (plain dict).
    inference_support: defaults to support_from_training(training_conditions);
        may only be narrower.
    train_info must describe real training; load_checkpoint() requires
        epochs > 0 and num_train_windows > 0.
    """
    tc = json.loads(json.dumps(dict(training_conditions)))
    L = int(tc["history_len"])
    support = json.loads(json.dumps(inference_support if inference_support is not None
                                    else support_from_training(tc)))
    validate_support(support, L, tc)
    ckpt = {
        "format": CHECKPOINT_FORMAT,
        "feature_version": FEATURE_VERSION,
        "history_len": L,
        "model_config": dict(model.config),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "training_conditions": tc,
        "inference_support": support,
        # JSON round-trip: train_info must hold only plain types so the file stays
        # loadable with torch.load(weights_only=True) (e.g. torch.__version__ is a
        # TorchVersion object, numpy scalars are not allow-listed).
        "train_info": json.loads(json.dumps(dict(train_info), default=str)),
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

    fmt = ckpt.get("format") if isinstance(ckpt, dict) else None
    if fmt != CHECKPOINT_FORMAT and fmt not in LEGACY_FORMATS:
        raise GRUCheckpointError("{} is not a GRU motion checkpoint (expected format '{}'). {}".format(
            path, CHECKPOINT_FORMAT, hint))
    legacy = fmt in LEGACY_FORMATS
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

    if legacy:
        # No recorded conditions: never infer capabilities; restrict to the
        # narrowest input class any one-step model was trained on.
        training_conditions = None
        support = conservative_support(history_len)
    else:
        training_conditions = ckpt.get("training_conditions")
        support = ckpt.get("inference_support")
        if not isinstance(training_conditions, dict):
            raise GRUCheckpointError("{}: missing training_conditions".format(path))
        try:
            validate_support(support, history_len, training_conditions)
        except ValueError as e:
            raise GRUCheckpointError("{}: invalid inference_support: {}".format(path, e))

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
            "format": fmt, "legacy_checkpoint": legacy,
            "training_conditions": training_conditions, "inference_support": support,
            "num_params": sum(p.numel() for p in model.parameters())}
    return model, meta
