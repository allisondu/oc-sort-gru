"""
Train the GRU motion model on DanceTrack TRAIN ground truth.

    # research runs: fixed sequence-level split
    python -m gru_motion.train --split_config configs/gru/dancetrack_split.json \
        --training_mode standard --epochs 30 --batch_size 512 --out_dir gru_runs/<name>

    # smoke tests: ad-hoc sequence selection
    python -m gru_motion.train --max_seqs 2 --max_windows 2000 --epochs 3 \
        --training_mode standard --out_dir gru_runs/smoke/<name> --tag smoke

Training modes (see gru_motion/data.py): "standard" (all L history frames
observed) and "partial" (>= --min_valid_obs observed, older frames may be
missing, most recent frame observed). Both predict exactly one frame ahead.

Outputs in <out_dir> (all loadable by GRUBoxPredictor / --gru_ckpt):
  best.pt       weights of the epoch with the LOWEST holdout loss (strict
                improvement; ties keep the earliest epoch; epoch 0 = untrained
                is never eligible). Written whenever holdout loss improves,
                rewritten at the end with the full loss history. Primary
                checkpoint for evaluation.
  last.pt       weights after the final completed epoch.
  summary.json  human-readable copy of both checkpoints' train_info.
Without holdout sequences only last.pt is written, and only with
--allow_no_holdout (smoke tests); research runs must have a holdout.
Holdout loss is computed in eval mode under torch.no_grad(); holdout data
never produces gradients. Loss is Smooth-L1 on standardised targets; Adam.

Detector-noise augmentation (--box_noise jitter_v1, see gru_motion/augment.py):
  * history boxes of every training window get fresh size-relative jitter
    EVERY EPOCH (numpy stream keyed by (--train_noise_seed, epoch)); targets
    stay the clean GT box, encoded relative to the noisy last observation;
    masks are unchanged (no missing frames are introduced).
  * normalisation statistics are computed on the CLEAN training windows (same
    convention and values as the clean-trained model); noisy inputs are
    standardised with those statistics.
  * two holdout losses per epoch on the same holdout windows: clean
    (`holdout_loss`) and noisy (`holdout_noisy_loss`, ONE fixed perturbation
    drawn once from --holdout_noise_seed, independent of the training seed).
  * best.pt is selected on --selection_metric (auto: noisy holdout when noise
    is enabled, clean holdout otherwise).
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .augment import BoxNoiseConfig, epoch_rng, holdout_rng, perturb_windows
from .data import (DANCETRACK_VAL_SEQUENCES, TRAINING_MODES, SamplingSpec, build_raw_windows, encode_raw,
                   enumerate_samples, list_sequences, load_sequences, load_split_config, normalization_stats,
                   subsample)
from .features import FEATURE_VERSION
from .model import GRUMotionModel, save_checkpoint


# Named noise presets. jitter_v1 is a MODELLING ASSUMPTION (no train-split detector
# outputs were available to fit it): ~2.5% of box size per coordinate, i.i.d. per frame.
NOISE_PRESETS = {
    "none": BoxNoiseConfig(),
    "jitter_v1": BoxNoiseConfig(center_x_std=0.025, center_y_std=0.025, log_w_std=0.025, log_h_std=0.025,
                                apply_prob=1.0, clip_sigma=3.0),
}
SELECTION_METRICS = ("auto", "holdout_clean", "holdout_noisy")


def noise_config(args):
    base = NOISE_PRESETS[args.box_noise].to_dict()
    for k in ("center_x_std", "center_y_std", "log_w_std", "log_h_std", "apply_prob", "clip_sigma"):
        v = getattr(args, "noise_" + k)
        if v is not None:
            base[k] = v
    return BoxNoiseConfig.from_dict(base)


def noise_realised(clean, noisy, mask):
    """Summary of the perturbation actually applied (valid slots only)."""
    m = mask > 0
    c, n = clean[m], noisy[m]
    changed = np.any(n != c, axis=-1)
    return {"frac_slots_perturbed": float(changed.mean()),
            "frac_windows_perturbed": float(np.any(np.any(noisy != clean, axis=-1), axis=-1).mean()),
            "std_dcx_over_w": float(((n[:, 0] - c[:, 0]) / c[:, 2]).std()),
            "std_dcy_over_h": float(((n[:, 1] - c[:, 1]) / c[:, 3]).std()),
            "std_dlogw": float(np.log(n[:, 2] / c[:, 2]).std()),
            "std_dlogh": float(np.log(n[:, 3] / c[:, 3]).std())}


def pick_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")      # tiny GRU: CPU beats MPS on Apple Silicon
    return torch.device(name)


def git_commit():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        c = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", root, "diff", "--quiet", "HEAD"]) != 0
        return c + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


@torch.no_grad()
def eval_loss(model, x, y_norm, batch_size):
    model.eval()
    tot = 0.0
    for i in range(0, len(x), batch_size):
        tot += F.smooth_l1_loss(model(x[i:i + batch_size]), y_norm[i:i + batch_size], reduction="sum").item()
    return tot / (len(x) * y_norm.shape[1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="datasets_gru/dancetrack_gt")
    ap.add_argument("--split_config", default=None,
                    help="fixed train/holdout split JSON (configs/gru/dancetrack_split.json); "
                         "excludes --seqs/--max_seqs/--holdout_seqs")
    ap.add_argument("--training_mode", choices=TRAINING_MODES, required=True)
    ap.add_argument("--min_valid_obs", type=int, default=None,
                    help="partial mode only (default 3); standard mode always uses history_len")
    ap.add_argument("--max_trailing_gap", type=int, default=0,
                    help="must be 0 (gap-aware training is not implemented yet)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--max_seqs", type=int, default=None)
    ap.add_argument("--max_windows", type=int, default=None)
    ap.add_argument("--holdout_seqs", nargs="*", default=[],
                    help="train-split sequences held out for monitoring (excluded from training)")
    ap.add_argument("--val_guard_dir", default="datasets/dancetrack/val",
                    help="refuse to train on any sequence name found here")
    ap.add_argument("--history_len", type=int, default=10)
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smooth_l1_beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu / cuda / mps / auto")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="", help="free-text purpose, e.g. 'smoke' (stored in train_info)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow_no_holdout", action="store_true",
                    help="smoke tests only: train without holdout; writes last.pt only (no best.pt)")
    ap.add_argument("--early_stop_patience", type=int, default=0,
                    help="stop after N epochs without holdout improvement (0 = off, default)")
    ap.add_argument("--box_noise", choices=sorted(NOISE_PRESETS), default="none",
                    help="detector-like localisation noise on history boxes (training + noisy holdout)")
    for k, h in (("center_x_std", "cx std, fraction of box width"), ("center_y_std", "cy std, fraction of height"),
                 ("log_w_std", "log-width std"), ("log_h_std", "log-height std"),
                 ("apply_prob", "probability a training window is perturbed"),
                 ("clip_sigma", "truncate draws at +-k std (0 = off)")):
        ap.add_argument("--noise_" + k, type=float, default=None, help="override preset: " + h)
    ap.add_argument("--train_noise_seed", type=int, default=None, help="training-noise seed (default: --seed)")
    ap.add_argument("--holdout_noise_seed", type=int, default=542,
                    help="seed of the fixed noisy-holdout perturbation (keep fixed across runs)")
    ap.add_argument("--selection_metric", choices=SELECTION_METRICS, default="auto",
                    help="best.pt criterion; auto = holdout_noisy if noise is enabled else holdout_clean")
    ap.add_argument("--dry_run", action="store_true",
                    help="load data, build windows and normalisation stats, report, then exit (no training, no save)")
    args = ap.parse_args(argv)

    if args.split != "train":
        sys.exit("GRU training uses the DanceTrack train split only (got --split {})".format(args.split))
    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")
    for pth in (best_path, last_path, os.path.join(args.out_dir, "model.pt")):
        if os.path.exists(pth) and not args.overwrite:
            sys.exit("{} exists; pass --overwrite or choose another --out_dir".format(pth))
    if args.early_stop_patience < 0:
        sys.exit("--early_stop_patience must be >= 0")
    try:
        noise = noise_config(args)
    except ValueError as e:
        sys.exit(str(e))
    train_noise_seed = args.seed if args.train_noise_seed is None else args.train_noise_seed
    sel_metric = args.selection_metric
    if sel_metric == "auto":
        sel_metric = "holdout_noisy" if noise.enabled else "holdout_clean"
    if sel_metric == "holdout_noisy" and not noise.enabled:
        sys.exit("--selection_metric holdout_noisy requires --box_noise")
    sel_key = {"holdout_clean": "holdout_loss", "holdout_noisy": "holdout_noisy_loss"}[sel_metric]

    t_start = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True   # reproducible cuDNN GRU kernels on CUDA
    torch.backends.cudnn.benchmark = False
    device = pick_device(args.device)

    # ---- data -----------------------------------------------------------------
    try:
        spec = SamplingSpec.make(args.training_mode, args.history_len, args.min_valid_obs, args.max_trailing_gap)
    except ValueError as e:
        sys.exit(str(e))
    split_info = None
    if args.split_config:
        if args.seqs or args.max_seqs is not None or args.holdout_seqs:
            sys.exit("--split_config cannot be combined with --seqs / --max_seqs / --holdout_seqs")
        try:
            cfg, train_dirs, hold_dirs, split_sha = load_split_config(args.split_config, args.data_root, args.split)
        except ValueError as e:
            sys.exit(str(e))
        split_info = {"path": os.path.abspath(args.split_config), "sha256": split_sha,
                      "split_id": cfg.get("split_id"), "split_version": cfg.get("split_version"),
                      "seed": cfg.get("seed")}
    else:
        all_dirs = list_sequences(args.data_root, args.split, args.seqs)
        hold = set(args.holdout_seqs)
        train_dirs = [d for d in all_dirs if os.path.basename(d) not in hold]
        if args.max_seqs is not None:
            train_dirs = train_dirs[:args.max_seqs]
        hold_dirs = [d for d in all_dirs if os.path.basename(d) in hold]
    if not train_dirs:
        sys.exit("no training sequences found under {}/{}".format(args.data_root, args.split))
    if not hold_dirs and not args.allow_no_holdout:
        sys.exit("no holdout sequences: best-checkpoint selection needs a holdout partition "
                 "(use --split_config or --holdout_seqs; --allow_no_holdout is for smoke tests only)")
    if args.early_stop_patience and not hold_dirs:
        sys.exit("--early_stop_patience requires holdout sequences")
    names = [os.path.basename(d) for d in train_dirs + hold_dirs]
    clash = sorted(set(names) & DANCETRACK_VAL_SEQUENCES)
    if os.path.isdir(args.val_guard_dir):
        clash = sorted(set(clash) | (set(names) & set(os.listdir(args.val_guard_dir))))
    if clash:
        sys.exit("refusing: sequences {} belong to the DanceTrack validation split".format(clash))

    data, seq_info = load_sequences(train_dirs)
    all_samples = enumerate_samples(data, spec)
    samples = subsample(all_samples, args.max_windows, args.seed)
    raw_boxes, raw_masks, raw_tgt = build_raw_windows(data, samples, spec)
    feats, targets, _ = encode_raw(raw_boxes, raw_masks, raw_tgt)        # CLEAN training windows
    print("mode {} (min_valid_obs {}, max_trailing_gap {}) | train: {} sequences, {} ids, {} candidate "
          "windows -> {} used; feats {} targets {}".format(
              spec.mode, spec.min_valid_obs, spec.max_trailing_gap, len(train_dirs),
              sum(s["ids"] for s in seq_info), len(all_samples), len(samples), feats.shape, targets.shape))

    in_mean, in_std, out_mean, out_std = normalization_stats(feats, targets)   # clean training windows only
    print("box noise: {}{}".format(noise.to_dict() if noise.enabled else "none (clean GT histories)",
                                   " | train noise seed {}, holdout noise seed {}".format(
                                       train_noise_seed, args.holdout_noise_seed) if noise.enabled else ""))
    if args.dry_run:
        nv = feats[..., 4].sum(1)
        print("dry run | holdout sequences: {} | windows with < L observations: {} ({:.2%})".format(
            [os.path.basename(d) for d in hold_dirs], int((nv < spec.history_len).sum()),
            float((nv < spec.history_len).mean())))
        print("dry run | in_mean {} in_std {}\n        | out_mean {} out_std {}".format(
            np.round(in_mean, 4), np.round(in_std, 4), np.round(out_mean, 4), np.round(out_std, 4)))
        print("dry run | split {} | nothing trained or saved".format(split_info))
        return None
    model = GRUMotionModel(args.hidden_size, args.num_layers, args.dropout)
    model.set_normalization(in_mean, in_std, out_mean, out_std)
    model.to(device)

    x = torch.from_numpy(feats).to(device)                               # clean (eval_train_loss)
    y = model.normalize_targets(torch.from_numpy(targets).to(device))
    hx = hy = hxn = hyn = None
    holdout_noise_info = None
    if hold_dirs:
        hdata, _ = load_sequences(hold_dirs)
        hsamples = enumerate_samples(hdata, spec)
        if not hsamples:
            sys.exit("holdout sequences yield no windows under mode {}".format(spec.mode))
        hb, hm, ht = build_raw_windows(hdata, hsamples, spec)
        hfeats, htargets, _ = encode_raw(hb, hm, ht)
        hx = torch.from_numpy(hfeats).to(device)
        hy = model.normalize_targets(torch.from_numpy(htargets).to(device))
        if noise.enabled:
            # ONE fixed perturbation of the holdout histories (every window perturbed), drawn once
            hcfg = BoxNoiseConfig.from_dict(dict(noise.to_dict(), apply_prob=1.0))
            hbn = perturb_windows(hb, hm, hcfg, holdout_rng(args.holdout_noise_seed))
            hnf, hnt, _ = encode_raw(hbn, hm, ht)
            hxn = torch.from_numpy(hnf).to(device)
            hyn = model.normalize_targets(torch.from_numpy(hnt).to(device))
            import hashlib
            holdout_noise_info = {"seed": args.holdout_noise_seed, "config": hcfg.to_dict(),
                                  "features_sha256": hashlib.sha256(hnf.tobytes()).hexdigest(),
                                  "realised": noise_realised(hb, hbn, hm)}

    def noisy_train_tensors(epoch):
        nb = perturb_windows(raw_boxes, raw_masks, noise, epoch_rng(train_noise_seed, epoch))
        nf, nt, _ = encode_raw(nb, raw_masks, raw_tgt)
        return (torch.from_numpy(nf).to(device), model.normalize_targets(torch.from_numpy(nt).to(device)),
                noise_realised(raw_boxes, nb, raw_masks))

    # ---- train ----------------------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(args.seed)
    params0 = {k: v.detach().clone() for k, v in model.named_parameters()}
    def holdout_losses():
        return {"holdout_loss": eval_loss(model, hx, hy, 4096) if hx is not None else None,
                "holdout_noisy_loss": eval_loss(model, hxn, hyn, 4096) if hxn is not None else None}

    def fmt_hold(rec):
        out = "" if rec["holdout_loss"] is None else " | holdout clean {:.5f}".format(rec["holdout_loss"])
        if rec["holdout_noisy_loss"] is not None:
            out += " | holdout noisy {:.5f}".format(rec["holdout_noisy_loss"])
        return out

    history = [dict({"epoch": 0, "eval_train_loss": eval_loss(model, x, y, 4096)}, **holdout_losses())]
    print("epoch 0 | eval train (clean) {:.5f}{}".format(history[0]["eval_train_loss"], fmt_hold(history[0])))

    selection = {"metric": "{} Smooth-L1 (beta {}), mean per output element in standardised space, "
                           "pooled over all holdout windows".format(
                               "fixed-noise holdout" if sel_metric == "holdout_noisy" else "clean holdout",
                               args.smooth_l1_beta),
                 "selection_metric": sel_metric, "loss_key": sel_key,
                 "rule": "lowest selection loss over epochs >= 1; strict improvement, ties keep earliest epoch",
                 "holdout_sequences": [os.path.basename(d) for d in hold_dirs]}
    augmentation = None
    if noise.enabled:
        augmentation = {"type": "box_localisation_noise", "preset": args.box_noise, "config": noise.to_dict(),
                        "applied_to": "history observations only (valid slots); masks unchanged; "
                                      "no missing frames introduced",
                        "target": "clean GT box at the target frame, encoded relative to the noisy last "
                                  "observed box",
                        "resampling": "fresh noise every epoch: numpy default_rng([train_noise_seed, 1, epoch])",
                        "train_noise_seed": train_noise_seed,
                        "normalization": "statistics from CLEAN training windows; noisy inputs standardised "
                                         "with them",
                        "noisy_holdout": holdout_noise_info,
                        "parameter_source": "modelling assumption (no train-split detector outputs available); "
                                            "not fitted to any validation data"}
    common = {
        "tag": args.tag,
        "num_train_windows": len(samples),
        "num_candidate_windows": len(all_samples),
        "train_sequences": [os.path.basename(d) for d in train_dirs],
        "holdout_sequences": [os.path.basename(d) for d in hold_dirs],
        "sequence_stats": seq_info,
        "data_root": os.path.abspath(args.data_root),
        "sampling": spec.to_dict(),
        "augmentation": augmentation,
        "split": split_info,
        "num_holdout_windows": int(len(hx)) if hx is not None else 0,
        "feature_version": FEATURE_VERSION,
        "normalization": {"in_mean": in_mean.tolist(), "in_std": in_std.tolist(),
                          "out_mean": out_mean.tolist(), "out_std": out_std.tolist(),
                          "computed_on": "CLEAN training windows only (valid, non-reference slots)"
                                         + ("; noisy inputs use these same statistics" if noise.enabled else "")},
        "config": {k: v for k, v in vars(args).items() if k not in ("overwrite",)},
        "seed": args.seed,
        "device": str(device),
        "git_commit": git_commit(),
        "torch_version": str(torch.__version__),
    }

    def write(path, state, kind, weights_epoch, status):
        """Save a checkpoint whose weights come from `weights_epoch`."""
        m = GRUMotionModel(args.hidden_size, args.num_layers, args.dropout)
        m.load_state_dict(state)
        hr = history[weights_epoch]
        info = dict(common)
        info.update({
            "epochs": weights_epoch,                 # epochs of training behind THESE weights
            "epochs_completed": len(history) - 1,    # epochs run when this file was written
            "checkpoint": {"kind": kind, "weights_epoch": weights_epoch, "holdout_loss": hr["holdout_loss"],
                           "holdout_noisy_loss": hr["holdout_noisy_loss"], "selection_metric": sel_metric,
                           "selection_loss": hr[sel_key], "status": status,
                           "selection": selection if kind == "best" else None},
            "loss_history": list(history),
            "param_delta_l2": float(sum(((state[k].cpu() - params0[k].cpu()) ** 2).sum().item()
                                        for k in params0) ** 0.5),
            "elapsed_seconds": time.time() - t_start,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        save_checkpoint(path, m, spec.training_conditions(augmentation), info)
        return info

    best_loss, best_epoch, best_state, since_best, stopped_early = float("inf"), None, None, 0, False
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        xt, yt, realised = (x, y, None) if not noise.enabled else noisy_train_tensors(epoch)
        model.train()
        perm = torch.randperm(len(x), generator=g).to(device)
        tot, nb = 0.0, 0
        for i in range(0, len(x), args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = F.smooth_l1_loss(model(xt[idx]), yt[idx], beta=args.smooth_l1_beta)
            if not torch.isfinite(loss):
                sys.exit("non-finite loss at epoch {} batch {}".format(epoch, nb))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        del xt, yt
        rec = dict({"epoch": epoch, "train_loss": tot / nb, "eval_train_loss": eval_loss(model, x, y, 4096)},
                   **holdout_losses())
        rec["seconds"] = time.time() - te
        if realised is not None:
            rec["train_noise"] = realised
        history.append(rec)
        improved = hx is not None and rec[sel_key] < best_loss
        if improved:
            best_loss, best_epoch, since_best = rec[sel_key], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            write(best_path, best_state, "best", best_epoch, "provisional (training in progress)")
        elif hx is not None:
            since_best += 1
        print("epoch {} | train{} {:.5f} | eval train (clean) {:.5f}{} | {:.2f}s{}".format(
            epoch, " (noisy)" if realised else "", rec["train_loss"], rec["eval_train_loss"], fmt_hold(rec),
            rec["seconds"], " | best" if improved else ""))
        if args.early_stop_patience and since_best >= args.early_stop_patience:
            stopped_early = True
            print("early stop: no holdout improvement for {} epochs".format(since_best))
            break

    last_epoch = len(history) - 1
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    common["early_stopped"] = stopped_early
    summary = {"last": write(last_path, last_state, "last", last_epoch, "final")}
    if best_state is not None:
        summary["best"] = write(best_path, best_state, "best", best_epoch, "final")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("last.pt = epoch {}{} | {:.1f}s total".format(
        last_epoch, "" if best_state is None else " | best.pt = epoch {} ({} {:.5f})".format(
            best_epoch, sel_metric, best_loss), time.time() - t_start))
    return best_path if best_state is not None else last_path


if __name__ == "__main__":
    main()
