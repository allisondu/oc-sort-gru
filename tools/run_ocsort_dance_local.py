"""
Device-agnostic (CUDA / Apple MPS / CPU) re-implementation of the inference loop in
tools/run_ocsort_dance.py + MOTEvaluatorDance.evaluate_ocsort, for running the
original OC-SORT baseline on machines without CUDA.

The tracker (trackers/ocsort_tracker) and all of its hyper-parameters are untouched:
tracker arguments come from the repo's own utils/args.py make_parser(), and the
detector preprocessing / postprocessing are the repo's own preproc() / postprocess().

Differences from the official script (execution only, not algorithmic):
  * Iterates datasets/dancetrack/<split>/<seq>/img1 directly (no COCO json needed),
    optionally restricted to a subset of sequences via --seqs.
  * Runs in two timed stages: (1) YOLOX detection, cached per sequence under
    --det_cache; (2) OC-SORT tracking from the cached detections. This gives separate
    detector FPS and tracking-only FPS, and lets tracking be re-run without the detector.
  * FP16 is only used on CUDA.

Example:
  python tools/run_ocsort_dance_local.py -f exps/example/mot/yolox_dancetrack_val.py \
      -c pretrained/ocsort_dance_model.pth.tar --fuse --expn baseline \
      --seqs dancetrack0004 dancetrack0005
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from trackers.ocsort_tracker.ocsort import OCSort
from utils.args import make_parser
from utils.utils import write_results_no_score

RGB_MEANS = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def rss_mb():
    import resource
    # ru_maxrss is bytes on macOS, KB on Linux; this is peak RSS of the process
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def mps_mem_str(device):
    if device.type != "mps":
        return ""
    return " | mps alloc {:.0f} MB / driver {:.0f} MB".format(
        torch.mps.current_allocated_memory() / 2**20, torch.mps.driver_allocated_memory() / 2**20)


def load_model(exp, args, device):
    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if args.fuse:
        model = fuse_model(model)
    model = model.to(device)
    half = args.fp16 and device.type == "cuda"
    if args.fp16 and not half:
        logger.warning("--fp16 requested but device is {}; running in FP32".format(device))
    if half:
        model = model.half()
    # decode_outputs uses legacy tensor-type strings (e.g. 'torch.mps.FloatTensor') that MPS
    # rejects, so run the head undecoded and decode on CPU, as run_ocsort_dance.py does for TRT
    model.head.decode_in_inference = False
    return model, half


def save_partial(path, dets, img_hw, det_time, frame_times, device, exp):
    tmp = path + ".tmp"
    torch.save({"dets": dets, "img_hw": img_hw, "det_time": det_time, "frame_times": frame_times,
                "device": str(device), "test_size": exp.test_size}, tmp)
    os.replace(tmp, path)


def detect_sequence(model, half, device, exp, img_files, partial_file, save_every, log_every):
    """Returns {frame_id: Tensor[N,7] or None} in network-input coordinates, plus timing.

    Progress is checkpointed to partial_file every save_every frames and resumed from it,
    so an interrupted run only loses the frames since the last checkpoint.
    """
    dets, det_time, img_hw, frame_times = {}, 0.0, None, []
    if os.path.exists(partial_file):
        c = torch.load(partial_file, weights_only=False)
        dets, img_hw, det_time = c["dets"], c["img_hw"], c["det_time"]
        frame_times = c.get("frame_times", [])
        logger.info("  resuming from {} ({} frames already detected)".format(partial_file, len(dets)))
    for i, f in enumerate(img_files):
        frame_id = i + 1
        if frame_id in dets:
            continue
        img = cv2.imread(f)
        img_hw = img.shape[:2]
        t0 = time.time()
        x, _ = preproc(img, exp.test_size, RGB_MEANS, STD)
        x = torch.from_numpy(x).unsqueeze(0).to(device)
        if half:
            x = x.half()
        with torch.no_grad():
            out = model(x).float().cpu()
            out = model.head.decode_outputs(out, dtype=out.type())
            out = postprocess(out, exp.num_classes, exp.test_conf, exp.nmsthre)[0]
        sync(device)
        dt = time.time() - t0
        det_time += dt
        frame_times.append(dt)
        dets[frame_id] = out
        if len(dets) % log_every == 0 or len(dets) <= 3:
            logger.info("  detected {}/{} frames | last {:.3f}s | avg {:.2f} FPS | rss {:.0f} MB{}".format(
                len(dets), len(img_files), dt, len(dets) / det_time, rss_mb(), mps_mem_str(device)))
        if len(dets) % save_every == 0:
            save_partial(partial_file, dets, img_hw, det_time, frame_times, device, exp)
    save_partial(partial_file, dets, img_hw, det_time, frame_times, device, exp)
    return dets, img_hw, det_time


def build_tracker(args, gru_predictor=None):
    """Same OC-SORT hyper-parameters for both trackers; only the class differs."""
    kwargs = dict(iou_threshold=args.iou_thresh, asso_func=args.asso, delta_t=args.deltat,
                  inertia=args.inertia, use_byte=args.use_byte)
    if args.tracker == "ocsort":
        return OCSort(det_thresh=args.track_thresh, **kwargs)
    from trackers.ocsort_gru import OCSortGRU
    return OCSortGRU(det_thresh=args.track_thresh, predictor=gru_predictor,
                     max_gap=args.gru_max_gap, min_obs=args.gru_min_obs, **kwargs)


def track_sequence(args, dets, img_hw, img_size, gru_predictor=None):
    """Mirrors the tracking part of MOTEvaluatorDance.evaluate_ocsort."""
    tracker = build_tracker(args, gru_predictor)
    results = []
    t0 = time.time()
    for frame_id in sorted(dets):
        out = dets[frame_id]
        # clone: OCSort.update scales boxes in place via a numpy view of CPU tensors
        online_targets = tracker.update(None if out is None else out.clone(), img_hw, img_size)
        online_tlwhs, online_ids = [], []
        for t in online_targets:
            tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
            tid = t[4]
            if tlwh[2] * tlwh[3] > args.min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        results.append((frame_id, online_tlwhs, online_ids))
    elapsed = time.time() - t0
    if getattr(tracker, "stats", None):
        logger.info("  GRU usage: {}".format(tracker.stats))
    return results, elapsed


def main():
    parser = make_parser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--data_root", default="datasets/dancetrack")
    parser.add_argument("--seqs", nargs="*", default=None, help="subset of sequences (default: all in split)")
    parser.add_argument("--run_device", default="auto", help="auto/cuda/mps/cpu (--device is taken by utils/args.py)")
    parser.add_argument("--det_cache", default="dance_detections_cache")
    parser.add_argument("--track_only", action="store_true", help="require cached detections; skip the detector")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="only process the first N frames of each sequence (smoke tests)")
    parser.add_argument("--save_every", type=int, default=25, help="checkpoint partial detections every N frames")
    parser.add_argument("--log_every", type=int, default=25)
    # --- tracker selection (oc-sort-gru). Default reproduces the baseline exactly. ---
    parser.add_argument("--tracker", choices=["ocsort", "ocsort_gru"], default="ocsort")
    parser.add_argument("--gru_ckpt", default=None,
                        help="trained GRU checkpoint (gru_motion.save_checkpoint); required for ocsort_gru. "
                             "Architecture and history length are read from it.")
    parser.add_argument("--gru_device", default="cpu", help="device for GRU inference (cpu/cuda/mps)")
    parser.add_argument("--gru_max_gap", type=int, default=1,
                        help="use GRU only if the track was observed within this many frames; else Kalman")
    parser.add_argument("--gru_min_obs", type=int, default=3,
                        help="minimum real observations in the history window; else Kalman")
    args = parser.parse_args()
    if args.tracker == "ocsort" and args.gru_ckpt:
        parser.error("--gru_ckpt given but --tracker is ocsort; pass --tracker ocsort_gru")

    exp = get_exp(args.exp_file, args.name)
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    if not args.expn:
        args.expn = exp.exp_name

    # Load + validate the GRU before doing any work, so a missing/untrained
    # checkpoint fails immediately instead of after detection.
    gru_predictor = None
    if args.tracker == "ocsort_gru":
        from gru_motion import GRUBoxPredictor, GRUCheckpointError
        try:
            gru_predictor = GRUBoxPredictor.from_checkpoint(args.gru_ckpt, device=args.gru_device)
        except GRUCheckpointError as e:
            logger.error(str(e))
            sys.exit(2)
        L = gru_predictor.history_len
        if not (1 <= args.gru_max_gap <= L - 1 and 2 <= args.gru_min_obs <= L):
            parser.error("need 1 <= --gru_max_gap <= {} and 2 <= --gru_min_obs <= {} (history_len={})".format(
                L - 1, L, L))

    out_dir = os.path.join(exp.output_dir, args.expn)
    results_folder = os.path.join(out_dir, "{}_{}".format(args.expn, args.split))
    os.makedirs(results_folder, exist_ok=True)
    cache_dir = os.path.join(args.det_cache, args.split)
    os.makedirs(cache_dir, exist_ok=True)
    logger.add(os.path.join(out_dir, "run_log.txt"), mode="a")
    logger.info("Args: {}".format(args))
    tracker_cfg = {"tracker": args.tracker, "track_thresh": args.track_thresh, "iou_thresh": args.iou_thresh,
                   "asso": args.asso, "deltat": args.deltat, "inertia": args.inertia, "use_byte": args.use_byte,
                   "min_box_area": args.min_box_area}
    if gru_predictor is not None:
        tracker_cfg.update({"gru_checkpoint": gru_predictor.meta, "gru_device": args.gru_device,
                            "gru_max_gap": args.gru_max_gap, "gru_min_obs": args.gru_min_obs})
        logger.info("GRU checkpoint: {} (sha256 {}, history_len {}, {} params)".format(
            gru_predictor.meta["path"], gru_predictor.meta["sha256"][:12],
            gru_predictor.history_len, gru_predictor.meta["num_params"]))
    with open(os.path.join(out_dir, "tracker_config.json"), "w") as f:
        json.dump(tracker_cfg, f, indent=2, default=str)

    seq_dirs = sorted(glob.glob(os.path.join(args.data_root, args.split, "dancetrack*")))
    if args.seqs:
        seq_dirs = [d for d in seq_dirs if os.path.basename(d) in set(args.seqs)]
    assert seq_dirs, "no sequences found"

    device = pick_device(args.run_device)
    model = half = None
    timing = {}
    for sd in seq_dirs:
        seq = os.path.basename(sd)
        img_files = sorted(glob.glob(os.path.join(sd, "img1", "*.jpg")))
        n_total = len(img_files)
        if args.max_frames:
            img_files = img_files[:args.max_frames]
        cache_file = os.path.join(cache_dir, "{}.pt".format(seq))
        partial_file = os.path.join(cache_dir, "{}.partial.pt".format(seq))
        if os.path.exists(cache_file):
            c = torch.load(cache_file, weights_only=False)
            dets, img_hw, det_time = c["dets"], tuple(c["img_hw"]), c["det_time"]
            logger.info("{}: loaded cached detections ({} frames)".format(seq, len(dets)))
        elif args.track_only:
            assert os.path.exists(partial_file), "no cached detections for {}".format(seq)
            c = torch.load(partial_file, weights_only=False)
            dets, img_hw, det_time = c["dets"], tuple(c["img_hw"]), c["det_time"]
            logger.info("{}: loaded PARTIAL cached detections ({} frames)".format(seq, len(dets)))
        else:
            if model is None:
                logger.info("Loading detector on {}".format(device))
                model, half = load_model(exp, args, device)
            logger.info("{}: detecting {} frames on {}".format(seq, len(img_files), device))
            dets, img_hw, det_time = detect_sequence(model, half, device, exp, img_files, partial_file,
                                                     args.save_every, args.log_every)
            if len(dets) == n_total:
                os.replace(partial_file, cache_file)
        if args.max_frames and len(dets) > args.max_frames:
            # cached det_time covers every cached frame; rescale to the frames used
            det_time *= args.max_frames / len(dets)
            dets = {k: v for k, v in dets.items() if k <= args.max_frames}
        if len(dets) < n_total:
            logger.warning("{}: tracking only {}/{} frames".format(seq, len(dets), n_total))

        results, track_time = track_sequence(args, dets, img_hw, exp.test_size, gru_predictor)
        write_results_no_score(os.path.join(results_folder, "{}.txt".format(seq)), results)
        n = len(dets)
        timing[seq] = {"frames": n, "det_time_s": det_time, "track_time_s": track_time,
                       "det_fps": n / det_time, "track_fps": n / track_time}
        logger.info("{}: {} frames | detector {:.2f} FPS | tracker-only {:.1f} FPS".format(
            seq, n, n / det_time, n / track_time))

    tot_n = sum(v["frames"] for v in timing.values())
    tot_det = sum(v["det_time_s"] for v in timing.values())
    tot_trk = sum(v["track_time_s"] for v in timing.values())
    timing["_total"] = {"frames": tot_n, "det_time_s": tot_det, "track_time_s": tot_trk,
                        "det_fps": tot_n / tot_det, "track_fps": tot_n / tot_trk,
                        "det_plus_track_fps": tot_n / (tot_det + tot_trk), "device": str(device)}
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    logger.info("TOTAL {} frames | detector {:.2f} FPS | tracker-only {:.1f} FPS | det+track {:.2f} FPS".format(
        tot_n, tot_n / tot_det, tot_n / tot_trk, tot_n / (tot_det + tot_trk)))


if __name__ == "__main__":
    main()
