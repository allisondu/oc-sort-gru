# OC-SORT baseline on DanceTrack val (local, Apple M2 Pro / MPS)

Original OC-SORT tracker and pretrained YOLOX detector (`pretrained/ocsort_dance_model.pth.tar`),
unmodified. `git diff HEAD -- trackers utils yolox exps` is empty; tracker params are the
`utils/args.py` defaults (det_thresh 0.6, iou 0.3, max_age 30, min_hits 3, delta_t 3, inertia 0.2,
IoU association, no BYTE). FP32 on MPS (official script uses FP16 on CUDA). No interpolation.

## Results (official DanceTrack TrackEval, all 25 val sequences, measured 2026-10-20)

|                         | HOTA  | IDF1  | AssA  | MOTA  | DetA  | IDSW |
| ----------------------- | ----- | ----- | ----- | ----- | ----- | ---- |
| this run                | 51.54 | 50.85 | 34.46 | 87.29 | 77.35 | 1902 |
| docs/MODEL_ZOO.md       | 52.1  | 51.6  | 35.3  | 87.3  | 77.2  | –    |

Detection-side metrics match; association-side metrics are 0.6–0.8 lower. Per-sequence numbers:
`trackeval/val_all/pedestrian_detailed.csv`; full TrackEval output: `logs/trackeval_val_all.log`.

Speed (`logs/timing_full_val.json`): 25,508 frames, detector 4.46 FPS on MPS (FP32, 800x1440),
tracker-only ~2000 FPS, det+track 4.45 FPS. Full val detection takes ~95 min.

## Why the earlier smoke runs stopped

* The 21:19 / 21:20 runs: no traceback, no crash report, no OOM. Measured MPS speed is 0.22 s/frame
  (4.4 FPS) with flat memory (MPS alloc 395 MB, peak RSS ~2.4 GB), and the old runner logged only
  every 200 frames and saved detections only after a whole sequence. Each run had been detecting
  for under a minute when it was stopped externally (the second was launched 56 s after the first),
  i.e. before the first progress line (~45 s) and long before the only save (~4.5 min). So it was
  not a failure: the runs were interrupted while working, and the script lost all progress.
* Separate crash at 21:16 (`~/Library/Logs/DiagnosticReports/python3.10-2026-10-20-211614.ips`):
  SIGABRT < 1 s after launch in `libomp __kmp_register_library_startup` via openblas — duplicate
  OpenMP runtime (env has `lib/libomp.dylib`, `lib/libiomp5.dylib`, torch's and skimage's copies).
  The runner's import chain does not trigger it (verified).
* TrackEval wrapper bug found along the way: on macOS the `USE_PARALLEL` pool uses `spawn`, so
  workers re-imported NumPy 2.2 without the `np.float` aliases and crashed. Fixed by forcing `fork`.

## Fixes to the scripts

* `tools/run_ocsort_dance_local.py`: `--max_frames`; detections checkpointed to
  `<seq>.partial.pt` every `--save_every` (25) frames (atomic write) and resumed on rerun, promoted
  to `<seq>.pt` once complete; per-frame time, peak RSS and MPS memory in progress logs
  (`--log_every`).
* `scripts/run_trackeval.py`: `fork` start method (above).
* `scripts/verify_tracker.py`: standalone check that runs the unmodified `OCSort` on cached
  detections, prints per-frame IDs / ID lifetimes and draws boxes + IDs.

## Diagnostics

* `logs/diag/smoke_mps_10f.log`: 10 frames MPS, 0.22 s/frame (first frame 0.79 s), peak RSS 1.5 GB.
* `logs/diag/smoke_cpu_5f.log`: 5 frames CPU, 1.05 s/frame, peak RSS 2.8 GB. MPS vs CPU boxes
  differ by < 0.0006 px with identical detection counts.
* `logs/diag/resume_test.log`: resume from a 10-frame checkpoint to 15 frames.
* `logs/diag/verify_tracker_10f.log`, `visualizations/verify_10f/`: IDs 1–3 held over 10 frames.
* `visualizations/dancetrack0004/`: sample frames + `verify.json` (mp4 is gitignored);
  0004 uses 8 IDs for 4 GT dancers (HOTA 43.4, IDSW 38).

## Reproduce

```
python tools/run_ocsort_dance_local.py -f exps/example/mot/yolox_dancetrack_val.py \
    -c pretrained/ocsort_dance_model.pth.tar --fuse --expn baseline --run_device mps
python baseline_results/scripts/run_trackeval.py --results YOLOX_outputs/baseline/baseline_val \
    --out baseline_results/trackeval/val_all
```
Detections are cached in `dance_detections_cache/val/` (25 × `<seq>.pt`), so rerunning the first
command only re-runs tracking (~13 s total); `--track_only` refuses to start the detector.
