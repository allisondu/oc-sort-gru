"""
Runs DanceTrack's official TrackEval (external/DanceTrack/TrackEval/scripts/run_mot_challenge.py)
unmodified, after restoring the numpy aliases (np.float/np.int/np.bool) that TrackEval uses but
NumPy >= 1.24 removed. Optionally evaluates only a subset of sequences by writing a seqmap.

Usage (from the OC_SORT root):
  python baseline_results/scripts/run_trackeval.py \
      --results YOLOX_outputs/baseline/baseline_val [--seqs dancetrack0004 ...] \
      [--out baseline_results/logs/trackeval_val]
"""
import argparse
import multiprocessing
import os
import runpy
import shutil
import sys

import numpy as np

for _name, _typ in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _typ)

# macOS defaults to 'spawn', whose pool workers re-import numpy without the aliases above;
# 'fork' makes TrackEval's USE_PARALLEL workers inherit the patched module
multiprocessing.set_start_method("fork", force=True)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRACKEVAL = os.path.join(ROOT, "external", "DanceTrack", "TrackEval")

p = argparse.ArgumentParser()
p.add_argument("--results", required=True, help="folder with <seq>.txt tracker outputs")
p.add_argument("--gt", default=os.path.join(ROOT, "datasets", "dancetrack", "val"))
p.add_argument("--seqs", nargs="*", default=None, help="default: every <seq>.txt in --results")
p.add_argument("--out", default=None, help="folder to copy TrackEval summary/detailed csv into")
p.add_argument("--cores", type=int, default=4)
a = p.parse_args()

results = os.path.abspath(a.results)
seqs = a.seqs or sorted(f[:-4] for f in os.listdir(results) if f.endswith(".txt"))

# TrackEval layout: TRACKERS_FOLDER/<tracker>/data/<seq>.txt
work = os.path.join(os.path.dirname(results), "trackeval_work")
tracker_data = os.path.join(work, "trackers", "ocsort", "data")
shutil.rmtree(work, ignore_errors=True)
os.makedirs(tracker_data)
for s in seqs:
    shutil.copy(os.path.join(results, s + ".txt"), tracker_data)
seqmap = os.path.join(work, "seqmap.txt")
with open(seqmap, "w") as f:
    f.write("name\n" + "".join(s + "\n" for s in seqs))

sys.path.insert(0, TRACKEVAL)
sys.argv = [
    "run_mot_challenge.py",
    "--SPLIT_TO_EVAL", "val",
    "--METRICS", "HOTA", "CLEAR", "Identity",
    "--GT_FOLDER", os.path.abspath(a.gt),
    "--SEQMAP_FILE", seqmap,
    "--SKIP_SPLIT_FOL", "True",
    "--TRACKERS_FOLDER", os.path.join(work, "trackers"),
    "--TRACKERS_TO_EVAL", "ocsort",
    "--TRACKER_SUB_FOLDER", "data",
    "--USE_PARALLEL", "True",
    "--NUM_PARALLEL_CORES", str(a.cores),
    "--PLOT_CURVES", "False",
]
runpy.run_path(os.path.join(TRACKEVAL, "scripts", "run_mot_challenge.py"), run_name="__main__")

if a.out:
    os.makedirs(a.out, exist_ok=True)
    src = os.path.join(work, "trackers", "ocsort")
    for f in os.listdir(src):
        if f.endswith(".csv") or f.endswith(".txt"):
            shutil.copy(os.path.join(src, f), a.out)
