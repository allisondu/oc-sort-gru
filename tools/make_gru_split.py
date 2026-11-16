"""
Create the fixed, sequence-level train/holdout split of the DanceTrack TRAIN
set for GRU motion-model experiments. Run ONCE; the output is committed and
reused by every training run (gru_motion/train.py --split_config). It refuses
to overwrite an existing split file.

Whole video sequences are held out (never individual windows: neighbouring
windows overlap almost entirely, so a window-level split would leak).

    # v1 (random)
    python tools/make_gru_split.py --method random --seed 0 --num_holdout 5 \
        --out configs/gru/dancetrack_split.json
    # v2 (stratified, recommended)
    python tools/make_gru_split.py --method stratified --seed 0 \
        --stats configs/gru/dancetrack_train_sequence_stats.json --out configs/gru/dancetrack_split_v2.json

Stratified method (dataset characteristics only, never model results):
  1. Sort sequences by objs_mean (crowding) and cut into num_holdout equal
     bins (quintiles for 5); every candidate holdout takes exactly ONE
     sequence per bin -> sparse-to-crowded coverage by construction.
  2. Enumerate all such candidates and keep those that satisfy, for each
     secondary metric m in COVERAGE_METRICS (percentile rank among all train
     sequences): min rank <= 0.35, max rank >= 0.65 and mean rank in
     [0.35, 0.65]; at least one 1280x720 and one 1920x1080 sequence; and a
     holdout share of annotated boxes in [0.09, 0.16].
  3. Pick one feasible candidate uniformly at random with the given seed
     (feasible list sorted, so the choice is reproducible).
"""
import argparse
import hashlib
import json
import os
import sys

import itertools

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gru_motion.data import DANCETRACK_VAL_SEQUENCES, list_sequences  # noqa: E402


COVERAGE_METRICS = ("cv_err_median", "turn_rate", "overlap_rate", "gap_rate")


def pct_rank(values):
    v = np.asarray(values, dtype=np.float64)
    return np.array([((v < x).sum() + 0.5 * ((v == x).sum() - 1)) / (len(v) - 1) for x in v])


def stratified_holdout(stats, k, seed):
    rows = sorted(stats, key=lambda r: r["sequence"])
    names = [r["sequence"] for r in rows]
    ranks = {m: dict(zip(names, pct_rank([r[m] for r in rows]))) for m in COVERAGE_METRICS}
    res = {r["sequence"]: r["resolution"] for r in rows}
    boxes = {r["sequence"]: r["boxes"] for r in rows}
    total = float(sum(boxes.values()))
    by_crowd = sorted(rows, key=lambda r: (r["objs_mean"], r["sequence"]))
    bins = [[r["sequence"] for r in b] for b in np.array_split(np.array(by_crowd, dtype=object), k)]
    feasible = []
    for combo in itertools.product(*bins):
        ok = True
        for m in COVERAGE_METRICS:
            rk = [ranks[m][n] for n in combo]
            if not (min(rk) <= 0.35 and max(rk) >= 0.65 and 0.35 <= np.mean(rk) <= 0.65):
                ok = False
                break
        if not ok:
            continue
        rs = {res[n] for n in combo}
        if "1280x720" not in rs or "1920x1080" not in rs:
            continue
        share = sum(boxes[n] for n in combo) / total
        if not (0.09 <= share <= 0.16):
            continue
        feasible.append(tuple(sorted(combo)))
    feasible = sorted(set(feasible))
    if not feasible:
        raise SystemExit("no feasible stratified holdout; relax the constraints")
    pick = feasible[int(np.random.default_rng(seed).integers(len(feasible)))]
    info = {"crowding_bins": bins, "num_candidates": int(np.prod([len(b) for b in bins])),
            "num_feasible": len(feasible), "coverage_metrics": list(COVERAGE_METRICS),
            "holdout_box_share": sum(boxes[n] for n in pick) / total,
            "holdout_percentile_ranks": {m: {n: round(float(ranks[m][n]), 3) for n in pick}
                                         for m in COVERAGE_METRICS}}
    return list(pick), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="datasets_gru/dancetrack_gt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_holdout", type=int, default=5)
    ap.add_argument("--out", default="configs/gru/dancetrack_split.json")
    ap.add_argument("--expected_num_sequences", type=int, default=40)
    ap.add_argument("--method", choices=["random", "stratified"], required=True)
    ap.add_argument("--stats", default="configs/gru/dancetrack_train_sequence_stats.json",
                    help="output of tools/analyze_dancetrack_sequences.py (stratified only)")
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit("{} already exists; the split is fixed. Delete it deliberately to regenerate.".format(args.out))

    names = [os.path.basename(d) for d in list_sequences(args.data_root, "train")]
    if len(names) != args.expected_num_sequences:
        sys.exit("found {} train sequences, expected {}; fetch them first or pass --expected_num_sequences".format(
            len(names), args.expected_num_sequences))
    bad = sorted(set(names) & DANCETRACK_VAL_SEQUENCES)
    if bad:
        sys.exit("validation sequences found in train data: {}".format(bad))

    manifest = os.path.join(args.data_root, "train", "FETCH_MANIFEST.json")
    man_sha = hashlib.sha256(open(manifest, "rb").read()).hexdigest() if os.path.exists(manifest) else None

    extra = {}
    if args.method == "random":
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(names))                 # names are sorted -> deterministic
        hold = sorted(names[i] for i in perm[:args.num_holdout])
        version, method = 1, ("numpy.random.default_rng(seed).permutation over sorted sequence names; "
                              "first num_holdout -> holdout")
    else:
        raw = open(args.stats, "rb").read()
        stats = json.loads(raw)["sequences"]
        if sorted(r["sequence"] for r in stats) != sorted(names):
            sys.exit("stats file does not cover exactly the train sequences")
        hold, extra = stratified_holdout(stats, args.num_holdout, args.seed)
        extra["stats_file"] = {"path": os.path.relpath(args.stats), "sha256": hashlib.sha256(raw).hexdigest()}
        version, method = 2, ("stratified: one sequence per crowding (objs_mean) bin; coverage constraints on "
                              "percentile ranks of {} (min<=0.35, max>=0.65, mean in [0.35,0.65]); >=1 1280x720 "
                              "and >=1 1920x1080; holdout box share in [0.09,0.16]; seeded uniform choice among "
                              "feasible candidates (see tools/make_gru_split.py)").format(", ".join(COVERAGE_METRICS))
    train = sorted(n for n in names if n not in set(hold))
    cfg = {
        "dataset": "DanceTrack train split (huggingface.co/datasets/noahcao/dancetrack, train1.zip + train2.zip)",
        "split_id": "dancetrack-train-gru-holdout-v{}".format(version),
        "split_version": version,
        "unit": "whole video sequence",
        "method": method,
        "seed": args.seed,
        "num_train_sequences": len(train),
        "num_holdout_sequences": len(hold),
        "train_sequences": train,
        "holdout_sequences": hold,
        "excluded_official_validation_sequences": sorted(DANCETRACK_VAL_SEQUENCES),
        "dataset_manifest": {"path": os.path.relpath(manifest), "sha256": man_sha},
    }
    if extra:
        cfg["selection_details"] = extra
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print("wrote {}: {} train / {} holdout -> holdout = {}".format(args.out, len(train), len(hold), hold))


if __name__ == "__main__":
    main()
