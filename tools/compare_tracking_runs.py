"""
Compare two TrackEval runs (baseline OC-SORT vs OCSortGRU) on DanceTrack val and
summarise GRU usage from --gru_log_decisions CSVs. Analysis only: no thresholds
or models are selected from these numbers.

    python tools/compare_tracking_runs.py --out gru_runs/evaluations/tracking_standard_v1_val \
        --base_name eval_baseline_ocsort --gru_name eval_gru_standard_v1 \
        [--assoc_diag gru_runs/evaluations/tracking_standard_v1_val/assoc_diag/summary.json]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

METRICS = [("HOTA", "HOTA___AUC", 100), ("AssA", "AssA___AUC", 100), ("DetA", "DetA___AUC", 100),
           ("IDF1", "IDF1", 100), ("MOTA", "MOTA", 100), ("IDSW", "IDSW", 1), ("Frag", "Frag", 1),
           ("IDs", "IDs", 1)]


def load_detailed(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["seq"]] = {name: float(r[col]) * scale for name, col, scale in METRICS}
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_name", default="eval_baseline_ocsort")
    ap.add_argument("--gru_name", default="eval_gru_standard_v1")
    ap.add_argument("--timing_name", default=None, help="GRU run without decision logging (clean timing)")
    ap.add_argument("--assoc_diag", default=None)
    ap.add_argument("--val_root", default="datasets/dancetrack")
    args = ap.parse_args()
    out = args.out
    base = load_detailed(os.path.join(out, "trackeval_" + args.base_name, "pedestrian_detailed.csv"))
    gru = load_detailed(os.path.join(out, "trackeval_" + args.gru_name, "pedestrian_detailed.csv"))
    seqs = sorted(s for s in base if s != "COMBINED")

    def timing(name):
        t = json.load(open(os.path.join("YOLOX_outputs", name, "timing.json")))["_total"]
        return {"tracker_only_fps": t["track_fps"], "det_plus_track_fps": t["det_plus_track_fps"],
                "track_time_s": t["track_time_s"], "frames": t["frames"]}
    timings = {"baseline": timing(args.base_name), "gru_with_decision_logging": timing(args.gru_name)}
    if args.timing_name:
        timings["gru_no_logging"] = timing(args.timing_name)

    # ---- GRU usage from decision logs -----------------------------------------
    usage, usage_tot = {}, {}
    for s in seqs:
        rows = list(csv.DictReader(open(os.path.join("YOLOX_outputs", args.gru_name, "gru_decisions", s + ".csv"))))
        c = {}
        for r in rows:
            c[r["decision"]] = c.get(r["decision"], 0) + 1
        recent = [r for r in rows if r["frames_since_last_obs"] == "1"]
        u = {"track_predictions": len(rows), **c,
             "gru_fraction": c.get("gru_used", 0) / max(len(rows), 1),
             "recently_observed_predictions": len(recent),
             "gru_fraction_of_recently_observed": sum(r["decision"] == "gru_used" for r in recent) / max(len(recent), 1)}
        usage[s] = u
        for k, v in c.items():
            usage_tot[k] = usage_tot.get(k, 0) + v
        usage_tot["track_predictions"] = usage_tot.get("track_predictions", 0) + len(rows)
        usage_tot["recently_observed_predictions"] = usage_tot.get("recently_observed_predictions", 0) + len(recent)
        usage_tot["recent_gru"] = usage_tot.get("recent_gru", 0) + sum(r["decision"] == "gru_used" for r in recent)
    usage_tot["gru_fraction"] = usage_tot.get("gru_used", 0) / usage_tot["track_predictions"]
    usage_tot["gru_fraction_of_recently_observed"] = usage_tot["recent_gru"] / usage_tot["recently_observed_predictions"]

    # ---- descriptive val sequence statistics (same definitions as the train-split analysis)
    from analyze_dancetrack_sequences import sequence_stats
    vstats = {s: sequence_stats(os.path.join(args.val_root, "val", s)) for s in seqs}

    # ---- per sequence deltas + correlations ----------------------------------
    rows = []
    for s in seqs:
        r = {"sequence": s}
        for name, _, _ in METRICS:
            r["base_" + name], r["gru_" + name] = base[s][name], gru[s][name]
            r["delta_" + name] = gru[s][name] - base[s][name]
        r.update({"gru_fraction": usage[s]["gru_fraction"], "objs_mean": vstats[s]["objs_mean"],
                  "cv_err_median": vstats[s]["cv_err_median"], "turn_rate": vstats[s]["turn_rate"],
                  "overlap_rate": vstats[s]["overlap_rate"], "speed_median": vstats[s]["speed_median"]})
        rows.append(r)
    with open(os.path.join(out, "per_sequence.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    corr = {}
    for d in ("delta_HOTA", "delta_AssA", "delta_IDSW", "delta_IDF1"):
        corr[d] = {x: spearman([r[d] for r in rows], [r[x] for r in rows])
                   for x in ("gru_fraction", "objs_mean", "cv_err_median", "turn_rate", "overlap_rate", "speed_median")}
    deltas = {name: {"improved": int(sum((r["delta_" + name] < 0) if name in ("IDSW", "Frag", "IDs")
                                         else (r["delta_" + name] > 0) for r in rows)),
                     "worse": int(sum((r["delta_" + name] > 0) if name in ("IDSW", "Frag", "IDs")
                                      else (r["delta_" + name] < 0) for r in rows)),
                     "mean": float(np.mean([r["delta_" + name] for r in rows])),
                     "median": float(np.median([r["delta_" + name] for r in rows]))}
              for name, _, _ in METRICS}
    # sign-flip permutation test on per-sequence HOTA/AssA/IDF1/IDSW deltas (25 sequences as units)
    rng = np.random.default_rng(0)
    perm = {}
    for name in ("HOTA", "AssA", "IDF1", "IDSW"):
        v = np.array([r["delta_" + name] for r in rows])
        sims = (rng.choice([-1, 1], size=(20000, len(v))) * v).mean(1)
        perm[name] = {"mean_delta": float(v.mean()), "two_sided_p": float((np.abs(sims) >= abs(v.mean()) - 1e-12).mean())}

    # non-learned direct-substitution controls (tools/diag_direct_baselines.py), if present
    controls = {}
    for name in ("diag_direct_const_vel", "diag_direct_last_obs"):
        p = os.path.join(out, "diag_controls", "trackeval_" + name, "pedestrian_detailed.csv")
        if os.path.exists(p):
            c = load_detailed(p)
            dh = [c[x]["HOTA"] - base[x]["HOTA"] for x in seqs]
            controls[name] = {"overall": c["COMBINED"],
                              "delta": {k: c["COMBINED"][k] - base["COMBINED"][k] for k in base["COMBINED"]},
                              "per_seq_dHOTA_mean": float(np.mean(dh)), "per_seq_dHOTA_sd": float(np.std(dh)),
                              "per_seq_abs_dHOTA_mean": float(np.mean(np.abs(dh))),
                              "seqs_improved_HOTA": int(sum(v > 0 for v in dh)),
                              "corr_with_gru_dHOTA": float(np.corrcoef(dh, [gru[x]["HOTA"] - base[x]["HOTA"] for x in seqs])[0, 1])}
    gdh = [gru[x]["HOTA"] - base[x]["HOTA"] for x in seqs]
    controls["gru_reference"] = {"per_seq_dHOTA_mean": float(np.mean(gdh)), "per_seq_dHOTA_sd": float(np.std(gdh)),
                                 "per_seq_abs_dHOTA_mean": float(np.mean(np.abs(gdh)))}
    diag = json.load(open(args.assoc_diag)) if args.assoc_diag and os.path.exists(args.assoc_diag) else None
    summary = {"overall": {"baseline": base["COMBINED"], "gru": gru["COMBINED"],
                           "delta": {k: gru["COMBINED"][k] - base["COMBINED"][k] for k in base["COMBINED"]}},
               "per_sequence_delta_counts": deltas, "sign_flip_permutation": perm,
               "spearman_delta_vs_sequence_properties": corr, "timing": timings,
               "gru_usage_total": usage_tot, "gru_usage_per_sequence": usage,
               "val_sequence_stats": vstats, "association_diagnostic": diag, "direct_substitution_controls": controls}
    json.dump(summary, open(os.path.join(out, "comparison.json"), "w"), indent=2)
    write_report(os.path.join(out, "report.md"), summary, rows, out)
    print(json.dumps(summary["overall"]["delta"], indent=1))


def write_report(path, S, rows, out):
    cfg = json.load(open(os.path.join(out, "evaluation_config.json"))) if os.path.exists(
        os.path.join(out, "evaluation_config.json")) else {}
    b, g, d = S["overall"]["baseline"], S["overall"]["gru"], S["overall"]["delta"]
    L = ["# OC-SORT vs OC-SORT + standard GRU (direct prediction) — DanceTrack val, 25 sequences", ""]
    if cfg:
        L += ["Configuration: `{}`".format(json.dumps(cfg)), ""]
    L += ["## Overall (TrackEval COMBINED)", "", "| metric | baseline | GRU | delta |", "|---|---|---|---|"]
    for k in ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW", "Frag", "IDs"):
        L.append("| {} | {:.3f} | {:.3f} | {:+.3f} |".format(k, b[k], g[k], d[k]))
    L += ["", "Timing: `{}`".format(json.dumps(S["timing"])), "",
          "Per-sequence direction (improved / worse of 25): " + ", ".join(
              "{} {}/{}".format(k, v["improved"], v["worse"]) for k, v in S["per_sequence_delta_counts"].items()), "",
          "Sign-flip permutation test over 25 sequences: `{}`".format(json.dumps(S["sign_flip_permutation"])), "",
          "## GRU usage", "", "`{}`".format(json.dumps(S["gru_usage_total"])), "",
          "## Per sequence", "",
          "| seq | HOTA base | HOTA GRU | dHOTA | dAssA | dDetA | dIDF1 | dMOTA | IDSW base | IDSW GRU | GRU frac | objs/frame | cv_err med |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| {} | {:.2f} | {:.2f} | {:+.2f} | {:+.2f} | {:+.2f} | {:+.2f} | {:+.2f} | {:.0f} | {:.0f} | {:.3f} | {:.1f} | {:.4f} |".format(
            r["sequence"][-4:], r["base_HOTA"], r["gru_HOTA"], r["delta_HOTA"], r["delta_AssA"], r["delta_DetA"],
            r["delta_IDF1"], r["delta_MOTA"], r["base_IDSW"], r["gru_IDSW"], r["gru_fraction"], r["objs_mean"],
            r["cv_err_median"]))
    L += ["", "## Spearman correlation of per-sequence deltas with sequence properties (n=25; descriptive)", "",
          "`{}`".format(json.dumps({k: {x: round(v, 3) for x, v in d2.items()}
                                    for k, d2 in S["spearman_delta_vs_sequence_properties"].items()}))]
    if S.get("direct_substitution_controls"):
        L += ["", "## Diagnostic controls: non-learned predictors in the GRU's place (same eligibility policy)", "",
              "| run | HOTA | AssA | DetA | IDF1 | MOTA | IDSW | per-seq |dHOTA| mean | seqs HOTA up | corr with GRU per-seq dHOTA |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        L.append("| baseline (Kalman) | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.0f} | - | - | - |".format(
            *(b[k] for k in ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW"))))
        gr = S["direct_substitution_controls"]["gru_reference"]
        L.append("| GRU | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.0f} | {:.2f} | {} | 1 |".format(
            *(g[k] for k in ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW")), gr["per_seq_abs_dHOTA_mean"],
            S["per_sequence_delta_counts"]["HOTA"]["improved"]))
        for n, c in S["direct_substitution_controls"].items():
            if n == "gru_reference":
                continue
            o = c["overall"]
            L.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.0f} | {:.2f} | {} | {:.2f} |".format(
                n, o["HOTA"], o["AssA"], o["DetA"], o["IDF1"], o["MOTA"], o["IDSW"], c["per_seq_abs_dHOTA_mean"],
                c["seqs_improved_HOTA"], c["corr_with_gru_dHOTA"]))
    if S["association_diagnostic"]:
        L += ["", "## First-round association diagnostic (separate run)", "",
              "`{}`".format(json.dumps(S["association_diagnostic"]["totals"]))]
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
