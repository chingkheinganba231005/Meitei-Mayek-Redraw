"""Every number of the paper that comes from an earlier run, recomputed from the results files.

    python -m redraw.paper_numbers                      # reads results/, writes results/paper_numbers.json

Accuracies are in percent, differences in percentage points. Error counts of single networks are the
mean over seeds (not an observed count); counts of seed averages and of McNemar tests are observed.
Paired comparisons across training runs use the exact signed-rank test with ties (``stats``).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp

from .manifest import write_json
from .stats import paired_runs, signed_rank_p

FAMILIES = {"control": "baseline_fp32", "ours": "redraw_hdr_hr", "coupled": "redraw_hdr"}
N_TEST = 12794


def _pct(x):
    return round(100 * float(x), 4)


def _sd(v):
    return round(100 * float(np.std(v, ddof=1)), 4) if len(v) > 1 else None


def final_numbers(f):
    out = {}
    t = f["test"]
    for fam, name in FAMILIES.items():
        keys = sorted(k for k in t if k.split("/")[0] == name)
        rows = {}
        for view in ("tta", "id"):
            for head in ("logits", "img", "vec"):
                v = [t[k]["test"][view].get(head) for k in keys]
                if all(a is not None for a in v):
                    rows[f"{view}_{head}"] = {"runs": [_pct(a) for a in v], "mean": _pct(np.mean(v)), "sd": _sd(v)}
        rows["mean_errors_tta"] = round(float(np.mean([t[k]["errors_tta"] for k in keys])), 1)
        if "parse_test" in t[keys[0]]:
            rows["rendered_ink"] = {m: [round(t[k]["parse_test"][m], 4) for k in keys]
                                    for m in ("precision", "recall", "f", "strokes")}
            p, r = np.mean(rows["rendered_ink"]["precision"]), np.mean(rows["rendered_ink"]["recall"])
            rows["rendered_ink"]["harmonic_mean_of_mean_p_and_r"] = round(float(2 * p * r / (p + r)), 4)
        if "verify_test" in t[keys[0]]:
            v = [t[k]["verify_test"] for k in keys]
            rows["check_1view"] = {"candidate_recall_mean": _pct(np.mean([a["topk_recall"] for a in v])),
                                   "energy_alone_right_mean": _pct(np.mean([a["cost_only"] for a in v])),
                                   "wrong_to_right": sum(a["rescued"] for a in v),
                                   "right_to_wrong": sum(a["broken"] for a in v),
                                   "per_seed": [(a["rescued"], a["broken"]) for a in v]}
        s = f["summary"].get(name, {})
        if s:
            rows["seed_average_tta"] = {"accuracy": _pct(s["average"]), "errors": s["average_errors"],
                                        "ci95": [_pct(a) for a in s["average_ci95"]]}
        out[fam] = rows
    c, o = out["control"], out["ours"]
    for view in ("tta", "id"):
        d = [round(b - a, 4) for a, b in zip(c[f"{view}_logits"]["runs"], o[f"{view}_logits"]["runs"])]
        out[f"ours_minus_control_{view}"] = {"per_seed": d, "mean": round(float(np.mean(d)), 4),
                                             "images_per_seed": [round(x / 100 * N_TEST) for x in d]}
    out["mcnemar"] = f["mcnemar_test"]
    out["wrong_by_all_seven"] = f["wrong_by_all"]
    return out


def reduced_data_numbers(f):
    """Seed-paired differences (ours - control) with reduced data and with all data, and the tests."""
    low = f["low_data_summary"]
    reduced = []
    for frac in ("0.02", "0.05", "0.1"):
        reduced += [100 * d for d in low[frac]["honest_minus_control"]]
    t = f["test"]
    full = [100 * (t[f"redraw_hdr_hr/full_s{s}"]["test"]["tta"]["logits"] - t[f"baseline_fp32/full_s{s}"]["test"]["tta"]["logits"])
            for s in range(3)]

    def summary(d):
        d = np.round(np.asarray(d), 6)
        return {"differences": d.round(4).tolist(), "n": len(d), "ahead": int((d > 0).sum()), "behind": int((d < 0).sum()),
                "mean": round(float(d.mean()), 4), "exact_wilcoxon_p_ties": round(signed_rank_p(d), 4),
                "t_test_p": round(float(ttest_1samp(d, 0).pvalue), 4) if len(d) > 2 else None}

    out = {"reduced (9 pairs)": summary(reduced), "full (3 pairs)": summary(full), "all (12 pairs)": summary(reduced + full)}
    for frac in ("0.02", "0.05", "0.1"):
        out[f"coupled_minus_control_{frac}"] = round(100 * float(np.mean(low[frac]["redraw_hdr"]) - np.mean(low[frac]["baseline_fp32"])), 4)
    out["note"] = ("the subsets of different seeds are drawn independently (not nested); the same subset and seed is "
                   "used for both networks of a pair")
    return out


def robustness_numbers(f):
    rb = f["robustness_by_family"]
    out = {fam: round(100 * float(np.mean([v[0] for v in rb[name].values()])), 2) for fam, name in FAMILIES.items()}
    strong = {}
    for key in rb["baseline_fp32"]:
        c = rb["baseline_fp32"][key][0]
        strong[key] = {"ours_minus_control": round(100 * (rb["redraw_hdr_hr"][key][0] - c), 2),
                       "coupled_minus_control": round(100 * (rb["redraw_hdr"][key][0] - c), 2)}
    return {"mean_over_14_conditions": out, "by_condition": strong,
            "note": "corruption before preprocessing, one view; coupled routing has one seed, the others three"}


def familiarity_numbers(f):
    fam = f["familiarity_test"]
    nn = fam["pixel_nearest_neighbour"]
    nets = [a for k, v in fam["networks"].items() for a, _ in v.values()]
    by_family = {}
    for fam_name, name in FAMILIES.items():
        keys = [k for k in fam["networks"] if k.split("/")[0] == name]
        by_family[fam_name] = {b: _pct(np.mean([fam["networks"][k][b][0] for k in keys])) for b in ("0", "1", "2", "3")}
    means = [v for d in by_family.values() for v in d.values()]
    return {"pixel_nearest_neighbour_all": _pct(nn["all"]),
            "pixel_nearest_neighbour_by_bin": {b: _pct(a) for b, (a, n) in nn["by_bin"].items()},
            "networks_range_over_bins": [_pct(min(nets)), _pct(max(nets))],
            "mean_over_seeds_by_bin": by_family, "mean_over_seeds_range": [min(means), max(means)],
            "least_familiar_10_mcnemar": {"ours_average_vs_control_average": fam.get("mcnemar_least_familiar_10"),
                                          "coupled_vs_control_seed0": fam.get("mcnemar_least_familiar_10_coupled")},
            "bins": "least familiar 5%, 5-10%, 10-25%, most familiar 75% (centred 32 x 32 thumbnails)"}


def development_numbers(r3, diag):
    rows = {}
    for name, v in r3["runs"].items():
        e = {"final": _pct(v["val"]["tta"]["logits"]), "image": _pct(v["val"]["tta"].get("img", v["val"]["tta"]["logits"])),
             "train_minutes": v.get("train_minutes")}
        if "vec" in v["val"]["tta"]:
            e["strokes"] = _pct(v["val"]["tta"]["vec"])
        if "parse_val" in v:
            e["rendered_ink_f"] = round(v["parse_val"]["f"], 3)
        if "verify_val" in v:
            e["energy_alone_right"] = _pct(v["verify_val"]["cost_only"])
        rows[name] = e
    nb = diag["neighbours"]
    return {"networks": rows, "nearest_training_label_agrees_all": _pct(nb["top1_same_label_all"]),
            "nearest_training_label_agrees_on_errors": _pct(nb["top1_same_label_on_classifier_errors"])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    res = Path(args.results)
    f = json.loads((res / "final.json").read_text())
    out = {"test": final_numbers(f), "reduced_data": reduced_data_numbers(f), "robustness": robustness_numbers(f),
           "familiarity": familiarity_numbers(f)}
    if (res / "round3.json").exists() and (res / "diagnostics.json").exists():
        out["development"] = development_numbers(json.loads((res / "round3.json").read_text()),
                                                 json.loads((res / "diagnostics.json").read_text()))
    write_json(res / "paper_numbers.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()


__all__ = ["final_numbers", "reduced_data_numbers", "robustness_numbers", "familiarity_numbers", "paired_runs"]
