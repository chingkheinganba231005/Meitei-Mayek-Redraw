# %% [markdown]
# # Final run: train on training + validation, evaluate the test set once
#
# Everything was fixed on validation in rounds 1 to 3, before this notebook looks at a single test
# image, and nothing is changed after:
#
# - recipe: ConvNeXt-T (ImageNet-22k), 128 px, 40 epochs, geometric augmentation, weight average,
#   label smoothing 0.1 (ImageNet-12k, ConvNeXt V2 and 224 px were all worse on validation);
# - the method: `redraw_hdr_hr`: stroke parsing, stroke head and the hypothesis check, with
#   honest routing;
# - the control: `baseline_fp32`, the same backbone and recipe without any of it;
# - the shape-bias ablation: `redraw_hdr`, the method with coupled routing.
#
# | | what | A100 time |
# |---|---|---|
# | 1 | train 3 seeds of the method, 3 of the control and 1 of the coupled model on training + validation (72,330 images) | 3 h 20 min |
# | 2 | test set: single networks (mean and s.d. over seeds), seed averages, paired tests, intervals, hard pairs, parse, check, test-time redrawing | 25 min |
# | 3 | familiar and unfamiliar handwriting: accuracy by how close a test image is to its most similar training image (pixels, no network involved) | 5 min |
# | 4 | damaged scans, test set | 15 min |
# | 5 | scarce data: two more seeds at 2, 5 and 10%, all three seeds evaluated on test (optional) | 1 h 45 min |
#
# About 6 hours in all; every network and prediction is kept on Drive, so after a disconnect run
# all cells again. Section 5 can be switched off (`RUN_LOW_DATA`). New files take about 3 GB on
# Drive. Results: `MyDrive/redraw/final/results.json` and the figures next to it.

# %%
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
FAST = os.environ.get("FAST") == "1"  # tiny CPU smoke test of the code, not a real run

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0.9", "scikit-image"], check=True)

REPO_URL = os.environ.get("REPO_URL", "https://github.com/chingkheinganba231005/Meitei-Mayek-Redraw.git")
REPO = Path(os.environ.get("REPO", "/content/redraw"))
CODE_ZIP = Path(os.environ.get("CODE_ZIP", "/content/drive/MyDrive/redraw_code.zip"))


def get_code():
    if IN_COLAB:
        shutil.rmtree(REPO, ignore_errors=True)  # always start from the current code
    if REPO.exists():
        return "local copy"
    clone = ["git", "clone", "-q", "--depth", "1"]
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")  # fail instead of asking for a password

    def cloned(url):
        ok = subprocess.run(clone + [url, str(REPO)], capture_output=True, env=env).returncode == 0
        if ok and not (REPO / "redraw" / "__init__.py").exists():  # an empty or unrelated repository
            shutil.rmtree(REPO, ignore_errors=True)
            ok = False
        return ok

    if CODE_ZIP.exists():  # a zip on Drive wins, so a newer zip is never shadowed by an older repository
        tmp = REPO.parent / "_code"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(CODE_ZIP) as z:
            z.extractall(tmp)
        package = next(tmp.rglob("redraw/__init__.py")).parent
        shutil.move(str(package.parent), str(REPO))
        return f"zip {CODE_ZIP}"
    if cloned(REPO_URL):
        return "GitHub"
    token = None
    if IN_COLAB:
        try:
            from google.colab import userdata
            token = userdata.get("GH_TOKEN")
        except Exception:
            token = None
    if token and cloned(REPO_URL.replace("https://", f"https://x-access-token:{token}@")):
        return "GitHub (private, with GH_TOKEN)"
    raise SystemExit(f"Could not get the code: no {CODE_ZIP}, and the repository could not be cloned. "
                     "If the repository is private, add the GH_TOKEN secret.")


print("code from", get_code())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
for _name in [n for n in sys.modules if n == "redraw" or n.startswith("redraw.")]:
    del sys.modules[_name]  # a re-run must not keep the code imported the first time

DATA = Path(os.environ.get("DATA", "/content/data"))
CACHE = Path(os.environ.get("CACHE", "/content/cache"))
WORK = Path(os.environ.get("WORK", "/content/drive/MyDrive/redraw"))
OUT = WORK / "final"
for folder in (DATA, CACHE, OUT):
    folder.mkdir(parents=True, exist_ok=True)

# %%
import json
import math
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from redraw import __version__
from redraw.analysis import accuracy_by_bin, bootstrap_ci, errors_in_pairs, familiarity, mcnemar
from redraw.augment import TTA_VIEWS
from redraw.data import Store, build_cache, load_index, stratified_subset
from redraw.model import CONFIGS
from redraw.render import curve_points
from redraw.robustness import CORRUPTIONS, corrupt_file
from redraw.split import URL, download, make_split
from redraw.train import cached_preds, load_model, parse_eval, predict, predict_ttr, train, verify_eval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("redraw", __version__, "|", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

RUN_ROBUSTNESS = True    # section 4, about 15 minutes
RUN_LOW_DATA = True      # section 5, about 1 h 45 min

IMG = 64 if FAST else 128
PAIRS = [(44, 25), (46, 9), (11, 47), (33, 34)]
SEEDS = (0, 1, 2)
FAMILIES = {"baseline_fp32": "plain fine-tuning", "redraw_hdr_hr": "redrawing, honest routing (the method)",
            "redraw_hdr": "redrawing, coupled routing"}
FINAL = [("baseline_fp32", s) for s in SEEDS] + [("redraw_hdr_hr", s) for s in SEEDS] + [("redraw_hdr", 0)]
BAR_ERRORS = 241   # errors of the earlier six-network average (98.12%) on the same 12,794 test images
MILD = dict(bend=0.02, tail=0.1)


def fast(cfg):
    """FAST: shrink a config to a two-epoch resnet18 for the CPU smoke test (no effect otherwise)."""
    if not FAST:
        return dict(cfg)
    return dict(cfg, arch="resnet18", epochs=min(cfg["epochs"], 2), pretrained=False, batch=16, K=4, hdr_S=4)


RESULTS = OUT / "results.json"
R = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
for _key in ("test", "low_data", "robustness"):
    R.setdefault(_key, {})
R["protocol"] = ("Recipe, method, control and seeds fixed on validation (rounds 1-3) before any test image was "
                 "used; final networks trained on training + validation; the test set is predicted once and "
                 "the predictions are cached.")


def save():
    RESULTS.write_text(json.dumps(R, indent=2, default=float))


def pct(x):
    return f"{100 * x:.2f}%"

# %% [markdown]
# ## Data (the same split as rounds 1-3)

# %%
ZIP = Path(os.environ.get("ZIP", WORK / "TUMMHCD-TEST-TRAIN.zip"))
if not (ZIP.exists() and zipfile.is_zipfile(ZIP)):
    drive_root = Path("/content/drive/MyDrive")
    found = [p for d in ("*", "*/*", "*/*/*") for p in drive_root.glob(f"{d}/TUMMHCD-TEST-TRAIN.zip")]
    found = [p for p in found if zipfile.is_zipfile(p)]
    if found:
        ZIP = found[0]
        print("using", ZIP)
    else:
        print("downloading TUMMHCD ...")
        download(URL, ZIP)

SPLITS, summary = make_split(ZIP, DATA)
print(summary)
if not FAST:
    assert (summary["train"], summary["val"], summary["test"]) == (61504, 10826, 12794), summary
INDEX, LABELS, IDX = load_index(SPLITS)
STORE = Store(build_cache(INDEX.file.tolist(), IMG, CACHE), DEVICE, IDX["train"])
full, te = IDX["full"], IDX["test"]
y = LABELS[te]


def final_path(name, tag):
    return WORK / "runs" / name / tag / "final.pt"


def right(p):
    return p.argmax(1) == y

# %% [markdown]
# ## 1. Train on training + validation
#
# No validation checks during training: the validation images are training images now.

# %%
for name, s in FINAL:
    tag = f"full_s{s}"
    cfg = fast(dict(CONFIGS[name], seed=s))
    if not final_path(name, tag).exists():
        t0 = time.time()
        model = train(name, cfg, full, tag, STORE, LABELS, WORK)
        print(f"{name}/{tag}: {(time.time() - t0) / 60:.0f} min")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
print("all final networks trained")

# %% [markdown]
# ## 2. The test set
#
# Each network with one view (`id`) and with the five test-time views (`tta`); the stroke networks
# also with test-time redrawing (the five views, each re-written four times along the network's
# own parse). Then the mean and standard deviation over seeds, the average of the seeds of each
# family, exact McNemar tests (method against control, seed against seed and average against
# average), bootstrap intervals, the hard pairs, parse quality and what the hypothesis check does.

# %%
for name, s in FINAL:
    key = f"{name}/full_s{s}"
    if key in R["test"]:
        continue
    tag = f"full_s{s}"
    cfg = fast(dict(CONFIGS[name], seed=s))
    model = load_model(final_path(name, tag), cfg, device=DEVICE)
    stem = f"{name}_{tag}"
    p_id = cached_preds(WORK, stem, "test_id", lambda: predict(model, te, cfg, STORE))
    p_tta = cached_preds(WORK, stem, "test_tta", lambda: predict(model, te, cfg, STORE, views=TTA_VIEWS))
    e = {"name": name, "seed": s,
         "test": {v: {h: float(right(p[h]).mean()) for h in p} for v, p in (("id", p_id), ("tta", p_tta))},
         "errors_tta": int((~right(p_tta["logits"])).sum()),
         "pair_errors_tta": errors_in_pairs(p_tta["logits"], y, PAIRS)}
    if cfg.get("strokes"):
        p_ttr = cached_preds(WORK, stem, "test_tta5_ttr4_mild",
                             lambda: predict_ttr(model, te, STORE, n=4, views=TTA_VIEWS, strength=MILD))
        e["tta5_ttr4_mild"] = float(right(p_ttr["logits"]).mean())
        e["parse_test"] = parse_eval(model, te, STORE)
    if cfg.get("hdr"):
        e["verify_test"] = verify_eval(model, te, STORE, LABELS)
    R["test"][key] = e
    save()
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

P = {f"{n}/{s}": cached_preds(WORK, f"{n}_full_s{s}", "test_tta", lambda: None)["logits"] for n, s in FINAL}
OK = {k: right(p) for k, p in P.items()}


def members(family):
    return [k for k in P if k.split("/")[0] == family]


summary = {}
for family in FAMILIES:
    ks = members(family)
    accs = [float(OK[k].mean()) for k in ks]
    avg = np.mean([P[k] for k in ks], 0)
    summary[family] = {
        "seeds": len(ks), "single": accs, "single_mean": float(np.mean(accs)),
        "single_sd": float(np.std(accs, ddof=1)) if len(ks) > 1 else None,
        "single_img_head": [R["test"][f"{k.split('/')[0]}/full_s{k.split('/')[1]}"]["test"]["tta"].get("img") for k in ks],
        "average": float(right(avg).mean()), "average_errors": int((~right(avg)).sum()),
        "average_ci95": bootstrap_ci(right(avg)), "average_pair_errors": errors_in_pairs(avg, y, PAIRS),
    }
all_avg = np.mean(list(P.values()), 0)
summary["all_seven"] = {"average": float(right(all_avg).mean()), "average_errors": int((~right(all_avg)).sum()),
                        "average_ci95": bootstrap_ci(right(all_avg))}
R["summary"] = summary
tests = {}
for s in SEEDS:
    if f"redraw_hdr_hr/{s}" in OK and f"baseline_fp32/{s}" in OK:
        tests[f"method_s{s}_vs_control_s{s}"] = mcnemar(OK[f"redraw_hdr_hr/{s}"], OK[f"baseline_fp32/{s}"])
avg_m = right(np.mean([P[k] for k in members("redraw_hdr_hr")], 0))
avg_c = right(np.mean([P[k] for k in members("baseline_fp32")], 0))
tests["method_average_vs_control_average"] = mcnemar(avg_m, avg_c)
tests["coupled_s0_vs_control_s0"] = mcnemar(OK["redraw_hdr/0"], OK["baseline_fp32/0"])
tests["coupled_s0_vs_method_s0"] = mcnemar(OK["redraw_hdr/0"], OK["redraw_hdr_hr/0"])
R["mcnemar_test"] = {k: {"a_only": a, "b_only": b, "p": p} for k, (a, b, p) in tests.items()}
n_wrong = np.sum([~OK[k] for k in OK], 0)
R["wrong_by_all"] = {"count": int((n_wrong == len(OK)).sum()),
                     "per_pair": {f"{a:03d}/{b:03d}": int(((n_wrong == len(OK)) & np.isin(y, [a, b])).sum())
                                  for a, b in PAIRS}}
save()

if len(te) == 12794:
    print(f"test images: {len(te)}; the earlier six-network average made {BAR_ERRORS} errors ({pct(1 - BAR_ERRORS / len(te))})\n")
for family, label in FAMILIES.items():
    sm = summary[family]
    sd = f" +- {100 * sm['single_sd']:.2f}" if sm["single_sd"] is not None else ""
    print(f"{label}")
    print(f"   single networks: {', '.join(pct(a) for a in sm['single'])}  ->  mean {pct(sm['single_mean'])}{sd}")
    print(f"   average of its {sm['seeds']} seed(s): {pct(sm['average'])} ({sm['average_errors']} errors, "
          f"95% interval {pct(sm['average_ci95'][0])} - {pct(sm['average_ci95'][1])}), hard pairs {sm['average_pair_errors']}")
print(f"average of all seven networks: {pct(summary['all_seven']['average'])} ({summary['all_seven']['average_errors']} errors)")
print()
for k, v in R["mcnemar_test"].items():
    print(f"{k}: +{v['a_only']}/-{v['b_only']}, p = {v['p']:.3g}")
print("\nwrong for all seven networks:", R["wrong_by_all"])
for k, e in R["test"].items():
    extra = ""
    if "tta5_ttr4_mild" in e:
        extra += f"  with test-time redrawing {pct(e['tta5_ttr4_mild'])}  parse F {e['parse_test']['f']:.3f}"
    if "verify_test" in e:
        q = e["verify_test"]
        extra += f"  check: top-3 {pct(q['topk_recall'])}, cost alone {pct(q['cost_only'])}, +{q['rescued']}/-{q['broken']}"
    print(f"{k:22s} id {pct(e['test']['id']['logits'])}  tta {pct(e['test']['tta']['logits'])}{extra}")

# %% [markdown]
# ## 3. Familiar and unfamiliar handwriting
#
# TUMMHCD's test images were drawn at random from the same writers as the training images, and
# most of them have a near copy in the training set (the same writer wrote each character several
# times). How close each test image is to its most similar training image is measured on 32 x 32
# thumbnails (centred cosine similarity; no network involved), and accuracy is reported for the
# least familiar 5%, the next 5%, the next 15% and the most familiar 75%. The least familiar images
# are the nearest thing this benchmark has to handwriting from a new writer. The same is done on
# validation for the round-3 networks (trained on the training split only).

# %%
BIN_NAMES = ["least familiar 5%", "5-10%", "10-25%", "most familiar 75%"]


def thumbs(idx, side=32, bs=4096):
    out = []
    for i in range(0, len(idx), bs):
        x = STORE.fetch(idx[i:i + bs], "gray")
        out.append(F.interpolate(x, size=(side, side), mode="area").flatten(1))
    return torch.cat(out)


def familiarity_bins(query_idx, ref_idx):
    """Similarity of each query image to its most similar reference image, its bin (by rank, so the
    bins hold exactly 5, 5, 15 and 75% of the images even with ties) and that reference's label."""
    sim, nn = familiarity(thumbs(query_idx), thumbs(ref_idx))
    sim, nn = sim.cpu().numpy(), nn.cpu().numpy()
    rank = np.empty(len(sim))
    rank[np.argsort(sim, kind="stable")] = (np.arange(len(sim)) + 0.5) / len(sim)
    return sim, np.digitize(rank, [0.05, 0.10, 0.25]), LABELS[ref_idx][nn]


sim_te, bin_te, nn_label_te = familiarity_bins(te, full)
FAM = {"edges": np.quantile(sim_te, [0.05, 0.10, 0.25]).tolist(),
       "pixel_nearest_neighbour": {"all": float((nn_label_te == y).mean()),
                                   "by_bin": accuracy_by_bin(nn_label_te == y, bin_te)},
       "networks": {k: accuracy_by_bin(v, bin_te) for k, v in OK.items()},
       "averages": {fam_: accuracy_by_bin(right(np.mean([P[k] for k in members(fam_)], 0)), bin_te)
                    for fam_ in FAMILIES}}
least10 = bin_te <= 1
FAM["mcnemar_least_familiar_10"] = dict(zip(("a_only", "b_only", "p"), mcnemar(avg_m[least10], avg_c[least10])))
FAM["mcnemar_least_familiar_10_coupled"] = dict(zip(("a_only", "b_only", "p"),
                                                    mcnemar(OK["redraw_hdr/0"][least10], OK["baseline_fp32/0"][least10])))
R["familiarity_test"] = FAM
print("similarity of the most similar training image, bin edges:", [round(v, 3) for v in FAM["edges"]])
print("nearest training image in pixels, as a classifier:", pct(FAM["pixel_nearest_neighbour"]["all"]), "|",
      "  ".join(f"{BIN_NAMES[b]} {pct(a)}" for b, (a, n) in FAM["pixel_nearest_neighbour"]["by_bin"].items()))
for fam_, label in FAMILIES.items():
    rows = [FAM["networks"][k] for k in members(fam_)]
    cells = []
    for b in range(4):
        accs = [r_[b][0] for r_ in rows if b in r_]
        cells.append(f"{BIN_NAMES[b]} {pct(np.mean(accs))}" + (f" +- {100 * np.std(accs, ddof=1):.2f}" if len(accs) > 1 else ""))
    print(f"{label}: " + "   ".join(cells))
m = FAM["mcnemar_least_familiar_10"]
print(f"least familiar 10%, method average vs control average: +{m['a_only']}/-{m['b_only']} (p = {m['p']:.3g})")
m = FAM["mcnemar_least_familiar_10_coupled"]
print(f"least familiar 10%, coupled vs control (seed 0): +{m['a_only']}/-{m['b_only']} (p = {m['p']:.3g})")

# the same on validation, for the round-3 networks (training split only), where their predictions exist
va, tr = IDX["val"], IDX["train"]
y_va = LABELS[va]
sim_va, bin_va, nn_label_va = familiarity_bins(va, tr)
val_rows = {}
for n in ("baseline_fp32", "baseline", "redraw", "redraw_hdr", "redraw_hdr_hr", "redraw_hdr_e2e", "redraw_norender"):
    f = WORK / "preds" / f"{n}_dev_val_tta.npz"
    if f.exists():
        p = cached_preds(WORK, f"{n}_dev", "val_tta", lambda: None)["logits"]
        val_rows[n] = accuracy_by_bin(p.argmax(1) == y_va, bin_va)
R["familiarity_val"] = {"pixel_nearest_neighbour": accuracy_by_bin(nn_label_va == y_va, bin_va), "networks": val_rows}
for n, r_ in val_rows.items():
    print(f"validation {n:18s} " + "   ".join(f"{BIN_NAMES[b]} {pct(a)}" for b, (a, _) in r_.items()))
save()

# %% [markdown]
# ## 4. Damaged scans (test set)
#
# Every test image corrupted before preprocessing (seven kinds, two strengths, fixed seeds) and
# classified with one view by each final network.

# %%
LEVELS = [(kind, level) for kind, levels in CORRUPTIONS.items() for level in levels]
if RUN_ROBUSTNESS:
    todo = [(k, lv) for k, lv in LEVELS if set(P) - set(R["robustness"].get(f"{k}_{lv}", {}))]
    if todo:
        models = {f"{n}/{s}": (load_model(final_path(n, f"full_s{s}"), fast(dict(CONFIGS[n], seed=s)), device=DEVICE),
                               fast(dict(CONFIGS[n], seed=s))) for n, s in FINAL}
        files = INDEX.file.to_numpy()[te].tolist()
        for kind, level in todo:
            arrays = build_cache(files, IMG, CACHE / "corrupt_test", fn=partial(corrupt_file, kind=kind, level=level),
                                 tag=f"{IMG}_{kind}_{level}")
            damaged = Store(arrays, DEVICE, stats=STORE.stats, meta_norm=STORE.meta_norm)
            row = R["robustness"].setdefault(f"{kind}_{level}", {})
            for k, (m_, cfg) in models.items():
                if k not in row:
                    row[k] = float(right(predict(m_, np.arange(len(te)), cfg, damaged, heads=("logits",))["logits"]).mean())
            save()
            del damaged, arrays
        del models
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    R["robustness_by_family"] = {}
    print(f"{'corruption':18s} " + " ".join(f"{label.split(' (')[0][:26]:>28s}" for label in FAMILIES.values()))
    for k in [f"{a}_{b}" for a, b in LEVELS]:
        cells = []
        for fam_ in FAMILIES:
            accs = [R["robustness"][k][m] for m in members(fam_)]
            R["robustness_by_family"].setdefault(fam_, {})[k] = (float(np.mean(accs)),
                                                                 float(np.std(accs, ddof=1)) if len(accs) > 1 else None)
            cells.append(f"{pct(np.mean(accs))}" + (f" +- {100 * np.std(accs, ddof=1):.2f}" if len(accs) > 1 else ""))
        print(f"{k:18s} " + " ".join(f"{c:>28s}" for c in cells))
    save()

# %% [markdown]
# ## 5. Scarce data (optional)
#
# Round 3 trained the control, the coupled and the honest network on 2, 5 and 10% of every class
# of the training split (seed 0). Here seeds 1 and 2 are added (a new subset and a new training
# seed each, the same subset for the three networks), and all three seeds are evaluated on test.

# %%
LOW_FRACS = (0.02, 0.05, 0.10)
LOW_NETS = ("baseline_fp32", "redraw_hdr", "redraw_hdr_hr")
LOW_STEPS = 4 if FAST else 2000


def low_cfg(cfg, n_images):
    bs = min(cfg.get("batch", 256), n_images)
    epochs = max(1, math.ceil(LOW_STEPS / max(1, n_images // bs)))
    return dict(cfg, epochs=epochs, val_every=10**9, warmup_steps=max(1, LOW_STEPS // 10),
                ckpt_every=max(1, epochs // 10), log_every=max(1, epochs // 4))


if RUN_LOW_DATA:
    for s in SEEDS:
        for frac in LOW_FRACS:
            sub = stratified_subset(IDX["train"], LABELS, frac, seed=s)
            tag = f"low{round(100 * frac):02d}" + (f"_s{s}" if s else "")  # seed 0 is round 3's run
            for name in LOW_NETS:
                key = f"{name}/{tag}"
                if key in R["low_data"]:
                    continue
                base = dict(CONFIGS[name], seed=s) if s else dict(CONFIGS[name])
                cfg = low_cfg(fast(base), len(sub))
                model = train(name, cfg, sub, tag, STORE, LABELS, WORK)
                p = cached_preds(WORK, f"{name}_{tag}", "test_tta",
                                 lambda: predict(model, te, cfg, STORE, views=TTA_VIEWS))
                R["low_data"][key] = {"fraction": frac, "seed": s, "images": int(len(sub)),
                                      "test_tta": {h: float(right(v).mean()) for h, v in p.items()}}
                save()
                del model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
    R["low_data_summary"] = {}
    for frac in LOW_FRACS:
        line = f"{round(100 * frac):3d}%:"
        accs = {}
        for name in LOW_NETS:
            accs[name] = [R["low_data"][k]["test_tta"]["logits"] for k in R["low_data"]
                          if k.startswith(name + "/") and R["low_data"][k]["fraction"] == frac]
            line += f"   {name} {pct(np.mean(accs[name]))} +- {100 * np.std(accs[name], ddof=1):.2f} ({len(accs[name])})" \
                if len(accs[name]) > 1 else f"   {name} {pct(np.mean(accs[name]))}"
        diff = [a - b for a, b in zip(accs["redraw_hdr_hr"], accs["baseline_fp32"])]
        R["low_data_summary"][f"{frac}"] = {n: accs[n] for n in LOW_NETS} | {"honest_minus_control": diff}
        print(line + f"   honest - control: {', '.join(f'{100 * d:+.2f}' for d in diff)}")
    save()

# %% [markdown]
# ## 6. Figures

# %%
fig, ax = plt.subplots(figsize=(6, 3.2))
width = 0.25
for j, (fam_, label) in enumerate(FAMILIES.items()):
    rows = [FAM["networks"][k] for k in members(fam_)]
    means = [np.mean([r_.get(b, (np.nan,))[0] for r_ in rows]) * 100 for b in range(4)]
    sds = [np.std([r_.get(b, (np.nan,))[0] for r_ in rows], ddof=1) * 100 if len(rows) > 1 else 0 for b in range(4)]
    ax.bar(np.arange(4) + (j - 1) * width, means, width, yerr=sds, capsize=2, label=label.split(" (")[0])
ax.set_xticks(range(4), BIN_NAMES, fontsize=8)
ax.set_ylabel("test accuracy (%)")
lo = min(min(r_[b][0] for b in r_) for r_ in FAM["networks"].values()) * 100
ax.set_ylim(max(0, lo - 3), 100)
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUT / "familiarity.png", dpi=150)
plt.show()

if RUN_LOW_DATA:
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for name, label in zip(LOW_NETS, ("plain fine-tuning", "redrawing, coupled", "redrawing, honest")):
        m_ = [100 * np.mean(R["low_data_summary"][f"{f}"][name]) for f in LOW_FRACS]
        s_ = [100 * np.std(R["low_data_summary"][f"{f}"][name], ddof=1) if len(R["low_data_summary"][f"{f}"][name]) > 1 else 0
              for f in LOW_FRACS]
        ax.errorbar([100 * f for f in LOW_FRACS], m_, yerr=s_, marker="o", capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_xticks([100 * f for f in LOW_FRACS], [f"{round(100 * f)}%" for f in LOW_FRACS])
    ax.set_xlabel("share of the training split")
    ax.set_ylabel("test accuracy (%)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "low_data.png", dpi=150)
    plt.show()

if RUN_ROBUSTNESS:
    heavy = [f"{k}_{levels[-1]}" for k, levels in CORRUPTIONS.items()]
    fig, ax = plt.subplots(figsize=(7, 3))
    for j, (fam_, label) in enumerate(FAMILIES.items()):
        vals = [100 * R["robustness_by_family"][fam_][k][0] for k in heavy]
        ax.bar(np.arange(len(heavy)) + (j - 1) * width, vals, width, label=label.split(" (")[0])
    ax.set_xticks(range(len(heavy)), [k.replace("_", " ") for k in heavy], fontsize=7, rotation=20)
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(80, 100)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "damaged_scans.png", dpi=150)
    plt.show()

# test images of the hard pairs: the image, the parse of the method (seed 0), and the two
# hypotheses of its pair bent onto it by the fitter with their costs (lower = better explanation)
colors = plt.cm.tab20(np.linspace(0, 1, 20))


def draw_strokes(ax, pts, width_, exist, scale=60):
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for k in np.argsort(-exist):
        if exist[k] > 0.5:
            ax.plot(pts[k, :, 0], pts[k, :, 1], color=colors[k % 20], lw=max(0.5, width_[k] * scale),
                    solid_capstyle="round")


model = load_model(final_path("redraw_hdr_hr", "full_s0"), fast(dict(CONFIGS["redraw_hdr_hr"], seed=0)), device=DEVICE)
rng = np.random.default_rng(0)
pick = np.concatenate([rng.choice(te[y == c], min(2, int((y == c).sum())), replace=False)
                       for pair in PAIRS for c in pair if (y == c).any()])
x = STORE.fetch(pick, "gray")
cand = torch.tensor([next(p for p in PAIRS if LABELS[i] in p) for i in pick], device=DEVICE)
with torch.no_grad():
    o = model(STORE.normalize(x, "gray"), draw=False)
    _, inter = model.backbone.forward_intermediates(STORE.normalize(x, "gray"), indices=model.taps)
    mem, mpos = model.decoder(inter)[3:]
    tmpl, fitted = model.hdr.fit(mem, mpos, cand)
    target = F.interpolate(x, size=(model.hdr_G, model.hdr_G), mode="area").squeeze(1)
    bent = model.hdr.bend(tmpl, fitted)
    cost = model.hdr.cost(bent, tmpl[2], fitted, target, G=model.hdr_G).cpu().numpy()
ppts = curve_points(o["ctrl"].float(), 64).cpu().numpy()
pw, pe = o["width"].float().cpu().numpy(), o["exist"].float().cpu().numpy()
bpts = curve_points(bent[0].float(), 64).cpu().numpy()
bwidth, bexist = bent[1].float().cpu().numpy(), torch.sigmoid(bent[2]).float().cpu().numpy()
rows_n = int(np.ceil(len(pick) / 2))
fig, axes = plt.subplots(rows_n, 8, figsize=(11, rows_n * 1.4), squeeze=False)
for ax in axes.flat:
    ax.axis("off")
for i in range(len(pick)):
    r_, c_ = divmod(i, 2)
    axes[r_, 4 * c_].imshow(1 - x[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[r_, 4 * c_].set_title(f"test, class {LABELS[pick[i]]:03d}", fontsize=6)
    draw_strokes(axes[r_, 4 * c_ + 1], ppts[i], pw[i], pe[i])
    axes[r_, 4 * c_ + 1].set_title("parse", fontsize=6)
    chosen = int(np.argmin(cost[i]))
    for j in range(2):
        ax = axes[r_, 4 * c_ + 2 + j]
        draw_strokes(ax, bpts[i, j, 0], bwidth[i, j, 0], bexist[i, j, 0], scale=40)
        ok_ = cand[i, j].item() == LABELS[pick[i]]
        ax.set_title(f"as {cand[i, j].item():03d}: {cost[i, j]:.3f}", fontsize=6,
                     color="black" if j != chosen else ("green" if ok_ else "red"))
plt.tight_layout()
plt.savefig(OUT / "test_parses_and_hypotheses.png", dpi=150)
plt.show()
del model

# %% [markdown]
# ## Summary
#
# Send `MyDrive/redraw/final/results.json` back (the figures too, if you like).

# %%
print("TEST SET (12,794 images), 5-view test-time augmentation")
for fam_, label in FAMILIES.items():
    sm = summary[fam_]
    sd = f" +- {100 * sm['single_sd']:.2f}" if sm["single_sd"] is not None else ""
    print(f"  {label}: single {pct(sm['single_mean'])}{sd}; average of seeds {pct(sm['average'])} "
          f"({sm['average_errors']} errors)")
print(f"  all seven: {pct(summary['all_seven']['average'])} ({summary['all_seven']['average_errors']} errors)")
if len(te) == 12794:
    print(f"  earlier six-network average: {pct(1 - BAR_ERRORS / len(te))} ({BAR_ERRORS} errors)")
print("results:", RESULTS)
