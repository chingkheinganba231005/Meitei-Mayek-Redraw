# %% [markdown]
# # Reading by redrawing, round 3: honest routing, backbone and resolution, and uses of the parse
#
# What rounds 1-2 and the diagnostics showed (validation, one seed each):
#
# - No stroke component raised accuracy: everything sits within noise of the 97.90% baseline.
# - Each of them **lowered the image head of its own network** by 0.1 to 0.3 points: the
#   backbone pays for learning to redraw, because every loss reached every part (coupled training).
# - The remaining errors are mostly conflicts the image cannot settle. I (class 025) versus
#   I lonsum (044) alone is a third of all errors, and the nearest training images of a mistaken
#   image nearly always carry the label it was mistaken for.
#
# Round 3 asks seven questions, all on the **validation split only** (the test set stays untouched):
#
# | | question | new runs | GPU time |
# |---|---|---|---|
# | A | Does **honest routing** remove that cost? Drawing never changes what the recogniser sees, and reading never moves the strokes | `redraw_hdr_hr` | 30 min |
# | B | Is another ConvNeXt-T pretraining better here (ImageNet-12k, or ConvNeXt V2)? | `baseline_in12k`, `baseline_v2` | 25 min |
# | C | Does 224 px input (the backbones' own resolution) help the baseline and the method? Uses the best backbone from B | `baseline224`, `redraw_hdr_hr224` | 95 min |
# | D | Does **test-time redrawing** add accuracy? Predictions are averaged over re-writings made from the network's own parse | none | 15 min |
# | E | Which networks complement each other? | none | 1 min |
# | F | Does the method help when training data is scarce (2, 5, 10% of it)? | 9 short runs | 45 min |
# | G | How do the networks cope with damaged scans? | none | 25 min |
#
# About 4 hours in all. Every finished network and every prediction is kept on Drive, so after a
# disconnect just run all cells again. To save time, set `RUN_LOW_DATA` or `RUN_ROBUSTNESS` to
# False below. Results: `MyDrive/redraw/round3/results.json` and the figures next to it.

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
OUT = WORK / "round3"
for folder in (DATA, CACHE, OUT):
    folder.mkdir(parents=True, exist_ok=True)

# %%
import json
import math
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch

from redraw import __version__
from redraw.analysis import errors_in_pairs, mcnemar, pairwise_average
from redraw.augment import TTA_VIEWS
from redraw.data import Store, build_cache, load_index, stratified_subset
from redraw.model import CONFIGS
from redraw.render import curve_points
from redraw.robustness import CORRUPTIONS, corrupt_file
from redraw.split import URL, download, make_split
from redraw.strokewarp import rewrite_noise, stroke_warp
from redraw.train import cached_preds, fetch, load_model, parse_eval, predict, predict_ttr, train, verify_eval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("redraw", __version__, "|", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

RUN_LOW_DATA = True      # section F, about 45 minutes
RUN_ROBUSTNESS = True    # section G, about 25 minutes

IMG = 64 if FAST else 128
PAIRS = [(44, 25), (46, 9), (11, 47), (33, 34)]
V1 = "convnext_tiny.fb_in22k_ft_in1k"   # the backbone of rounds 1-2
# networks from rounds 1-2 and the diagnostics, reused if they are on Drive; the first two are
# needed as references and are trained if missing
REFERENCE = ["baseline_fp32", "redraw_hdr"]
OLD = REFERENCE + ["baseline", "redraw", "redraw_norender", "redraw_hdr_sw", "redraw_hdr_e2e", "redraw_hdr_nohead"]


def fast(cfg):
    """FAST: shrink a config to a two-epoch resnet18 for the CPU smoke test (no effect otherwise)."""
    if not FAST:
        return dict(cfg)
    cfg = dict(cfg, arch="resnet18", epochs=min(cfg["epochs"], 2), pretrained=False, batch=16, K=4, hdr_S=4)
    if cfg.get("res"):
        cfg["res"] = 96
    if "strokewarp" in cfg:
        cfg["strokewarp"] = dict(cfg["strokewarp"], warmup=0)
    return cfg


RESULTS = OUT / "results.json"
R = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
for _key in ("runs", "ttr", "low_data", "robustness"):
    R.setdefault(_key, {})


def save():
    RESULTS.write_text(json.dumps(R, indent=2, default=float))


def pct(x):
    return f"{100 * x:.2f}%"

# %% [markdown]
# ## 1. Data (the same split as rounds 1-2)

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
va = IDX["val"]
y = LABELS[va]

# %% [markdown]
# ## 2. Helpers
#
# `run` trains a network on the training split (or reuses the copy on Drive) and evaluates it on
# validation: every head with one view and with the five test-time views, errors in the four
# hard pairs, parse quality (strokes against ink) and, with the hypothesis check, what the check
# contributes (`verify`: top-3 recall, accuracy of the cheapest redrawing alone, images it turns
# from wrong to right / right to wrong).

# %%
def final_path(name, tag="dev"):
    return WORK / "runs" / name / tag / "final.pt"


def accuracy(p, idx=None):
    return float((p.argmax(1) == (y if idx is None else LABELS[idx])).mean())


def pair_accuracy(p, a, b):
    m = (y == a) | (y == b)
    return float((p[m].argmax(1) == y[m]).mean()) if m.any() else float("nan")


def check_same_recipe(name, cfg, tag="dev"):
    """Networks are reused by name: refuse a saved one trained with another backbone, input size or routing."""
    f = final_path(name, tag)
    if name in R["runs"]:
        saved = R["runs"][name]["cfg"]
    elif f.exists():
        saved = torch.load(f, map_location="cpu", weights_only=False)["cfg"]
    else:
        return
    for key in ("arch", "res", "route"):
        if saved.get(key) != cfg.get(key):
            raise SystemExit(f"{f} was trained with {key}={saved.get(key)!r}, but this run asks for "
                             f"{cfg.get(key)!r}. Move that folder away first.")


def evaluate(name, model, cfg):
    p_id = cached_preds(WORK, f"{name}_dev", "val_id", lambda: predict(model, va, cfg, STORE))
    p_tta = cached_preds(WORK, f"{name}_dev", "val_tta", lambda: predict(model, va, cfg, STORE, views=TTA_VIEWS))
    history = torch.load(final_path(name), map_location="cpu", weights_only=False)["history"]
    entry = {
        "cfg": cfg,
        "val": {v: {h: accuracy(p[h]) for h in p} for v, p in (("id", p_id), ("tta", p_tta))},
        "val_errors_tta": int((p_tta["logits"].argmax(1) != y).sum()),
        "pair_errors_tta": errors_in_pairs(p_tta["logits"], y, PAIRS),
        "pairs_tta": {f"{a:03d}/{b:03d}": pair_accuracy(p_tta["logits"], a, b) for a, b in PAIRS},
        "train_minutes": round(sum(h["sec"] for h in history) / 60, 1),
        "history": history,
    }
    if cfg.get("strokes") and cfg.get("render_weight", 1.0) > 0:
        entry["parse_val"] = parse_eval(model, va, STORE)
    if cfg.get("hdr"):
        entry["verify_val"] = verify_eval(model, va, STORE, LABELS)
    return entry


def run(name, cfg, train_if_missing=True):
    check_same_recipe(name, cfg)
    if "val" in R["runs"].get(name, {}):
        return True
    if not (train_if_missing or final_path(name).exists()):
        print(f"{name}: not on Drive, skipped")
        return False
    model = train(name, cfg, IDX["train"], "dev", STORE, LABELS, WORK, val_idx=va)
    R["runs"][name] = evaluate(name, model, cfg)
    save()
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return True


def show(names):
    head = f"{'network':18s} {'final':>7s} {'fused':>7s} {'image':>7s} {'strokes':>7s}   errors  " + \
           "  ".join(f"{a:03d}/{b:03d}" for a, b in PAIRS) + "   parse F   check"
    print(head)
    for n in names:
        if n not in R["runs"]:
            continue
        e = R["runs"][n]
        acc = [e["val"]["tta"].get(h) for h in ("logits", "fused", "img", "vec")]
        line = f"{n:18s} " + " ".join(f"{pct(a) if a is not None else '-':>7s}" for a in acc)
        line += f"   {e['val_errors_tta']:6d}  " + "  ".join(f"{v:7d}" for v in e["pair_errors_tta"].values())
        line += f"   {e['parse_val']['f']:.3f}" if "parse_val" in e else "       -"
        if "verify_val" in e:
            q = e["verify_val"]
            line += f"   top-3 {pct(q['topk_recall'])}, cost alone {pct(q['cost_only'])}, +{q['rescued']}/-{q['broken']}"
        print(line)


def paired(a, b, view="val_tta"):
    """McNemar test between two networks on validation (their TTA predictions)."""
    pa = cached_preds(WORK, f"{a}_dev", view, lambda: None)["logits"]
    pb = cached_preds(WORK, f"{b}_dev", view, lambda: None)["logits"]
    a_only, b_only, p = mcnemar(pa.argmax(1) == y, pb.argmax(1) == y)
    R.setdefault("mcnemar_val_tta", {})[f"{a}_vs_{b}"] = {"a_only": a_only, "b_only": b_only, "p": p}
    print(f"{a} vs {b}: only {a} right on {a_only}, only {b} right on {b_only} (p = {p:.3g})")


for name in OLD:
    run(name, fast(CONFIGS[name]), train_if_missing=name in REFERENCE)
save()
show(OLD)

# %% [markdown]
# ## A. Honest routing
#
# `redraw_hdr_hr` is `redraw_hdr` with honest routing everywhere: the stroke decoder and the
# hypothesis fitter read the backbone's feature maps through a stop-gradient (so drawing never
# changes what the recogniser sees), and the stroke head reads features where the strokes are but
# cannot move them (so the parse stays an explanation of the ink). The backbone learns from
# classification alone. What to look at: is its **image** column back at the baseline's level,
# and is the **final** column at least as good? Is the parse still as good (parse F)?

# %%
run("redraw_hdr_hr", fast(CONFIGS["redraw_hdr_hr"]))
show(["baseline_fp32", "redraw_hdr", "redraw_hdr_hr"])
paired("redraw_hdr_hr", "baseline_fp32")
paired("redraw_hdr_hr", "redraw_hdr")
save()

# %% [markdown]
# ## B. Which ConvNeXt-T pretraining?
#
# Same recipe, same 128 px input, three pretrainings: ImageNet-22k (rounds 1-2), ImageNet-12k, and
# ConvNeXt V2 (masked-autoencoder pretraining, then ImageNet-22k). The best one on validation is
# used for the 224 px runs in C.

# %%
BACKBONES = {V1: "baseline_fp32", "convnext_tiny.in12k_ft_in1k": "baseline_in12k",
             "convnextv2_tiny.fcmae_ft_in22k_in1k": "baseline_v2"}
for arch, name in BACKBONES.items():
    run(name, fast(CONFIGS[name]))
scores = {arch: R["runs"][name]["val"]["tta"]["logits"] for arch, name in BACKBONES.items()}
R["chosen_arch"] = max(scores, key=scores.get)  # ties keep the first, the backbone of rounds 1-2
show(list(BACKBONES.values()))
print("chosen for the 224 px runs:", R["chosen_arch"])
save()

# %% [markdown]
# ## C. 224 px input
#
# The cached 128 px images are resized to 224 px on the GPU, before augmentation. Same 40 epochs.
# This answers how much of the remaining gap is resolution, for the baseline and for the method.

# %%
ARCH = R["chosen_arch"]
run("baseline224", fast(dict(CONFIGS["baseline224"], arch=ARCH)))
run("redraw_hdr_hr224", fast(dict(CONFIGS["redraw_hdr_hr224"], arch=ARCH)))
show([BACKBONES[ARCH], "baseline224", "redraw_hdr_hr", "redraw_hdr_hr224"])
paired("baseline224", BACKBONES[ARCH])
paired("redraw_hdr_hr224", "baseline224")
save()

# %% [markdown]
# ## D. Test-time redrawing
#
# A network that parses strokes can make its own test-time views: it reads the image, then
# re-writes it a few times along the strokes it found (StrokeWarp: inner control points move so
# curvature changes, free tails grow or shrink; joints and pen width stay), and averages its
# predictions over the original and the re-writings. The random draws are fixed, so the numbers
# repeat. `ttr4` = original + 4 re-writings; `mild` uses half the bending; `tta5_ttr4_mild`
# re-writes each of the 5 ordinary test-time views. A plain CNN cannot do this (it has no parse).

# %%
MILD = dict(bend=0.02, tail=0.1)
TTR_NETS = ["redraw", "redraw_hdr", "redraw_hdr_sw", "redraw_hdr_hr", "redraw_hdr_hr224"]
for name in TTR_NETS:
    if name not in R["runs"] or name in R["ttr"]:
        continue
    cfg = R["runs"][name]["cfg"]
    model = load_model(final_path(name), cfg, device=DEVICE)
    tag = f"{name}_dev"
    p = {
        "id": cached_preds(WORK, tag, "val_id", lambda: None)["logits"],
        "tta5": cached_preds(WORK, tag, "val_tta", lambda: None)["logits"],
        "ttr4_mild": cached_preds(WORK, tag, "val_ttr4_mild",
                                  lambda: predict_ttr(model, va, STORE, n=4, strength=MILD))["logits"],
        "ttr4": cached_preds(WORK, tag, "val_ttr4", lambda: predict_ttr(model, va, STORE, n=4))["logits"],
        "tta5_ttr4_mild": cached_preds(WORK, tag, "val_tta5_ttr4_mild",
                                       lambda: predict_ttr(model, va, STORE, n=4, views=TTA_VIEWS,
                                                           strength=MILD))["logits"],
    }
    entry = {k: accuracy(v) for k, v in p.items()}
    for k in ("ttr4_mild", "tta5_ttr4_mild"):
        ref = "id" if k == "ttr4_mild" else "tta5"
        a_only, b_only, pv = mcnemar(p[k].argmax(1) == y, p[ref].argmax(1) == y)
        entry[f"mcnemar_{k}_vs_{ref}"] = {"a_only": a_only, "b_only": b_only, "p": pv}
    entry["pair_errors_tta5_ttr4_mild"] = errors_in_pairs(p["tta5_ttr4_mild"], y, PAIRS)
    R["ttr"][name] = entry
    save()
    del model
for name, e in R["ttr"].items():
    m = e["mcnemar_tta5_ttr4_mild_vs_tta5"]
    print(f"{name:18s} " + "  ".join(f"{k} {pct(e[k])}" for k in ("id", "ttr4_mild", "ttr4", "tta5", "tta5_ttr4_mild"))
          + f"   (with vs without re-writings: +{m['a_only']}/-{m['b_only']}, p = {m['p']:.3g})")

# %% [markdown]
# ## E. Which networks complement each other?
#
# Accuracy of the plain average of every pair of networks (TTA predictions), and each network's
# best partner. A partner from a different family helping more than another copy of the same
# family would say the method adds diversity, not just another draw.

# %%
NETS = [n for n in R["runs"] if (WORK / "preds" / f"{n}_dev_val_tta.npz").exists()]
P = {n: cached_preds(WORK, f"{n}_dev", "val_tta", lambda: None)["logits"] for n in NETS}
pairs = pairwise_average(P, y)
R["diversity"] = {
    "single": {n: pairs[(n, n)] for n in NETS},
    "pairs": {f"{a}+{b}": v for (a, b), v in pairs.items() if a != b},
    "average_of_all": float((np.mean([P[n] for n in NETS], 0).argmax(1) == y).mean()),
    "networks": NETS,
}
for n in NETS:
    partner = max((m for m in NETS if m != n), key=lambda m: pairs.get((n, m), pairs.get((m, n))), default=None)
    if partner:
        both = pairs.get((n, partner), pairs.get((partner, n)))
        print(f"{n:18s} alone {pct(pairs[(n, n)])}   best partner {partner:18s} together {pct(both)}")
print("average of all", len(NETS), "networks:", pct(R["diversity"]["average_of_all"]))
save()

# %% [markdown]
# ## F. Scarce training data
#
# The same three networks trained on 2, 5 and 10% of every class of the training split (the same
# images for all three), for 2000 steps each, and evaluated on the whole validation split:
# plain fine-tuning (`baseline_fp32`), coupled redrawing (`redraw_hdr`) and honest redrawing
# (`redraw_hdr_hr`). If learning to redraw is worth anything as a prior, it should show here.

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
    for frac in LOW_FRACS:
        sub = stratified_subset(IDX["train"], LABELS, frac, seed=0)
        tag = f"low{round(100 * frac):02d}"
        for name in LOW_NETS:
            key = f"{name}/{tag}"
            if key in R["low_data"]:
                continue
            cfg = low_cfg(fast(CONFIGS[name]), len(sub))
            model = train(name, cfg, sub, tag, STORE, LABELS, WORK, val_idx=va)
            p_tta = cached_preds(WORK, f"{name}_{tag}", "val_tta",
                                 lambda: predict(model, va, cfg, STORE, views=TTA_VIEWS))
            entry = {"fraction": frac, "images": int(len(sub)), "epochs": cfg["epochs"],
                     "val_tta": {h: accuracy(p) for h, p in p_tta.items()}}
            if cfg.get("hdr"):
                entry["verify_val"] = verify_eval(model, va, STORE, LABELS)
            R["low_data"][key] = entry
            save()
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    for frac in LOW_FRACS:
        tag = f"low{round(100 * frac):02d}"
        cells = []
        for name in LOW_NETS:
            e = R["low_data"].get(f"{name}/{tag}")
            if e:
                cells.append(f"{name} {pct(e['val_tta']['logits'])}"
                             + (f" (image head {pct(e['val_tta']['img'])})" if "vec" in e["val_tta"] else ""))
        print(f"{round(100 * frac):3d}% of the training data:  " + "   ".join(cells))

# %% [markdown]
# ## G. Damaged scans
#
# Every validation image corrupted before preprocessing (noise, blur, rotation, low contrast,
# uneven lighting, salt and pepper, low resolution; two strengths each; fixed random seeds), then
# classified with one view. Clean accuracy (one view) is in the first column.

# %%
ROBUST = [n for n in ("baseline_fp32", "redraw", "redraw_hdr", "redraw_hdr_hr", "baseline224", "redraw_hdr_hr224")
          if n in R["runs"]]
LEVELS = [(kind, level) for kind, levels in CORRUPTIONS.items() for level in levels]
if RUN_ROBUSTNESS:
    todo = [(k, lv) for k, lv in LEVELS if f"{k}_{lv}" not in R["robustness"]
            or set(ROBUST) - set(R["robustness"][f"{k}_{lv}"])]
    if todo:
        models = {n: load_model(final_path(n), R["runs"][n]["cfg"], device=DEVICE) for n in ROBUST}
        files = INDEX.file.to_numpy()[va].tolist()
        for kind, level in todo:
            arrays = build_cache(files, IMG, CACHE / "corrupt", fn=partial(corrupt_file, kind=kind, level=level),
                                 tag=f"{IMG}_{kind}_{level}")
            damaged = Store(arrays, DEVICE, stats=STORE.stats, meta_norm=STORE.meta_norm)
            row = R["robustness"].setdefault(f"{kind}_{level}", {})
            for n, m in models.items():
                if n not in row:
                    row[n] = accuracy(predict(m, np.arange(len(va)), R["runs"][n]["cfg"], damaged,
                                              heads=("logits",))["logits"])
            save()
            del damaged, arrays
        del models
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    keys = [f"{k}_{lv}" for k, lv in LEVELS]
    print(f"{'network':18s} {'clean':>7s} " + " ".join(f"{k:>14s}" for k in keys) + "     mean")
    R["robustness_mean"] = {}
    for n in ROBUST:
        accs = [R["robustness"][k][n] for k in keys]
        R["robustness_mean"][n] = float(np.mean(accs))
        print(f"{n:18s} {pct(R['runs'][n]['val']['id']['logits']):>7s} "
              + " ".join(f"{pct(a):>14s}" for a in accs) + f"   {pct(np.mean(accs))}")
    save()

# %% [markdown]
# ## H. Figures
#
# 1. The parse with coupled and with honest routing, on images of the hard pairs and a few others
#    (image, coupled strokes, honest strokes).
# 2. Test-time redrawing: an image and four re-writings made from the honest network's own parse.
# 3. Accuracy against the share of training data.

# %%
colors = plt.cm.tab20(np.linspace(0, 1, 20))


def draw_strokes(ax, pts, width, exist, scale=60):
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for k in np.argsort(-exist):
        if exist[k] > 0.5:
            ax.plot(pts[k, :, 0], pts[k, :, 1], color=colors[k % 20], lw=max(0.5, width[k] * scale),
                    solid_capstyle="round")


rng = np.random.default_rng(0)
classes = [c for pair in PAIRS for c in pair] + list(rng.choice(55, 4, replace=False))
pick = np.array([rng.choice(va[y == c]) for c in classes if (y == c).any()])
parses = {}
for name in ("redraw_hdr", "redraw_hdr_hr"):
    m = load_model(final_path(name), R["runs"][name]["cfg"], device=DEVICE)
    x = fetch(STORE, pick, m.input_res)
    with torch.no_grad():
        o = m(STORE.normalize(x, "gray"), draw=False)
    parses[name] = (curve_points(o["ctrl"].float(), 64).cpu().numpy(), o["width"].float().cpu().numpy(),
                    o["exist"].float().cpu().numpy())
    if name == "redraw_hdr_hr":
        honest, x_honest, o_honest = m, x, o
    else:
        del m
x = fetch(STORE, pick, None)
fig, axes = plt.subplots(int(np.ceil(len(pick) / 4)), 12, figsize=(14, 1.3 * np.ceil(len(pick) / 4)), squeeze=False)
for ax in axes.flat:
    ax.axis("off")
for i in range(len(pick)):
    r, c = divmod(i, 4)
    axes[r, 3 * c].imshow(1 - x[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[r, 3 * c].set_title(f"{LABELS[pick[i]]:03d}", fontsize=7)
    for j, name in enumerate(("redraw_hdr", "redraw_hdr_hr")):
        pts, width, exist = parses[name]
        draw_strokes(axes[r, 3 * c + 1 + j], pts[i], width[i], exist[i])
        axes[r, 3 * c + 1 + j].set_title("coupled" if j == 0 else "honest", fontsize=6)
plt.tight_layout()
plt.savefig(OUT / "parses_coupled_vs_honest.png", dpi=130)
plt.show()

# test-time redrawing examples (mild strength, as in section D)
rows = min(6, len(pick))
fig, axes = plt.subplots(rows, 5, figsize=(6, 1.25 * rows), squeeze=False)
sw = dict(CONFIGS["redraw_hdr_sw"]["strokewarp"], **MILD)
for r_ in range(5):
    xr = x_honest
    if r_:
        noise = rewrite_noise(*o_honest["ctrl"].shape[:2], sw["bend"], sw["tail"],
                              torch.Generator().manual_seed(r_), x_honest.device)
        xr = stroke_warp(x_honest, o_honest["ctrl"], o_honest["exist"], noise=noise, **sw)
    for i in range(rows):
        axes[i, r_].imshow(1 - xr[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, r_].axis("off")
        if i == 0:
            axes[i, r_].set_title("image" if r_ == 0 else f"re-writing {r_}", fontsize=7)
plt.tight_layout()
plt.savefig(OUT / "test_time_redrawing.png", dpi=130)
plt.show()
del honest

if RUN_LOW_DATA and R["low_data"]:
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for name, label in zip(LOW_NETS, ("plain fine-tuning", "coupled redrawing", "honest redrawing")):
        xs = [f for f in LOW_FRACS if f"{name}/low{round(100 * f):02d}" in R["low_data"]]
        ys = [100 * R["low_data"][f"{name}/low{round(100 * f):02d}"]["val_tta"]["logits"] for f in xs]
        ax.plot([100 * f for f in xs], ys, marker="o", label=label)
    ax.set_xscale("log")
    ax.set_xticks([100 * f for f in LOW_FRACS], [f"{round(100 * f)}%" for f in LOW_FRACS])
    ax.set_xlabel("share of the training split")
    ax.set_ylabel("validation accuracy (%)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "low_data.png", dpi=150)
    plt.show()

# %% [markdown]
# ## Summary
#
# Everything above in one place; the numbers are in `results.json`, with the figures next to it.

# %%
print("all networks (validation, 5-view TTA):")
show(list(R["runs"]))
print()
for k, v in R.get("mcnemar_val_tta", {}).items():
    print(f"{k}: +{v['a_only']}/-{v['b_only']}, p = {v['p']:.3g}")
print("\nbackbone for 224 px:", R.get("chosen_arch"))
for name, e in R["ttr"].items():
    print(f"test-time redrawing {name}: TTA {pct(e['tta5'])} -> with re-writings {pct(e['tta5_ttr4_mild'])}")
print("average of all networks:", pct(R["diversity"]["average_of_all"]))
if R.get("robustness_mean"):
    print("mean accuracy on damaged scans:", {n: pct(v) for n, v in R["robustness_mean"].items()})
print("results:", RESULTS)
