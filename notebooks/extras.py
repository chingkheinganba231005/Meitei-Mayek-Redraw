# %% [markdown]
# # Additional experiments: templates, faithfulness, the routing dial and a second script
#
# These experiments extend the final run without changing its test protocol: sections A and B
# use the TUMMHCD test set exactly as the final run did, and section C uses a second dataset with
# the recipe unchanged (nothing is tuned on it).
#
# | | what | GPU time |
# |---|---|---|
# | A | TUMMHCD, the final networks, no training: the learned stroke templates of all 55 classes; characters redrawn as vector graphics (also saved as SVG); selective prediction (does the check's evidence tell when the recogniser is wrong?); erasing strokes (does the stroke the explanation names decide what the networks read?) | 30 min |
# | B | TUMMHCD, the routing dial: a quarter of, and all of, the drawing gradient let into the backbone (one seed each, training + validation), test set and damaged scans | 1 h 40 min |
# | C | A second script: the Devanagari Handwritten Character Dataset (46 classes, 92,000 images, downloaded from UCI), same recipe: 3 seeds of the control, 3 of ours, 1 coupled; test once; damaged scans; 2% and 5% of the data with 3 seeds; parses | 4 h 40 min |
#
# Everything is kept on Drive, so after a disconnect run all cells again. Switch parts off with the
# flags below (`RUN_DIAL`, `RUN_DHCD`, `RUN_DHCD_LOW_DATA`). New files take about 4 GB on Drive.
# Results: `MyDrive/redraw/extras/results.json` and the figures next to it.

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
DATA_D = Path(os.environ.get("DATA_D", "/content/data_dhcd"))
CACHE_D = Path(os.environ.get("CACHE_D", "/content/cache_dhcd"))
WORK_D = WORK / "dhcd"
OUT = WORK / "extras"
for folder in (DATA, CACHE, OUT, DATA_D, CACHE_D, WORK_D):
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
from redraw.analysis import (bootstrap_ci, errors_in_pairs, mcnemar, rank_average, risk_at, risk_coverage)
from redraw.augment import TTA_VIEWS
from redraw.data import Store, build_cache, load_index, stratified_subset
from redraw.model import CONFIGS
from redraw.render import curve_points
from redraw.robustness import CORRUPTIONS, corrupt_file
from redraw.split import URL, download, make_split, make_split_folders
from redraw.train import cached_preds, check_outputs, load_model, parse_eval, predict, train, verify_eval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("redraw", __version__, "|", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

RUN_DIAL = True            # section B, about 1 h 40 min
RUN_DHCD = True            # section C, about 3 h 50 min without its scarce-data part
RUN_DHCD_LOW_DATA = True   # section C, scarce data, about 50 min more

IMG = 64 if FAST else 128
PAIRS = [(44, 25), (46, 9), (11, 47), (33, 34)]
SEEDS = (0, 1, 2)
HEAVY = [(kind, levels[-1]) for kind, levels in CORRUPTIONS.items()]   # the stronger level of each corruption
DHCD_URL = "https://archive.ics.uci.edu/static/public/389/devanagari+handwritten+character+dataset.zip"


def fast(cfg):
    """FAST: shrink a config to a two-epoch resnet18 for the CPU smoke test (no effect otherwise)."""
    if not FAST:
        return dict(cfg)
    return dict(cfg, arch="resnet18", epochs=min(cfg["epochs"], 2), pretrained=False, batch=16, K=4, hdr_S=4)


RESULTS = OUT / "results.json"
R = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
for _key in ("selective", "dial", "dhcd"):
    R.setdefault(_key, {})


def save():
    RESULTS.write_text(json.dumps(R, indent=2, default=float))


def pct(x):
    return f"{100 * x:.2f}%"


def final_path(work, name, tag):
    return work / "runs" / name / tag / "final.pt"


def npz_cache(path, make):
    """float32 arrays saved once (cached_preds stores float16, too coarse for energies)."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = make()
        np.savez(path.with_suffix(".tmp.npz"), **arrays)
        path.with_suffix(".tmp.npz").replace(path)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}

# %% [markdown]
# ## TUMMHCD (the same split as before)

# %%
ZIP = Path(os.environ.get("ZIP", WORK / "TUMMHCD-TEST-TRAIN.zip"))
if not (ZIP.exists() and zipfile.is_zipfile(ZIP)):
    drive_root = Path("/content/drive/MyDrive")
    found = [p for d in ("*", "*/*", "*/*/*") for p in drive_root.glob(f"{d}/TUMMHCD-TEST-TRAIN.zip")]
    found = [p for p in found if zipfile.is_zipfile(p)]
    if found:
        ZIP = found[0]
    else:
        download(URL, ZIP)
SPLITS, summary = make_split(ZIP, DATA)
if not FAST:
    assert (summary["train"], summary["val"], summary["test"]) == (61504, 10826, 12794), summary
INDEX, LABELS, IDX = load_index(SPLITS)
STORE = Store(build_cache(INDEX.file.tolist(), IMG, CACHE), DEVICE, IDX["train"])
full, te = IDX["full"], IDX["test"]
y = LABELS[te]
for name, s in [("baseline_fp32", s) for s in SEEDS] + [("redraw_hdr_hr", s) for s in SEEDS] + [("redraw_hdr", 0)]:
    assert final_path(WORK, name, f"full_s{s}").exists(), f"run the final notebook first ({name}/full_s{s} missing)"

# %% [markdown]
# ## A1. The learned stroke templates
#
# The first of the two templates of every class, as learned by our network (seed 0) from the
# training images: cubic B\'ezier strokes, drawn with the template's own width; faint strokes are
# ones the template rarely writes.

# %%
colors = plt.cm.tab20(np.linspace(0, 1, 20))
OURS0 = load_model(final_path(WORK, "redraw_hdr_hr", "full_s0"), fast(dict(CONFIGS["redraw_hdr_hr"], seed=0)),
                   device=DEVICE)


def draw_template(ax, ctrl, width, prior, title=None, scale=50):
    pts = curve_points(torch.as_tensor(ctrl), 64).numpy()
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=7)
    for s_ in range(pts.shape[0]):
        ax.plot(pts[s_, :, 0], pts[s_, :, 1], color=colors[s_ % 20], lw=max(0.6, float(width[s_]) * scale),
                alpha=float(np.clip(prior[s_], 0.08, 1)), solid_capstyle="round")


ctrl, twidth, tprior = (t.detach().float().cpu() for t in OURS0.hdr.template_params())
tprior = torch.sigmoid(tprior).numpy()
n_cls = ctrl.shape[0]
cols = 11
rows_n = int(np.ceil(n_cls / cols))
fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 1.05, rows_n * 1.15), squeeze=False)
for c, ax in enumerate(axes.flat):
    if c < n_cls:
        draw_template(ax, ctrl[c, 0], twidth[c, 0], tprior[c, 0], f"{c:03d}")
    else:
        ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "templates_all.pdf")
plt.savefig(OUT / "templates_all.png", dpi=160)
plt.show()

hard = [c for pair in PAIRS for c in pair]
fig, axes = plt.subplots(2, len(hard), figsize=(len(hard) * 1.1, 2.4), squeeze=False)
for j, c in enumerate(hard):
    for m in range(2):
        draw_template(axes[m, j], ctrl[c, m], twidth[c, m], tprior[c, m], f"{c:03d}, template {m + 1}" if m == 0 else None)
plt.tight_layout()
plt.savefig(OUT / "templates_hard_pairs.pdf")
plt.show()

# %% [markdown]
# ## A2. Characters redrawn as vector graphics
#
# Each test character next to the network's parse, drawn as a vector graphic (here plotted at any
# size, and also written as SVG files to `extras/svg/`). The parse is a by-product of recognition:
# no stroke labels were used.

# %%
rng = np.random.default_rng(1)
classes = hard + list(rng.choice(n_cls, 8, replace=False))
pick = np.array([rng.choice(te[y == c]) for c in classes if (y == c).any()])
x = STORE.fetch(pick, "gray")
with torch.no_grad():
    o = OURS0(STORE.normalize(x, "gray"), draw=False)
pts = curve_points(o["ctrl"].float(), 96).cpu().numpy()
pw, pe = o["width"].float().cpu().numpy(), o["exist"].float().cpu().numpy()
cols = 8
rows_n = int(np.ceil(len(pick) / (cols // 2)))
fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 1.15, rows_n * 1.2), squeeze=False)
for ax in axes.flat:
    ax.axis("off")
for i in range(len(pick)):
    r_, c_ = divmod(i, cols // 2)
    axes[r_, 2 * c_].imshow(1 - x[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[r_, 2 * c_].set_title(f"{LABELS[pick[i]]:03d}", fontsize=7)
    ax = axes[r_, 2 * c_ + 1]
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    for k in np.argsort(-pe[i]):
        if pe[i, k] > 0.5:
            ax.plot(pts[i, k, :, 0], pts[i, k, :, 1], color="black", lw=max(0.8, pw[i, k] * 60), solid_capstyle="round")
plt.tight_layout()
plt.savefig(OUT / "vector_redraw.pdf")
plt.show()

svg_dir = OUT / "svg"
svg_dir.mkdir(exist_ok=True)
c_np = o["ctrl"].float().cpu().numpy()
for i in range(min(8, len(pick))):
    size = 256
    paths = []
    for k in range(c_np.shape[1]):
        if pe[i, k] > 0.5:
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = (c_np[i, k] * size).tolist()
            paths.append(f'<path d="M {x0:.1f} {y0:.1f} C {x1:.1f} {y1:.1f}, {x2:.1f} {y2:.1f}, {x3:.1f} {y3:.1f}" '
                         f'stroke="black" stroke-width="{pw[i, k] * size:.1f}" fill="none" stroke-linecap="round"/>')
    (svg_dir / f"test_{pick[i]}_class{LABELS[pick[i]]:03d}.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<rect width="100%" height="100%" fill="white"/>{"".join(paths)}</svg>')
print("SVG files:", sorted(p.name for p in svg_dir.iterdir())[:8])
del OURS0
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# %% [markdown]
# ## A3. Selective prediction: does the check know when the recogniser is wrong?
#
# A recogniser that may abstain keeps the images it is most sure of. The usual measure of sureness is
# the top softmax probability. The hypothesis check offers a second one that a plain network does not
# have: how much better the chosen class's templates explain the ink than the best rival's (the
# energy margin; negative when the check prefers another class). We report the error rate among the
# kept images at 90, 95 and 98% coverage and the area under the whole risk-coverage curve (AURC,
# lower is better), for the softmax alone, the energy margin alone and the average of their ranks
# (no weight is fitted, so nothing is tuned on the test set).

# %%
def selective_scores(name, s):
    stem = f"{name}_full_s{s}"
    p = cached_preds(WORK, stem, "test_tta", lambda: None)["logits"]
    pred = p.argmax(1)
    scores = {"softmax": p.max(1)}
    if name != "baseline_fp32":
        def make():
            m = load_model(final_path(WORK, name, f"full_s{s}"), fast(dict(CONFIGS[name], seed=s)), device=DEVICE)
            return check_outputs(m, te, STORE)
        chk = npz_cache(WORK / "preds" / f"{stem}_test_check.npz", make)
        cand, energy = chk["cand"], chk["energy"]
        in_cand = cand == pred[:, None]
        e_pred = np.where(in_cand.any(1), (energy * in_cand).sum(1), np.inf)
        e_rival = np.where(in_cand, np.inf, energy).min(1)
        scores["energy margin"] = np.where(np.isfinite(e_pred), e_rival - e_pred, -np.inf)
        scores["softmax + energy (rank average)"] = rank_average(scores["softmax"], scores["energy margin"])
    return pred == y, scores


rows = {}
for name, s in [("baseline_fp32", s) for s in SEEDS] + [("redraw_hdr_hr", s) for s in SEEDS]:
    ok, scores = selective_scores(name, s)
    for label, sc in scores.items():
        key = f"{'control' if name == 'baseline_fp32' else 'ours'}: {label}"
        _, _, aurc = risk_coverage(sc, ok)
        rows.setdefault(key, []).append({"aurc": aurc, **{f"risk@{c}": risk_at(sc, ok, c) for c in (0.90, 0.95, 0.98)},
                                         "accuracy": float(ok.mean())})
R["selective"] = {k: {m: [r[m] for r in v] for m in v[0]} for k, v in rows.items()}
save()
print(f"{'score':48s} {'AURC x100':>10s} {'err@90%':>9s} {'err@95%':>9s} {'err@98%':>9s}   (mean over seeds)")
for k, v in R["selective"].items():
    print(f"{k:48s} {100 * np.mean(v['aurc']):10.3f} " + " ".join(f"{100 * np.mean(v[f'risk@{c}']):8.2f}%" for c in (0.90, 0.95, 0.98)))

# %% [markdown]
# ## A4. Erasing strokes: is the explanation faithful?
#
# The parse splits each character into strokes, so an explanation can be tested by deleting what it
# points at. For every test image one stroke of our network's parse is erased (its pixels set to
# paper) and the networks read the image again. The stroke is chosen in five ways: the one the stroke
# reader finds most important (leaving it out of the parse lowers the reader's support for the
# network's class the most; this is the explanation under test), the least important one, a random
# one, the largest one (most ink), and the one with the most gradient x input for the network that
# reads the image (a standard pixel attribution, summed over each stroke's pixels). If the explanation
# is faithful, erasing its most important stroke changes the decision far more often than erasing a
# random or the largest stroke, and its least important stroke rarely matters. The control network
# reads the same erased images: if it changes its mind on the same strokes, the explanation points at
# strokes that decide the character, not at quirks of one network. One view (no test-time
# augmentation); images whose parse has fewer than two strokes are left out; each seed of ours is
# paired with the control of the same seed.

# %%
from redraw.charset import CLASSES
from redraw.erase import CHOOSERS, erase, erase_test, explain

ER = R.setdefault("erase", {})
NETS = {"ours": "redraw_hdr_hr", "control": "baseline_fp32"}
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9893"


def load_final(name, s):
    return load_model(final_path(WORK, name, f"full_s{s}"), fast(dict(CONFIGS[name], seed=s)), device=DEVICE)


for s in SEEDS:
    if f"s{s}" in ER:
        continue
    nets = {lab: load_final(name, s) for lab, name in NETS.items()}
    t0 = time.time()
    e = erase_test(nets["ours"], nets, te, STORE, seed=s)
    valid = e["strokes"] >= 2
    row = {"images": int(valid.sum()), "strokes": float(e["strokes"][valid].mean()),
           "intact accuracy": {lab: float((e["pred"][lab] == y)[valid].mean()) for lab in nets},
           "changed": {c: {lab: float((e["erased"][c][lab] != e["pred"][lab])[valid].mean()) for lab in nets}
                       for c in CHOOSERS},
           "accuracy": {c: {lab: float((e["erased"][c][lab] == y)[valid].mean()) for lab in nets} for c in CHOOSERS},
           "ink erased": {k: float(v[valid].mean()) for k, v in e["ink_erased"].items()},
           "same stroke as reader, most": {k: float((v == e["stroke"]["reader, most"])[valid].mean())
                                           for k, v in e["stroke"].items() if k != "reader, most"}}
    ER[f"s{s}"] = row
    save()
    if s == 0:
        np.savez(OUT / "erase_s0.npz", most=e["stroke"]["reader, most"], strokes=e["strokes"],
                 **{f"{c}|{lab}": e["erased"][c][lab] for c in CHOOSERS for lab in nets},
                 **{f"pred|{lab}": e["pred"][lab] for lab in nets})
    print(f"seed {s}: {(time.time() - t0) / 60:.1f} min")
    del nets
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def mean_over_seeds(get):
    return float(np.mean([get(ER[f"s{s}"]) for s in SEEDS]))


print(f"{'stroke erased':18s} {'ink erased':>11s} " + " ".join(f"{lab + ' changes':>15s}" for lab in NETS)
      + "   (share of test images, mean over seeds)")
for c in CHOOSERS:
    ink_keys = [c] if c != "gradient x input" else [f"{c}, {lab}" for lab in NETS]
    ink = np.mean([mean_over_seeds(lambda r, k=k: r["ink erased"][k]) for k in ink_keys])
    print(f"{c:18s} {pct(ink):>11s} " + " ".join(f"{pct(mean_over_seeds(lambda r, lab=lab: r['changed'][c][lab])):>15s}"
                                                 for lab in NETS))
print("intact accuracy (one view):", {lab: pct(mean_over_seeds(lambda r, lab=lab: r["intact accuracy"][lab])) for lab in NETS})

fig, ax = plt.subplots(figsize=(6.4, 2.6))
xs = np.arange(len(CHOOSERS))
for j, (lab, col) in enumerate((("ours", ORANGE), ("control", BLUE))):
    vals = np.array([[100 * ER[f"s{s}"]["changed"][c][lab] for s in SEEDS] for c in CHOOSERS])
    ax.bar(xs + (j - 0.5) * 0.38, vals.mean(1), 0.36, color=col, label=f"{lab} network")
    for i in range(len(CHOOSERS)):
        ax.scatter(np.full(len(SEEDS), xs[i] + (j - 0.5) * 0.38), vals[i], s=8, color="black", zorder=3)
ax.set_xticks(xs, CHOOSERS, fontsize=8)
ax.set_ylabel("decision changed (% of images)")
ax.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(OUT / "erase.pdf")
plt.show()

# examples: erasing the stroke the reader names changes both networks' minds
z = np.load(OUT / "erase_s0.npz")
flip = ((z["reader, most|ours"] != z["pred|ours"]) & (z["reader, most|control"] != z["pred|control"])
        & (z["pred|ours"] == y) & (z["strokes"] >= 2))
rng = np.random.default_rng(0)
shown, pick_i = set(), []
for i in rng.permutation(np.flatnonzero(flip)):
    if y[i] not in shown:
        shown.add(y[i])
        pick_i.append(i)
    if len(pick_i) == 8:
        break
print(f"{flip.mean():.2%} of the test images change both networks' decisions when the reader's stroke is erased")
if pick_i:
    model = load_final("redraw_hdr_hr", 0)
    x = STORE.fetch(te[pick_i], "gray")
    o, _, _, own = explain(model, x, STORE)
    x_er = erase(x, own, torch.as_tensor(z["most"][pick_i], device=x.device)).cpu().numpy()
    pts = curve_points(o["ctrl"].float(), 64).cpu().numpy()
    ex, wd = o["exist"].float().cpu().numpy(), o["width"].float().cpu().numpy()
    name = lambda c: CLASSES[int(c)].name if int(c) < len(CLASSES) else str(int(c))
    rows_n = int(np.ceil(len(pick_i) / 4))
    fig, axes = plt.subplots(rows_n, 12, figsize=(12, 1.5 * rows_n), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for j, i in enumerate(pick_i):
        r_, c_ = divmod(j, 4)
        a0, a1, a2 = axes[r_, 3 * c_:3 * c_ + 3]
        a0.imshow(1 - x[j, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        a0.set_title(name(y[i]), fontsize=7)
        a1.imshow(1 - 0.3 * x[j, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1, extent=(0, 1, 1, 0))
        a1.set_xlim(0, 1); a1.set_ylim(1, 0); a1.set_aspect("equal")
        for k in range(pts.shape[1]):
            if ex[j, k] > 0.5:
                hot = k == z["most"][i]
                a1.plot(pts[j, k, :, 0], pts[j, k, :, 1], color=ORANGE if hot else GREY,
                        lw=max(0.8, wd[j, k] * 50), solid_capstyle="round", zorder=3 if hot else 2)
        a1.set_title("decisive stroke", fontsize=6)
        a2.imshow(1 - x_er[j, 0], cmap="gray", vmin=0, vmax=1)
        a2.set_title(f"ours: {name(z['reader, most|ours'][i])}\ncontrol: {name(z['reader, most|control'][i])}", fontsize=6)
    plt.tight_layout()
    plt.savefig(OUT / "erase_examples.png", dpi=160)
    plt.show()
    del model

# %% [markdown]
# ## B. The routing dial
#
# Honest routing lets none of the drawing gradient into the backbone; coupled routing lets all of it
# in and also lets the stroke reader move the strokes. Here the stroke reader stays honest and a
# fraction of the drawing gradient (`draw_grad` = 0.25 or 1) reaches the backbone. One seed each,
# trained on training + validation like the final networks, evaluated on the same test set and on the
# stronger level of each corruption. If the shape bias is caused by the drawing gradient, it should
# grow with the dial.

# %%
DIAL = [("redraw_hdr_hr_g25", 0.25), ("redraw_hdr_hr_g100", 1.0)]
if RUN_DIAL:
    for name, _ in DIAL:
        cfg = fast(dict(CONFIGS[name], seed=0))
        if not final_path(WORK, name, "full_s0").exists():
            model = train(name, cfg, full, "full_s0", STORE, LABELS, WORK)
            del model
        if name not in R["dial"]:
            model = load_model(final_path(WORK, name, "full_s0"), cfg, device=DEVICE)
            p = cached_preds(WORK, f"{name}_full_s0", "test_tta", lambda: predict(model, te, cfg, STORE, views=TTA_VIEWS))
            R["dial"][name] = {"test_tta": {h: float((v.argmax(1) == y).mean()) for h, v in p.items()},
                               "parse_test": parse_eval(model, te, STORE)}
            save()
            del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    # damaged scans, the stronger level of each corruption, all networks of the dial (seed 0)
    DIAL_NETS = {"control": ("baseline_fp32", "full_s0"), "dial 0 (ours)": ("redraw_hdr_hr", "full_s0"),
                 "dial 0.25": ("redraw_hdr_hr_g25", "full_s0"), "dial 1": ("redraw_hdr_hr_g100", "full_s0"),
                 "coupled": ("redraw_hdr", "full_s0")}
    rob = R["dial"].setdefault("robustness", {})
    files = INDEX.file.to_numpy()[te].tolist()
    todo = [(k, lv) for k, lv in HEAVY if set(DIAL_NETS) - set(rob.get(f"{k}_{lv}", {}))]
    if todo:
        models = {lab: (load_model(final_path(WORK, n, t), fast(dict(CONFIGS[n], seed=0)), device=DEVICE),
                        fast(dict(CONFIGS[n], seed=0))) for lab, (n, t) in DIAL_NETS.items()}
        for kind, level in todo:
            arrays = build_cache(files, IMG, CACHE / "corrupt_test", fn=partial(corrupt_file, kind=kind, level=level),
                                 tag=f"{IMG}_{kind}_{level}")
            damaged = Store(arrays, DEVICE, stats=STORE.stats, meta_norm=STORE.meta_norm)
            row = rob.setdefault(f"{kind}_{level}", {})
            for lab, (m_, cfg) in models.items():
                if lab not in row:
                    row[lab] = float((predict(m_, np.arange(len(te)), cfg, damaged, heads=("logits",))["logits"].argmax(1)
                                      == y).mean())
            save()
            del damaged, arrays
        del models
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    print("clean test accuracy (5 views):", {n: pct(e["test_tta"]["logits"]) for n, e in R["dial"].items() if n != "robustness"})
    print(f"{'corruption':18s} " + " ".join(f"{lab:>14s}" for lab in DIAL_NETS))
    for k, lv in HEAVY:
        row = rob[f"{k}_{lv}"]
        print(f"{k + '_' + str(lv):18s} " + " ".join(f"{pct(row[lab]):>14s}" for lab in DIAL_NETS))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    labs = [lab for lab in DIAL_NETS if lab != "control"]
    xs = np.arange(len(HEAVY))
    w = 0.8 / len(labs)
    for j, lab in enumerate(labs):
        d = [100 * (rob[f"{k}_{lv}"][lab] - rob[f"{k}_{lv}"]["control"]) for k, lv in HEAVY]
        ax.bar(xs + (j - (len(labs) - 1) / 2) * w, d, w, label=lab)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs, [f"{k} {lv}" for k, lv in HEAVY], fontsize=7, rotation=15)
    ax.set_ylabel("change against control (points)")
    ax.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    plt.savefig(OUT / "dial.pdf")
    plt.show()
    save()

# %% [markdown]
# ## C. A second script: Devanagari
#
# The Devanagari Handwritten Character Dataset (Acharya et al., 2015): 36 consonants and 10 digits,
# 32 x 32 images, 78,200 training and 13,800 test images. Same preprocessing, same recipe, same
# configurations as on TUMMHCD; nothing is tuned on this dataset, and its test set is predicted once.
# If the download fails, put the zip from https://archive.ics.uci.edu/dataset/389 on Drive as
# `MyDrive/redraw/dhcd/dhcd.zip` and run again.

# %%
if RUN_DHCD:
    ZIP_D = Path(os.environ.get("ZIP_D", WORK_D / "dhcd.zip"))
    if not (ZIP_D.exists() and zipfile.is_zipfile(ZIP_D)):
        print("downloading the Devanagari dataset ...")
        download(DHCD_URL, ZIP_D)
    SPLITS_D, summary_d = make_split_folders(ZIP_D, DATA_D)
    print(summary_d)
    INDEX_D, LABELS_D, IDX_D = load_index(SPLITS_D)
    NUM_D = summary_d["classes"]
    NAMES_D = json.loads((SPLITS_D / "classes.json").read_text())
    STORE_D = Store(build_cache(INDEX_D.file.tolist(), IMG, CACHE_D), DEVICE, IDX_D["train"])
    full_d, te_d = IDX_D["full"], IDX_D["test"]
    y_d = LABELS_D[te_d]
    FINAL_D = [("baseline_fp32", s) for s in SEEDS] + [("redraw_hdr_hr", s) for s in SEEDS] + [("redraw_hdr", 0)]
    D = R["dhcd"]
    D["summary"] = summary_d
    for name, s in FINAL_D:
        if not final_path(WORK_D, name, f"full_s{s}").exists():
            t0 = time.time()
            model = train(name, fast(dict(CONFIGS[name], seed=s)), full_d, f"full_s{s}", STORE_D, LABELS_D, WORK_D,
                          num_classes=NUM_D)
            print(f"{name}/full_s{s}: {(time.time() - t0) / 60:.0f} min")
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    tests_d = D.setdefault("test", {})
    for name, s in FINAL_D:
        key = f"{name}/full_s{s}"
        if key in tests_d:
            continue
        cfg = fast(dict(CONFIGS[name], seed=s))
        model = load_model(final_path(WORK_D, name, f"full_s{s}"), cfg, num_classes=NUM_D, device=DEVICE)
        p = cached_preds(WORK_D, f"{name}_full_s{s}", "test_tta", lambda: predict(model, te_d, cfg, STORE_D, views=TTA_VIEWS))
        e = {"test_tta": {h: float((v.argmax(1) == y_d).mean()) for h, v in p.items()},
             "errors": int((p["logits"].argmax(1) != y_d).sum())}
        if cfg.get("strokes"):
            e["parse_test"] = parse_eval(model, te_d, STORE_D)
        if cfg.get("hdr"):
            e["verify_test"] = verify_eval(model, te_d, STORE_D, LABELS_D)
        tests_d[key] = e
        save()
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    P_D = {f"{n}/{s}": cached_preds(WORK_D, f"{n}_full_s{s}", "test_tta", lambda: None)["logits"] for n, s in FINAL_D}
    OK_D = {k: v.argmax(1) == y_d for k, v in P_D.items()}
    D["mcnemar"] = {f"ours_s{s}_vs_control_s{s}": dict(zip(("a_only", "b_only", "p"),
                                                             mcnemar(OK_D[f"redraw_hdr_hr/{s}"], OK_D[f"baseline_fp32/{s}"])))
                    for s in SEEDS}
    for fam in ("baseline_fp32", "redraw_hdr_hr", "redraw_hdr"):
        ks = [k for k in P_D if k.split("/")[0] == fam]
        accs = [float(OK_D[k].mean()) for k in ks]
        avg = np.mean([P_D[k] for k in ks], 0).argmax(1) == y_d
        D.setdefault("families", {})[fam] = {"single": accs, "mean": float(np.mean(accs)),
                                             "sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
                                             "average": float(avg.mean()), "average_ci95": bootstrap_ci(avg)}
    save()
    for fam, v in D["families"].items():
        sd = f" +- {100 * v['sd']:.2f}" if v["sd"] is not None else ""
        print(f"DHCD {fam:15s} single {pct(v['mean'])}{sd}  ({', '.join(pct(a) for a in v['single'])});  average {pct(v['average'])}")
    for k, v in D["mcnemar"].items():
        print(f"DHCD {k}: +{v['a_only']}/-{v['b_only']}, p = {v['p']:.3g}")
    for k, e in tests_d.items():
        extra = f"  parse F {e['parse_test']['f']:.3f}" if "parse_test" in e else ""
        if "verify_test" in e:
            q = e["verify_test"]
            extra += f"  check: top-3 {pct(q['topk_recall'])}, cost alone {pct(q['cost_only'])}, +{q['rescued']}/-{q['broken']}"
        print(f"DHCD {k:24s} {pct(e['test_tta']['logits'])}{extra}")

# %% [markdown]
# ### C2. Devanagari: damaged scans (stronger level of each corruption)

# %%
if RUN_DHCD:
    rob_d = D.setdefault("robustness", {})
    files_d = INDEX_D.file.to_numpy()[te_d].tolist()
    todo = [(k, lv) for k, lv in HEAVY if set(P_D) - set(rob_d.get(f"{k}_{lv}", {}))]
    if todo:
        models = {f"{n}/{s}": (load_model(final_path(WORK_D, n, f"full_s{s}"), fast(dict(CONFIGS[n], seed=s)),
                                          num_classes=NUM_D, device=DEVICE), fast(dict(CONFIGS[n], seed=s)))
                  for n, s in FINAL_D}
        for kind, level in todo:
            arrays = build_cache(files_d, IMG, CACHE_D / "corrupt_test", fn=partial(corrupt_file, kind=kind, level=level),
                                 tag=f"{IMG}_{kind}_{level}")
            damaged = Store(arrays, DEVICE, stats=STORE_D.stats, meta_norm=STORE_D.meta_norm)
            row = rob_d.setdefault(f"{kind}_{level}", {})
            for k, (m_, cfg) in models.items():
                if k not in row:
                    row[k] = float((predict(m_, np.arange(len(te_d)), cfg, damaged, heads=("logits",))["logits"].argmax(1)
                                    == y_d).mean())
            save()
            del damaged, arrays
        del models
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    print(f"{'corruption':18s} {'control':>16s} {'ours':>16s} {'coupled':>10s}   (change against the control mean)")
    for k, lv in HEAVY:
        row = rob_d[f"{k}_{lv}"]
        c = [row[f"baseline_fp32/{s}"] for s in SEEDS]
        o_ = [row[f"redraw_hdr_hr/{s}"] for s in SEEDS]
        print(f"{k + '_' + str(lv):18s} {pct(np.mean(c)):>16s} {100 * (np.mean(o_) - np.mean(c)):+15.2f} "
              f"{100 * (row['redraw_hdr/0'] - np.mean(c)):+10.2f}")
    save()

# %% [markdown]
# ### C3. Devanagari: scarce data (2 and 5% of every class, three seeds)

# %%
LOW_STEPS = 4 if FAST else 2000


def low_cfg(cfg, n_images):
    bs = min(cfg.get("batch", 256), n_images)
    epochs = max(1, math.ceil(LOW_STEPS / max(1, n_images // bs)))
    return dict(cfg, epochs=epochs, val_every=10**9, warmup_steps=max(1, LOW_STEPS // 10),
                ckpt_every=max(1, epochs // 10), log_every=max(1, epochs // 4))


if RUN_DHCD and RUN_DHCD_LOW_DATA:
    low_d = D.setdefault("low_data", {})
    for s in SEEDS:
        for frac in (0.02, 0.05):
            sub = stratified_subset(IDX_D["train"], LABELS_D, frac, seed=s)
            tag = f"low{round(100 * frac):02d}_s{s}"
            for name in ("baseline_fp32", "redraw_hdr_hr"):
                key = f"{name}/{tag}"
                if key in low_d:
                    continue
                cfg = low_cfg(fast(dict(CONFIGS[name], seed=s)), len(sub))
                model = train(name, cfg, sub, tag, STORE_D, LABELS_D, WORK_D, num_classes=NUM_D)
                p = cached_preds(WORK_D, f"{name}_{tag}", "test_tta", lambda: predict(model, te_d, cfg, STORE_D, views=TTA_VIEWS))
                low_d[key] = {"fraction": frac, "seed": s, "images": int(len(sub)),
                              "test_tta": float((p["logits"].argmax(1) == y_d).mean())}
                save()
                del model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
    for frac in (0.02, 0.05):
        c = [low_d[f"baseline_fp32/low{round(100 * frac):02d}_s{s}"]["test_tta"] for s in SEEDS]
        o_ = [low_d[f"redraw_hdr_hr/low{round(100 * frac):02d}_s{s}"]["test_tta"] for s in SEEDS]
        print(f"DHCD {round(100 * frac)}%: control {pct(np.mean(c))}, ours {pct(np.mean(o_))}, "
              f"ours - control per seed {[round(100 * (a - b), 2) for a, b in zip(o_, c)]}")
    save()

# %% [markdown]
# ### C4. Devanagari: parses and the hypothesis check
#
# Test images of the eight classes the control confuses most, with the parse of our network (seed 0)
# and the first template of its two best candidates bent onto the image, with the energy.

# %%
if RUN_DHCD:
    model = load_model(final_path(WORK_D, "redraw_hdr_hr", "full_s0"), fast(dict(CONFIGS["redraw_hdr_hr"], seed=0)),
                       num_classes=NUM_D, device=DEVICE)
    ctl = np.mean([P_D[f"baseline_fp32/{s}"] for s in SEEDS], 0).argmax(1)
    err_by_class = np.bincount(y_d[ctl != y_d], minlength=NUM_D)
    worst = np.argsort(-err_by_class)[:8]
    rng = np.random.default_rng(0)
    pick = np.array([rng.choice(te_d[y_d == c]) for c in worst if (y_d == c).any()])
    x = STORE_D.fetch(pick, "gray")
    with torch.no_grad():
        o = model(STORE_D.normalize(x, "gray"), draw=False, ink=x)
        _, inter = model.backbone.forward_intermediates(STORE_D.normalize(x, "gray"), indices=model.taps)
        mem, mpos = model.decoder(inter)[3:]
        cand = o["cand"][:, :2]
        tmpl, fitted = model.hdr.fit(mem, mpos, cand)
        target = F.interpolate(x, size=(model.hdr_G, model.hdr_G), mode="area").squeeze(1)
        bent = model.hdr.bend(tmpl, fitted)
        cost = model.hdr.cost(bent, tmpl[2], fitted, target, G=model.hdr_G).cpu().numpy()
    ppts = curve_points(o["ctrl"].float(), 64).cpu().numpy()
    pw, pe = o["width"].float().cpu().numpy(), o["exist"].float().cpu().numpy()
    bpts = curve_points(bent[0].float(), 64).cpu().numpy()
    bwidth, bexist = bent[1].float().cpu().numpy(), torch.sigmoid(bent[2]).float().cpu().numpy()
    fig, axes = plt.subplots(int(np.ceil(len(pick) / 2)), 8, figsize=(11, 1.4 * np.ceil(len(pick) / 2)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i in range(len(pick)):
        r_, c_ = divmod(i, 2)
        axes[r_, 4 * c_].imshow(1 - x[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[r_, 4 * c_].set_title(NAMES_D[LABELS_D[pick[i]]], fontsize=6)
        ax = axes[r_, 4 * c_ + 1]
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
        for k in np.argsort(-pe[i]):
            if pe[i, k] > 0.5:
                ax.plot(ppts[i, k, :, 0], ppts[i, k, :, 1], color=colors[k % 20], lw=max(0.5, pw[i, k] * 60), solid_capstyle="round")
        ax.set_title("parse", fontsize=6)
        chosen = int(np.argmin(cost[i]))
        for j in range(2):
            ax = axes[r_, 4 * c_ + 2 + j]
            ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
            for k in range(bpts.shape[3]):
                if bexist[i, j, 0, k] > 0.5:
                    ax.plot(bpts[i, j, 0, k, :, 0], bpts[i, j, 0, k, :, 1], color=colors[k % 20],
                            lw=max(0.5, bwidth[i, j, 0, k] * 40), solid_capstyle="round")
            ok_ = cand[i, j].item() == LABELS_D[pick[i]]
            ax.set_title(f"as {NAMES_D[cand[i, j].item()][:14]}: {cost[i, j]:.3f}", fontsize=5.5,
                         color="black" if j != chosen else ("green" if ok_ else "red"))
    plt.tight_layout()
    plt.savefig(OUT / "dhcd_parses.png", dpi=150)
    plt.show()
    ctrl_d, tw_d, tp_d = (t.detach().float().cpu() for t in model.hdr.template_params())
    tp_d = torch.sigmoid(tp_d).numpy()
    cols = 10
    rows_n = int(np.ceil(NUM_D / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 1.05, rows_n * 1.15), squeeze=False)
    for c, ax in enumerate(axes.flat):
        if c < NUM_D:
            draw_template(ax, ctrl_d[c, 0], tw_d[c, 0], tp_d[c, 0], NAMES_D[c][:14])
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "dhcd_templates.pdf")
    plt.show()
    del model

# %% [markdown]
# ## Summary
#
# All numbers are in `MyDrive/redraw/extras/results.json`, with the figures next to it
# (`templates_all.pdf`, `vector_redraw.pdf`, `erase.pdf`, `erase_examples.png`, `dial.pdf`,
# `dhcd_parses.png`, `dhcd_templates.pdf`).

# %%
print("results:", RESULTS)
print(json.dumps({k: v for k, v in R.items() if k != "dhcd"}, indent=1, default=float)[:3000])
if RUN_DHCD:
    print(json.dumps({k: v for k, v in R["dhcd"].items() if k in ("families", "mcnemar", "summary")}, indent=1, default=float))
