# %% [markdown]
# # Reading by redrawing, round 1: does redrawing help?
#
# Four networks with the same backbone (ConvNeXt-T), the same recipe and the
# same 40 epochs. They are compared on the **validation split only**; the test
# set is not touched in this round.
#
# | name | what it is |
# |---|---|
# | `redraw` | stroke decoder, renderer and stroke head added to the CNN |
# | `redraw_hdr` | `redraw` plus the hypothesis check: the top 3 candidates are redrawn from their class's learned stroke templates and re-ranked by how well that works |
# | `baseline` | the plain fine-tuned network |
# | `redraw_norender` | `redraw`, but its strokes are never compared with the ink (are the gains from redrawing, or just from extra layers?) |
#
# About 3 to 3.5 hours on an A100. Every model is saved to Google Drive, so after
# a disconnect just run all cells again: finished models are reused and an
# interrupted one continues from its last checkpoint. (If you already ran an
# earlier version of this notebook, its finished models are reused too.)
#
# **Before running.** Runtime → Change runtime type → **A100 GPU**. The dataset
# zip (`TUMMHCD-TEST-TRAIN.zip`) is found automatically if it is anywhere in the
# top three folder levels of your Drive (for example in `tummhcd98` from the
# first project). The code comes from `redraw_code.zip` at the top level of My
# Drive, or from the GitHub repository once it exists (a private repository
# needs a Colab secret `GH_TOKEN`, key icon on the left; it is never printed).

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

    # the zip on Drive wins while it exists, so a newer zip is never shadowed by an older repository
    if CODE_ZIP.exists():
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

DATA = Path(os.environ.get("DATA", "/content/data"))                       # unpacked dataset + split CSVs
CACHE = Path(os.environ.get("CACHE", "/content/cache"))                    # preprocessed image arrays
WORK = Path(os.environ.get("WORK", "/content/drive/MyDrive/redraw"))       # checkpoints and predictions
OUT = WORK / "round1"
for folder in (DATA, CACHE, OUT):
    folder.mkdir(parents=True, exist_ok=True)

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binomtest

from redraw import __version__
from redraw.augment import TTA_VIEWS
from redraw.data import Store, build_cache, load_index
from redraw.model import CONFIGS
from redraw.render import curve_points
from redraw.split import URL, download, make_split
from redraw.train import cached_preds, load_model, parse_eval, predict, train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("redraw", __version__, "|", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

IMG = 64 if FAST else 128
RUNS = ["redraw", "redraw_hdr", "baseline", "redraw_norender"]  # the methods first, so a problem shows early
CFG = {k: dict(CONFIGS[k]) for k in RUNS}
if FAST:
    CFG = {k: dict(v, arch="resnet18", epochs=1, pretrained=False, batch=16, K=4, hdr_S=4) for k, v in CFG.items()}
PAIRS = [(44, 25), (46, 9), (11, 47), (33, 34)]

RESULTS = OUT / "results.json"
R = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
R.setdefault("runs", {})


def save():
    RESULTS.write_text(json.dumps(R, indent=2, default=float))


def pct(x):
    return f"{100 * x:.2f}%"

# %% [markdown]
# ## 1. Data
#
# Same split as the first project: 15% of every training class for validation,
# seed 42, giving 61,504 / 10,826 / 12,794 images.

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
ARRAYS = build_cache(INDEX.file.tolist(), IMG, CACHE)
STORE = Store(ARRAYS, DEVICE, IDX["train"])
va = IDX["val"]
R["data"] = {s: int(len(i)) for s, i in IDX.items()}
save()

# %% [markdown]
# ## 2. Train and compare
#
# Each network is trained on the training split and evaluated on validation,
# with and without test-time augmentation (5 views). Reported per network:
# `final` (what the network outputs), `fused` (image head + stroke head, before
# the hypothesis check), and each head alone.

# %%
def accuracy(p, idx):
    return float((p.argmax(1) == LABELS[idx]).mean())


def pair_accuracy(p, idx, a, b):
    y = LABELS[idx]
    m = (y == a) | (y == b)
    return float((p[m].argmax(1) == y[m]).mean())


for name in RUNS:
    cfg = CFG[name]
    if "val" in R["runs"].get(name, {}):
        print(f"{name}: done before, reusing")
        continue
    model = train(name, cfg, IDX["train"], "dev", STORE, LABELS, WORK, val_idx=va)
    p_id = cached_preds(WORK, f"{name}_dev", "val_id", lambda: predict(model, va, cfg, STORE))
    p_tta = cached_preds(WORK, f"{name}_dev", "val_tta", lambda: predict(model, va, cfg, STORE, views=TTA_VIEWS))
    history = torch.load(WORK / "runs" / name / "dev" / "final.pt", map_location="cpu", weights_only=False)["history"]
    entry = {
        "cfg": cfg,
        "val": {v: {h: accuracy(p[h], va) for h in p} for v, p in (("id", p_id), ("tta", p_tta))},
        "val_errors_tta": int((p_tta["logits"].argmax(1) != LABELS[va]).sum()),
        "pairs_tta": {f"{a:03d}/{b:03d}": pair_accuracy(p_tta["logits"], va, a, b) for a, b in PAIRS},
        "train_minutes": round(sum(h["sec"] for h in history) / 60, 1),
        "history": history,
    }
    if cfg.get("strokes") and cfg.get("render_weight", 1.0) > 0:
        entry["parse_val"] = parse_eval(model, va, STORE)
    R["runs"][name] = entry
    save()
    print(name, json.dumps({k: entry[k] for k in ("val", "val_errors_tta", "pairs_tta", "train_minutes")}, indent=1))
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# %% [markdown]
# ## 3. Summary
#
# The McNemar tests count the validation images where exactly one of two
# networks is right (with test-time augmentation).

# %%
for name in RUNS:
    e = R["runs"][name]
    line = f"{name:16s}"
    for v in ("id", "tta"):
        for h, label in (("logits", "final"), ("fused", "fused"), ("img", "img"), ("vec", "vec")):
            acc = e["val"][v].get(h)
            line += f"  {v}/{label}: {pct(acc) if acc is not None else '-':>7s}"
    if "parse_val" in e:
        line += f"  parse F {e['parse_val']['f']:.3f}, {e['parse_val']['strokes']:.1f} strokes"
    print(line)
print()
for name in RUNS:
    e = R["runs"][name]
    print(f"{name:16s} errors {e['val_errors_tta']:4d}   hard pairs: "
          + "  ".join(f"{k} {pct(v)}" for k, v in e["pairs_tta"].items()))
print()
y = LABELS[va]
ok = {n: cached_preds(WORK, f"{n}_dev", "val_tta", lambda: None)["logits"].argmax(1) == y for n in RUNS}
R["mcnemar_val_tta"] = {}
for a, b in (("redraw_hdr", "baseline"), ("redraw_hdr", "redraw"), ("redraw", "baseline"), ("redraw", "redraw_norender")):
    only_a, only_b = int((ok[a] & ~ok[b]).sum()), int((~ok[a] & ok[b]).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    R["mcnemar_val_tta"][f"{a}_vs_{b}"] = {"a_only": only_a, "b_only": only_b, "p": p}
    print(f"{a} vs {b}: only {a} right on {only_a} images, only {b} right on {only_b} (p = {p:.3g})")
save()

# %% [markdown]
# ## 4. What the network redraws
#
# Left: validation images. Right: the free strokes the `redraw_hdr` network put
# on them (one colour per stroke). Rows: the four hard pairs, then random classes.

# %%
model = load_model(WORK / "runs" / "redraw_hdr" / "dev" / "final.pt", CFG["redraw_hdr"], device=DEVICE)
rng = np.random.default_rng(0)
classes = [c for pair in PAIRS for c in pair] + list(rng.choice(55, 4, replace=False))
pick = np.concatenate([rng.choice(va[LABELS[va] == c], min(6, int((LABELS[va] == c).sum())), replace=False)
                       for c in classes])
x = STORE.fetch(pick, "gray")
with torch.no_grad():
    o = model(STORE.normalize(x, "gray"), ink=x)
pts = curve_points(o["ctrl"].float(), 64).cpu().numpy()
exist, width = o["exist"].float().cpu().numpy(), o["width"].float().cpu().numpy()
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


cols = 12
rows_n = int(np.ceil(len(pick) / (cols // 2)))
fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 1.2, rows_n * 1.25), squeeze=False)
for ax in axes.flat:
    ax.axis("off")
for i in range(len(pick)):
    r, c = divmod(i, cols // 2)
    axes[r, 2 * c].imshow(1 - x[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[r, 2 * c].set_title(f"{LABELS[pick[i]]:03d}", fontsize=7)
    draw_strokes(axes[r, 2 * c + 1], pts[i], width[i], exist[i])
plt.tight_layout()
plt.savefig(OUT / "parses.png", dpi=130)
plt.show()

# %% [markdown]
# ## 5. The learned class templates
#
# The first of the two templates of every class, as learned from the training
# images (stroke opacity = how likely the stroke is to be written).

# %%
ctrl, twidth, prior = (t.detach().float().cpu() for t in model.hdr.template_params())
tpts = curve_points(ctrl, 64).numpy()
p_written = torch.sigmoid(prior).numpy()
fig, axes = plt.subplots(5, 11, figsize=(13, 6.2))
for c, ax in enumerate(axes.flat):
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    if c >= tpts.shape[0]:
        continue
    ax.set_title(f"{c:03d}", fontsize=7)
    for s in range(tpts.shape[2]):
        ax.plot(tpts[c, 0, s, :, 0], tpts[c, 0, s, :, 1], color=colors[s % 20],
                lw=max(0.5, float(twidth[c, 0, s]) * 50), alpha=float(np.clip(p_written[c, 0, s], 0.05, 1)))
plt.tight_layout()
plt.savefig(OUT / "templates.png", dpi=130)
plt.show()

# %% [markdown]
# ## 6. The hypothesis check at work
#
# Validation images of the hard pairs: the image, then the first template of
# each class of its pair bent onto it by the fitter, with the cost (lower =
# better explanation). A green cost marks the hypothesis the check prefers when
# that is the right class; red marks a wrong preference.

# %%
pairs_pick = np.concatenate([rng.choice(va[LABELS[va] == c], min(3, int((LABELS[va] == c).sum())), replace=False)
                             for pair in PAIRS for c in pair])
x = STORE.fetch(pairs_pick, "gray")
cand = torch.tensor([next(p for p in PAIRS if LABELS[i] in p) for i in pairs_pick], device=DEVICE)
with torch.no_grad():
    _, inter = model.backbone.forward_intermediates(STORE.normalize(x, "gray"), indices=model.taps)
    mem, mpos = model.decoder(inter)[3:]
    tmpl, fitted = model.hdr.fit(mem, mpos, cand)
    target = F.interpolate(x, size=(model.hdr_G, model.hdr_G), mode="area").squeeze(1)
    bent = model.hdr.bend(tmpl, fitted)
    cost = model.hdr.cost(bent, tmpl[2], fitted, target, G=model.hdr_G).cpu().numpy()
bpts = curve_points(bent[0].float(), 64).cpu().numpy()             # (B, 2, M, S, 64, 2)
bwidth, bexist = bent[1].float().cpu().numpy(), torch.sigmoid(bent[2]).float().cpu().numpy()
rows_n = int(np.ceil(len(pairs_pick) / 3))
fig, axes = plt.subplots(rows_n, 9, figsize=(12, rows_n * 1.45), squeeze=False)
for ax in axes.flat:
    ax.axis("off")
for i in range(len(pairs_pick)):
    r, c = divmod(i, 3)
    axes[r, 3 * c].imshow(1 - x[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[r, 3 * c].set_title(f"{LABELS[pairs_pick[i]]:03d}", fontsize=7)
    chosen = int(np.argmin(cost[i]))
    for j in range(2):
        ax = axes[r, 3 * c + 1 + j]
        draw_strokes(ax, bpts[i, j, 0], bwidth[i, j, 0], bexist[i, j, 0], scale=40)
        right = cand[i, j].item() == LABELS[pairs_pick[i]]
        color = "black" if j != chosen else ("green" if right else "red")
        ax.set_title(f"as {cand[i, j].item():03d}: {cost[i, j]:.3f}", fontsize=6, color=color)
plt.tight_layout()
plt.savefig(OUT / "hypotheses.png", dpi=130)
plt.show()
print("results:", RESULTS)
