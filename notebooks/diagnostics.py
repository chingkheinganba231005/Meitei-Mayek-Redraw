# %% [markdown]
# # Diagnostics: where could the remaining accuracy come from?
#
# Rounds 1 and 2 showed that none of the stroke-based variants beats plain
# fine-tuning on validation (all within about 0.1%). Before building anything
# else, this notebook measures, on the **validation split only**, three possible
# sources of the missing accuracy:
#
# 1. **The corrected baseline.** Rounds 1-2 computed the classification loss in
#    bfloat16 (a bug, the same for every network). `baseline_fp32` retrains the
#    baseline with a float32 loss: about 11 minutes.
# 2. **Diversity.** How much do the eight trained networks disagree? Accuracy of
#    their average, how many images all of them get wrong, and per hard pair.
# 3. **Memory.** TUMMHCD was split at random per class, so most validation images
#    have training images by the same writer (each wrote every character several
#    times). Does a nearest-neighbour memory of the training set, in the
#    network's own feature space, fix errors the classifier makes, especially on
#    the pair 044/025? The combination is tuned on one half of validation and
#    scored on the other half (and the other way round), so the gain is not
#    tuned on the images it is measured on.
#
# About 15 minutes on a large Colab GPU in total. Results go to `MyDrive/redraw/diagnostics/`.

# %%
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
FAST = os.environ.get("FAST") == "1"

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0.9", "scikit-image"], check=True)

REPO_URL = os.environ.get("REPO_URL", "https://github.com/chingkheinganba231005/Meitei-Mayek-Redraw.git")
REPO = Path(os.environ.get("REPO", "/content/redraw"))
CODE_ZIP = Path(os.environ.get("CODE_ZIP", "/content/drive/MyDrive/redraw_code.zip"))


def get_code():
    if IN_COLAB:
        shutil.rmtree(REPO, ignore_errors=True)
    if REPO.exists():
        return "local copy"
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    if CODE_ZIP.exists():
        tmp = REPO.parent / "_code"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(CODE_ZIP) as z:
            z.extractall(tmp)
        shutil.move(str(next(tmp.rglob("redraw/__init__.py")).parent.parent), str(REPO))
        return f"zip {CODE_ZIP}"
    if subprocess.run(["git", "clone", "-q", "--depth", "1", REPO_URL, str(REPO)], capture_output=True,
                      env=env).returncode == 0 and (REPO / "redraw" / "__init__.py").exists():
        return "GitHub"
    raise SystemExit("Could not get the code (no zip on Drive, and the repository could not be cloned).")


print("code from", get_code())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
for _name in [n for n in sys.modules if n == "redraw" or n.startswith("redraw.")]:
    del sys.modules[_name]

DATA = Path(os.environ.get("DATA", "/content/data"))
CACHE = Path(os.environ.get("CACHE", "/content/cache"))
WORK = Path(os.environ.get("WORK", "/content/drive/MyDrive/redraw"))
OUT = WORK / "diagnostics"
for folder in (DATA, CACHE, OUT):
    folder.mkdir(parents=True, exist_ok=True)

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from redraw import __version__
from redraw.augment import TTA_VIEWS
from redraw.data import Store, build_cache, load_index
from redraw.model import CONFIGS
from redraw.split import URL, download, make_split
from redraw.train import amp_dtype, cached_preds, load_model, predict, train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
print("redraw", __version__, "|", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
IMG = 64 if FAST else 128
PAIRS = [(44, 25), (46, 9), (11, 47), (33, 34)]
RUNS = ["baseline", "redraw", "redraw_norender", "redraw_hdr", "redraw_hdr_sw", "redraw_hdr_e2e",
        "redraw_hdr_nohead", "baseline_fp32"]
D = {}


def save():
    (OUT / "diagnostics.json").write_text(json.dumps(D, indent=2, default=float))


def pct(x):
    return f"{100 * x:.2f}%"

# %%
ZIP = Path(os.environ.get("ZIP", WORK / "TUMMHCD-TEST-TRAIN.zip"))
if not (ZIP.exists() and zipfile.is_zipfile(ZIP)):
    found = [p for d in ("*", "*/*", "*/*/*") for p in Path("/content/drive/MyDrive").glob(f"{d}/TUMMHCD-TEST-TRAIN.zip")]
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
tr, va = IDX["train"], IDX["val"]
y_va = LABELS[va]

# %% [markdown]
# ## 1. The corrected baseline (float32 loss)

# %%
cfg_fp32 = dict(CONFIGS["baseline_fp32"])
if FAST:
    cfg_fp32 = dict(cfg_fp32, arch="resnet18", epochs=1, pretrained=False, batch=16)
model = train("baseline_fp32", cfg_fp32, tr, "dev", STORE, LABELS, WORK, val_idx=va)
p = cached_preds(WORK, "baseline_fp32_dev", "val_tta", lambda: predict(model, va, cfg_fp32, STORE, views=TTA_VIEWS))
cached_preds(WORK, "baseline_fp32_dev", "val_id", lambda: predict(model, va, cfg_fp32, STORE))
D["baseline_fp32_val_tta"] = float((p["logits"].argmax(1) == y_va).mean())
print("baseline_fp32 validation accuracy (TTA):", pct(D["baseline_fp32_val_tta"]))
save()

# %% [markdown]
# ## 2. Diversity of the eight networks

# %%
P = {}
for n in RUNS:
    f = WORK / "preds" / f"{n}_dev_val_tta.npz"
    if f.exists():
        P[n] = cached_preds(WORK, f"{n}_dev", "val_tta", lambda: None)["logits"]
wrong = {n: P[n].argmax(1) != y_va for n in P}
acc = {n: float(1 - w.mean()) for n, w in wrong.items()}
avg = np.mean([P[n] for n in P], 0)
n_wrong = np.sum([wrong[n] for n in P], 0)
D["diversity"] = {
    "networks": list(P),
    "single": acc,
    "average_of_all": float((avg.argmax(1) == y_va).mean()),
    "any_right": float((n_wrong < len(P)).mean()),
    "wrong_by_all": int((n_wrong == len(P)).sum()),
    "wrong_by_some_not_all": int(((n_wrong > 0) & (n_wrong < len(P))).sum()),
    "wrong_by_all_per_pair": {f"{a:03d}/{b:03d}": int(((n_wrong == len(P)) & np.isin(y_va, [a, b])).sum())
                              for a, b in PAIRS},
}
for n in P:
    print(f"{n:18s} {pct(acc[n])}")
print("average of all:", pct(D["diversity"]["average_of_all"]), "| some network right:", pct(D["diversity"]["any_right"]))
print("wrong by all networks:", D["diversity"]["wrong_by_all"], "per pair:", D["diversity"]["wrong_by_all_per_pair"])
save()

# %% [markdown]
# ## 3. Memory: nearest training images in the network's feature space

# %%
@torch.no_grad()
def features(net, idx, bs=1024):
    """Pooled features (the input of the classifier), L2-normalised."""
    net.eval()
    amp = amp_dtype(DEVICE)
    out = []
    for i in range(0, len(idx), bs):
        x = STORE.normalize(STORE.fetch(idx[i:i + bs], "gray"), "gray")
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            f = net.backbone(x)
        out.append(F.normalize(f.float(), dim=1))
    return torch.cat(out)


f_tr, f_va = features(model, tr), features(model, va)
K = 50
sims, nbrs = [], []
for i in range(0, len(f_va), 2048):
    s, j = (f_va[i:i + 2048] @ f_tr.T).topk(min(K, len(tr)), dim=1)
    sims.append(s.cpu())
    nbrs.append(j.cpu())
sims, nbrs = torch.cat(sims).numpy(), torch.cat(nbrs).numpy()
lab_nb = LABELS[tr][nbrs]                                    # (N_val, K) labels of the neighbours


def knn_probs(k, tau):
    w = np.exp((sims[:, :k] - sims[:, :1]) / tau)
    out = np.zeros((len(va), 55))
    np.add.at(out, (np.arange(len(va))[:, None].repeat(k, 1), lab_nb[:, :k]), w)
    return out / out.sum(1, keepdims=True)


clf = p["logits"]
grid = [(k, tau, lam) for k in (1, 5, 10, 20, 50) if k <= K for tau in (0.01, 0.03, 0.1) for lam in (0.1, 0.2, 0.3, 0.5, 0.7)]
cache = {(k, tau): knn_probs(k, tau) for k in (1, 5, 10, 20, 50) if k <= K for tau in (0.01, 0.03, 0.1)}
D["knn_alone"] = {f"k{k}_t{tau}": float((v.argmax(1) == y_va).mean()) for (k, tau), v in cache.items()}
half = np.random.default_rng(0).permutation(len(va)) < len(va) // 2
cv = []
for pick_on, score_on in ((half, ~half), (~half, half)):
    best = max(grid, key=lambda g: (((1 - g[2]) * clf[pick_on] + g[2] * cache[g[:2]][pick_on]).argmax(1)
                                     == y_va[pick_on]).mean())
    comb = (1 - best[2]) * clf + best[2] * cache[best[:2]]
    cv.append({"chosen": best, "classifier": float((clf[score_on].argmax(1) == y_va[score_on]).mean()),
               "combined": float((comb[score_on].argmax(1) == y_va[score_on]).mean())})
D["memory_cv"] = cv
gain = np.mean([c["combined"] - c["classifier"] for c in cv])
print("kNN alone (best):", pct(max(D["knn_alone"].values())))
for c in cv:
    print(f"  chosen k={c['chosen'][0]}, tau={c['chosen'][1]}, weight={c['chosen'][2]}: classifier {pct(c['classifier'])} -> with memory {pct(c['combined'])}")
print("held-out gain from memory:", f"{100 * gain:+.2f} points")

# the combination chosen on the whole of validation, only to look at where it helps (not a result)
best_all = max(grid, key=lambda g: (((1 - g[2]) * clf + g[2] * cache[g[:2]]).argmax(1) == y_va).mean())
comb = (1 - best_all[2]) * clf + best_all[2] * cache[best_all[:2]]
D["memory_per_pair"] = {}
for a, b in PAIRS:
    m = np.isin(y_va, [a, b])
    D["memory_per_pair"][f"{a:03d}/{b:03d}"] = {"classifier": float((clf[m].argmax(1) == y_va[m]).mean()),
                                               "with_memory": float((comb[m].argmax(1) == y_va[m]).mean())}
err = clf.argmax(1) != y_va
D["neighbours"] = {
    "top1_similarity_percentiles": np.percentile(sims[:, 0], [5, 25, 50, 75, 95]).tolist(),
    "top1_same_label_all": float((lab_nb[:, 0] == y_va).mean()),
    "top1_same_label_on_classifier_errors": float((lab_nb[err, 0] == y_va[err]).mean()),
    "top1_similarity_on_errors_median": float(np.median(sims[err, 0])),
    "top1_similarity_on_correct_median": float(np.median(sims[~err, 0])),
}
print(json.dumps(D["memory_per_pair"], indent=1))
print(json.dumps(D["neighbours"], indent=1))
save()

# %% [markdown]
# ## 4. Classifier errors and their nearest training images
#
# Each row: a validation image the classifier gets wrong (true / predicted
# class), then its five nearest training images (class, cosine similarity).

# %%
rng = np.random.default_rng(0)
err_idx = np.flatnonzero(err)
show = np.concatenate([rng.permutation(err_idx[np.isin(y_va[err_idx], [44, 25])])[:8],
                       rng.permutation(err_idx[~np.isin(y_va[err_idx], [44, 25])])[:8]])
fig, axes = plt.subplots(len(show), 6, figsize=(7.5, 1.3 * len(show)), squeeze=False)
for r, i in enumerate(show):
    imgs = [STORE.fetch(va[i:i + 1], "gray")] + [STORE.fetch(tr[nbrs[i, j]:nbrs[i, j] + 1], "gray") for j in range(5)]
    titles = [f"{y_va[i]:03d}->{clf[i].argmax():03d}"] + [f"{lab_nb[i, j]:03d} {sims[i, j]:.2f}" for j in range(5)]
    for c, (im, t) in enumerate(zip(imgs, titles)):
        ax = axes[r, c]
        ax.imshow(1 - im[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=6, color="red" if c and lab_nb[i, c - 1] != y_va[i] else "black")
        ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "errors_and_neighbours.png", dpi=120)
plt.show()
print("results:", OUT / "diagnostics.json")
