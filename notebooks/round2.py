# %% [markdown]
# # Reading by redrawing, round 2: the learning methods
#
# Round 1 trained `baseline`, `redraw`, `redraw_hdr` and `redraw_norender`; they are
# reused here, not retrained (if round 1 has not been run, they are trained first).
# Round 2 adds three networks, all compared on the **validation split only**:
#
# | name | question it answers |
# |---|---|
# | `redraw_hdr_sw` | does StrokeWarp help? Half of each batch is deformed stroke by stroke along the network's own parse: inner control points move (curvature), free tails grow or shrink, joints and pen width stay |
# | `redraw_hdr_e2e` | does honest training matter? The same network trained end to end, so the fitter may learn to fit wrong classes badly and every template learns from every class |
# | `redraw_hdr_nohead` | does the stroke head earn its place? The CNN proposes and the redrawing verifies, nothing else |
#
# About 4 hours on an A100 for the three new networks (if time is short, the
# first two matter most; remove `redraw_hdr_nohead` from `NEW` to skip it).
# Finished models are reused after a disconnect: just run all cells again.

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
OUT = WORK / "round2"
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
from redraw.strokewarp import stroke_warp
from redraw.train import cached_preds, load_model, parse_eval, predict, train, verify_eval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("redraw", __version__, "|", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

IMG = 64 if FAST else 128
OLD = ["baseline", "redraw", "redraw_hdr", "redraw_norender"]          # from round 1
NEW = ["redraw_hdr_sw", "redraw_hdr_e2e", "redraw_hdr_nohead"]
RUNS = OLD + NEW
CFG = {k: dict(CONFIGS[k]) for k in RUNS}
if FAST:
    CFG = {k: dict(v, arch="resnet18", epochs=2, pretrained=False, batch=16, K=4, hdr_S=4) for k, v in CFG.items()}
    for k in CFG:
        if "strokewarp" in CFG[k]:
            CFG[k]["strokewarp"] = dict(CFG[k]["strokewarp"], warmup=0)
PAIRS = [(44, 25), (46, 9), (11, 47), (33, 34)]

RESULTS = OUT / "results.json"
R = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
R.setdefault("runs", {})


def save():
    RESULTS.write_text(json.dumps(R, indent=2, default=float))


def pct(x):
    return f"{100 * x:.2f}%"

# %% [markdown]
# ## 1. Data (the same split as round 1)

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
save()

# %% [markdown]
# ## 2. Train (new networks) and evaluate (all)
#
# For every network with the hypothesis check, `verify` reports what the check
# contributes on validation: how often the true class is among the 3 candidates,
# how often the cheapest redrawing alone is right, and how many images the check
# turns from wrong to right (`rescued`) or right to wrong (`broken`).

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
    if cfg.get("hdr"):
        entry["verify_val"] = verify_eval(model, va, STORE, LABELS)
    R["runs"][name] = entry
    save()
    print(name, json.dumps({k: entry[k] for k in ("val", "val_errors_tta", "pairs_tta", "train_minutes")
                            + (("verify_val",) if "verify_val" in entry else ())}, indent=1))
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# %% [markdown]
# ## 3. Summary

# %%
for name in RUNS:
    e = R["runs"][name]
    line = f"{name:18s}"
    for v in ("id", "tta"):
        for h, label in (("logits", "final"), ("fused", "fused"), ("img", "img"), ("vec", "vec")):
            acc = e["val"][v].get(h)
            line += f"  {v}/{label}: {pct(acc) if acc is not None else '-':>7s}"
    print(line)
print()
for name in RUNS:
    e = R["runs"][name]
    extra = ""
    if "verify_val" in e:
        q = e["verify_val"]
        extra = (f"   check: top-3 recall {pct(q['topk_recall'])}, cost alone {pct(q['cost_only'])}, "
                 f"rescued {q['rescued']}, broken {q['broken']}")
    print(f"{name:18s} errors {e['val_errors_tta']:4d}  " + "  ".join(f"{k} {pct(v)}" for k, v in e["pairs_tta"].items()) + extra)
print()
y = LABELS[va]
ok = {n: cached_preds(WORK, f"{n}_dev", "val_tta", lambda: None)["logits"].argmax(1) == y for n in RUNS}
R["mcnemar_val_tta"] = {}
for a, b in (("redraw_hdr_sw", "redraw_hdr"), ("redraw_hdr", "redraw_hdr_e2e"), ("redraw_hdr_nohead", "redraw_hdr"),
             ("redraw_hdr_sw", "baseline"), ("redraw_hdr", "baseline")):
    only_a, only_b = int((ok[a] & ~ok[b]).sum()), int((~ok[a] & ok[b]).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    R["mcnemar_val_tta"][f"{a}_vs_{b}"] = {"a_only": only_a, "b_only": only_b, "p": p}
    print(f"{a} vs {b}: only {a} right on {only_a} images, only {b} right on {only_b} (p = {p:.3g})")
save()

# %% [markdown]
# ## 4. StrokeWarp on validation images
#
# Each row: a validation image, then three StrokeWarp versions of it, made with
# the strokes the `redraw_hdr_sw` network finds on it.

# %%
model = load_model(WORK / "runs" / "redraw_hdr_sw" / "dev" / "final.pt", CFG["redraw_hdr_sw"], device=DEVICE)
rng = np.random.default_rng(1)
classes = [c for pair in PAIRS for c in pair] + list(rng.choice(55, 4, replace=False))
pick = np.array([rng.choice(va[LABELS[va] == c]) for c in classes])
x = STORE.fetch(pick, "gray")
with torch.no_grad():
    o = model(STORE.normalize(x, "gray"), draw=False)
sw = dict(CFG["redraw_hdr_sw"]["strokewarp"])
warps = [stroke_warp(x, o["ctrl"], o["exist"], **sw) for _ in range(3)]
fig, axes = plt.subplots(len(pick) // 2, 8, figsize=(10, len(pick) // 2 * 1.3), squeeze=False)
for i in range(len(pick)):
    r, c = divmod(i, 2)
    for j, im in enumerate([x] + warps):
        ax = axes[r, 4 * c + j]
        ax.imshow(1 - im[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if j == 0:
            ax.set_title(f"{LABELS[pick[i]]:03d}", fontsize=7)
plt.tight_layout()
plt.savefig(OUT / "strokewarp.png", dpi=130)
plt.show()

# %% [markdown]
# ## 5. Honest versus end-to-end templates
#
# The first template of the eight classes of the hard pairs, learned with honest
# training (top) and end to end (bottom).

# %%
colors = plt.cm.tab20(np.linspace(0, 1, 20))
hard = [c for pair in PAIRS for c in pair]
fig, axes = plt.subplots(2, len(hard), figsize=(len(hard) * 1.3, 3))
for row, name in enumerate(("redraw_hdr", "redraw_hdr_e2e")):
    m = load_model(WORK / "runs" / name / "dev" / "final.pt", CFG[name], device=DEVICE)
    ctrl, width, prior = (t.detach().float().cpu() for t in m.hdr.template_params())
    tp = curve_points(ctrl, 64).numpy()
    written = torch.sigmoid(prior).numpy()
    for j, c in enumerate(hard):
        ax = axes[row, j]
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{name.replace('redraw_hdr', 'honest').replace('honest_e2e', 'end to end')} {c:03d}", fontsize=6)
        for s in range(tp.shape[2]):
            ax.plot(tp[c, 0, s, :, 0], tp[c, 0, s, :, 1], color=colors[s % 20],
                    lw=max(0.5, float(width[c, 0, s]) * 50), alpha=float(np.clip(written[c, 0, s], 0.05, 1)))
    del m
plt.tight_layout()
plt.savefig(OUT / "templates_honest_vs_e2e.png", dpi=130)
plt.show()
print("results:", RESULTS)
