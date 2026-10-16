# %% [markdown]
# # Revision 3: the reconstruction, the explanation and the check
#
# These analyses use the final networks (seeds 0-2 of the method and the control) and the test set
# the paper already reports; no network is chosen here. Settings are fixed in advance: foreground is a
# value above 0.5 on a 64 px canvas, tolerances are 0, 1 and 1.5 px, and the skeleton tolerance and
# the stroke budgets are reported as curves, not tuned.
#
# | | what | GPU time |
# |---|---|---|
# | 1 | reconstruction: our parses (all strokes and the k most certain), skeleton tracing with Bezier fitting at four tolerances, per-image optimisation of 4-16 strokes with our renderer and loss; overlap, tolerant precision/recall/F, centreline distance, curve counts, length, overlap between strokes, time per image | 45 min |
# | 2 | stability: parses of shifted, rotated and zoomed images mapped back, and parses of the three seeds | 5 min |
# | 3 | probes: linear and small MLP classifiers on the frozen stroke geometry and features | 10 min |
# | 4 | interventions: erasing the ink of one stroke chosen six ways (including random and ink-matched controls), leaving a stroke out of what the reader sees, and a classifier trained on the drawings alone | 45 min |
# | 5 | the check: what it changes per seed, energy margins, selective prediction, and refitting the fitter by gradient descent (validation) | 10 min |
# | 6 | figures with fixed selection rules | 5 min |
#
# Results: `MyDrive/redraw/revision/` (`recon.json`, `stability.json`, `probes.json`,
# `interventions.json`, `checker.json`, `figures/`), per-image arrays next to them.

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0.9", "scikit-image", "pytest"], check=True)

REPO_URL = os.environ.get("REPO_URL", "https://github.com/chingkheinganba231005/meitei-mayek-redraw.git")
BRANCH = os.environ.get("BRANCH", "revision")
REPO = Path(os.environ.get("REPO", "/content/redraw"))
CODE_ZIP = Path(os.environ.get("CODE_ZIP", "/content/drive/MyDrive/redraw_code.zip"))


def get_code():
    if IN_COLAB:
        shutil.rmtree(REPO, ignore_errors=True)  # always start from the current code
    if REPO.exists():
        return "local copy"
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")  # fail instead of asking for a password

    def cloned(url):
        cmd = ["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, url, str(REPO)]
        ok = subprocess.run(cmd, capture_output=True, env=env).returncode == 0
        if ok and not (REPO / "redraw" / "experiments.py").exists():  # an unrelated or older repository
            shutil.rmtree(REPO, ignore_errors=True)
            ok = False
        return ok

    if CODE_ZIP.exists():  # a zip on Drive wins, so a newer zip is never shadowed by an older repository
        tmp = REPO.parent / "_code"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(CODE_ZIP) as z:
            z.extractall(tmp)
        shutil.move(str(next(tmp.rglob("redraw/__init__.py")).parent.parent), str(REPO))
        return f"zip {CODE_ZIP}"
    if cloned(REPO_URL):
        return f"GitHub, branch {BRANCH}"
    token = None
    if IN_COLAB:
        try:
            from google.colab import userdata
            token = userdata.get("GH_TOKEN")
        except Exception:
            token = None
    if token and cloned(REPO_URL.replace("https://", f"https://x-access-token:{token}@")):
        return f"GitHub, branch {BRANCH} (private, with GH_TOKEN)"
    raise SystemExit(f"Could not get the code: no {CODE_ZIP}, and branch {BRANCH} of {REPO_URL} could not be "
                     "cloned. If the repository is private, add the GH_TOKEN secret.")


print("code from", get_code())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
for _name in [n for n in sys.modules if n == "redraw" or n.startswith("redraw.")]:
    del sys.modules[_name]  # a re-run must not keep the code imported the first time

DATA = Path(os.environ.get("DATA", "/content/data"))
CACHE = Path(os.environ.get("CACHE", "/content/cache"))
WORK = Path(os.environ.get("WORK", "/content/drive/MyDrive/redraw"))  # networks of rounds 1-3 and the final run

# %%
from redraw import experiments as X

ctx = X.setup(WORK, DATA, CACHE, fast=FAST)

# %%
import json


def f3(x):
    return "-" if x is None else f"{x:.3f}"

# %% [markdown]
# ## 1. Reconstruction

# %%
REC = X.run_recon(ctx)
print(f"{'method':22s} {'IoU':>6s} {'F@0':>6s} {'F@1':>6s} {'F@1.5':>6s} {'centre px':>9s} {'curves':>6s} "
      f"{'length':>7s} {'overlap':>7s} {'ms/img':>7s}")
for key, m in REC["methods"].items():
    t = REC["seconds_per_image"].get(key, {}).get("seconds")
    print(f"{key:22s} {f3(m['iou']['mean']):>6s} {f3(m['f@0']['mean']):>6s} {f3(m['f@1']['mean']):>6s} "
          f"{f3(m['f@1.5']['mean']):>6s} {m['centreline']['mean']:9.2f} {m['curves']['mean']:6.1f} "
          f"{m['arc_length']['mean']:7.1f} {m['redundancy']['mean']:7.3f} {'' if t is None else f'{1000 * t:7.2f}'}")

# %% [markdown]
# ## 2. Stability

# %%
ST = X.run_stability(ctx)
print(json.dumps(ST, indent=1)[:3000])

# %% [markdown]
# ## 3. Probes on frozen representations

# %%
PR = X.run_probes(ctx)
for seed, rows in PR.items():
    if seed.startswith("seed"):
        for k, v in rows.items():
            print(f"{seed} {k:40s} {100 * v['test_accuracy']:.2f}%  ({v['dimensions']} numbers)")

# %% [markdown]
# ## 4. Interventions

# %%
IV = X.run_interventions(ctx)
for key, v in IV.items():
    if key.startswith("erase/"):
        print(key, "share of decisions changed:", json.dumps(v["changed"]))
        print("   share of ink erased:", json.dumps(v["ink_erased"]))
    elif key.startswith("representation/"):
        print(key, "fused changed:", v["fused_changed"], "reader changed:", v["reader_changed"])
print("from the drawings alone:", IV.get("from_drawings"))

# %% [markdown]
# ## 5. The hypothesis check

# %%
CH = X.run_checker(ctx)
for key, v in CH.get("per_seed", {}).items():
    print(key, v["transitions"], "| candidate recall", f3(v["candidate_recall"]),
          "| energy alone right among the candidates", f3(v["energy_alone_right_among_candidates"]))
    for score, s in v["selective"].items():
        print(f"   {score:42s} AURC {s['aurc']:.4f}  risk at 95% coverage {s['risk@0.95']:.4f}")
print("refitting:", CH.get("refit"))

# %% [markdown]
# ## 6. Figures

# %%
FIG = X.run_figures(ctx)
print(sorted(p.name for p in (ctx.out / "figures").glob("*.pdf")))

# %%
X.export(ctx)
