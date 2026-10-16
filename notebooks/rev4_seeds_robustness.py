# %% [markdown]
# # Revision 4: more seeds, damaged images and the routing dial
#
# | | what | GPU time |
# |---|---|---|
# | 1 | seeds 3 and 4 of the control and the method on training + validation, test set (five seeds in all; the number was fixed before these runs and is not extended) | 1 h 50 min |
# | 2 | seeds 1 and 2 of coupled routing | 1 h 20 min |
# | 3 | damaged images, one view, every final network: corruption of the raw scan before preprocessing (the whole pipeline, crop failures counted), the same damage after preprocessing with the framing fixed, and texture changes that keep the shape against shape changes that keep the texture | 1 h 15 min |
# | 4 | optional: the routing dial, a quarter, half and all of the drawing gradient let into the backbone, development split, scored on validation only | 1 h 45 min |
#
# Every corruption uses the same random draws for every network. Results: `extra_seeds.json`,
# `coupled_seeds.json`, `robustness.json`, `dial.json` in `MyDrive/redraw/revision/`.

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
RUN_DIAL = True


def pct(x):
    return "-" if x is None else f"{100 * x:.2f}%"

# %% [markdown]
# ## 1. Seeds 3 and 4

# %%
ES = X.run_extra_seeds(ctx)
for view in ("test_tta", "test_id"):
    a = ES.get(f"all_seeds_{view}")
    if a:
        print(f"{view}: control {pct(a['baseline_fp32_mean'])} +- {100 * (a['baseline_fp32_sd'] or 0):.2f}, "
              f"ours {pct(a['redraw_hdr_hr_mean'])} +- {100 * (a['redraw_hdr_hr_sd'] or 0):.2f}, "
              f"per seed {[round(100 * d, 2) for d in a['paired']['differences']]}, "
              f"exact Wilcoxon p = {a['paired'].get('wilcoxon_exact_p')}")

# %% [markdown]
# ## 2. Coupled routing, seeds 1 and 2

# %%
CS = X.run_coupled_seeds(ctx)
print({k: pct(v["test_5view"]["logits"]) for k, v in CS.items()})

# %% [markdown]
# ## 3. Damaged images

# %%
RB = X.run_robustness(ctx)
print("crop failures (raw):", RB.get("crop_failures"))
for cond, s in RB["summary"].items():
    print(cond, {f: pct(v["mean_over_conditions"]) for f, v in s["families"].items()})

# %% [markdown]
# ## 4. The routing dial (optional, validation only)

# %%
if RUN_DIAL:
    DL = X.run_dial(ctx)
    for key, e in DL["runs"].items():
        print(f"share {key}: validation {pct(e['val_5view'])} (5 views), ink F {e['rendered_ink_val']['f']:.3f}, "
              + ", ".join(f"{k} {pct(v)}" for k, v in e.items() if "/" in k))

# %%
X.export(ctx)
