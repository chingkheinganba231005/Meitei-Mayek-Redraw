# %% [markdown]
# # Revision 2: what the stroke reader uses, and what each routing rule does
#
# Every network here is trained on the development split (61,504 images) and scored on validation
# (10,826 images); the test set is not used. All configurations of one seed see the same images in the
# same order with the same augmentation (`matched_aug`), so their differences are not differences of
# data. The hypothesis check is off in all of them, so the two routing rules are isolated; the check is
# studied on its own in rev3.
#
# | configuration | question |
# |---|---|
# | `control_m` | image-only control |
# | `rt_none` | routing factorial: neither rule (coupled routing) |
# | `rt_backbone` | the drawing loss kept out of the backbone, nothing else |
# | `rt_geometry` | the reader's loss kept away from the stroke geometry, nothing else |
# | `rt_both` | both rules: honest routing (the method without the check) |
# | `rd_geometry` | the reader sees only the geometry of the strokes |
# | `rd_features` | the reader sees only backbone features sampled along the strokes |
# | `rd_grid` | the reader samples backbone features along a fixed grid of curves instead of the strokes |
# | `rd_free` | learned stroke slots trained by classification alone, no rendering loss |
# | `sequential` | staged training: decoder and reader trained on the frozen `control_m` of the same seed |
#
# For the coupled network the cosine between the drawing and classification gradients on the shared
# backbone is also measured.
#
# Decided before the runs: if the geometry-only reader is far below `rt_both`, the paper presents a
# hybrid recogniser with a visual reconstruction, not a stroke bottleneck; if the fixed grid matches
# `rt_both`, the accuracy gain may come from an extra feature head; if staged training matches
# `rt_both`, the paper says so and claims less novelty; if neither rule changes accuracy by more than
# the spread between seeds, the paper no longer claims that routing improves accuracy.
#
# Three seeds, seed after seed, so an interrupted run leaves complete matrices for the first seeds.
# GPU time: about 3 h per seed, 10 h in all (in round 3 the control took 11 min and a stroke network
# without the check 20 min). New files: about 4 GB on Drive. Results: `MyDrive/redraw/revision/core.json`.

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
SEEDS = (0, 1, 2)


def pct(x):
    return "-" if x is None else f"{100 * x:.2f}"


C = X.run_core(ctx, seeds=SEEDS)

# %%
print(f"{'configuration':13s} {'seeds':>5s} {'5 views':>14s} {'1 view':>7s} {'image':>7s} {'reader':>7s} "
      f"{'vs control':>22s} {'ink F':>6s} {'min':>5s}")
for name, e in C["summary"].items():
    v5 = e["val_5view_logits"]
    sd = f"+-{100 * v5['sd']:.2f}" if v5["sd"] is not None else ""
    vs = e.get("val_5view_vs_control")
    vs = f"{100 * vs['mean']:+.2f} ({vs['wins']}-{vs['losses']}-{vs['ties']})" if vs else ""
    ink = f"{e['rendered_ink_f_mean']:.3f}" if "rendered_ink_f_mean" in e else "-"
    print(f"{name:13s} {len(e['seeds']):5d} {pct(v5['mean']) + sd:>14s} {pct(e['val_1view_logits']['mean']):>7s} "
          f"{pct(e.get('val_5view_img', {}).get('mean')):>7s} {pct(e.get('val_5view_vec', {}).get('mean')):>7s} "
          f"{vs:>22s} {ink:>6s} {e['train_minutes_mean']:5.0f}")
for key, run in C["runs"].items():
    if "gradient_conflict" in run:
        g = run["gradient_conflict"]
        print(f"{key}: cosine(drawing, classification) on the shared backbone {g['cosine_mean']:+.3f} +- {g['cosine_sd']:.3f}")

# %%
X.export(ctx)
