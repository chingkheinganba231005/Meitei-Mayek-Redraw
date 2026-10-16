# %% [markdown]
# # Revision 5: a second dataset
#
# The Devanagari Handwritten Character Dataset (Acharya et al., 2015; UCI 389): 46 classes (36
# consonants and 10 digits), 32 x 32 images, 78,200 training and 13,800 test images, written by people
# other than the TUMMHCD writers. It was chosen before any result on it was seen. Preprocessing,
# recipe and configurations are those of TUMMHCD; only the number of classes changes. Validation is
# carved out as for TUMMHCD but not used to tune anything, and the test set is predicted once.
#
# Three seeds each of the control, the method and coupled routing on training + validation; test set
# with one and five views; the stronger level of each fixed-framing corruption.
#
# GPU time: about 5 h. If the download fails, put the zip from https://archive.ics.uci.edu/dataset/389
# on Drive as `MyDrive/redraw/dhcd/dhcd.zip`. Results: `MyDrive/redraw/revision/dhcd/second_dataset.json`.

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
def pct(x):
    return "-" if x is None else f"{100 * x:.2f}%"


DATA_D = Path(os.environ.get("DATA_D", "/content/data_dhcd"))
CACHE_D = Path(os.environ.get("CACHE_D", "/content/cache_dhcd"))
ctx2 = X.setup_second(ctx, DATA_D, CACHE_D)
SD = X.run_second_dataset(ctx2)

# %%
for view in ("test_5view", "test_1view"):
    for name, v in SD.get(f"families_{view}", {}).items():
        print(f"{view} {name:14s} {pct(v['mean'])} +- {100 * (v['sd'] or 0):.2f} {[round(100 * a, 2) for a in v['runs']]}")
    for other in ("redraw_hdr_hr", "redraw_hdr"):
        p = SD.get(f"paired_{other}_vs_control_{view}")
        if p:
            print(f"   {other} - control per seed {[round(100 * d, 2) for d in p['differences']]}")
for key, t in SD.get("paired_tests_5view", {}).items():
    b = t["paired_bootstrap_ours_minus_control"]
    print(f"{key}: only control right {t['control_only_right']}, only ours right {t['ours_only_right']}, "
          f"McNemar p = {t['mcnemar_p']:.3g}, [{100 * b['low']:+.2f}, {100 * b['high']:+.2f}] points")

# %%
X.export(ctx)
