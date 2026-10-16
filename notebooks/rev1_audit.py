# %% [markdown]
# # Revision 1: checks before any new training
#
# Nothing is trained here. The notebook checks the code, and the saved results the paper rests on.
#
# | | what | GPU time |
# |---|---|---|
# | 1 | unit tests of the routing, renderer, check, reconstruction and statistics code | 2 min |
# | 2 | which loss trains which parameters, for the method and every configuration of the revision (real architecture, random weights) | 5 min |
# | 3 | the learned scales of the hypothesis check; sampled against exact rendering on real parses | 3 min |
# | 4 | from the saved test predictions: accuracy of every head with one and five views, per-class recall and macro-F1, McNemar tests and paired bootstrap intervals, the comparison across seeds, what the check changes | 2 min |
# | 5 | cost on this GPU: parameters, FLOPs, latency (batch 1) and throughput, peak memory, training step time, preprocessing | 10 min |
#
# Run it first, and once more at the very end: section 4 then also covers seeds 3 and 4 (notebook rev4).
# Results: `MyDrive/redraw/revision/audit.json` and `bench.json`, and a ZIP in `revision/exports`.

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


def pct(x):
    return "-" if x is None else f"{100 * x:.2f}%"

# %% [markdown]
# ## 1. Unit tests
#
# The experiments start only if these pass.

# %%
TESTS = ["tests/test_audit.py", "tests/test_render_checker_audit.py", "tests/test_recon.py",
         "tests/test_stats_bench_manifest.py", "tests/test_experiments.py", "tests/test_routing.py"]
r = subprocess.run([sys.executable, "-m", "pytest", "-q", *TESTS], cwd=REPO, capture_output=True, text=True)
print(r.stdout[-1500:])
assert r.returncode == 0, "a unit test failed: fix the code before running any experiment"

# %% [markdown]
# ## 2-4. Gradient contract, check scales, renderer and the saved test predictions

# %%
A = X.run_audit(ctx)
print(A["gradient_contract"]["redraw_hdr_hr"]["markdown"])
print()
for name, c in A["gradient_contract"].items():
    print(f"{name:20s} {'as intended' if not c['mismatches'] else c['mismatches']}")
print("\nlearned check scales:", json.dumps(A.get("check_scales"), indent=1))
if "renderer" in A:
    print("renderer on real parses:", json.dumps(A["renderer"], indent=1))

# %%
for key, t in A.get("paired_tests", {}).items():
    if "across_seeds" in key:
        print(f"{key}: ours - control per seed {[round(100 * d, 3) for d in t['differences']]} points, "
              f"wins {t['wins']}, losses {t['losses']}, exact Wilcoxon p = {t.get('wilcoxon_exact_p')}")
    else:
        b = t["paired_bootstrap_ours_minus_control"]
        print(f"{key}: only control right {t['control_only_right']}, only ours right {t['ours_only_right']}, "
              f"McNemar p = {t['mcnemar_p']:.3g}, difference {100 * b['difference']:+.2f} points "
              f"[{100 * b['low']:+.2f}, {100 * b['high']:+.2f}]")
for key, v in A.get("check_transitions", {}).items():
    print("check:", key, v)
for key, v in A.get("per_class", {}).items():
    if key.endswith("test_tta"):
        print(f"{key}: macro-F1 {v['macro_f1']:.4f}, lowest class recall {pct(v['min_recall'])}")

# %% [markdown]
# ## 5. Cost

# %%
B = X.run_bench(ctx)
print(B["environment"]["device"])
for lab, e in B["networks"].items():
    print(f"{lab:8s} parameters {e['parameters']['total'] / 1e6:.1f} M, {e['flops_1view'] / 1e9:.2f} GFLOPs per view, "
          f"latency {e['latency_batch1_1view']['median_ms']:.1f} ms (1 view) / {e['latency_batch1_5view']['median_ms']:.1f} ms "
          f"(5 views), {e['throughput_1view']['images_per_s']:.0f} images/s, train step {e['train_step']['seconds_per_step']:.3f} s")
print(json.dumps(B["ours_over_control"], indent=1))

# %%
X.export(ctx)
