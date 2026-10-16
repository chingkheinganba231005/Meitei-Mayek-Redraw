"""Paired comparisons at the right level.

Three different questions, three tools:

    paired_runs        independently trained pairs (same seed / subset for both networks), all scored on
                       the same images: is one network ahead across training runs? Conditional on the test
                       set; it says nothing about new writers.
    mcnemar            (in ``analysis``) two fixed trained networks on the same images: are the images only
                       one of them gets right split evenly? Treats the images as the sample.
    paired_bootstrap   two fixed systems on the same images: an interval for the accuracy difference when
                       images are resampled (images, not writers, are the unit; TUMMHCD has no writer ids).
"""

import numpy as np
from scipy.stats import binomtest, rankdata, ttest_1samp, wilcoxon


def signed_rank_p(d, decimals=9):
    """Two-sided Wilcoxon signed-rank p-value, exact for up to 20 non-zero differences: every assignment
    of signs is enumerated, with mid-ranks for tied absolute differences (accuracy differences on one test
    set are multiples of 1/N, so ties are common; they are restored by rounding to ``decimals``).
    Zero differences are dropped. Larger samples use scipy's normal approximation with tie correction."""
    d = np.round(np.asarray(d, float), decimals)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    if n > 20:
        return float(wilcoxon(d, alternative="two-sided", method="approx").pvalue)
    ranks = rankdata(np.abs(d))
    observed = ranks[d > 0].sum()
    signs = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1
    w = signs @ ranks
    centre = ranks.sum() / 2
    return float(np.mean(np.abs(w - centre) >= abs(observed - centre) - 1e-9))


def paired_runs(a, b):
    """a, b: accuracies of paired runs (same length). Differences b - a, wins/losses/ties, the exact
    two-sided Wilcoxon signed-rank test (ties handled, see ``signed_rank_p``), the sign test and a
    one-sample t-test on the differences."""
    d = np.round(np.asarray(b, float) - np.asarray(a, float), 9)
    out = {"n": int(len(d)), "differences": d.tolist(), "mean": float(d.mean()),
           "sd": float(d.std(ddof=1)) if len(d) > 1 else None,
           "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}
    nz = d[d != 0]
    if len(nz) >= 1:
        out["wilcoxon_exact_p"] = signed_rank_p(nz)
        out["sign_p"] = float(binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue)
    if len(d) > 1 and d.std() > 0:
        out["t_p"] = float(ttest_1samp(d, 0.0).pvalue)
    return out


def paired_bootstrap(ok_a, ok_b, n=10000, seed=0, level=0.95):
    """Interval for accuracy(b) - accuracy(a) on the same images, resampling images."""
    ok_a, ok_b = np.asarray(ok_a, bool), np.asarray(ok_b, bool)
    diff = ok_b.astype(np.int8) - ok_a.astype(np.int8)
    rng = np.random.default_rng(seed)
    stats = np.empty(n)
    for i in range(0, n, 500):
        m = min(500, n - i)
        stats[i:i + m] = diff[rng.integers(0, len(diff), (m, len(diff)))].mean(1)
    lo, hi = np.quantile(stats, [(1 - level) / 2, (1 + level) / 2])
    return {"difference": float(diff.mean()), "low": float(lo), "high": float(hi),
            "only_b_right": int((diff > 0).sum()), "only_a_right": int((diff < 0).sum())}


def per_class(y, pred, num_classes):
    """Recall per class, its support, and macro-F1."""
    y, pred = np.asarray(y), np.asarray(pred)
    recall, f1, support = np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes, int)
    for c in range(num_classes):
        t, p = y == c, pred == c
        support[c] = int(t.sum())
        tp = int((t & p).sum())
        recall[c] = tp / max(1, support[c])
        prec = tp / max(1, int(p.sum()))
        f1[c] = 2 * prec * recall[c] / (prec + recall[c]) if prec + recall[c] > 0 else 0.0
    return {"recall": recall.tolist(), "support": support.tolist(), "macro_f1": float(f1.mean()),
            "min_recall": float(recall[support > 0].min()) if (support > 0).any() else None}
