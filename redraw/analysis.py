"""Comparing networks on the same images: paired tests, confidence intervals, how well pairs combine."""

import itertools

import numpy as np
from scipy.stats import binomtest


def mcnemar(ok_a, ok_b):
    """Exact McNemar test on two boolean "right" vectors: (only a right, only b right, two-sided p)."""
    ok_a, ok_b = np.asarray(ok_a, bool), np.asarray(ok_b, bool)
    a_only, b_only = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return a_only, b_only, float(p)


def bootstrap_ci(ok, n=10000, seed=0, level=0.95):
    """Percentile bootstrap interval of the accuracy of a boolean "right" vector."""
    ok = np.asarray(ok, bool)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(0, n, 500):
        m = min(500, n - i)
        means[i:i + m] = ok[rng.integers(0, len(ok), (m, len(ok)))].mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def pairwise_average(probs, labels):
    """Accuracy of the plain average of every pair of networks: {(a, b): accuracy}, plus singles as (a, a)."""
    names = list(probs)
    out = {(a, a): float((probs[a].argmax(1) == labels).mean()) for a in names}
    for a, b in itertools.combinations(names, 2):
        out[(a, b)] = float(((probs[a] + probs[b]).argmax(1) == labels).mean())
    return out


def errors_in_pairs(p, labels, pairs):
    """Number of images of each confusable pair (a, b) that are mistaken for the other class of the pair."""
    pred = p.argmax(1)
    return {f"{a:03d}/{b:03d}": int((((labels == a) & (pred == b)) | ((labels == b) & (pred == a))).sum())
            for a, b in pairs}
