"""Comparing networks on the same images: paired tests, confidence intervals, how well pairs combine."""

import itertools

import numpy as np
import torch
import torch.nn.functional as F
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


def familiarity(query, ref, chunk=4096):
    """How familiar each query image is: cosine similarity of its most similar reference image, and its index.

    ``query`` (N, D) and ``ref`` (M, D) are flattened images (any model-free representation, e.g.
    32 x 32 thumbnails); each is centred and normalised here. Returns two tensors of length N.
    """
    def prep(x):
        x = x.float()
        return F.normalize(x - x.mean(1, keepdim=True), dim=1)

    q, r = prep(query), prep(ref)
    best, arg = [], []
    for i in range(0, len(q), chunk):
        v, j = (q[i:i + chunk] @ r.T).max(1)
        best.append(v)
        arg.append(j)
    return torch.cat(best), torch.cat(arg)


def accuracy_by_bin(ok, bins):
    """Accuracy of a boolean "right" vector within each bin of an integer bin vector: {bin: (accuracy, n)}."""
    ok, bins = np.asarray(ok, bool), np.asarray(bins)
    return {int(b): (float(ok[bins == b].mean()), int((bins == b).sum())) for b in np.unique(bins)}


def risk_coverage(score, correct):
    """Selective prediction: keep the images with the highest ``score`` first.

    Returns (coverage, risk, aurc): for k = 1..N kept images, the share kept and the error rate among
    them, and the area under that curve (lower is better).
    """
    score, correct = np.asarray(score, float), np.asarray(correct, bool)
    order = np.argsort(-score, kind="stable")
    errors = np.cumsum(~correct[order])
    k = np.arange(1, len(score) + 1)
    coverage, risk = k / len(score), errors / k
    return coverage, risk, float(risk.mean())


def risk_at(score, correct, coverage):
    """Error rate among the ``coverage`` share of images with the highest score."""
    cov, risk, _ = risk_coverage(score, correct)
    return float(risk[min(len(risk) - 1, max(0, int(np.ceil(coverage * len(risk))) - 1))])


def rank_average(*scores):
    """Average of the ranks of several scores (higher is more confident), without any fitted weight."""
    from scipy.stats import rankdata
    return np.mean([rankdata(np.asarray(s, float)) for s in scores], 0)
