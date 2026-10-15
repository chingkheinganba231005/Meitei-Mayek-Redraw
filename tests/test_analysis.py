import numpy as np

import torch

from redraw.analysis import (accuracy_by_bin, bootstrap_ci, errors_in_pairs, familiarity, mcnemar,
                             pairwise_average)
from redraw.data import stratified_subset


def test_mcnemar_counts_and_p():
    a = np.array([1, 1, 1, 0, 1, 0], bool)
    b = np.array([1, 0, 0, 0, 1, 1], bool)
    a_only, b_only, p = mcnemar(a, b)
    assert (a_only, b_only) == (2, 1) and 0 < p <= 1
    assert mcnemar(a, a) == (0, 0, 1.0)


def test_bootstrap_interval_contains_the_accuracy():
    ok = np.random.default_rng(0).random(2000) < 0.9
    lo, hi = bootstrap_ci(ok, n=2000)
    assert lo < ok.mean() < hi and hi - lo < 0.04


def test_pairwise_average_and_pair_errors():
    labels = np.array([0, 1, 2, 2])
    p1 = np.eye(3)[[0, 2, 2, 2]]          # one mistake: 1 -> 2
    p2 = np.eye(3)[[0, 1, 1, 2]] * 0.9    # one mistake: 2 -> 1
    out = pairwise_average({"a": p1, "b": p2}, labels)
    assert out[("a", "a")] == 0.75 and out[("b", "b")] == 0.75 and ("a", "b") in out
    assert errors_in_pairs(p1, labels, [(1, 2)]) == {"001/002": 1}


def test_stratified_subset_takes_the_same_share_of_every_class():
    labels = np.repeat(np.arange(4), [100, 50, 20, 3])
    idx = np.arange(len(labels))
    sub = stratified_subset(idx, labels, 0.1, seed=1)
    counts = np.bincount(labels[sub], minlength=4)
    assert counts.tolist() == [10, 5, 2, 1] and len(np.unique(sub)) == len(sub)
    assert np.array_equal(sub, stratified_subset(idx, labels, 0.1, seed=1))


def test_familiarity_finds_the_near_duplicate():
    torch.manual_seed(0)
    ref = torch.rand(50, 64)
    query = torch.cat([ref[7:8] * 2 + 0.01 * torch.randn(1, 64), torch.rand(1, 64)])  # a brighter copy of 7, and a stranger
    sim, idx = familiarity(query, ref, chunk=1)
    assert idx[0] == 7 and sim[0] > 0.99 and sim[1] < sim[0]


def test_accuracy_by_bin():
    out = accuracy_by_bin([1, 0, 1, 1], [0, 0, 1, 1])
    assert out == {0: (0.5, 2), 1: (1.0, 2)}


def test_risk_coverage_orders_by_confidence():
    from redraw.analysis import rank_average, risk_at, risk_coverage
    score = np.array([0.9, 0.8, 0.7, 0.1])
    correct = np.array([True, True, True, False])
    cov, risk, aurc = risk_coverage(score, correct)
    assert risk.tolist() == [0, 0, 0, 0.25] and cov[-1] == 1 and abs(aurc - 0.0625) < 1e-9
    assert risk_at(score, correct, 0.75) == 0 and risk_at(score, correct, 1.0) == 0.25
    assert risk_coverage(-score, correct)[2] > aurc  # the wrong order is worse
    assert rank_average([1, 2, 3], [3, 2, 1]).tolist() == [2, 2, 2]


def test_split_from_class_folders(tmp_path):
    import json
    import zipfile

    import cv2

    from helpers import glyph
    from redraw.split import make_split_folders
    with zipfile.ZipFile(tmp_path / "d.zip", "w") as z:
        for split, n in (("Train", 10), ("Test", 3)):
            for name in ("character_10_x", "character_2_y", "digit_0"):
                for i in range(n):
                    ok, buf = cv2.imencode(".png", 255 - glyph(i))
                    z.writestr(f"Data/{split}/{name}/{i}.png", buf.tobytes())
    split_dir, summary = make_split_folders(tmp_path / "d.zip", tmp_path / "data")
    assert summary == {"classes": 3, "train": 27, "val": 3, "test": 9}
    assert json.loads((split_dir / "classes.json").read_text()) == ["character_2_y", "character_10_x", "digit_0"]
