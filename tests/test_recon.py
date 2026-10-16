import numpy as np
import torch

from redraw.recon import (affine_view, evaluate, fit_beziers, map_points, match_curves, optimise_strokes,
                          pad_strokes, skeleton_curves, tolerant_prf, trace_skeleton, union_chamfer)
from redraw.render import curve_points, render

R = 64


def letter_t():
    """A 'T' drawn with two strokes, as ink in [0, 1] at 128 px, with its strokes."""
    ctrl = torch.tensor([[[[0.2, 0.25], [0.4, 0.25], [0.6, 0.25], [0.8, 0.25]],
                          [[0.5, 0.25], [0.5, 0.45], [0.5, 0.65], [0.5, 0.85]]]])
    width, exist = torch.tensor([[0.07, 0.07]]), torch.tensor([[1.0, 1.0]])
    ink, _, _ = render(ctrl, width, exist, G=128, T=64, exact=True)
    return ink.unsqueeze(1), ctrl, width, exist


def test_the_true_strokes_score_well_and_a_shifted_copy_needs_tolerance():
    ink, ctrl, width, exist = letter_t()
    m = evaluate(ctrl, width, exist, ink, R=R)
    assert m["iou"][0] > 0.9 and m["f@1"][0] > 0.98 and m["centreline"][0] < 1.0
    assert m["curves"][0] == 2 and 70 < m["arc_length"][0] < 90
    shifted = evaluate(ctrl + torch.tensor([2.0 / R, 0.0]), width, exist, ink, R=R)
    assert shifted["f@0"][0] < m["f@0"][0] and shifted["f@1.5"][0] > shifted["f@0"][0]


def test_tolerant_scores_on_masks():
    a = np.zeros((1, 16, 16), bool)
    a[0, 4:12, 8] = True
    b = np.roll(a, 1, axis=2)
    s = tolerant_prf(a, a)
    assert s["f@0"][0] == 1.0
    s = tolerant_prf(b, a)
    assert s["f@0"][0] == 0.0 and s["f@1"][0] == 1.0
    assert np.isnan(tolerant_prf(np.zeros_like(a), a)["precision@0"][0])


def test_skeleton_baseline_recovers_a_two_stroke_letter():
    ink, _, _, _ = letter_t()
    small = torch.nn.functional.adaptive_avg_pool2d(ink, R)[0, 0].numpy()
    c, w = skeleton_curves(small, tol=1.0)
    assert 2 <= len(c) <= 6 and (w > 0).all()
    m = evaluate(*pad_strokes([(c, w)]), ink, R=R)
    assert m["f@1"][0] > 0.9


def test_skeleton_tracing_keeps_a_ring_whole():
    yy, xx = np.mgrid[:40, :40]
    ring = np.abs(np.hypot(yy - 20, xx - 20) - 12) < 1.5
    from skimage.morphology import skeletonize
    paths = trace_skeleton(skeletonize(ring))
    assert len(paths) == 1 and len(paths[0]) > 50


def test_one_cubic_fits_points_of_a_cubic():
    ctrl = torch.tensor([[[0.1, 0.2], [0.3, 0.9], [0.7, 0.1], [0.9, 0.8]]])
    pts = curve_points(ctrl, 40)[0].numpy()
    assert len(fit_beziers(pts, tol=0.01)) == 1      # 0.6 px on a 64 px canvas


def test_per_image_optimisation_lowers_the_drawing_loss():
    ink, _, _, _ = letter_t()
    target = torch.nn.functional.adaptive_avg_pool2d(ink, 48)[:, 0]
    _, _, _, before = optimise_strokes(target, K=4, steps=0)
    c, w, e, after = optimise_strokes(target, K=4, steps=60, lr=0.05)
    assert after.item() < 0.7 * before.item() and c.shape == (1, 4, 4, 2)


def test_parse_of_a_moved_image_maps_back():
    x = torch.zeros(1, 1, 64, 64)
    x[0, 0, 40, 20] = 1.0                       # a dot at (x, y) = (20.5, 40.5) / 64
    moved, theta = affine_view(x, angle=10, scale=1.05, dx=0.03, dy=-0.02)
    yx = np.unravel_index(moved[0, 0].argmax().item(), (64, 64))
    p = torch.tensor([[[(yx[1] + 0.5) / 64, (yx[0] + 0.5) / 64]]], dtype=torch.float32)
    back = map_points(p, theta)[0, 0]
    assert torch.allclose(back, torch.tensor([20.5 / 64, 40.5 / 64]), atol=1.5 / 64)


def test_stability_measures_ignore_order_and_direction():
    ctrl = torch.rand(5, 4, 2)
    pts = curve_points(ctrl, 16).numpy()
    on = np.ones(5, bool)
    swapped = pts[[2, 0, 4, 1, 3]][:, ::-1]
    assert union_chamfer(pts, on, swapped, on) < 1e-9
    cost, unmatched = match_curves(pts, on, swapped, on)
    assert cost < 1e-9 and unmatched == 0
