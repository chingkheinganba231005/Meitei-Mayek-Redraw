import torch

from redraw.render import curve_points, distance_to_ink, fit_losses, parse_score, render, render_loss


def straight(y=0.5):
    """One horizontal stroke across the middle of the canvas."""
    ctrl = torch.tensor([[[[0.2, y], [0.4, y], [0.6, y], [0.8, y]]]])
    return ctrl, torch.tensor([[0.12]]), torch.tensor([[1.0]])


def test_curve_ends_at_first_and_last_control_point():
    ctrl = torch.rand(2, 3, 4, 2)
    pts = curve_points(ctrl, 16)
    assert torch.allclose(pts[..., 0, :], ctrl[..., 0, :]) and torch.allclose(pts[..., -1, :], ctrl[..., 3, :])


def test_render_draws_ink_along_the_stroke_only():
    img, d, pts = render(*straight(), G=32)
    assert img.shape == (1, 32, 32) and d.shape == (1, 1, 32 * 32) and pts.shape == (1, 1, 32, 2)
    assert img[0, 16, 16] > 0.9 and img[0, 16, 2] < 0.05 and img[0, 2, 16] < 0.05


def test_missing_stroke_draws_nothing():
    ctrl, w, _ = straight()
    img, _, _ = render(ctrl, w, torch.zeros(1, 1), G=32)
    assert img.max() < 1e-4


def test_distance_to_ink():
    t = torch.zeros(1, 16, 16)
    t[0, 8, 8] = 1
    d = distance_to_ink(t) * 16
    assert d[0, 8, 8] == 0 and 2.5 <= d[0, 8, 11] <= 3.5 and d[0, 8, 0] > d[0, 8, 11]


def test_losses_pull_a_misplaced_stroke_onto_the_ink():
    target, _, _ = render(*straight(0.5), G=32)
    target = (target > 0.5).float()
    ctrl, w, e = straight(0.3)
    ctrl.requires_grad_(True)
    img, d, pts = render(ctrl, w, e, G=32)
    loss = render_loss(fit_losses(img, d, pts, e, target))
    loss.backward()
    assert torch.isfinite(loss) and ctrl.grad[..., 1].mean() < 0  # gradient descent moves y down (towards 0.5)


def test_parse_score():
    img, _, _ = render(*straight(), G=32)
    p, r, f = parse_score(img, (img > 0.5).float())
    assert p.item() == 1 and r.item() == 1 and f.item() > 0.99
    empty = torch.zeros_like(img)
    assert parse_score(empty, (img > 0.5).float())[2].item() == 0
