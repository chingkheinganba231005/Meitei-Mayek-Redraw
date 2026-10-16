import torch

from redraw.model import make_model
from redraw.render import fit_losses, pixel_grid, render, sample_map

G = 48


def line(x0, y0, x1, y1):
    t = torch.linspace(0, 1, 4).view(4, 1)
    return (torch.tensor([x0, y0]) * (1 - t) + torch.tensor([x1, y1]) * t).view(1, 1, 4, 2)


def px(img, x, y):
    return float(img[0, int(y * G), int(x * G)])


def test_stroke_ends_are_drawn_and_nothing_beyond_them():
    img, _, _ = render(line(0.2, 0.5, 0.8, 0.5), torch.tensor([[0.06]]), torch.tensor([[1.0]]), G=G)
    assert px(img, 0.21, 0.5) > 0.5 and px(img, 0.79, 0.5) > 0.5
    assert px(img, 0.12, 0.5) < 0.05 and px(img, 0.88, 0.5) < 0.05
    assert px(img, 0.5, 0.6) < 0.01  # half the width is 0.03 of the canvas; edges are soft over about a pixel


def test_a_closed_curve_draws_a_ring():
    ctrl = torch.tensor([[[[0.5, 0.25], [0.95, 0.75], [0.05, 0.75], [0.5, 0.25]]]])
    img, _, _ = render(ctrl, torch.tensor([[0.04]]), torch.tensor([[1.0]]), G=G, T=64)
    assert px(img, 0.5, 0.25) > 0.5 and px(img, 0.5, 0.5) < 0.05


def test_soft_union_of_crossing_strokes_stays_below_one():
    ctrl = torch.cat([line(0.2, 0.5, 0.8, 0.5), line(0.5, 0.2, 0.5, 0.8)], 1)
    img, _, _ = render(ctrl, torch.tensor([[0.06, 0.06]]), torch.tensor([[1.0, 1.0]]), G=G)
    assert img.max() <= 1.0 and px(img, 0.5, 0.5) > 0.9


def test_a_stroke_that_does_not_exist_draws_nothing():
    img, _, _ = render(line(0.2, 0.5, 0.8, 0.5), torch.tensor([[0.06]]), torch.tensor([[0.0]]), G=G)
    assert img.abs().max() < 1e-5


def test_losses_are_finite_on_an_empty_target():
    img, d, pts = render(line(0.2, 0.5, 0.8, 0.5), torch.tensor([[0.06]]), torch.tensor([[1.0]]), G=G)
    parts = fit_losses(img, d, pts, torch.tensor([[1.0]]), torch.zeros(1, G, G))
    assert all(torch.isfinite(v) for v in parts.values())


def _iou(a, b):
    a, b = a > 0.5, b > 0.5
    return (a & b).flatten(1).sum(1) / (a | b).flatten(1).sum(1).clamp_min(1)


def test_sampled_distance_leaves_gaps_in_thin_long_strokes_and_the_exact_one_does_not():
    g = torch.Generator().manual_seed(0)
    ctrl = (0.5 + 0.8 * (torch.rand(64, 1, 4, 2, generator=g) - 0.5)).clamp(0, 1)
    exist = torch.ones(64, 1)
    reference, _, _ = render(ctrl, torch.full((64, 1), 0.02), exist, G=G, T=1024)
    thin_sampled, _, _ = render(ctrl, torch.full((64, 1), 0.02), exist, G=G, T=32)
    thin_exact, _, _ = render(ctrl, torch.full((64, 1), 0.02), exist, G=G, T=32, exact=True)
    assert _iou(thin_sampled, reference).mean() < 0.9      # beads: the known limitation
    assert _iou(thin_exact, reference).mean() > 0.97
    # at pen-like widths (5-10% of the character box) the samples nearly suffice, even for long curves
    wide = 0.05 + 0.05 * torch.rand(64, 1, generator=g)
    a, _, _ = render(ctrl, wide, exist, G=G, T=32)
    b, _, _ = render(ctrl, wide, exist, G=G, T=1024)
    assert _iou(a, b).mean() > 0.95


def test_bilinear_sampling_uses_pixel_centres():
    xs = (torch.arange(G).float() + 0.5) / G
    m = xs.view(1, 1, G).expand(1, G, G)          # value = x coordinate of the pixel centre
    pts = torch.tensor([[[0.3, 0.5], [0.71, 0.2]]])
    assert torch.allclose(sample_map(m, pts), pts[..., 0], atol=1e-5)
    assert torch.allclose(pixel_grid(G, "cpu")[0], torch.tensor([0.5 / G, 0.5 / G]))


def _checker():
    torch.manual_seed(0)
    model = make_model(dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=True, K=4, hdr=True,
                            hdr_k=3, hdr_M=2, hdr_S=4, route="honest"), 10).eval()
    for head in (model.hdr.offset, model.hdr.affine):
        torch.nn.init.normal_(head[-1].weight, std=0.05)
    return model


def test_check_never_sees_the_label_at_test_time():
    model = _checker()
    x = torch.rand(6, 1, 64, 64)
    with torch.no_grad():
        o = model(x, draw=False, ink=x)
    assert torch.equal(o["cand"], o["fused"].topk(3, 1).indices)


def test_bonus_is_centred_over_candidates_and_favours_lower_energy():
    model = _checker()
    x = torch.rand(6, 1, 64, 64)
    with torch.no_grad():
        o = model(x, draw=False, ink=x)
    bonus, cand, cost = o["bonus"], o["cand"], o["cost"]
    on = torch.zeros_like(bonus, dtype=torch.bool).scatter(1, cand, True)
    assert (bonus[~on] == 0).all()
    assert torch.allclose(bonus.gather(1, cand).sum(1), torch.zeros(len(x)), atol=1e-5)
    assert torch.equal(bonus.gather(1, cand).argsort(1, descending=True), cost.argsort(1))
    assert torch.allclose(o["logits"], o["fused"].float() + bonus, atol=1e-5)
