import torch

from redraw.render import curve_points, render
from redraw.strokewarp import bend_strokes, stroke_warp


def strokes(*segments):
    """Straight strokes given as ((x0, y0), (x1, y1)) pairs, as control points (1, K, 4, 2)."""
    rows = []
    for (x0, y0), (x1, y1) in segments:
        rows.append([[x0 + (x1 - x0) * f, y0 + (y1 - y0) * f] for f in (0, 1 / 3, 2 / 3, 1)])
    ctrl = torch.tensor([rows])
    exist = torch.ones(1, len(segments))
    img, _, _ = render(ctrl, torch.full((1, len(segments)), 0.06), exist, G=64)
    return img.unsqueeze(1), ctrl, exist


def no_noise(K):
    return torch.zeros(1, K, 2, 2), torch.zeros(1, K, 2)


def ink_extent_x(img, row_band):
    cols = img[0, 0, row_band].amax(0) > 0.5
    idx = torch.nonzero(cols).flatten()
    return idx.min().item(), idx.max().item()


def test_no_noise_keeps_the_image():
    img, ctrl, exist = strokes(((0.3, 0.5), (0.7, 0.5)))
    same = stroke_warp(img, ctrl, exist, noise=no_noise(1))
    assert (same - img).abs().max() < 0.02


def test_a_free_tail_grows_and_a_joined_end_stays():
    # an L: a horizontal stroke whose left end meets a vertical stroke; its right end is free
    img, ctrl, exist = strokes(((0.3, 0.5), (0.7, 0.5)), ((0.3, 0.2), (0.3, 0.5)))
    inner, tails = no_noise(2)
    tails[0, 0] = torch.tensor([0.5, 0.5])           # ask both ends of the horizontal stroke to grow 50%
    new = bend_strokes(ctrl, exist, noise=(inner, tails))
    assert torch.allclose(new[0, 0, 0], ctrl[0, 0, 0])              # joined end: not moved
    assert abs(new[0, 0, 3, 0].item() - 0.9) < 1e-5                 # free end: 0.7 + 0.5 * 0.4
    out = stroke_warp(img, ctrl, exist, noise=(inner, tails))
    left0, right0 = ink_extent_x(img, slice(30, 34))
    left1, right1 = ink_extent_x(out, slice(30, 34))
    assert abs(left1 - left0) <= 1 and right1 >= right0 + 8          # the tail got longer, the joint held


def test_bending_moves_the_middle_but_not_the_ends():
    img, ctrl, exist = strokes(((0.2, 0.5), (0.8, 0.5)))
    inner = torch.tensor([[[[0.0, 0.1], [0.0, 0.1]]]])               # both inner points down by 0.1
    new = bend_strokes(ctrl, exist, noise=(inner, torch.zeros(1, 1, 2)))
    mid = curve_points(new, 33)[0, 0, 16]
    assert abs(mid[1].item() - 0.575) < 1e-4 and torch.allclose(new[0, 0, [0, 3]], ctrl[0, 0, [0, 3]])
    out = stroke_warp(img, ctrl, exist, noise=(inner, torch.zeros(1, 1, 2)), sigma=0.05)
    centre_col = out[0, 0, :, 31:33].mean(1)
    assert torch.argmax(centre_col).item() > torch.argmax(img[0, 0, :, 31:33].mean(1)).item() + 2


def test_mask_and_missing_strokes():
    img, ctrl, exist = strokes(((0.3, 0.5), (0.7, 0.5)))
    x = img.repeat(2, 1, 1, 1)
    c, e = ctrl.repeat(2, 1, 1, 1), exist.repeat(2, 1)
    noise = (torch.full((2, 1, 2, 2), 0.08), torch.zeros(2, 1, 2))
    out = stroke_warp(x, c, e, mask=torch.tensor([False, True]), noise=noise)
    assert torch.equal(out[0], x[0]) and not torch.equal(out[1], x[1])
    still = stroke_warp(img, ctrl, torch.zeros(1, 1), noise=(noise[0][:1], noise[1][:1]))
    assert (still - img).abs().max() < 1e-4                          # a stroke that does not exist moves nothing
    assert out.min() >= 0 and out.max() <= 1
