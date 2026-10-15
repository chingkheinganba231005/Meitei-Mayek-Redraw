"""StrokeWarp: deform a handwritten character stroke by stroke, along strokes the network found itself.

Two writers, or one writer twice, rarely differ by a smooth warp of the whole
character. They bend individual strokes a little more or less, and they stop
the free ends (tails) of strokes a little earlier or later. StrokeWarp makes
exactly those changes, using the network's own parse of the image:

- bend: the two inner control points of every stroke move by a small random
  amount while its end points stay put, so the stroke's curvature changes
  but strokes that meet stay joined;
- tails: an end point that touches no other stroke moves along the stroke's
  own direction, lengthening or shortening that tail.

Pen width is never changed (it carries how large the character was written).
Every output pixel then looks back to where its ink came from: a softmax-weighted
average, over the nearest samples of the bent strokes, of how far each sample
travelled (bent and original samples correspond one to one along each curve).
The image is resampled with that field, so the real ink, with its real texture,
is what moves. The parse comes from the network being trained (its
weight-averaged copy); no stroke labels are involved.
"""

import torch
import torch.nn.functional as F

from .render import curve_points

# p: share of each batch that is warped; warmup: epochs before the first warp (the parse must be
# usable first); bend: s.d. of the inner control-point moves; tail: largest change of a free tail,
# as a fraction of its stroke's chord; free: an end point farther than this from every other stroke
# is a free end; sigma: reach of a stroke over nearby pixels; grid: resolution of the field.
# Lengths are fractions of the canvas.
STROKEWARP = dict(p=0.5, warmup=10, bend=0.035, tail=0.2, free=0.06, sigma=0.03, grid=32)


def rewrite_noise(B, K, bend, tail, generator=None, device=None):
    """The random draws of ``bend_strokes`` (inner moves, tail changes), from ``generator`` if given."""
    inner = torch.randn(B, K, 2, 2, generator=generator) * bend
    ends = (torch.rand(B, K, 2, generator=generator) * 2 - 1) * tail
    return inner.to(device), ends.to(device)


def bend_strokes(ctrl, exist, bend=0.035, tail=0.2, free=0.06, noise=None):
    """New control points: inner points moved (curvature), free end points moved along the stroke.

    ``noise`` = (inner (B, K, 2, 2), tail (B, K, 2)) overrides the random draws (for tests); the tail
    values are fractions of the chord, positive = longer.
    """
    B, K = ctrl.shape[:2]
    ctrl = ctrl.float()
    inner, ends = noise if noise is not None else (
        torch.randn(B, K, 2, 2, device=ctrl.device) * bend,
        (torch.rand(B, K, 2, device=ctrl.device) * 2 - 1) * tail)
    new = ctrl.clone()
    new[:, :, 1:3] = ctrl[:, :, 1:3] + inner

    # which end points are free: no sample of another existing stroke within ``free``
    pts = curve_points(ctrl, 32)                                   # (B, K, T, 2)
    end_pts = torch.stack([ctrl[:, :, 0], ctrl[:, :, 3]], 2)       # (B, K, 2, 2)
    d = torch.cdist(end_pts.reshape(B, K * 2, 2), pts.reshape(B, K * 32, 2)).view(B, K, 2, K, 32)
    d = d.min(-1).values                                           # (B, K, 2 ends, K other strokes)
    own = torch.eye(K, dtype=torch.bool, device=ctrl.device).view(1, K, 1, K)
    alive = (exist.float() > 0.5).view(B, 1, 1, K)
    d = d.masked_fill(own | ~alive, float("inf"))
    is_free = d.min(-1).values > free                              # (B, K, 2)

    chord = (ctrl[:, :, 3] - ctrl[:, :, 0]).norm(dim=-1, keepdim=True)          # (B, K, 1)
    for e, (tip, prev) in enumerate(((0, 1), (3, 2))):
        direction = ctrl[:, :, tip] - ctrl[:, :, prev]
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        move = direction * chord * ends[:, :, e:e + 1] * is_free[:, :, e:e + 1].float()
        new[:, :, tip] = new[:, :, tip] + move
    return new.clamp(0.0, 1.0)


def stroke_warp(x, ctrl, exist, bend=0.035, tail=0.2, free=0.06, sigma=0.03, grid=32, mask=None,
                noise=None, **unused):
    """Warp images ``x`` (B, C, H, W) along their strokes ``ctrl`` (B, K, 4, 2) / ``exist`` (B, K).

    ``mask`` (B,) bool selects the images to warp; the others come back unchanged.
    """
    B, C, H, W = x.shape
    dev = x.device
    T = 32
    pts = curve_points(ctrl.float(), T)
    moved = curve_points(bend_strokes(ctrl, exist, bend, tail, free, noise), T)
    back = (pts - moved).reshape(B, -1, 2)                         # from each moved sample back to its ink
    dest = moved.reshape(B, -1, 2)
    on = (exist.float() > 0.5).unsqueeze(-1).expand(B, ctrl.shape[1], T).reshape(B, -1)

    # backward displacement field on a coarse grid, softmax-weighted over the nearby moved samples
    g = (torch.arange(grid, device=dev, dtype=torch.float32) + 0.5) / grid
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    q = torch.stack([gx, gy], -1).view(1, -1, 2).expand(B, -1, -1)
    logit = -torch.cdist(q, dest).pow(2) / (2 * sigma ** 2)
    logit = logit.masked_fill(on.unsqueeze(1) == 0, float("-inf"))
    w = torch.softmax(logit, -1).nan_to_num(0.0)                  # an image without strokes: no motion
    field = (w @ back).view(B, grid, grid, 2).permute(0, 3, 1, 2)
    field = F.interpolate(field, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

    # the output pixel at p shows the ink that was at p + field(p)
    ys = (torch.arange(H, device=dev, dtype=torch.float32) + 0.5) / H
    xs = (torch.arange(W, device=dev, dtype=torch.float32) + 0.5) / W
    by, bx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([bx, by], -1).unsqueeze(0)
    out = F.grid_sample(x.float(), (base + field) * 2 - 1, mode="bilinear", padding_mode="zeros",
                        align_corners=False).clamp(0, 1).to(x.dtype)
    if mask is not None:
        out = torch.where(mask.view(B, 1, 1, 1), out, x)
    return out
