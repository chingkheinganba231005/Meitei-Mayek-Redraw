"""Differentiable rendering of pen strokes, and the losses that fit them to the ink.

Coordinates are in [0, 1] x [0, 1], x to the right and y down, the frame of
the network input. A stroke is a cubic Bezier curve (four control points), a
width and an existence probability. Rendering measures the distance from
every pixel centre to each stroke (T samples along the curve) and turns it
into soft ink; the strokes are combined with a soft union. Nothing here has
learnable parameters, so the only way for the network to lower the
reconstruction loss is to put its strokes where the ink is.
"""

import torch
import torch.nn.functional as F


def bernstein(T, device=None, dtype=None):
    t = torch.linspace(0, 1, T, device=device, dtype=dtype)
    return torch.stack([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3], -1)  # (T, 4)


def curve_points(ctrl, T=32):
    """(..., 4, 2) control points -> (..., T, 2) points along the curve."""
    return torch.einsum("tk,...kd->...td", bernstein(T, ctrl.device, ctrl.dtype), ctrl)


_GRIDS = {}


def pixel_grid(G, device, dtype=torch.float32):
    """Pixel centres of a G x G raster as (G*G, 2) (x, y) pairs."""
    key = (G, str(device), dtype)
    if key not in _GRIDS:
        c = (torch.arange(G, device=device, dtype=dtype) + 0.5) / G
        yy, xx = torch.meshgrid(c, c, indexing="ij")
        _GRIDS[key] = torch.stack([xx, yy], -1).view(-1, 2)
    return _GRIDS[key]


def stroke_distance(pts, grid):
    """Distance from each pixel to the nearest sample of each stroke, (B, K, P).

    With 32 samples per curve this is within a fraction of a pixel of the exact
    distance to the curve for strokes of ordinary length.
    """
    B, K, T, _ = pts.shape
    d = torch.cdist(grid.unsqueeze(0).expand(B * K, -1, -1), pts.reshape(B * K, T, 2))
    return d.min(-1).values.view(B, K, -1)


def render(ctrl, width, exist, G=48, T=32, tau=None):
    """Soft raster of the strokes.

    ctrl (B, K, 4, 2), width (B, K) and exist (B, K), all in [0, 1].
    Returns the image (B, G, G), the per-stroke distances (B, K, G*G) and the
    curve samples (B, K, T, 2).
    """
    B, K = ctrl.shape[:2]
    tau = tau or 0.5 / G  # edge softness: half a pixel
    pts = curve_points(ctrl, T)
    d = stroke_distance(pts, pixel_grid(G, ctrl.device, ctrl.dtype))
    cover = torch.sigmoid((width.unsqueeze(-1) / 2 - d) / tau) * exist.unsqueeze(-1)
    img = 1 - torch.exp(torch.log1p(-cover.clamp(max=1 - 1e-6)).sum(1))  # 1 - prod(1 - c)
    return img.view(B, G, G), d, pts


def distance_to_ink(target, thresh=0.5, steps=None):
    """Approximate distance (canvas units) from every pixel to the nearest ink pixel.

    Repeated 3x3 dilations, alternating a square and a cross, give an octagonal
    approximation of the Euclidean distance, capped at ``steps`` pixels
    (default: half the canvas). target: (B, G, G) in [0, 1].
    """
    B, G, _ = target.shape
    reached = (target > thresh).float().unsqueeze(1)
    dist = torch.zeros_like(reached)
    cross = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], device=target.device,
                         dtype=reached.dtype).view(1, 1, 3, 3)
    for i in range(steps or G // 2):
        dist = dist + (1 - reached)
        reached = F.max_pool2d(reached, 3, 1, 1) if i % 2 == 0 else (F.conv2d(reached, cross, padding=1) > 0).to(reached.dtype)
    return dist.squeeze(1) / G


def sample_map(m, pts):
    """Bilinear samples of a (B, G, G) map at points (B, N, 2) in [0, 1] -> (B, N)."""
    return F.grid_sample(m.unsqueeze(1), pts.unsqueeze(1) * 2 - 1, mode="bilinear", padding_mode="border",
                         align_corners=False).view(pts.shape[0], -1)


def fit_losses(img, d, pts, exist, target, reduce=True):
    """Reconstruction losses, averaged over the batch (or per image with ``reduce=False``).

    bce:   pixel fit, ink and paper weighted equally
    prec:  the samples of every existing stroke should lie on ink
    cover: every ink pixel should be close to some existing stroke
    count: mean existence probability (fewer strokes are preferred)
    """
    B, K, T, _ = pts.shape
    t = target.reshape(B, -1)
    r = img.reshape(B, -1).clamp(1e-5, 1 - 1e-5)
    w_ink = t.sum(1, keepdim=True).clamp_min(1.0)
    w_bg = (1 - t).sum(1, keepdim=True).clamp_min(1.0)
    bce = -(t * torch.log(r) / w_ink + (1 - t) * torch.log(1 - r) / w_bg).sum(1) / 2

    off_ink = sample_map(distance_to_ink(target), pts.reshape(B, K * T, 2)).view(B, K, T).mean(2)
    prec = (off_ink * exist).sum(1) / exist.sum(1).clamp_min(1e-3)

    # a stroke that does not exist counts as 0.1 of the canvas further away
    near = (d + 0.1 * (1 - exist).unsqueeze(-1)).min(1).values
    cover = (near * t).sum(1) / w_ink.squeeze(1)
    parts = {"bce": bce, "prec": prec, "cover": cover, "count": exist.mean(1)}
    return {k: v.mean() for k, v in parts.items()} if reduce else parts


RENDER_WEIGHTS = {"bce": 1.0, "prec": 2.0, "cover": 2.0, "count": 0.02}


def render_loss(parts, weights=RENDER_WEIGHTS):
    return sum(weights[k] * parts[k] for k in weights)


@torch.no_grad()
def parse_score(img, target, tol=1.5):
    """How well the strokes redraw the character: (precision, recall, F) per image.

    Precision is the share of redrawn ink within ``tol`` pixels of real ink,
    recall the share of real ink within ``tol`` pixels of redrawn ink. The
    tolerance makes the score fair at the low rendering resolution, where a
    one-pixel offset would otherwise count as a miss.
    """
    G = target.shape[-1]
    r, t = img > 0.5, target > 0.5
    steps = int(tol) + 2
    near_t = distance_to_ink(target, steps=steps) * G <= tol
    near_r = distance_to_ink(img, steps=steps) * G <= tol
    p = (r & near_t).flatten(1).sum(1) / r.flatten(1).sum(1).clamp_min(1)
    q = (t & near_r).flatten(1).sum(1) / t.flatten(1).sum(1).clamp_min(1)
    return p, q, 2 * p * q / (p + q).clamp_min(1e-6)
