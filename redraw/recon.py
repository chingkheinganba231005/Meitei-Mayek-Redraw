"""How well a set of strokes redraws a character, two baselines, and how stable a parse is.

Metrics (``evaluate``) compare a rendering of the strokes with the ink on one evaluation canvas
(``R`` x ``R`` pixels, default 64): the ink image (128 px, ink 1, paper 0) is area-averaged to R, the
strokes are rendered at R with the exact polyline distance (``render(..., exact=True)``), and the
foreground of both is value > 0.5. Per image:

    iou, dice                overlap of the two foregrounds, no tolerance
    precision@t, recall@t    share of drawn pixels within t pixels of ink, and of ink pixels within t
                             pixels of drawn ones (exact Euclidean distance between pixel centres; on a
                             pixel grid t = 0.5 is the same as t = 0)
    f@t                      their harmonic mean, per image
    centreline               symmetric mean distance (pixels) between the curve centrelines and the
                             skeleton of the ink; also each direction
    curves, arc_length, redundancy
                             strokes that exist (> 0.5), their total length in pixels, and how much the
                             drawn strokes overlap (sum of the stroke areas / area of their union - 1)

Baselines: ``skeleton_curves`` (threshold, skeletonise, trace the skeleton graph, fit cubic Beziers,
estimate widths from the distance transform; a junction splits a pen stroke into several curves) and
``optimise_strokes`` (fit K strokes to each image separately by gradient descent with the same renderer
and drawing loss as the network, i.e. iterative rather than amortised analysis by synthesis).

Stability: ``affine_view`` / ``map_points`` move a parse of a transformed image back to the original
frame; ``union_chamfer`` compares the drawn centrelines as point sets (indifferent to how the ink is
split into strokes and to their order or direction) and ``match_curves`` matches strokes one to one.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .render import RENDER_WEIGHTS, curve_points, fit_losses, render

TOLS = (0.0, 1.0, 1.5)


# ---------------------------------------------------------------- metrics

def ink_canvas(x, R):
    """Ink images (B, 1, H, W) in [0, 1] -> (B, R, R) by area averaging."""
    return F.adaptive_avg_pool2d(x.float(), R).squeeze(1)


def tolerant_prf(pred, true, tols=TOLS):
    """Per-image precision, recall and F at each tolerance (pixels). pred, true: (B, R, R) bool numpy.
    An empty drawing has precision nan; empty ink has recall nan."""
    out = {f"{k}@{t:g}": np.full(len(pred), np.nan) for t in tols for k in ("precision", "recall", "f")}
    for i, (p, q) in enumerate(zip(pred, true)):
        to_true = ndimage.distance_transform_edt(~q) if q.any() else None
        to_pred = ndimage.distance_transform_edt(~p) if p.any() else None
        for t in tols:
            pr = float((to_true[p] <= t + 1e-6).mean()) if p.any() and to_true is not None else np.nan
            rc = float((to_pred[q] <= t + 1e-6).mean()) if q.any() and to_pred is not None else np.nan
            out[f"precision@{t:g}"][i], out[f"recall@{t:g}"][i] = pr, rc
            out[f"f@{t:g}"][i] = 2 * pr * rc / (pr + rc) if pr + rc > 0 else (0.0 if pr == pr and rc == rc else np.nan)
    return out


def overlap(pred, true):
    inter = (pred & true).reshape(len(pred), -1).sum(1)
    a, b = pred.reshape(len(pred), -1).sum(1), true.reshape(len(true), -1).sum(1)
    union = a + b - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
        dice = np.where(a + b > 0, 2 * inter / np.maximum(a + b, 1), 1.0)
    return iou, dice


def centreline_distance(curves, true, R):
    """curves: per image an (N, 2) array of centreline points in canvas units; true: (B, R, R) bool.
    Returns (symmetric, skeleton->curves, curves->skeleton) mean distances in pixels (nan if either is empty)."""
    sym, s2c, c2s = (np.full(len(true), np.nan) for _ in range(3))
    for i, (c, q) in enumerate(zip(curves, true)):
        sk = np.argwhere(skeletonize(q))
        if len(c) == 0 or len(sk) == 0:
            continue
        sk = (sk[:, ::-1] + 0.5)                      # pixel centres as (x, y) in pixels
        cp = np.asarray(c, np.float64) * R
        a = cKDTree(cp).query(sk)[0].mean()
        b = cKDTree(sk).query(cp)[0].mean()
        s2c[i], c2s[i], sym[i] = a, b, (a + b) / 2
    return sym, s2c, c2s


def stroke_stats(ctrl, width, exist, R, thresh=0.5):
    """Number of existing strokes, their total length (pixels) and their overlap (redundancy)."""
    on = exist > thresh
    pts = curve_points(ctrl.float(), 64)
    length = (pts[:, :, 1:] - pts[:, :, :-1]).norm(dim=-1).sum(-1) * R
    _, d, _ = render(ctrl.float(), width.float(), exist.float(), G=R, T=64, exact=True)
    each = (d <= width.float().unsqueeze(-1) / 2) & on.unsqueeze(-1)            # (B, K, P) per-stroke masks
    union = each.any(1).sum(1).clamp_min(1)
    redundancy = each.sum((1, 2)) / union - 1
    redundancy = torch.where(each.any(1).any(1), redundancy, torch.zeros_like(redundancy))
    return {"curves": on.sum(1).cpu().numpy(), "arc_length": (length * on).sum(1).cpu().numpy(),
            "redundancy": redundancy.float().cpu().numpy()}


def evaluate(ctrl, width, exist, ink, R=64, tols=TOLS, thresh=0.5):
    """All reconstruction metrics for a batch: strokes (B, K, ...) against ink images (B, 1, H, W)."""
    target = ink_canvas(ink, R)
    drawn, _, _ = render(ctrl.float(), width.float(), exist.float(), G=R, T=64, exact=True)
    pred, true = (drawn > 0.5).cpu().numpy(), (target > 0.5).cpu().numpy()
    out = dict(zip(("iou", "dice"), overlap(pred, true)))
    out.update(tolerant_prf(pred, true, tols))
    pts = curve_points(ctrl.float(), 64).cpu().numpy()
    on = (exist > thresh).cpu().numpy()
    curves = [pts[i][on[i]].reshape(-1, 2) for i in range(len(pts))]
    out["centreline"], out["centreline_ink_to_strokes"], out["centreline_strokes_to_ink"] = \
        centreline_distance(curves, true, R)
    out.update(stroke_stats(ctrl, width, exist, R, thresh))
    return out


# ---------------------------------------------------------------- baseline 1: skeleton tracing

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_skeleton(skel):
    """Split a skeleton (bool H x W) into pixel paths between endpoints and junctions; closed loops
    become one path each. Returns a list of (n, 2) integer arrays of (row, col)."""
    pix = set(map(tuple, np.argwhere(skel)))

    def nbrs(p):
        return [(p[0] + dy, p[1] + dx) for dy, dx in _NEIGHBOURS if (p[0] + dy, p[1] + dx) in pix]

    degree = {p: len(nbrs(p)) for p in pix}
    nodes = {p for p, d in degree.items() if d != 2}
    seen, paths = set(), []
    for n in nodes:
        if degree[n] == 0:
            paths.append(np.array([n]))
            continue
        for m in nbrs(n):
            if (n, m) in seen:
                continue
            path, prev, cur = [n, m], n, m
            seen.update({(n, m), (m, n)})
            while cur not in nodes:
                nxt = [q for q in nbrs(cur) if q != prev and (cur, q) not in seen]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                seen.update({(prev, cur), (cur, prev)})
                path.append(cur)
            paths.append(np.array(path))
    on_path = {p for path in paths for p in map(tuple, path)}
    for start in sorted(pix - on_path - nodes):                    # loops with no endpoint or junction
        if start in on_path:
            continue
        path, prev, cur = [start], None, start
        while True:
            nxt = [q for q in nbrs(cur) if q != prev and q not in on_path and q not in path]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
        path.append(start)
        on_path.update(map(tuple, path))
        paths.append(np.array(path))
    return paths


def _fit_cubic(points):
    """Least-squares cubic Bezier through ordered points (m, 2) with fixed end points.
    Returns control points (4, 2), the largest deviation and where it is."""
    m = len(points)
    p0, p3 = points[0], points[-1]
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    dense = np.linspace(0, 1, 256)
    basis = np.stack([(1 - dense) ** 3, 3 * (1 - dense) ** 2 * dense, 3 * (1 - dense) * dense ** 2, dense ** 3], 1)
    if m < 4 or d[-1] == 0:
        ctrl = np.stack([p0, p0 + (p3 - p0) / 3, p0 + 2 * (p3 - p0) / 3, p3])
    else:
        t = d / d[-1]                                          # chord-length start, then reparametrise
        for _ in range(30):
            B = np.stack([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3], 1)
            rhs = points - np.outer(B[:, 0], p0) - np.outer(B[:, 3], p3)
            inner, *_ = np.linalg.lstsq(B[:, 1:3], rhs, rcond=None)
            ctrl = np.stack([p0, inner[0], inner[1], p3])
            t_new = dense[cKDTree(basis @ ctrl).query(points)[1]]
            t_new[0], t_new[-1] = 0.0, 1.0
            if np.abs(t_new - t).max() < 1e-3:
                break
            t = t_new
    dev = cKDTree(basis @ ctrl).query(points)[0]
    return ctrl, float(dev.max()), int(dev.argmax())


def fit_beziers(points, tol):
    """Cubic Beziers through ordered points, split at the worst point until every point is within ``tol``."""
    ctrl, err, worst = _fit_cubic(points)
    if err <= tol or len(points) < 7:
        return [ctrl]
    k = min(max(worst, 3), len(points) - 4)
    return fit_beziers(points[:k + 1], tol) + fit_beziers(points[k:], tol)


def skeleton_curves(ink, thresh=0.5, tol=1.0, min_pixels=3):
    """Strokes of one ink image (R x R, values in [0, 1]) by skeleton tracing.
    ``tol`` (pixels) sets how closely each Bezier follows its skeleton path: larger, fewer curves.
    Returns control points (n, 4, 2) and widths (n,) in canvas units."""
    R = ink.shape[-1]
    mask = ink > thresh
    if not mask.any():
        return np.zeros((0, 4, 2), np.float32), np.zeros(0, np.float32)
    skel = skeletonize(mask)
    dist = ndimage.distance_transform_edt(mask)
    ctrls, widths = [], []
    for path in trace_skeleton(skel):
        if len(path) < min_pixels and len(path) > 1:
            continue                                    # tiny links inside a junction
        xy = (path[:, ::-1].astype(np.float64) + 0.5) / R
        w = 2 * float(np.median(dist[path[:, 0], path[:, 1]])) / R
        for c in fit_beziers(xy, tol / R):
            ctrls.append(c)
            widths.append(max(w, 1.0 / R))
    if not ctrls:
        return np.zeros((0, 4, 2), np.float32), np.zeros(0, np.float32)
    return np.clip(np.array(ctrls, np.float32), 0, 1), np.array(widths, np.float32)


def pad_strokes(items, K=None):
    """List of (ctrl (n, 4, 2), width (n,)) -> padded tensors ctrl (B, K, 4, 2), width, exist."""
    K = K or max(1, max(len(c) for c, _ in items))
    B = len(items)
    ctrl, width, exist = torch.zeros(B, K, 4, 2), torch.full((B, K), 0.05), torch.zeros(B, K)
    for i, (c, w) in enumerate(items):
        n = min(len(c), K)
        if n:
            ctrl[i, :n], width[i, :n], exist[i, :n] = torch.as_tensor(c[:n]), torch.as_tensor(w[:n]), 1.0
    return ctrl, width, exist


# ---------------------------------------------------------------- baseline 2: per-image optimisation

def grid_init(B, K, device=None):
    """Short horizontal strokes on a grid of reference points, as the stroke decoder starts."""
    side = int(np.ceil(np.sqrt(K)))
    g = (torch.arange(side, dtype=torch.float32) + 0.5) / side
    ref = torch.stack(torch.meshgrid(g, g, indexing="xy"), -1).view(-1, 2)[:K]
    off = torch.tensor([-0.3, -0.1, 0.1, 0.3]).view(1, 4, 1) * torch.tensor([0.4, 0.0])
    ctrl = (ref.view(K, 1, 2) + off).clamp(0.02, 0.98)
    return ctrl.unsqueeze(0).expand(B, -1, -1, -1).clone().to(device)


def optimise_strokes(target, K=12, steps=300, lr=0.03, G=48, T=32, weights=RENDER_WEIGHTS):
    """Fit K strokes to each target (B, G, G) separately, minimising the drawing loss with Adam.
    Returns ctrl (B, K, 4, 2), width (B, K), exist (B, K) and the final per-image loss."""
    B, device = len(target), target.device
    c = torch.logit(grid_init(B, K, device)).requires_grad_(True)
    w = torch.full((B, K), -1.0, device=device, requires_grad=True)     # width 0.02 + 0.18 * sigmoid(-1)
    e = torch.zeros(B, K, device=device, requires_grad=True)
    opt = torch.optim.Adam([c, w, e], lr=lr)

    def loss_of():
        ctrl, width, exist = torch.sigmoid(c), 0.02 + 0.18 * torch.sigmoid(w), torch.sigmoid(e)
        img, d, pts = render(ctrl, width, exist, G=G, T=T)
        parts = fit_losses(img, d, pts, exist, target, reduce=False)
        return sum(weights[k] * parts[k] for k in weights)              # (B,): images do not interact

    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss_of().sum().backward()
            opt.step()
    with torch.no_grad():
        final = loss_of()
    return torch.sigmoid(c).detach(), (0.02 + 0.18 * torch.sigmoid(w)).detach(), torch.sigmoid(e).detach(), final


# ---------------------------------------------------------------- stability

def affine_view(x, angle=0.0, scale=1.0, dx=0.0, dy=0.0):
    """Rotate (degrees), zoom and shift (canvas units) images (B, C, H, W). Returns the moved images and
    theta (B, 2, 3), which maps a point of the moved image back to the original (normalised coordinates)."""
    B = x.shape[0]
    a = np.deg2rad(angle)
    theta = torch.tensor([[np.cos(a) / scale, -np.sin(a) / scale, -2 * dx],
                          [np.sin(a) / scale, np.cos(a) / scale, -2 * dy]], dtype=x.dtype, device=x.device)
    theta = theta.unsqueeze(0).expand(B, 2, 3)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False), theta


def map_points(points, theta):
    """Points (B, ..., 2) in [0, 1] of the moved image -> the same points in the original image."""
    shape = points.shape
    u = points.to(theta.dtype).reshape(shape[0], -1, 2) * 2 - 1
    v = torch.einsum("bij,bnj->bni", theta[:, :, :2], u) + theta[:, :, 2].unsqueeze(1)
    return ((v + 1) / 2).reshape(shape)


def union_chamfer(pts_a, on_a, pts_b, on_b):
    """Symmetric mean distance between the centreline points of the existing strokes of two parses
    (canvas units); pts (K, T, 2), on (K,) bool. nan if either parse is empty."""
    a, b = pts_a[on_a].reshape(-1, 2), pts_b[on_b].reshape(-1, 2)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float((cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()) / 2)


def match_curves(pts_a, on_a, pts_b, on_b):
    """One-to-one matching of the existing strokes of two parses (Hungarian, cost = mean point distance,
    whichever direction fits better). Returns the mean matched cost (canvas units) and the number of
    strokes left unmatched."""
    a, b = pts_a[on_a], pts_b[on_b]
    if len(a) == 0 or len(b) == 0:
        return np.nan, abs(len(a) - len(b))
    fwd = np.linalg.norm(a[:, None] - b[None], axis=-1).mean(-1)
    rev = np.linalg.norm(a[:, None] - b[None, :, ::-1], axis=-1).mean(-1)
    cost = np.minimum(fwd, rev)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean()), abs(len(a) - len(b))
