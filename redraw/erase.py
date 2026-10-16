"""Erasing strokes: does the stroke an explanation points at decide what the network reads?

The parse splits the ink into strokes, so an explanation of a decision can be tested by deleting what it
points at (a deletion test with strokes in place of pixel blocks). Every ink pixel is given to the
existing stroke whose edge is nearest, if it lies within ``margin`` of that edge, and a stroke is erased
by setting its pixels to paper. The stroke to erase is chosen in six ways (``CHOOSERS``):

    reader, most        the stroke whose absence from the parse lowers the stroke reader's support for
                        the network's class the most (the explanation under test)
    reader, least       the stroke whose absence lowers it the least
    random              any stroke of the parse (fixed draws)
    largest             the stroke with the most ink
    ink-matched         another stroke whose amount of ink is closest to that of "reader, most"
                        (separates "this stroke matters" from "this much ink matters")
    gradient x input    the stroke with the most gradient x input of the network that reads the image,
                        summed over the stroke's pixels (a standard pixel attribution, given the same
                        strokes; a first-order estimate of what erasing each stroke does)

Every erased image is read by every network ("gradient x input" by the network it was computed for).
"""

import numpy as np
import torch

from .render import curve_points, pixel_grid, stroke_distance
from .train import amp_dtype, fetch

CHOOSERS = ("reader, most", "reader, least", "random", "largest", "ink-matched", "gradient x input")


def stroke_pixels(ctrl, width, exist, size, margin=0.04, thresh=0.5, T=32, chunk=16):
    """Which pixels of a size x size image belong to which stroke: (B, K, size, size) bool.

    A pixel belongs to the existing stroke (existence above ``thresh``) whose edge is nearest, if it
    lies within ``margin`` (canvas units) of that edge; a pixel belongs to at most one stroke.
    """
    B, K = exist.shape
    grid = pixel_grid(size, ctrl.device)
    slots = torch.arange(K, device=ctrl.device).view(1, K, 1)
    out = []
    for i in range(0, B, chunk):
        sl = slice(i, i + chunk)
        edge = stroke_distance(curve_points(ctrl[sl].float(), T), grid) - width[sl].float().unsqueeze(-1) / 2
        edge = edge.masked_fill((exist[sl] <= thresh).unsqueeze(-1), float("inf"))
        near, owner = edge.min(1)
        own = (owner.unsqueeze(1) == slots) & (near <= margin).unsqueeze(1)
        out.append(own.view(-1, K, size, size))
    return torch.cat(out)


def reader_importance(model, xn, out, pred, thresh=0.5):
    """How much the stroke reader's log-probability of ``pred`` falls when each stroke is left out of
    the parse: (B, K), -inf for strokes that do not exist. ``out`` is the network's output on ``xn``."""
    if getattr(model, "head", None) is None:
        raise ValueError("reader importance needs a network with a stroke reader")
    _, inter = model.backbone.forward_intermediates(xn, indices=model.taps)
    pts = curve_points(out["ctrl"], model.T)
    width, exist = out["width"], out["exist"]

    def support(e):
        return model.head(pts, width, e, inter[0]).float().log_softmax(-1).gather(1, pred[:, None]).squeeze(1)

    base = support(exist)
    drops = []
    for k in range(exist.shape[1]):
        left_out = exist.clone()
        left_out[:, k] = 0.0
        drops.append(base - support(left_out))
    return torch.stack(drops, 1).masked_fill(exist <= thresh, float("-inf"))


def gradient_x_input(net, x, store, pred):
    """Gradient x input of the network's log-probability of ``pred`` with respect to the ink: (B, 1, H, W)."""
    amp = amp_dtype(store.device)
    with torch.enable_grad():
        xr = x.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            o = net(store.normalize(xr, "gray"), draw=False, ink=xr)
        lp = o["logits"].float().log_softmax(-1).gather(1, pred[:, None]).sum()
        grad, = torch.autograd.grad(lp, xr)
    return (grad * x).float()


def erase(x, own, s):
    """The images ``x`` (B, 1, H, W) with stroke ``s`` (B,) of each set to paper."""
    keep = ~own[torch.arange(len(x), device=x.device), s]
    return x * keep.unsqueeze(1).to(x.dtype)


@torch.no_grad()
def explain(parser, x, store, margin=0.04, thresh=0.5):
    """The parser's output on ``x``, its class, the reader's importance of each stroke and the pixels of
    each stroke."""
    amp = amp_dtype(store.device)
    xn = store.normalize(x, "gray")
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        o = parser(xn, draw=False, ink=x)
        pred = o["logits"].float().argmax(1)
        imp = reader_importance(parser, xn, o, pred, thresh)
    own = stroke_pixels(o["ctrl"], o["width"], o["exist"], x.shape[-1], margin, thresh)
    return o, pred, imp, own


@torch.no_grad()
def erase_test(parser, nets, idx, store, bs=64, margin=0.04, seed=0, thresh=0.5):
    """Erase one stroke of every image, chosen in each of the ``CHOOSERS`` ways, and let the networks read it.

    ``parser`` (a network with strokes and a stroke reader) supplies the strokes and the reader's choices;
    ``nets`` {name: network} read the images and may include ``parser``. Returns numpy arrays:
    ``pred[name]`` the class read from the intact image; ``erased[chooser][name]`` the class read after
    erasing; ``stroke[key]`` the stroke erased and ``ink_erased[key]`` the share of the ink it held,
    where ``key`` is the chooser, or "gradient x input, <name>" for that network's own choice;
    ``strokes`` the number of strokes in the parse.
    """
    parser.eval()
    for net in nets.values():
        net.eval()
    amp = amp_dtype(store.device)
    res = getattr(parser, "input_res", None)
    idx = np.asarray(idx)
    gen = torch.Generator().manual_seed(seed)
    shared = [c for c in CHOOSERS if c != "gradient x input"]
    keys = shared + [f"gradient x input, {n}" for n in nets]
    out = {"strokes": [], "pred": {n: [] for n in nets}, "erased": {c: {n: [] for n in nets} for c in CHOOSERS},
           "stroke": {k: [] for k in keys}, "ink_erased": {k: [] for k in keys}}

    def read(net, xv):
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            return net(store.normalize(xv, "gray"), draw=False, ink=xv)["logits"].float().argmax(1)

    for i in range(0, len(idx), bs):
        x = fetch(store, idx[i:i + bs], res)
        o, pred, imp, own = explain(parser, x, store, margin, thresh)
        exist = o["exist"].float() > thresh
        ink = (own.to(x.dtype) * x).flatten(2).sum(-1)                            # (B, K)
        total = x.flatten(1).sum(1).clamp_min(1e-6)
        draws = torch.rand(exist.shape, generator=gen).to(x.device)
        choice = {"reader, most": imp.argmax(1),
                  "reader, least": imp.masked_fill(~exist, float("inf")).argmin(1),
                  "random": draws.masked_fill(~exist, -1.0).argmax(1),
                  "largest": ink.masked_fill(~exist, -1.0).argmax(1)}
        most = choice["reader, most"]
        rows_ = torch.arange(len(x), device=x.device)
        gap = (ink - ink[rows_, most].unsqueeze(1)).abs().masked_fill(~exist, float("inf"))
        gap[rows_, most] = float("inf")
        choice["ink-matched"] = gap.argmin(1)
        preds = {n: read(net, x) for n, net in nets.items()}
        for n, net in nets.items():
            score = (gradient_x_input(net, x, store, preds[n]) * own.to(x.dtype)).flatten(2).sum(-1)
            choice[f"gradient x input, {n}"] = score.masked_fill(~exist, float("-inf")).argmax(1)
        rows = torch.arange(len(x), device=x.device)
        for key, s in choice.items():
            out["stroke"][key].append(s.cpu())
            out["ink_erased"][key].append((ink[rows, s] / total).cpu())
            erased = erase(x, own, s)
            if key in shared:
                for n, net in nets.items():
                    out["erased"][key][n].append(read(net, erased).cpu())
            else:
                n = key[len("gradient x input, "):]
                out["erased"]["gradient x input"][n].append(read(nets[n], erased).cpu())
        out["strokes"].append(exist.sum(1).cpu())
        for n in nets:
            out["pred"][n].append(preds[n].cpu())

    def cat(v):
        return {k: cat(w) for k, w in v.items()} if isinstance(v, dict) else torch.cat(v).numpy()

    return cat(out)
