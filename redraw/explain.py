"""Do the strokes matter to the decision, and what does the hypothesis check change?

Three kinds of intervention (see also ``erase.erase_test``, which edits the image):

    representation_test     leave one stroke out of what the stroke reader sees while the image, and so
                            the image classifier's logits, stay fixed. Measures how much the fused
                            decision depends on that stroke through the reader; the image classifier
                            is a bypass that can keep a decision unchanged.
    render_parses           draw every parse as an image, so that a separate classifier trained on the
                            drawings alone can measure how much class information the strokes keep.
                            This is not a test of the original network's reasoning.
    (erase.erase_test)      erase the ink of one stroke and let whole networks read the edited image.

Check analysis: ``check_transitions`` splits the decisions the check changes into right->wrong,
wrong->right and wrong->other-wrong; ``candidate_margins`` gives each image's energy margin;
``refit`` re-optimises the fitter's outputs for every candidate by gradient descent, to see whether the
one-pass fitter fits some classes worse than it could (an amortisation gap) and whether that changes
which candidate explains the ink best.
"""

import numpy as np
import torch

from .erase import reader_importance
from .render import curve_points, render
from .train import amp_dtype, fetch


def reader_logits_without(model, xn, out):
    """Stroke-reader logits with each stroke left out in turn: (B, K, C)."""
    _, inter = model.backbone.forward_intermediates(xn, indices=model.taps)
    pts = curve_points(out["ctrl"], model.T)
    width, exist = out["width"], out["exist"]
    logits = []
    for k in range(exist.shape[1]):
        e = exist.clone()
        e[:, k] = 0.0
        logits.append(model.head(pts, width, e, inter[0]).float())
    return torch.stack(logits, 1)


@torch.no_grad()
def representation_test(model, idx, store, bs=128, seed=0, thresh=0.5):
    """Leave out of the reader's input the stroke the reader finds most important, the least important or
    a random one; the image classifier's logits stay as they are. Numpy arrays per image: the fused class
    before and after, and the fall in the fused log-probability of the original class."""
    model.eval()
    amp = amp_dtype(store.device)
    res = getattr(model, "input_res", None)
    idx = np.asarray(idx)
    gen = torch.Generator().manual_seed(seed)
    rows = {"strokes": [], "pred": [], "vec_pred": []}
    keys = ("most", "least", "random")
    for k in keys:
        rows[f"pred_{k}"], rows[f"drop_{k}"], rows[f"vec_pred_{k}"] = [], [], []
    for i in range(0, len(idx), bs):
        x = fetch(store, idx[i:i + bs], res)
        xn = store.normalize(x, "gray")
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            o = model(xn, draw=False)
            without = reader_logits_without(model, xn, o)
        img, vec = o["img"].float(), o["vec"].float()
        fused = img + vec
        pred = fused.argmax(1)
        exist = o["exist"].float() > thresh
        lp = vec.log_softmax(-1).gather(1, pred[:, None])
        drop_reader = (lp - without.log_softmax(-1).gather(2, pred.view(-1, 1, 1).expand(-1, without.shape[1], 1))
                       .squeeze(-1)).masked_fill(~exist, float("-inf"))
        draws = torch.rand(exist.shape, generator=gen).to(x.device).masked_fill(~exist, -1.0)
        pick = {"most": drop_reader.argmax(1),
                "least": drop_reader.masked_fill(~exist, float("inf")).argmin(1),
                "random": draws.argmax(1)}
        r = torch.arange(len(x), device=x.device)
        base_lp = fused.log_softmax(-1).gather(1, pred[:, None]).squeeze(1)
        for k, s in pick.items():
            f2 = img + without[r, s]
            rows[f"pred_{k}"].append(f2.argmax(1).cpu())
            rows[f"vec_pred_{k}"].append(without[r, s].argmax(1).cpu())
            rows[f"drop_{k}"].append((base_lp - f2.log_softmax(-1).gather(1, pred[:, None]).squeeze(1)).cpu())
        rows["strokes"].append(exist.sum(1).cpu())
        rows["pred"].append(pred.cpu())
        rows["vec_pred"].append(vec.argmax(1).cpu())
    return {k: torch.cat(v).numpy() for k, v in rows.items()}


@torch.no_grad()
def render_parses(model, idx, store, R=128, bs=128, thresh=0.0):
    """The parse of every image drawn as an R x R ink image (uint8), with the exact polyline distance.
    Strokes are drawn with their existence probability, as the renderer does in training."""
    model.eval()
    amp = amp_dtype(store.device)
    res = getattr(model, "input_res", None)
    idx = np.asarray(idx)
    out = np.empty((len(idx), R, R), np.uint8)
    for i in range(0, len(idx), bs):
        x = fetch(store, idx[i:i + bs], res)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            o = model(store.normalize(x, "gray"), draw=False)
        exist = o["exist"].float() * (o["exist"].float() > thresh)
        img, _, _ = render(o["ctrl"].float(), o["width"].float(), exist, G=R, T=64, exact=True)
        out[i:i + len(x)] = (img.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()
    return out


def check_transitions(fused_pred, final_pred, y):
    """What the hypothesis check did to each decision (counts)."""
    f_ok, g_ok = fused_pred == y, final_pred == y
    changed = fused_pred != final_pred
    return {"unchanged right": int((f_ok & g_ok).sum()), "unchanged wrong": int((~f_ok & ~changed).sum()),
            "wrong -> right": int((~f_ok & g_ok).sum()), "right -> wrong": int((f_ok & ~g_ok).sum()),
            "wrong -> other wrong": int((~f_ok & ~g_ok & changed).sum()), "images": int(len(y))}


def candidate_margins(cand, energy, y):
    """Per image: is the true class among the candidates, the energy margin of the true class over the
    best rival candidate (positive: the true class explains the ink better), and whether the energy alone
    picks the true class among the candidates."""
    hit = cand == y[:, None]
    in_cand = hit.any(1)
    e_true = np.where(in_cand, np.where(hit, energy, 0).sum(1), np.nan)
    e_rival = np.where(hit, np.inf, energy).min(1)
    return {"in_candidates": in_cand, "margin": e_rival - e_true,
            "energy_pick_right": in_cand & (cand[np.arange(len(cand)), energy.argmin(1)] == y)}


def refit(model, x, store, steps=100, lr=0.05, labels=None):
    """Re-optimise the fitter's outputs (affine map, offsets, width and existence changes) for every
    candidate and template by gradient descent on the same energy, within the same bounds.
    Returns the candidates and their energies from the one-pass fitter and after refitting."""
    model.eval()
    hdr = model.hdr
    with torch.no_grad():
        xn = store.normalize(x, "gray")
        o = model(xn, draw=False, ink=x)
        _, inter = model.backbone.forward_intermediates(xn, indices=model.taps)
        mem, mpos = model.decoder(inter)[3:]
        cand = o["cand"]
        templates, fitted = hdr.fit(mem, mpos, cand)
        target = torch.nn.functional.interpolate(x[:, :1].float(), size=(model.hdr_G, model.hdr_G),
                                                 mode="area").squeeze(1)
        before = hdr.cost(hdr.bend(templates, fitted), templates[2], fitted, target, G=model.hdr_G)

    def inv_tanh(v, bound):
        return torch.atanh((v / bound).clamp(-0.999, 0.999))

    raw = {"a": inv_tanh(fitted["a"].float(), 1.0).clone().requires_grad_(True),
           "off": inv_tanh(fitted["off"].float(), hdr.max_offset).clone().requires_grad_(True),
           "dlogit": fitted["dlogit"].float().clone().requires_grad_(True),
           "ws": inv_tanh(torch.log(fitted["wscale"].float()) / hdr.width_range, 1.0).clone().requires_grad_(True)}
    opt = torch.optim.Adam(raw.values(), lr=lr)
    tmpl = tuple(t.detach().float() for t in templates)

    def energy():
        a = torch.tanh(raw["a"])
        f = {"a": a, "A": hdr._affine(a), "off": torch.tanh(raw["off"]) * hdr.max_offset, "dlogit": raw["dlogit"],
             "wscale": torch.exp(hdr.width_range * torch.tanh(raw["ws"]))}
        return hdr.cost(hdr.bend(tmpl, f), tmpl[2], f, target, G=model.hdr_G)

    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            energy().sum().backward()
            opt.step()
    with torch.no_grad():
        after = energy()
    return {"cand": cand.cpu().numpy(), "energy_fitter": before.float().cpu().numpy(),
            "energy_refit": after.float().cpu().numpy()}


__all__ = ["reader_logits_without", "representation_test", "render_parses", "check_transitions",
           "candidate_margins", "refit", "reader_importance"]
