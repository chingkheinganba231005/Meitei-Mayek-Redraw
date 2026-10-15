"""Training loop, weight averaging, prediction and stroke-parse evaluation.

A config (see ``model.CONFIGS``) sets the backbone and recipe, and for the
stroke models: ``K`` (strokes), ``render_weight`` (0 = strokes are never
compared with the ink), ``head_weight`` (weight of the two separate head
losses), ``jitter`` (noise on the stroke samples the stroke head sees) and
``new_lr`` (learning-rate multiplier for the parts that start from scratch).
With ``hdr`` the hypothesis check is added (``hdr_fit``, ``hdr_disc``,
``hdr_wrong``, ``template_lr``). ``strokewarp`` (a dict, see
``strokewarp.STROKEWARP``) deforms part of each batch stroke by stroke along the
parse of the network's own weight-averaged copy. ``route`` (see ``model.StrokeNet``) decides which
losses reach which parts. ``res`` resizes the cached images on the GPU before they enter (and before
augmentation). For the low-data runs, which have hundreds of very short epochs: ``val_every``
(epochs between validation checks, default 5), ``warmup_steps`` (learning-rate warm-up, default
3 epochs), ``ckpt_every`` (epochs between checkpoints, default 1) and ``log_every`` (default 1).

The loss is cross-entropy on the combined logits of the two heads, plus
``head_weight`` times the cross-entropy of each head on its own, plus
``render_weight`` times the reconstruction loss of the redrawn character (see
``render.fit_losses``). The hypothesis check adds two terms that train
different parts on purpose:

- fit: the cost of each candidate's bent template. It trains the fitter (and
  the backbone) to explain the image as well as it can under every
  hypothesis, right or wrong, and the true class's templates to explain
  images of their class. A template never learns from another class's images.
- margin: with the fitter frozen, the right class's templates must explain
  the image better than each wrong candidate's by ``hdr_m``. It trains only
  the templates. The fitter is never rewarded for fitting a wrong class
  badly, so its costs stay honest evidence.
- discrimination: cross-entropy (no label smoothing, so it can only ask for
  more evidence for the right class, never less) of the re-ranked logits with
  the fused logits held fixed. It trains the templates and the two scalars
  that turn costs into logits, on the images the two heads get wrong or nearly wrong.

Checkpoints: the resumable state is written to local disk after every epoch
and copied to ``work`` (e.g. Google Drive) every ``DRIVE_EVERY`` epochs.
"""

import copy
import math
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .augment import AUG_PRESETS, augment, view
from .files import must_write, write_file
from .model import make_model
from .render import fit_losses, parse_score, render_loss
from .strokewarp import STROKEWARP, rewrite_noise, stroke_warp

DRIVE_EVERY = 5
LOCAL = Path(os.environ.get("REDRAW_LOCAL_CKPT", Path(tempfile.gettempdir()) / "redraw_ckpt"))
HEADS = ("logits", "fused", "img", "vec")
TEMPLATES = ("hdr.t_ctrl", "hdr.t_width", "hdr.t_exist")


def safe_save(obj, path, tries=4):
    return write_file(lambda f: torch.save(obj, f), path, tries)


def _load_state(paths, device, log):
    best = None
    for p in paths:
        if not p.exists():
            continue
        try:
            ck = torch.load(p, map_location=device, weights_only=False)
        except Exception as e:  # half-written file after a crash
            log(f"ignoring unreadable checkpoint {p} ({type(e).__name__})")
            continue
        if best is None or ck["epoch"] > best["epoch"]:
            best = ck
    return best


def amp_dtype(device):
    return torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else None


class EMA:
    """Exponential moving average of the weights; this copy is what gets evaluated and kept."""

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay, self.updates = decay, 0
        for p in self.module.parameters():
            p.requires_grad_(False)
        pairs = list(zip(self.module.state_dict().values(), model.state_dict().values()))
        self.floats = [list(t) for t in zip(*[(e, m) for e, m in pairs if e.dtype.is_floating_point])]
        self.others = [(e, m) for e, m in pairs if not e.dtype.is_floating_point]

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        torch._foreach_lerp_(self.floats[0], self.floats[1], 1 - d)
        for e, m in self.others:
            e.copy_(m)


def param_groups(model, lr, wd, new_lr=1.0, template_lr=10.0):
    """Backbone at the base rate; new parts at ``new_lr`` times it; the class templates (a few
    thousand numbers) at ``template_lr`` times it and without weight decay."""
    new = {id(p) for p in model.new_parameters()}
    groups = {}
    for name, p in model.named_parameters():
        if name in TEMPLATES:
            key = ("template", False)
        else:
            decay = not (p.ndim <= 1 or name.endswith(".bias") or name.endswith(".ref"))
            key = ("new" if id(p) in new else "backbone", decay)
        groups.setdefault(key, []).append(p)
    scale = {"backbone": 1.0, "new": new_lr, "template": template_lr}
    return [{"params": ps, "weight_decay": wd if decay else 0.0, "lr_scale": scale[kind]}
            for (kind, decay), ps in groups.items()]


def fetch(store, idx, res=None):
    """Cached ink images (B, 1, H, W) in [0, 1], resized to ``res`` x ``res`` if it differs."""
    x = store.fetch(idx, "gray")
    if res and res != x.shape[-1]:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False).clamp_(0, 1)
    return x


def ink_target(x, G):
    """Augmented ink image (B, 1, H, W) in [0, 1] -> the G x G picture the strokes must redraw."""
    return F.interpolate(x[:, :1].float(), size=(G, G), mode="area").squeeze(1)


def compute_loss(model, out, x_ink, y, crit, cfg):
    # logits come out of the mixed-precision forward pass in bfloat16; the loss must be float32
    # (rounds 1 and 2 computed it in bfloat16, which only adds gradient noise, the same for every network)
    loss = crit(out["fused"].float(), y)
    parts = {}
    if "vec" in out:
        hw = cfg.get("head_weight", 0.5)
        loss = loss + hw * (crit(out["img"].float(), y) + crit(out["vec"].float(), y))
    if "render" in out and cfg.get("render_weight", 1.0) > 0:
        with torch.autocast(x_ink.device.type, enabled=False):
            parts = fit_losses(out["render"], out["dist"], out["pts"], out["exist"], ink_target(x_ink, model.G))
            loss = loss + cfg["render_weight"] * render_loss(parts)
    if "cost" in out:
        live = out["live"]
        w = torch.where(live, 1.0, cfg.get("hdr_wrong", 0.5))
        fit = (out["cost"] * w).sum() / w.sum()
        frozen = out["cost_frozen"]
        right = (frozen * live).sum(1, keepdim=True)          # exactly one true candidate per row
        margin = (F.relu(cfg.get("hdr_m", 0.1) + right - frozen) * ~live).sum() / (~live).sum().clamp_min(1)
        disc = F.cross_entropy(out["verified"], y)
        loss = (loss + cfg.get("hdr_fit", 1.0) * fit + cfg.get("hdr_margin", 1.0) * margin
                + cfg.get("hdr_disc", 1.0) * disc)
        parts.update(hdr_fit=fit, hdr_margin=margin, hdr_disc=disc)
    return loss, parts


@torch.no_grad()
def predict(model, idx, cfg, store, views=("id",), bs=512, heads=HEADS):
    """Softmax probabilities of each head, averaged over the test-time views: {head: (N, C)}."""
    model.eval()
    amp = amp_dtype(store.device)
    res = getattr(model, "input_res", None)
    idx = np.asarray(idx)
    out = {h: [] for h in heads}
    for i in range(0, len(idx), bs):
        x = fetch(store, idx[i:i + bs], res)
        acc = {}
        for v in views:
            xv = view(x, v)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                o = model(store.normalize(xv, "gray"), draw=False, ink=xv)
            for h in heads:
                if h in o:
                    acc[h] = acc.get(h, 0) + o[h].float().softmax(-1)
        for h in acc:
            out[h].append((acc[h] / len(views)).cpu())
    return {h: torch.cat(v).numpy() for h, v in out.items() if v}


@torch.no_grad()
def parse_eval(model, idx, store, bs=256):
    """How well the strokes redraw the characters: mean precision, recall, F and stroke count."""
    model.eval()
    amp = amp_dtype(store.device)
    res = getattr(model, "input_res", None)
    idx = np.asarray(idx)
    rows = []
    for i in range(0, len(idx), bs):
        x = fetch(store, idx[i:i + bs], res)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            o = model(store.normalize(x, "gray"))
        p, r, f = parse_score(o["render"].float(), ink_target(x, model.G))
        rows.append(torch.stack([p, r, f, (o["exist"] > 0.5).float().sum(1)], 1).cpu())
    m = torch.cat(rows).mean(0).tolist()
    return dict(zip(("precision", "recall", "f", "strokes"), m))


@torch.no_grad()
def verify_eval(model, idx, store, labels, bs=512):
    """What the hypothesis check contributes, on images ``idx``.

    topk_recall: the true class is among the candidates; cost_only: the cheapest redrawing alone
    picks the true class; fused / final: accuracy before / after the check; rescued / broken:
    images the check turns from wrong to right / from right to wrong.
    """
    model.eval()
    amp = amp_dtype(store.device)
    res = getattr(model, "input_res", None)
    idx = np.asarray(idx)
    rows = []
    for i in range(0, len(idx), bs):
        chunk = idx[i:i + bs]
        x = fetch(store, chunk, res)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            o = model(store.normalize(x, "gray"), draw=False, ink=x)
        y = torch.as_tensor(labels[chunk], device=x.device)
        pick = o["cand"].gather(1, o["cost"].float().argmin(1, keepdim=True)).squeeze(1)
        rows.append(torch.stack([(o["cand"] == y[:, None]).any(1), pick == y,
                                 o["fused"].argmax(1) == y, o["logits"].argmax(1) == y], 1).cpu())
    m = torch.cat(rows)
    return {"topk_recall": m[:, 0].float().mean().item(), "cost_only": m[:, 1].float().mean().item(),
            "fused": m[:, 2].float().mean().item(), "final": m[:, 3].float().mean().item(),
            "rescued": int((~m[:, 2] & m[:, 3]).sum()), "broken": int((m[:, 2] & ~m[:, 3]).sum())}


@torch.no_grad()
def predict_ttr(model, idx, store, n=4, views=("id",), strength=None, seed=0, bs=256, heads=("logits",)):
    """Test-time redrawing: softmax averaged over each test-time view and ``n`` re-writings of it.

    A re-writing is StrokeWarp (see ``strokewarp``) applied along the strokes the network itself
    finds in that view, with fixed random draws (``seed``), so the result is repeatable. ``strength``
    overrides StrokeWarp's ``bend`` / ``tail``. Only for networks with strokes. Returns {head: (N, C)}.
    """
    if not getattr(model, "strokes", False):
        raise ValueError("test-time redrawing needs a network that parses strokes")
    sw = dict(STROKEWARP, **(strength or {}))
    model.eval()
    amp = amp_dtype(store.device)
    res = getattr(model, "input_res", None)
    idx = np.asarray(idx)
    out = {h: [] for h in heads}

    def run(xv):
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            return model(store.normalize(xv, "gray"), draw=False, ink=xv)

    for b, i in enumerate(range(0, len(idx), bs)):
        x = fetch(store, idx[i:i + bs], res)
        acc, count = {}, 0
        for vi, v in enumerate(views):
            xv = view(x, v)
            o = run(xv)
            outs = [o]
            for r in range(n):
                g = torch.Generator().manual_seed(((seed * 1009 + b) * 101 + vi) * 101 + r)
                noise = rewrite_noise(*o["ctrl"].shape[:2], sw["bend"], sw["tail"], g, x.device)
                outs.append(run(stroke_warp(xv, o["ctrl"], o["exist"], noise=noise, **sw)))
            for oo in outs:
                for h in heads:
                    if h in oo:
                        acc[h] = acc.get(h, 0) + oo[h].float().softmax(-1)
            count += len(outs)
        for h in acc:
            out[h].append((acc[h] / count).cpu())
    return {h: torch.cat(v).numpy() for h, v in out.items() if v}


def load_model(path, cfg=None, num_classes=55, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = cfg or ck["cfg"]
    model = make_model(cfg, num_classes, pretrained=False, device=device)
    model.load_state_dict(ck["model"])
    return model.eval()


def train(name, cfg, train_idx, tag, store, labels, work, val_idx=None, num_classes=55, log=print):
    """Train one model and return its averaged copy. Resumes from ``work`` after a disconnect."""
    device = store.device
    out_dir = Path(work) / "runs" / name / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "final.pt").exists():
        return load_model(out_dir / "final.pt", cfg, num_classes, device)

    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    random.seed(seed)
    aug = AUG_PRESETS[cfg.get("aug", "full")]
    amp = amp_dtype(device)
    train_idx = np.asarray(train_idx)
    bs = min(cfg.get("batch", 256), len(train_idx))
    steps_per_epoch = max(1, len(train_idx) // bs)
    total = cfg["epochs"] * steps_per_epoch
    warm = cfg.get("warmup_steps") or max(1, min(3 * steps_per_epoch, total // 5))
    val_every = cfg.get("val_every", 5)
    ckpt_every, log_every = cfg.get("ckpt_every", 1), cfg.get("log_every", 1)
    res = cfg.get("res")

    def lr_at(step):
        if step < warm:
            return cfg["lr"] * (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return cfg["lr"] * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t)))

    model = make_model(cfg, num_classes, device=device)
    ema = EMA(model, cfg.get("ema", 0.9995))
    opt = torch.optim.AdamW(param_groups(model, cfg["lr"], cfg["wd"], cfg.get("new_lr", 2.5),
                                         cfg.get("template_lr", 10.0)), lr=cfg["lr"])
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.get("smoothing", 0.1))
    draw = cfg.get("strokes", True) and cfg.get("render_weight", 1.0) > 0
    jitter = cfg.get("jitter", 0.01)
    sw = dict(STROKEWARP, **cfg["strokewarp"]) if cfg.get("strokewarp") else None
    y_all = torch.as_tensor(labels, device=device)
    start, step, history = 1, 0, []
    local = LOCAL / name / tag / "last.pt"
    ck = _load_state([out_dir / "last.pt", local], device, log)
    if ck is not None:
        model.load_state_dict(ck["model"])
        ema.module.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        ema.updates, start, step, history = ck["ema_updates"], ck["epoch"] + 1, ck["step"], ck["history"]
        log(f"{name}/{tag}: resuming at epoch {start}")

    for epoch in range(start, cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        sums = {}
        gen = torch.Generator().manual_seed(seed * 1000 + epoch)
        perm = train_idx[torch.randperm(len(train_idx), generator=gen).numpy()]
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            x_ink = augment(fetch(store, idx, res), aug)
            if sw is not None and epoch > sw["warmup"]:
                with torch.no_grad(), torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    parse = ema.module(store.normalize(x_ink, "gray"), draw=False)
                chosen = torch.rand(len(idx), device=device) < sw["p"]
                x_ink = stroke_warp(x_ink, parse["ctrl"], parse["exist"], mask=chosen, **sw)
            y = y_all[torch.as_tensor(idx, device=device)]
            for group in opt.param_groups:
                group["lr"] = lr_at(step) * group["lr_scale"]
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                out = model(store.normalize(x_ink, "gray"), draw=draw, jitter=jitter, ink=x_ink, labels=y)
            loss, parts = compute_loss(model, out, x_ink, y, crit, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ema.update(model)
            for k, v in [("loss", loss)] + list(parts.items()):
                sums[k] = sums.get(k, 0) + v.detach().float()
            step += 1

        row = {"epoch": epoch, **{k: v.item() / steps_per_epoch for k, v in sums.items()},
               "lr": lr_at(step - 1), "sec": round(time.time() - t0, 1)}
        if val_idx is not None and len(val_idx) and (epoch % val_every == 0 or epoch == cfg["epochs"]):
            probs = predict(ema.module, val_idx, cfg, store)
            for h, p in probs.items():
                row[f"val_{h}"] = float((p.argmax(1) == labels[val_idx]).mean())
            if draw:
                row.update({f"parse_{k}": v for k, v in parse_eval(ema.module, val_idx[:2048], store).items()})
        history.append(row)
        if epoch % log_every == 0 or epoch in (start, cfg["epochs"]):
            log(f"{name}/{tag} " + "  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                                             for k, v in row.items()))
        if epoch % ckpt_every and epoch < cfg["epochs"]:
            continue
        state = {"model": model.state_dict(), "ema": ema.module.state_dict(), "opt": opt.state_dict(),
                 "ema_updates": ema.updates, "epoch": epoch, "step": step, "history": history}
        safe_save(state, local, tries=2)
        if epoch % (DRIVE_EVERY * ckpt_every) == 0 and epoch < cfg["epochs"]:
            err = safe_save(state, out_dir / "last.pt")
            if err is not None:
                log(f"{name}/{tag}: could not copy the checkpoint to {out_dir} ({err}); training goes on")

    final = {"model": ema.module.state_dict(), "cfg": cfg, "history": history}
    err = safe_save(final, out_dir / "final.pt", tries=6)
    if err is not None:
        backup = LOCAL / name / tag / "final.pt"
        safe_save(final, backup)
        raise RuntimeError(f"Could not write {out_dir / 'final.pt'} ({err}). Is the drive full? A copy is in "
                           f"{backup}; free some space and run the cell again, training will not restart.") from err
    (out_dir / "last.pt").unlink(missing_ok=True)
    local.unlink(missing_ok=True)
    return ema.module.eval()


def cached_preds(work, tag, split, make):
    """Load saved probabilities ({head: array}, float16 on disk) or compute and save them."""
    f = Path(work) / "preds" / f"{tag}_{split}.npz"
    if not f.exists():
        p = {h: v.astype(np.float16) for h, v in make().items()}
        must_write(lambda tmp: np.savez(tmp, **p), f)
    with np.load(f) as z:
        return {h: z[h].astype(np.float32) for h in z.files}
