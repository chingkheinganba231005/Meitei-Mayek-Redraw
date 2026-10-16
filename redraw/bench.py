"""Cost on clearly stated hardware: parameters, FLOPs, latency, throughput, memory, training step time.

FLOPs are counted with ``torch.utils.flop_counter`` (one multiply-add = 2 FLOPs) for one forward pass
of one view, including the hypothesis check. GPU timings use warm-up iterations and synchronise the
device; batch-1 latency and large-batch throughput answer different questions and are kept apart.
Compare timings only within one run of this module (same GPU).
"""

import time

import numpy as np
import torch

from .model import make_model
from .train import amp_dtype, param_groups


def parameters(model):
    groups = {"backbone": "backbone.", "image classifier": "img_head.", "stroke decoder": "decoder.",
              "stroke reader": "head.", "hypothesis check": "hdr."}
    out = {k: 0 for k in groups}
    total = 0
    for n, p in model.named_parameters():
        total += p.numel()
        for k, pre in groups.items():
            if n.startswith(pre):
                out[k] += p.numel()
    out["beta, gamma"] = sum(p.numel() for n, p in model.named_parameters() if n in ("beta", "gamma"))
    out["total"] = total
    return out


def flops(model, size=128, device="cpu"):
    from torch.utils.flop_counter import FlopCounterMode
    model = model.to(device).eval()
    x = torch.rand(1, 1, size, size, device=device)
    counter = FlopCounterMode(display=False)
    with counter:                       # with gradients enabled: the counter needs them for detach()
        model(x, draw=False, ink=x)
    return int(counter.get_total_flops())


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def latency(model, device, size=128, batch=1, views=1, iters=100, warmup=10):
    """Milliseconds per call (median and 90th percentile) for ``batch`` images and ``views`` test-time views."""
    model = model.to(device).eval()
    amp = amp_dtype(device)
    x = torch.rand(batch * views, 1, size, size, device=device)
    times = []
    for i in range(warmup + iters):
        _sync(device)
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            model(x, draw=False, ink=x)
        _sync(device)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90)),
            "batch": batch, "views": views}


@torch.no_grad()
def throughput(model, device, size=128, batch=256, views=1, iters=20, warmup=3):
    lat = latency(model, device, size, batch, views, iters, warmup)
    return {"images_per_s": batch / (lat["median_ms"] / 1000), "batch": batch, "views": views}


def peak_memory_mb(model, device, size=128, batch=256, views=1):
    if device != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats()
    latency(model, device, size, batch, views, iters=2, warmup=1)
    return torch.cuda.max_memory_allocated() / 2 ** 20


def train_step_time(cfg, device, batch=256, size=128, steps=20, warmup=3, num_classes=55):
    """Seconds per training step on random images (same batch, same GPU for every configuration)."""
    from .train import compute_loss
    model = make_model(dict(cfg, pretrained=False), num_classes, device=device).train()
    opt = torch.optim.AdamW(param_groups(model, 1e-4, 0.05), lr=1e-4)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    amp = amp_dtype(device)
    x = torch.rand(batch, 1, size, size, device=device)
    y = torch.randint(0, num_classes, (batch,), device=device)
    draw = cfg.get("strokes", True) and cfg.get("render_weight", 1.0) > 0
    times = []
    for i in range(warmup + steps):
        _sync(device)
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            out = model(x, draw=draw, ink=x, labels=y)
        loss, _ = compute_loss(model, out, x, y, crit, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        _sync(device)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return {"seconds_per_step": float(np.median(times)), "batch": batch}


def preprocess_ms(files, n=200):
    """CPU time to read and preprocess one scan (median over ``n`` files)."""
    from .preprocess import preprocess_file
    times = []
    for f in list(files)[:n]:
        t0 = time.perf_counter()
        preprocess_file(f)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)) if times else None
