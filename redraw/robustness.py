"""Synthetic corruptions of the raw scans, for the robustness evaluation.

Each corruption is applied to the original greyscale image before any
preprocessing, with a seed derived from the file path, so every run sees the
same corrupted copies.
"""

import zlib

import cv2
import numpy as np
from PIL import Image

from .preprocess import preprocess

CORRUPTIONS = {
    "noise": (0.05, 0.10),        # Gaussian noise, s.d. as a fraction of 255
    "blur": (0.6, 1.0),           # Gaussian blur sigma in pixels (images are ~24 px)
    "rotation": (10, 20),         # degrees, random sign
    "contrast": (0.5, 0.25),      # ink contrast kept
    "shading": (0.3, 0.6),        # linear illumination fall-off across the image
    "salt_pepper": (0.02, 0.05),  # fraction of pixels flipped to black or white
    "low_res": (16, 12),          # downsample to this many pixels, then back up
}


def corrupt(gray, kind, level, rng):
    g = gray.astype(np.float32)
    H, W = g.shape
    paper = float(np.median(g))
    if kind == "noise":
        g = g + rng.normal(0, level * 255, g.shape)
    elif kind == "blur":
        g = cv2.GaussianBlur(g, (0, 0), level)
    elif kind == "rotation":
        angle = level * rng.choice([-1, 1])
        M = cv2.getRotationMatrix2D(((W - 1) / 2, (H - 1) / 2), angle, 1.0)
        g = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=paper)
    elif kind == "contrast":
        g = paper + level * (g - paper)
    elif kind == "shading":
        a = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:H, 0:W]
        t = (np.cos(a) * xx / max(W - 1, 1) + np.sin(a) * yy / max(H - 1, 1))
        t = (t - t.min()) / max(np.ptp(t), 1e-6)
        g = g * (1 - level * t)
    elif kind == "salt_pepper":
        m = rng.random(g.shape)
        g = np.where(m < level / 2, 0, np.where(m > 1 - level / 2, 255, g))
    elif kind == "low_res":
        side = int(level)
        small = cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA)
        g = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(kind)
    return np.clip(g + 0.5, 0, 255).astype(np.uint8)


def corrupt_file(path, size, kind, level):
    """Read a dataset image, corrupt it, preprocess it (picklable for multiprocessing)."""
    gray = np.array(Image.open(path).convert("L"))
    rng = np.random.default_rng(zlib.crc32(f"{path}|{kind}|{level}".encode()))
    return preprocess(corrupt(gray, kind, level, rng), size)


# ------------------------------------------------------------------ after preprocessing
# The same kinds of damage applied to the 128 px ink images (ink 1, paper 0) with the framing fixed, so
# that a crop or threshold failure cannot be mistaken for a recognition failure. Severities match the
# raw ones: lengths given in pixels of a typical scan (about SCAN_PX wide) are scaled to the canvas.
SCAN_PX = 24
FRAMED = {
    "noise": (0.05, 0.10),        # Gaussian noise added to the ink values
    "blur": (0.6, 1.0),           # sigma in scan pixels
    "rotation": (10, 20),         # degrees, random sign
    "contrast": (0.5, 0.25),      # ink multiplied by this
    "shading": (0.3, 0.6),        # a linear ramp of ink-like darkness, in a random direction
    "salt_pepper": (0.02, 0.05),  # fraction of pixels set to ink or paper
    "low_res": (16, 12),          # ink reduced to this many pixels and enlarged again
}
# Changes of texture that keep the shape, and changes of shape that keep the texture (shape-bias test).
TEXTURE_SHAPE = {
    "outline": (2, 1),            # keep only a band this many pixels wide along the edges of the ink
    "stripes": (8, 4),            # ink multiplied by a sinusoidal grating with this period (pixels)
    "elastic": (0.03, 0.06),      # smooth random displacement, amplitude as a fraction of the canvas
}


def _gauss_blur(x, sigma):
    import torch
    import torch.nn.functional as F
    r = max(1, int(np.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def _random_ramp(B, H, W, gen, device, dtype):
    import torch
    a = torch.rand(B, generator=gen, device=device) * 2 * np.pi
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H, device=device), torch.linspace(0, 1, W, device=device), indexing="ij")
    t = torch.cos(a).view(B, 1, 1) * xx + torch.sin(a).view(B, 1, 1) * yy
    t = t - t.flatten(1).min(1).values.view(B, 1, 1)
    return (t / t.flatten(1).max(1).values.clamp_min(1e-6).view(B, 1, 1)).unsqueeze(1).to(dtype)


def _rotate(x, deg):
    import torch
    import torch.nn.functional as F
    a = deg * np.pi / 180
    theta = torch.stack([torch.stack([torch.cos(a), -torch.sin(a), torch.zeros_like(a)], 1),
                         torch.stack([torch.sin(a), torch.cos(a), torch.zeros_like(a)], 1)], 1).to(x.dtype)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def corrupt_framed(x, kind, level, seed=0):
    """Damage ink images x (B, 1, H, W) in [0, 1] after preprocessing (fixed framing). Same seed, same damage."""
    import torch
    import torch.nn.functional as F
    B, _, H, W = x.shape
    gen = torch.Generator(device=x.device).manual_seed(seed)
    if kind == "noise":
        y = x + torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype) * level
    elif kind == "blur":
        y = _gauss_blur(x, level * H / SCAN_PX)
    elif kind == "rotation":
        sign = torch.where(torch.rand(B, generator=gen, device=x.device) < 0.5, -1.0, 1.0)
        y = _rotate(x, sign * level)
    elif kind == "contrast":
        y = x * level
    elif kind == "shading":
        y = x + level * _random_ramp(B, H, W, gen, x.device, x.dtype)
    elif kind == "salt_pepper":
        m = torch.rand(x.shape, generator=gen, device=x.device)
        y = torch.where(m < level / 2, torch.ones_like(x), torch.where(m > 1 - level / 2, torch.zeros_like(x), x))
    elif kind == "low_res":
        side = int(level)
        y = F.interpolate(F.adaptive_avg_pool2d(x, side), size=(H, W), mode="bilinear", align_corners=False)
    else:
        raise ValueError(kind)
    return y.clamp(0, 1)


def texture_shape(x, kind, level, seed=0):
    """Change the texture of the ink but not its shape ("outline", "stripes"), or its shape but not its
    texture ("elastic"). x (B, 1, H, W) ink in [0, 1]."""
    import torch
    import torch.nn.functional as F
    B, _, H, W = x.shape
    gen = torch.Generator(device=x.device).manual_seed(seed)
    if kind == "outline":
        k = 2 * int(level) + 1
        inner = -F.max_pool2d(-(x > 0.5).float(), k, 1, k // 2)        # erosion of the ink mask
        y = x * (1 - inner)
    elif kind == "stripes":
        a = torch.rand(B, generator=gen, device=x.device) * np.pi
        yy, xx = torch.meshgrid(torch.arange(H, device=x.device, dtype=x.dtype),
                                torch.arange(W, device=x.device, dtype=x.dtype), indexing="ij")
        u = torch.cos(a).view(B, 1, 1) * xx + torch.sin(a).view(B, 1, 1) * yy
        y = x * (0.5 + 0.5 * torch.sin(2 * np.pi * u / level)).unsqueeze(1)
    elif kind == "elastic":
        warp = torch.randn(B, 2, 4, 4, generator=gen, device=x.device, dtype=x.dtype) * level * 2
        theta = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], device=x.device, dtype=x.dtype).expand(B, 2, 3)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        grid = grid + F.interpolate(warp, size=(H, W), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
        y = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    else:
        raise ValueError(kind)
    return y.clamp(0, 1)


def crop_failures(meta):
    """Images whose ink mask came out empty (the crop then falls back to the whole image)."""
    return int((np.asarray(meta)[:, 4] == 0).sum())
