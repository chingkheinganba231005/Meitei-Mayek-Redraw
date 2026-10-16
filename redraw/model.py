"""The recogniser: a pretrained CNN that also redraws what it reads.

    image -> backbone -> pooled features -> image head ---------------------+
                     \\                                                       +-> logits
                      -> stroke decoder -> K strokes -> stroke head --------+
                                              |
                                              +-> renderer -> compared with the input ink

The backbone is the unmodified timm classification network, so with the
stroke parts switched off this is exactly the plain fine-tuned baseline. The
stroke decoder has K queries that read the stride-8 and stride-16 feature
maps; each predicts one cubic Bezier stroke (control points, width,
existence). The strokes are rendered and compared with the input, which is
the only training signal they get: there are no stroke labels. The stroke
head turns every stroke into a token (its shape and size, plus the CNN
features sampled along the curve) and classifies the set of tokens with a
small transformer. The two heads are combined by adding their logits.

``route`` decides which losses may change which parts. "coupled" (rounds 1-2): every loss reaches
every part, so the reconstruction losses also reshape the backbone, and the stroke head can move the
strokes to wherever helps it classify. "honest": the stroke decoder and the hypothesis fitter read the
backbone's feature maps through a stop-gradient, so drawing never changes what the recogniser sees;
the stroke head reads features where the strokes are but cannot move them, so the parse stays an
explanation of the ink. The backbone then learns from classification alone. ``draw_grad`` (honest
routing only, default 0) lets that fraction of the drawing gradient back into the backbone: a dial
between honest (0) and coupled (1) for the backbone, while the stroke head still cannot move strokes.
``reader_moves_strokes`` lets the stroke head's loss move the strokes under honest routing; with
``draw_grad`` it gives the four corners of the routing factorial (coupled = 1 and True).

Controls for what the stroke head uses: ``reader_inputs`` ("both", "geometry": shape tokens only,
"features": backbone features sampled along the curves only) and ``reader_curves`` ("parse", or
"grid": a fixed grid of straight curves, the same for every image, so the head is an ordinary
feature-pooling head). ``frozen_backbone`` keeps the backbone and image classifier fixed (they are
loaded from a trained control): the classifier-first, decoder-second baseline.
"""

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .render import curve_points, fit_losses, render

CONFIGS = {
    # the plain fine-tuned network, our baseline
    "baseline": dict(arch="convnext_tiny.fb_in22k_ft_in1k", strokes=False, epochs=40, lr=2e-4, wd=0.05,
                     drop_path=0.1, aug="geometric"),
    # the full method
    "redraw": dict(arch="convnext_tiny.fb_in22k_ft_in1k", strokes=True, K=12, render_weight=1.0,
                   epochs=40, lr=2e-4, wd=0.05, drop_path=0.1, aug="geometric"),
    # same network, strokes never compared with the ink: K free attention slots
    "redraw_norender": dict(arch="convnext_tiny.fb_in22k_ft_in1k", strokes=True, K=12, render_weight=0.0,
                            epochs=40, lr=2e-4, wd=0.05, drop_path=0.1, aug="geometric"),
    # the method plus the hypothesis check: top-3 candidates redrawn from their learned class templates
    "redraw_hdr": dict(arch="convnext_tiny.fb_in22k_ft_in1k", strokes=True, K=12, render_weight=1.0,
                       hdr=True, hdr_k=3, hdr_M=2, hdr_S=8, hdr_fit=1.0, hdr_margin=0.5, hdr_m=0.1,
                       hdr_disc=1.0, hdr_wrong=0.5, template_lr=10.0,
                       epochs=40, lr=2e-4, wd=0.05, drop_path=0.1, aug="geometric"),
}
# ablation of the training principle: the same network trained end to end, so the fitter may
# learn to fit wrong classes badly and the templates learn from every class's images
CONFIGS["redraw_hdr_e2e"] = dict(CONFIGS["redraw_hdr"], honest=False)
# StrokeWarp: half of each batch is deformed stroke by stroke along the network's own parse
CONFIGS["redraw_hdr_sw"] = dict(CONFIGS["redraw_hdr"],
                                strokewarp=dict(p=0.5, warmup=10, bend=0.035, tail=0.2, free=0.06, sigma=0.03))
# the hypothesis check without the stroke head: the CNN proposes, the redrawing verifies
CONFIGS["redraw_hdr_nohead"] = dict(CONFIGS["redraw_hdr"], stroke_head=False)
# the baseline again, with the float32 loss (rounds 1-2 had a bfloat16 loss); new name, new checkpoint
CONFIGS["baseline_fp32"] = dict(CONFIGS["baseline"])
# round 3. Honest routing everywhere: reconstruction never reaches the backbone, classification never
# moves the strokes (see StrokeNet), on top of the honest training of the hypothesis check
CONFIGS["redraw_hdr_hr"] = dict(CONFIGS["redraw_hdr"], route="honest")
# backbone pilot at the same 128 px: ImageNet-12k pretraining, and ConvNeXt V2 (masked-autoencoder
# pretraining, then ImageNet-22k); ``res`` resizes the cached 128 px images on the GPU
CONFIGS["baseline_in12k"] = dict(CONFIGS["baseline"], arch="convnext_tiny.in12k_ft_in1k")
CONFIGS["baseline_v2"] = dict(CONFIGS["baseline"], arch="convnextv2_tiny.fcmae_ft_in22k_in1k")
CONFIGS["baseline224"] = dict(CONFIGS["baseline"], res=224)
CONFIGS["redraw_hdr_hr224"] = dict(CONFIGS["redraw_hdr_hr"], res=224)
# the routing dial: a quarter of, or all of, the drawing gradient reaches the backbone
CONFIGS["redraw_hdr_hr_g25"] = dict(CONFIGS["redraw_hdr_hr"], draw_grad=0.25)
CONFIGS["redraw_hdr_hr_g100"] = dict(CONFIGS["redraw_hdr_hr"], draw_grad=1.0)
CONFIGS["redraw_hdr_hr_g50"] = dict(CONFIGS["redraw_hdr_hr"], draw_grad=0.5)

# revision experiments: the check switched off to isolate the two routing rules (factorial), what the
# stroke head reads, and simpler alternatives. All use the float32 loss and a matched augmentation stream.
_NOCHECK = dict(CONFIGS["redraw"], route="honest", matched_aug=True)
CONFIGS["control_m"] = dict(CONFIGS["baseline"], matched_aug=True)
CONFIGS["rt_none"] = dict(_NOCHECK, draw_grad=1.0, reader_moves_strokes=True)      # = coupled
CONFIGS["rt_backbone"] = dict(_NOCHECK, draw_grad=0.0, reader_moves_strokes=True)  # drawing kept out of backbone
CONFIGS["rt_geometry"] = dict(_NOCHECK, draw_grad=1.0, reader_moves_strokes=False) # reading cannot move strokes
CONFIGS["rt_both"] = dict(_NOCHECK, draw_grad=0.0, reader_moves_strokes=False)     # both rules (honest)
CONFIGS["rd_geometry"] = dict(CONFIGS["rt_both"], reader_inputs="geometry")
CONFIGS["rd_features"] = dict(CONFIGS["rt_both"], reader_inputs="features")
CONFIGS["rd_grid"] = dict(CONFIGS["rt_both"], reader_curves="grid")
CONFIGS["rd_free"] = dict(CONFIGS["redraw_norender"], matched_aug=True)            # slots, no rendering loss
CONFIGS["sequential"] = dict(CONFIGS["rt_both"], frozen_backbone=True)            # needs init_from


def grid_curves(K):
    """K straight cubic curves on a fixed grid (half horizontal, half vertical), (K, 4, 2) in [0, 1]."""
    n_h = (K + 1) // 2
    n_v = K - n_h
    t = torch.tensor([0.1, 0.1 + 0.8 / 3, 0.1 + 1.6 / 3, 0.9])
    rows = [torch.stack([t, torch.full((4,), (i + 0.5) / n_h)], -1) for i in range(n_h)]
    cols = [torch.stack([torch.full((4,), (i + 0.5) / max(n_v, 1)), t], -1) for i in range(n_v)]
    return torch.stack(rows + cols)


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def sine_pos(h, w, d, device):
    """2-D sine positional encoding of an h x w map, (h*w, d)."""
    y = (torch.arange(h, device=device).float() + 0.5) / h
    x = (torch.arange(w, device=device).float() + 0.5) / w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    freq = 2 ** torch.arange(d // 4, device=device).float() * math.pi
    return torch.cat([f(c.reshape(-1, 1) * freq) for c in (xx, yy) for f in (torch.sin, torch.cos)], 1)


class Block(nn.Module):
    """Pre-norm transformer block: self-attention, optional cross-attention, MLP.

    ``bias`` (B, N) is added to the self-attention logits of each key; the
    stroke head uses it to ignore strokes that do not exist.
    """

    def __init__(self, d, heads, cross=False):
        super().__init__()
        self.heads, self.cross = heads, cross
        self.n1, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o1 = nn.Linear(d, 3 * d), nn.Linear(d, d)
        if cross:
            self.n2, self.nm = nn.LayerNorm(d), nn.LayerNorm(d)
            self.q2, self.kv2, self.o2 = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.ff = mlp(d, 4 * d, d)

    def _attn(self, q, k, v, mask=None):
        B, N, D = q.shape
        q, k, v = (t.reshape(B, -1, self.heads, D // self.heads).transpose(1, 2) for t in (q, k, v))
        mask = None if mask is None else mask.to(q.dtype)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, N, D)

    def forward(self, x, mem=None, mem_pos=None, pos=None, bias=None, mask=None):
        """``bias`` (B, N): added per key; ``mask`` (N, N): added to all self-attention logits."""
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        if pos is not None:
            q, k = q + pos, k + pos
        m = None
        if bias is not None:
            m = bias.view(x.shape[0], 1, 1, -1)
        if mask is not None:
            m = mask.view(1, 1, *mask.shape) if m is None else m + mask
        x = x + self.o1(self._attn(q, k, v, m))
        if self.cross:
            k, v = self.kv2(self.nm(mem)).chunk(2, -1)
            q = self.q2(self.n2(x) + (0 if pos is None else pos))
            x = x + self.o2(self._attn(q, k + (0 if mem_pos is None else mem_pos), v))
        return x + self.ff(self.n3(x))


class StrokeDecoder(nn.Module):
    """K stroke queries reading two feature maps.

    Each query has a learnable reference point (initialised on a grid over the
    canvas) and predicts its four control points as offsets from it, so the
    strokes start spread out instead of piled up in the middle.
    """

    def __init__(self, in_chs, d=256, K=12, layers=3, heads=8, reach=0.4):
        super().__init__()
        self.K, self.d, self.reach = K, d, reach
        self.proj = nn.ModuleList([nn.Conv2d(c, d, 1) for c in in_chs])
        self.level = nn.Parameter(torch.zeros(len(in_chs), d))
        self.query = nn.Parameter(torch.randn(K, d) * 0.02)
        self.qpos = nn.Parameter(torch.randn(K, d) * 0.02)
        side = math.ceil(math.sqrt(K))
        g = (torch.arange(side).float() + 0.5) / side
        ref = torch.stack(torch.meshgrid(g, g, indexing="xy"), -1).view(-1, 2)[:K]
        self.ref = nn.Parameter(torch.logit(ref.clamp(0.05, 0.95)))
        self.blocks = nn.ModuleList([Block(d, heads, cross=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.ctrl, self.width, self.exist = mlp(d, d, 8, 3), mlp(d, d // 2, 1), mlp(d, d // 2, 1)
        nn.init.zeros_(self.ctrl[-1].weight)
        with torch.no_grad():  # each stroke starts as a short horizontal segment on its reference point
            self.ctrl[-1].bias.copy_(torch.tensor([-0.3, 0.0, -0.1, 0.0, 0.1, 0.0, 0.3, 0.0]))
        nn.init.zeros_(self.width[-1].weight)
        nn.init.constant_(self.width[-1].bias, -1.0)
        nn.init.zeros_(self.exist[-1].weight)

    def forward(self, feats):
        mem, mpos = [], []
        for i, f in enumerate(feats):
            B, _, h, w = f.shape
            mem.append(self.proj[i](f).flatten(2).transpose(1, 2))
            mpos.append(sine_pos(h, w, self.d, f.device).to(mem[-1].dtype) + self.level[i])
        mem, mpos = torch.cat(mem, 1), torch.cat(mpos, 0).unsqueeze(0)
        x = self.query.unsqueeze(0).expand(B, -1, -1)
        pos = self.qpos.unsqueeze(0).expand(B, -1, -1)
        for blk in self.blocks:
            x = blk(x, mem=mem, mem_pos=mpos, pos=pos)
        x = self.norm(x)
        off = torch.tanh(self.ctrl(x).float()).view(B, self.K, 4, 2) * self.reach
        ctrl = (torch.sigmoid(self.ref).view(1, self.K, 1, 2) + off).clamp(0.0, 1.0)
        width = 0.02 + 0.18 * torch.sigmoid(self.width(x).float().squeeze(-1))
        exist = torch.sigmoid(self.exist(x).float().squeeze(-1))
        return ctrl, width, exist, mem, mpos


class StrokeHead(nn.Module):
    """Classifier over the set of strokes. It sees neither stroke order nor which query made a stroke."""

    def __init__(self, feat_ch, num_classes, d=256, layers=2, heads=8, T=32, inputs="both"):
        super().__init__()
        if inputs not in ("both", "geometry", "features"):
            raise ValueError(f"reader_inputs must be 'both', 'geometry' or 'features', not {inputs!r}")
        self.inputs = inputs
        self.geo = mlp(2 * T + 4, d, d)
        self.app = nn.Conv2d(feat_ch, d, 1)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, num_classes)

    def forward(self, pts, width, exist, fmap):
        B, K, T, _ = pts.shape
        length = (pts[:, :, 1:] - pts[:, :, :-1]).norm(dim=-1).sum(-1, keepdim=True)
        chord = (pts[:, :, -1] - pts[:, :, 0]).norm(dim=-1, keepdim=True)
        # width is relative to the glyph box, so it also tells how large the character was written
        geo = torch.cat([(pts - 0.5).flatten(2), width.unsqueeze(-1), length, chord, exist.unsqueeze(-1)], -1)
        a = self.app(fmap)
        app = F.grid_sample(a, (pts.reshape(B, K * T, 1, 2) * 2 - 1).to(a.dtype), mode="bilinear",
                            align_corners=False).view(B, -1, K, T).mean(-1).transpose(1, 2)
        if self.inputs == "geometry":
            tok = self.geo(geo.to(app.dtype))
        elif self.inputs == "features":
            tok = app
        else:
            tok = self.geo(geo.to(app.dtype)) + app
        x = torch.cat([self.cls.expand(B, -1, -1).to(app.dtype), tok], 1)
        bias = torch.cat([torch.zeros(B, 1, device=x.device), torch.log(exist.clamp_min(1e-4))], 1)
        for blk in self.blocks:
            x = blk(x, bias=bias)
        return self.out(self.norm(x[:, 0]))


class HypothesisRedrawer(nn.Module):
    """Redraws the character as each candidate class, starting from that class's learned strokes.

    Every class owns M templates of S strokes: cubic Beziers in a canonical
    frame, each with a width and a prior probability of being written at all.
    They are ordinary parameters, learned from the training images. For a
    candidate class, the fitter (a small transformer reading the same feature
    maps as the stroke decoder) predicts how to bend each template onto the
    image: one affine map per template and a bounded offset for every control
    point, plus a change of each stroke's width and existence. ``cost`` then
    renders the bent template and measures how badly it explains the image
    and how far it had to be bent. The cheaper the explanation, the more
    likely the class.
    """

    def __init__(self, num_classes, d=256, M=2, S=6, layers=2, heads=8, max_offset=0.12,
                 w_def=0.05, w_aff=0.05, w_kl=0.05, tau=0.02, width_range=0.3):
        super().__init__()
        self.C, self.M, self.S, self.d = num_classes, M, S, d
        self.max_offset, self.w_def, self.w_aff, self.w_kl, self.tau = max_offset, w_def, w_aff, w_kl, tau
        # the fitter may change a stroke's width only by exp(+-width_range): the width of a template
        # relative to the glyph box carries the size the character was written at (the zone cue)
        self.width_range = width_range
        g = torch.Generator().manual_seed(0)
        # start: S short strokes spread around the centre, jittered per class and template
        ang = torch.arange(S).float() / S * 2 * math.pi
        centre = 0.5 + 0.22 * torch.stack([ang.cos(), ang.sin()], -1)                      # (S, 2)
        along = torch.stack([-ang.sin(), ang.cos()], -1).view(S, 1, 2)                        # tangent direction
        base = centre.view(S, 1, 2) + torch.tensor([-0.12, -0.04, 0.04, 0.12]).view(1, 4, 1) * along  # (S, 4, 2)
        init = base.expand(num_classes, M, S, 4, 2) + 0.03 * torch.randn(num_classes, M, S, 4, 2, generator=g)
        # stored flat (C, M, S*8): 5-D parameters would break channels_last conversion of the model
        self.t_ctrl = nn.Parameter(torch.logit(init.clamp(0.03, 0.97)).reshape(num_classes, M, S * 8))
        self.t_width = nn.Parameter(torch.full((num_classes, M, S), math.log(0.06 / 0.12)))  # width 0.06
        self.t_exist = nn.Parameter(torch.full((num_classes, M, S), 1.5))                    # prior p = 0.82
        self.cls_emb = nn.Embedding(num_classes, d)
        self.tmpl_emb = nn.Parameter(torch.randn(M, d) * 0.02)
        self.slot_emb = nn.Parameter(torch.randn(S, d) * 0.02)
        self.param_emb = mlp(10, d, d)
        self.blocks = nn.ModuleList([Block(d, heads, cross=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.offset, self.exist, self.wscale = mlp(d, d, 8, 3), mlp(d, d // 2, 1), mlp(d, d // 2, 1)
        self.affine = mlp(d, d, 6)
        for head in (self.offset, self.exist, self.wscale, self.affine):  # start from the template itself
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def template_params(self):
        """Templates as (control points in [0, 1], widths, existence-prior logits)."""
        ctrl = torch.sigmoid(self.t_ctrl).view(self.C, self.M, self.S, 4, 2)
        return ctrl, 0.02 + 0.18 * torch.sigmoid(self.t_width), self.t_exist

    def fit(self, mem, mpos, cand, live=None):
        """Bend the templates of the candidate classes ``cand`` (B, k) onto the images.

        ``live`` (B, k) bool marks the candidates whose templates may learn from
        this fit (the true class); the others are detached, so a template only
        ever learns to explain images of its own class.
        Returns the template tensors (B, k, M, S, ...) and the fitter's outputs.
        """
        B, k = cand.shape
        M, S = self.M, self.S
        ctrl, width, prior = (t[cand] for t in self.template_params())          # (B, k, M, S, ...)
        if live is not None:
            m = live.view(B, k, 1, 1)
            ctrl = torch.where(m.unsqueeze(-1).unsqueeze(-1), ctrl, ctrl.detach())
            width, prior = torch.where(m, width, width.detach()), torch.where(m, prior, prior.detach())
        params = torch.cat([ctrl.flatten(-2), width.unsqueeze(-1), torch.sigmoid(prior).unsqueeze(-1)], -1)
        tok = (self.param_emb(params.to(mem.dtype)) + self.cls_emb(cand).view(B, k, 1, 1, -1)
               + self.tmpl_emb.view(1, 1, M, 1, -1) + self.slot_emb.view(1, 1, 1, S, -1))
        tok = tok.reshape(B, k * M * S, self.d)
        group = torch.arange(k * M * S, device=tok.device) // S          # strokes of one template see each other
        mask = torch.zeros(k * M * S, k * M * S, device=tok.device)
        mask.masked_fill_(group.view(-1, 1) != group.view(1, -1), float("-inf"))
        for blk in self.blocks:
            tok = blk(tok, mem=mem, mem_pos=mpos, mask=mask)
        tok = self.norm(tok).view(B, k, M, S, self.d)
        a = torch.tanh(self.affine(tok.mean(3)).float())                  # (B, k, M, 6)
        fitted = {
            "off": torch.tanh(self.offset(tok).float()).view(B, k, M, S, 4, 2) * self.max_offset,
            "dlogit": self.exist(tok).float().squeeze(-1),
            "wscale": torch.exp(self.width_range * torch.tanh(self.wscale(tok).float().squeeze(-1))),
            "a": a,
            "A": self._affine(a),
        }
        return (ctrl, width, prior), fitted

    @staticmethod
    def _affine(a):
        """Six bounded numbers -> 2x3 map about the canvas centre (scale, rotation, shear, shift)."""
        sx, sy = torch.exp(0.35 * a[..., 0]), torch.exp(0.35 * a[..., 1])
        rot, shear = 0.4 * a[..., 2], 0.3 * a[..., 3]
        c, s = torch.cos(rot), torch.sin(rot)
        lin = torch.stack([torch.stack([c * sx, (c * shear - s) * sy], -1),
                           torch.stack([s * sx, (s * shear + c) * sy], -1)], -2)    # (..., 2, 2)
        shift = 0.15 * a[..., 4:6]
        return torch.cat([lin, shift.unsqueeze(-1)], -1)                           # (..., 2, 3)

    def bend(self, templates, fitted, detach=False):
        """Apply the fitter's changes to the templates -> control points, widths, existence logits."""
        ctrl, width, prior = templates
        f = {k: v.detach() for k, v in fitted.items()} if detach else fitted
        A = f["A"]
        pts = torch.einsum("bkmij,bkmsnj->bkmsni", A[..., :2], ctrl - 0.5) + 0.5
        pts = (pts + A[..., 2].unsqueeze(3).unsqueeze(3) + f["off"]).clamp(0.0, 1.0)
        return pts, width * f["wscale"], prior + f["dlogit"]

    def cost(self, bent, prior, fitted, target, G=32, T=24):
        """How badly each bent template explains the image: (B, k) after the soft minimum over templates."""
        pts, width, logit = bent
        B, k, M, S = pts.shape[:4]
        n = B * k * M
        exist = torch.sigmoid(logit)
        img, d, samples = render(pts.reshape(n, S, 4, 2), width.reshape(n, S), exist.reshape(n, S), G=G, T=T)
        tgt = target.unsqueeze(1).expand(B, k * M, G, G).reshape(n, G, G)
        parts = fit_losses(img, d, samples, exist.reshape(n, S), tgt, reduce=False)
        recon = (parts["bce"] + 2 * parts["prec"] + 2 * parts["cover"]).view(B, k, M)
        deform = (fitted["off"] ** 2).mean((-1, -2, -3)) / self.max_offset ** 2
        bend_a = (fitted["a"] ** 2).mean(-1)
        q = torch.sigmoid(prior).clamp(1e-4, 1 - 1e-4)
        e = exist.clamp(1e-4, 1 - 1e-4)
        kl = (e * torch.log(e / q) + (1 - e) * torch.log((1 - e) / (1 - q))).mean(-1)
        energy = recon + self.w_def * deform + self.w_aff * bend_a + self.w_kl * kl     # (B, k, M)
        return -self.tau * torch.logsumexp(-energy / self.tau, dim=2)



class StrokeNet(nn.Module):
    def __init__(self, arch, num_classes=55, in_chans=1, drop_path=0.1, pretrained=True, strokes=True,
                 K=12, d=256, dec_layers=3, head_layers=2, G=48, T=32, hdr=False, hdr_k=3, hdr_M=2,
                 hdr_S=6, hdr_G=32, honest=True, stroke_head=True, route="coupled", draw_grad=0.0,
                 reader_moves_strokes=None, reader_inputs="both", reader_curves="parse", frozen_backbone=False):
        super().__init__()
        if route not in ("coupled", "honest"):
            raise ValueError(f"route must be 'coupled' or 'honest', not {route!r}")
        if not 0.0 <= draw_grad <= 1.0 or (draw_grad and route != "honest"):
            raise ValueError("draw_grad must be in [0, 1] and is only used with honest routing")
        if reader_moves_strokes is not None and route != "honest":
            raise ValueError("reader_moves_strokes is only used with honest routing (coupled always moves them)")
        if reader_curves not in ("parse", "grid"):
            raise ValueError(f"reader_curves must be 'parse' or 'grid', not {reader_curves!r}")
        self.route, self.draw_grad = route, draw_grad
        self.reader_moves_strokes = route == "coupled" if reader_moves_strokes is None else bool(reader_moves_strokes)
        self.reader_curves, self.frozen_backbone = reader_curves, frozen_backbone
        self.input_res = None  # side the cached images are resized to before entering (None: as cached)
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=in_chans,
                                          drop_path_rate=drop_path)
        self.img_head = nn.Linear(self.backbone.num_features, num_classes)
        self.strokes, self.G, self.T = strokes, G, T
        self.hdr = None
        if strokes:
            info = list(self.backbone.feature_info)
            self.taps = [next(i for i, f in enumerate(info) if f["reduction"] == r) for r in (8, 16)]
            chs = [info[i]["num_chs"] for i in self.taps]
            self.decoder = StrokeDecoder(chs, d=d, K=K, layers=dec_layers)
            self.head = (StrokeHead(chs[0], num_classes, d=d, layers=head_layers, T=T, inputs=reader_inputs)
                         if stroke_head else None)
            self.register_buffer("grid_ctrl", grid_curves(K), persistent=False)
            if hdr:
                self.hdr = HypothesisRedrawer(num_classes, d=d, M=hdr_M, S=hdr_S)
                self.hdr_k, self.hdr_G, self.honest = hdr_k, hdr_G, honest
                self.beta = nn.Parameter(torch.tensor(math.log(math.expm1(5.0))))   # softplus -> 5
                self.gamma = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))  # softplus -> 1

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_backbone:  # a fixed feature extractor: no dropout or stochastic depth
            self.backbone.eval()
        return self

    def freeze_backbone(self):
        """Backbone and image classifier fixed (after loading them from a trained control)."""
        for p in list(self.backbone.parameters()) + list(self.img_head.parameters()):
            p.requires_grad_(False)
        self.frozen_backbone = True
        return self.train(self.training)

    def new_parameters(self):
        """Parameters that start from scratch (they get a higher learning rate)."""
        mods = [self.img_head] + ([self.decoder] if self.strokes else [])
        mods += [self.head] if self.strokes and self.head is not None else []
        extra = [self.beta, self.gamma] if self.hdr is not None else []
        mods += [self.hdr] if self.hdr is not None else []
        return [p for m in mods for p in m.parameters()] + extra

    def forward(self, x, draw=True, jitter=0.0, ink=None, labels=None):
        """x: normalised input. ``ink`` (the same images before normalisation) is needed for the
        hypothesis check; ``labels`` (training only) make sure the true class is among the candidates."""
        if not self.strokes:
            logits = self.img_head(self.backbone(x))
            return {"img": logits, "fused": logits, "logits": logits}
        final, inter = self.backbone.forward_intermediates(x, indices=self.taps)
        img_logits = self.img_head(self.backbone.forward_head(final))
        honest = self.route == "honest"
        if honest:  # the decoder sees the features; only a fraction draw_grad of its gradient goes back
            g = self.draw_grad
            seen_by_decoder = [f.detach() + g * (f - f.detach()) if g else f.detach() for f in inter]
        else:
            seen_by_decoder = inter
        ctrl, width, exist, mem, mpos = self.decoder(seen_by_decoder)
        out = {"img": img_logits, "ctrl": ctrl, "width": width, "exist": exist}
        if self.head is not None:
            if self.reader_curves == "grid":  # the same straight curves for every image
                B = ctrl.shape[0]
                g = self.grid_ctrl.to(ctrl.dtype).unsqueeze(0).expand(B, -1, -1, -1)
                pts = curve_points(g, self.T)
                w_in = torch.full(width.shape, 0.06, device=width.device, dtype=width.dtype)
                e_in = torch.ones_like(exist)
            else:
                pts = curve_points(ctrl, self.T)
                w_in, e_in = width, exist
                if not self.reader_moves_strokes:  # the stroke head reads where the strokes are; it cannot move them
                    pts, w_in, e_in = pts.detach(), width.detach(), exist.detach()
            seen = (pts + jitter * torch.randn_like(pts)).clamp(0, 1) if jitter else pts
            out["vec"] = self.head(seen, w_in, e_in, inter[0])
            fused = img_logits + out["vec"]
        else:  # strokes only feed the hypothesis check and the reconstruction loss
            fused = img_logits
        out.update(fused=fused, logits=fused)
        if draw:
            with torch.autocast(x.device.type, enabled=False):
                out["render"], out["dist"], out["pts"] = render(ctrl.float(), width.float(), exist.float(),
                                                                G=self.G, T=self.T)
        if self.hdr is not None and ink is not None:
            out.update(self._verify(fused, mem, mpos, ink, labels, x.device.type))
        return out

    def _verify(self, fused, mem, mpos, ink, labels, device_type):
        """Redraw the image as each of the top-k candidate classes and re-rank them by how well it works."""
        k = min(self.hdr_k, fused.shape[1])
        cand = fused.detach().topk(k, 1).indices
        live = None
        if labels is not None:  # training: the true class is always one of the candidates
            hit = (cand == labels[:, None]).any(1)
            cand = cand.clone()
            cand[~hit, -1] = labels[~hit]
            live = cand == labels[:, None]
        target = F.interpolate(ink[:, :1].float(), size=(self.hdr_G, self.hdr_G), mode="area").squeeze(1)
        templates, fitted = self.hdr.fit(mem, mpos, cand, live if self.honest else None)
        with torch.autocast(device_type, enabled=False):
            e_fit = self.hdr.cost(self.hdr.bend(templates, fitted), templates[2], fitted, target, G=self.hdr_G)
            if self.training and self.honest:
                # the same cost, but only the templates (all candidates) can change it: the fitter is frozen here
                own = tuple(t[cand] for t in self.hdr.template_params())
                frozen = {key: v.detach() for key, v in fitted.items()}
                e_disc = self.hdr.cost(self.hdr.bend(own, frozen), own[2], frozen, target, G=self.hdr_G)
            else:
                # inference, or the end-to-end ablation: one cost, and every loss reaches every part
                e_disc = e_fit
            h = -F.softplus(self.beta) * e_disc
            bonus = torch.zeros(fused.shape, device=fused.device).scatter(
                1, cand, F.softplus(self.gamma) * (h - h.mean(1, keepdim=True)))
        out = {"cand": cand, "live": live, "cost": e_fit, "bonus": bonus,
               "verified": fused.detach().float() + bonus, "logits": fused.float() + bonus}
        if self.training:
            out["cost_frozen"] = e_disc
        return out


def make_model(cfg, num_classes=55, pretrained=None, device="cpu"):
    if pretrained is None:
        pretrained = cfg.get("pretrained", True)
    model = StrokeNet(cfg["arch"], num_classes, drop_path=cfg["drop_path"], pretrained=pretrained,
                      strokes=cfg.get("strokes", True), K=cfg.get("K", 12), G=cfg.get("G", 48),
                      hdr=cfg.get("hdr", False), hdr_k=cfg.get("hdr_k", 3), hdr_M=cfg.get("hdr_M", 2),
                      hdr_S=cfg.get("hdr_S", 6), honest=cfg.get("honest", True),
                      stroke_head=cfg.get("stroke_head", True), route=cfg.get("route", "coupled"),
                      draw_grad=cfg.get("draw_grad", 0.0), reader_moves_strokes=cfg.get("reader_moves_strokes"),
                      reader_inputs=cfg.get("reader_inputs", "both"), reader_curves=cfg.get("reader_curves", "parse"),
                      frozen_backbone=cfg.get("frozen_backbone", False))
    model.input_res = cfg.get("res")
    return model.to(device).to(memory_format=torch.channels_last)
