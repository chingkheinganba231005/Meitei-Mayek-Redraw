"""Which loss trains which parameters: an executable check of the gradient routing (Table 1 of the paper).

Every loss of the objective is computed from its own forward pass and back-propagated alone. For each
group of parameters the result is "none" (the loss is not connected to them: no gradient at all),
"zero" (connected, but every gradient is exactly zero) or "live", with the gradient norm.

    python -m redraw.audit --config redraw_hdr_hr            # the real architecture, random weights
    python -m redraw.audit --config redraw_hdr --tiny         # a small backbone, seconds on a CPU

A gradient that is "none" or "zero" means that loss cannot change those parameters in that step.
Stop-gradients block derivatives, not every influence: a backbone trained by classification still
changes the features the stroke decoder reads in later steps.
"""

import argparse
import json

import torch
import torch.nn.functional as F

from .model import CONFIGS, make_model
from .render import fit_losses, render_loss
from .train import TEMPLATES, ink_target

_PROJ = ("decoder.proj.", "decoder.level")
GROUPS = (
    ("backbone", lambda n: n.startswith("backbone.")),
    ("image classifier", lambda n: n.startswith("img_head.")),
    ("decoder: memory projections", lambda n: n.startswith(_PROJ)),
    ("decoder: queries and stroke outputs", lambda n: n.startswith("decoder.") and not n.startswith(_PROJ)),
    ("stroke reader", lambda n: n.startswith("head.")),
    ("fitter", lambda n: n.startswith("hdr.") and n not in TEMPLATES),
    ("templates", lambda n: n in TEMPLATES),
    ("beta, gamma", lambda n: n in ("beta", "gamma")),
)

# what honest routing with honest check training should give ("live"); every other cell must not be live
EXPECTED_HONEST = {
    "classification (fused)": {"backbone", "image classifier", "stroke reader"},
    "classification (image head)": {"backbone", "image classifier"},
    "classification (stroke reader)": {"backbone", "stroke reader"},
    "drawing": {"decoder: memory projections", "decoder: queries and stroke outputs"},
    "fit": {"decoder: memory projections", "fitter", "templates"},
    "margin": {"templates"},
    "discrimination": {"templates", "beta, gamma"},
}
# coupled routing additionally lets these losses reach these groups
EXTRA_COUPLED = {
    "classification (fused)": {"decoder: memory projections", "decoder: queries and stroke outputs"},
    "classification (stroke reader)": {"decoder: memory projections", "decoder: queries and stroke outputs"},
    "drawing": {"backbone"},
    "fit": {"backbone"},
}
DECODER = {"decoder: memory projections", "decoder: queries and stroke outputs"}


def perturb_zero_heads(model, std=0.02, seed=0):
    """Give the zero-initialised output layers small random weights, as after a few training steps
    (at step 0 they pass no gradient back, which would hide every path behind them)."""
    g = torch.Generator().manual_seed(seed)
    heads = []
    if getattr(model, "strokes", False):
        heads += [model.decoder.ctrl, model.decoder.width, model.decoder.exist]
    if getattr(model, "hdr", None) is not None:
        heads += [model.hdr.offset, model.hdr.affine, model.hdr.exist, model.hdr.wscale]
    with torch.no_grad():
        for head in heads:
            w = head[-1].weight
            w.copy_(torch.randn(w.shape, generator=g) * std)
    return model


def loss_terms(model, x, y, cfg, ink=None):
    """Every loss of the objective (see ``train.compute_loss``) as a separate scalar, from one forward pass.
    ``x`` is the network input; ``ink`` the same images before normalisation (default: ``x``)."""
    ink = x if ink is None else ink
    crit = torch.nn.CrossEntropyLoss(label_smoothing=cfg.get("smoothing", 0.1))
    draw = model.strokes and cfg.get("render_weight", 1.0) > 0
    out = model(x, draw=draw, ink=ink, labels=y)
    terms = {"classification (fused)": crit(out["fused"].float(), y)}
    if "vec" in out:
        terms["classification (image head)"] = crit(out["img"].float(), y)
        terms["classification (stroke reader)"] = crit(out["vec"].float(), y)
    if draw:
        parts = fit_losses(out["render"], out["dist"], out["pts"], out["exist"], ink_target(ink, model.G))
        terms["drawing"] = render_loss(parts)
    if "cost" in out:
        live = out["live"]
        w = torch.where(live, 1.0, cfg.get("hdr_wrong", 0.5))
        terms["fit"] = (out["cost"] * w).sum() / w.sum()
        frozen = out["cost_frozen"]
        right = (frozen * live).sum(1, keepdim=True)
        terms["margin"] = (F.relu(cfg.get("hdr_m", 0.1) + right - frozen) * ~live).sum() / (~live).sum().clamp_min(1)
        terms["discrimination"] = F.cross_entropy(out["verified"], y)
    return terms


def _status(grads):
    if all(g is None for g in grads):
        return "none", 0.0
    norm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads if g is not None)))
    return ("zero" if norm == 0.0 else "live"), norm


def gradient_contract(model, x, y, cfg, seed=0):
    """{loss: {group: {"status", "norm"}}} plus, for losses that reach the templates, the classes whose
    templates received a gradient ({loss: {"template_classes": [...]}})."""
    model.train()
    names = list(loss_terms(model, x, y, cfg))
    table = {}
    for name in names:
        torch.manual_seed(seed)
        model.zero_grad(set_to_none=True)
        loss = loss_terms(model, x, y, cfg)[name]
        if loss.requires_grad:          # a loss whose parameters are all frozen has no gradient at all
            loss.backward()
        params = dict(model.named_parameters())
        row = {}
        for group, member in GROUPS:
            grads = [p.grad for n, p in params.items() if member(n)]
            if grads:
                status, norm = _status(grads)
                row[group] = {"status": status, "norm": norm}
        if "hdr.t_ctrl" in params and params["hdr.t_ctrl"].grad is not None:
            per_class = params["hdr.t_ctrl"].grad.abs().flatten(1).sum(1)
            row["template_classes"] = [int(c) for c in torch.nonzero(per_class > 0).flatten()]
        table[name] = row
    model.zero_grad(set_to_none=True)
    return table


def expected_live(route="honest"):
    exp = {k: set(v) for k, v in EXPECTED_HONEST.items()}
    if route == "coupled":
        for k, v in EXTRA_COUPLED.items():
            exp[k] |= v
    return exp


def expected_for(cfg):
    """The groups each loss should train under configuration ``cfg`` (any routing, ablation or control):
    the drawing and fit losses reach the backbone with coupled routing or a non-zero ``draw_grad``; the
    classification losses reach the stroke geometry when the reader may move the strokes; a reader that
    sees geometry only does not reach the backbone; a frozen backbone and image classifier learn nothing."""
    route = cfg.get("route", "coupled")
    coupled = route == "coupled"
    into_backbone = coupled or cfg.get("draw_grad", 0.0) > 0
    moves = cfg.get("reader_moves_strokes")
    moves = coupled if moves is None else bool(moves)
    if cfg.get("reader_curves", "parse") == "grid":
        moves = False
    strokes = cfg.get("strokes", True)
    reader = strokes and cfg.get("stroke_head", True)
    exp = {"classification (fused)": {"backbone", "image classifier"} | ({"stroke reader"} if reader else set())}
    if reader:
        exp["classification (image head)"] = {"backbone", "image classifier"}
        exp["classification (stroke reader)"] = {"stroke reader"}
        if cfg.get("reader_inputs", "both") != "geometry":
            exp["classification (stroke reader)"].add("backbone")
        if moves:
            exp["classification (fused)"] |= DECODER
            exp["classification (stroke reader)"] |= DECODER
    if strokes and cfg.get("render_weight", 1.0) > 0:
        exp["drawing"] = set(DECODER) | ({"backbone"} if into_backbone else set())
    if strokes and cfg.get("hdr"):
        exp["fit"] = {"decoder: memory projections", "fitter", "templates"} | ({"backbone"} if into_backbone else set())
        exp["margin"] = {"templates"}
        exp["discrimination"] = {"templates", "beta, gamma"}
    if cfg.get("frozen_backbone"):
        for groups in exp.values():
            groups -= {"backbone", "image classifier"}
    return exp


def mismatches(table, expected):
    """Cells where the audit disagrees with the intended routing: (loss, group, found, expected)."""
    bad = []
    for loss, row in table.items():
        want = expected.get(loss, set())
        for group, cell in row.items():
            if group == "template_classes":
                continue
            is_live = cell["status"] == "live"
            if is_live != (group in want):
                bad.append((loss, group, cell["status"], "live" if group in want else "not live"))
    return bad


def markdown(table):
    groups = [g for g, _ in GROUPS if any(g in row for row in table.values())]
    lines = ["| loss | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    mark = {"live": "trains", "zero": "0", "none": "-"}
    for loss, row in table.items():
        cells = [mark[row[g]["status"]] if g in row else "" for g in groups]
        lines.append(f"| {loss} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def audit(name, tiny=False, size=64, batch=4, num_classes=55, seed=0):
    """The contract of configuration ``name`` with random weights (connectivity does not depend on them)."""
    cfg = dict(CONFIGS[name], pretrained=False, drop_path=0.0)
    if tiny:
        cfg.update(arch="resnet18", K=4, hdr_S=4)
    torch.manual_seed(seed)
    model = perturb_zero_heads(make_model(cfg, num_classes))
    if cfg.get("frozen_backbone"):
        model.freeze_backbone()
    x = torch.rand(batch, 1, size, size)
    y = torch.arange(batch) % num_classes
    return gradient_contract(model, x, y, cfg, seed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="redraw_hdr_hr")
    ap.add_argument("--tiny", action="store_true", help="resnet18 backbone, 4 strokes")
    ap.add_argument("--json", help="also write the table to this file")
    args = ap.parse_args(argv)
    table = audit(args.config, tiny=args.tiny)
    print(markdown(table))
    bad = mismatches(table, expected_for(CONFIGS[args.config]))
    print("\nmatches the intended routing" if not bad else f"\n{len(bad)} cells differ: {bad}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(table, f, indent=1)


if __name__ == "__main__":
    main()
