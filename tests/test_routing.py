import torch
import torch.nn.functional as F

from redraw.model import make_model
from redraw.render import fit_losses, render_loss
from redraw.train import ink_target

TINY = dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=True, K=4)


def _model(route, **extra):
    torch.manual_seed(0)
    model = make_model(dict(TINY, route=route, **extra), 5).train()
    # as after a few steps: the zero-initialised output layers have moved (at step 0 they pass no
    # gradient back at all, which would make every "not touched" check below pass for the wrong reason)
    heads = [model.decoder.ctrl, model.decoder.width, model.decoder.exist]
    if model.hdr is not None:
        heads += [model.hdr.offset, model.hdr.affine, model.hdr.exist, model.hdr.wscale]
    for head in heads:
        torch.nn.init.normal_(head[-1].weight, std=0.02)
    return model


def touched(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def reconstruction(model, x):
    out = model(x)
    parts = fit_losses(out["render"], out["dist"], out["pts"], out["exist"], ink_target(x, model.G))
    return render_loss(parts)


def test_honest_route_keeps_reconstruction_out_of_the_backbone():
    x = torch.rand(2, 1, 64, 64)
    honest = _model("honest")
    reconstruction(honest, x).backward()
    assert touched(honest.decoder) and not touched(honest.backbone)
    coupled = _model("coupled")
    reconstruction(coupled, x).backward()
    assert touched(coupled.decoder) and touched(coupled.backbone)


def test_honest_stroke_head_reads_the_strokes_but_cannot_move_them():
    x, y = torch.rand(2, 1, 64, 64), torch.tensor([0, 1])
    honest = _model("honest")
    F.cross_entropy(honest(x, draw=False)["vec"], y).backward()
    assert touched(honest.head) and touched(honest.backbone) and not touched(honest.decoder)
    coupled = _model("coupled")
    F.cross_entropy(coupled(x, draw=False)["vec"], y).backward()
    assert touched(coupled.decoder)


def test_honest_hypothesis_fit_stays_out_of_the_backbone():
    x, y = torch.rand(3, 1, 64, 64), torch.tensor([1, 2, 3])
    model = _model("honest", hdr=True, hdr_k=3)
    out = model(x, draw=False, ink=x, labels=y)
    out["cost"].mean().backward()
    assert model.hdr.offset[0].weight.grad.abs().sum() > 0 and not touched(model.backbone)
    coupled = _model("coupled", hdr=True, hdr_k=3)
    coupled(x, draw=False, ink=x, labels=y)["cost"].mean().backward()
    assert touched(coupled.backbone)


def test_honest_route_gives_the_same_outputs():
    x = torch.rand(2, 1, 64, 64)
    a, b = _model("honest").eval(), _model("coupled").eval()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        oa, ob = a(x), b(x)
    for k in ("logits", "ctrl", "render"):
        assert torch.allclose(oa[k], ob[k])


def test_unknown_route_is_refused():
    try:
        make_model(dict(TINY, route="sideways"), 5)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_draw_grad_dial_scales_the_drawing_gradient_into_the_backbone():
    x = torch.rand(2, 1, 64, 64)
    grads = {}
    for g in (0.0, 0.25, 1.0):
        model = _model("honest", draw_grad=g)
        reconstruction(model, x).backward()
        grads[g] = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                              for p in model.backbone.parameters()])
    assert grads[0.0].abs().sum() == 0 and grads[1.0].abs().sum() > 0
    assert torch.allclose(grads[0.25], 0.25 * grads[1.0], rtol=1e-4, atol=1e-8)


def test_draw_grad_needs_honest_routing():
    try:
        make_model(dict(TINY, route="coupled", draw_grad=0.5), 5)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
