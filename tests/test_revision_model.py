import pytest
import torch

from redraw import train as T
from redraw.augment import AUG_PRESETS, augment
from redraw.model import CONFIGS, grid_curves, make_model
from redraw.render import fit_losses, render_loss
from redraw.train import ink_target, load_backbone, param_groups

TINY = dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=True, K=4)


def _model(**cfg):
    torch.manual_seed(0)
    model = make_model(dict(TINY, **cfg), 5).train()
    for head in (model.decoder.ctrl, model.decoder.width, model.decoder.exist):
        torch.nn.init.normal_(head[-1].weight, std=0.02)  # zero-initialised heads pass no gradient at step 0
    return model


def grads(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    return {n: (None if p.grad is None else p.grad.clone()) for n, p in model.named_parameters()}


def live(g, prefix):
    return any(v is not None and v.abs().sum() > 0 for n, v in g.items() if n.startswith(prefix))


def draw_loss(model, x):
    out = model(x)
    return render_loss(fit_losses(out["render"], out["dist"], out["pts"], out["exist"], ink_target(x, model.G)))


def reader_loss(model, x, y):
    return torch.nn.functional.cross_entropy(model(x, draw=False)["vec"], y)


X, Y = torch.rand(2, 1, 64, 64), torch.tensor([1, 3])


def test_factorial_corner_with_both_rules_off_is_coupled_routing():
    coupled = _model(route="coupled")
    corner = _model(route="honest", draw_grad=1.0, reader_moves_strokes=True)
    corner.load_state_dict(coupled.state_dict())
    for loss in (draw_loss, lambda m, x: reader_loss(m, x, Y)):
        a, b = grads(coupled, loss(coupled, X)), grads(corner, loss(corner, X))
        for n in a:
            assert (a[n] is None) == (b[n] is None), n
            if a[n] is not None:
                assert torch.allclose(a[n], b[n], atol=1e-6), n


@pytest.mark.parametrize("draw_grad, moves, backbone_from_drawing, strokes_from_reading", [
    (0.0, False, False, False), (0.0, True, False, True), (1.0, False, True, False), (1.0, True, True, True)])
def test_each_routing_rule_switches_one_gradient_path(draw_grad, moves, backbone_from_drawing, strokes_from_reading):
    model = _model(route="honest", draw_grad=draw_grad, reader_moves_strokes=moves)
    assert live(grads(model, draw_loss(model, X)), "backbone.") == backbone_from_drawing
    g = grads(model, reader_loss(model, X, Y))
    assert live(g, "decoder.ctrl.") == strokes_from_reading
    assert live(g, "backbone.")  # the reader still trains the backbone through the features it samples


def test_reader_input_controls():
    geometry = _model(route="honest", reader_inputs="geometry")
    g = grads(geometry, reader_loss(geometry, X, Y))
    assert not live(g, "backbone.") and not live(g, "head.app.") and live(g, "head.geo.")
    features = _model(route="honest", reader_inputs="features")
    g = grads(features, reader_loss(features, X, Y))
    assert live(g, "backbone.") and live(g, "head.app.") and not live(g, "head.geo.")
    with pytest.raises(ValueError):
        _model(route="honest", reader_inputs="pixels")


def test_grid_curves_do_not_depend_on_the_parse():
    assert grid_curves(12).shape == (12, 4, 2) and grid_curves(12).min() >= 0 and grid_curves(12).max() <= 1
    model = _model(route="honest", reader_curves="grid").eval()
    with torch.no_grad():
        a = model(X, draw=False)["vec"]
        model.decoder.ctrl[-1].bias.add_(0.3)  # move every stroke
        b = model(X, draw=False)["vec"]
    assert torch.allclose(a, b, atol=1e-6)


def test_frozen_backbone_from_a_trained_control(tmp_path):
    control = make_model(dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=False), 5)
    torch.save({"model": control.state_dict()}, tmp_path / "control.pt")
    model = _model(route="honest", frozen_backbone=True)
    load_backbone(model, tmp_path / "control.pt").freeze_backbone()
    first = next(iter(control.backbone.state_dict()))
    assert torch.equal(model.backbone.state_dict()[first], control.backbone.state_dict()[first])
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert not any(p.requires_grad for p in model.img_head.parameters())
    model.train()
    assert not model.backbone.training and model.decoder.training
    trainable = {id(p) for g in param_groups(model, 1e-3, 0.0) for p in g["params"]}
    assert not trainable & {id(p) for p in model.backbone.parameters()}
    torch.save({"model": {"head.out.weight": torch.zeros(1)}}, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        load_backbone(model, tmp_path / "bad.pt")


def test_matched_augmentation_ignores_other_random_draws():
    x = torch.rand(4, 1, 32, 32)
    a = augment(x, AUG_PRESETS["geometric"], generator=torch.Generator().manual_seed(5))
    torch.rand(1000)  # what another network's dropout would consume
    b = augment(x, AUG_PRESETS["geometric"], generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)


def test_revision_configs_build():
    for name in ("control_m", "rt_none", "rt_backbone", "rt_geometry", "rt_both", "rd_geometry", "rd_features",
                 "rd_grid", "rd_free", "redraw_hdr_hr_g50"):
        cfg = dict(CONFIGS[name], arch="resnet18", pretrained=False, K=4, hdr_S=4)
        make_model(cfg, 5)
    assert CONFIGS["rt_both"]["matched_aug"] and not CONFIGS["rt_both"].get("hdr")


def test_frozen_backbone_needs_a_control(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "LOCAL", tmp_path / "local")

    class FakeStore:
        device = "cpu"

    with pytest.raises(ValueError):
        T.train("seq", dict(CONFIGS["sequential"], arch="resnet18", pretrained=False, K=4, epochs=1),
                [0, 1], "x", FakeStore(), torch.zeros(2, dtype=torch.long).numpy(), tmp_path / "work")


def test_run_meta():
    meta = T.run_meta("cpu", seconds=1.5)
    assert {"device", "torch", "timm", "python", "time", "seconds"} <= set(meta)
