import torch

from redraw.augment import AUG_PRESETS, TTA_VIEWS, augment, view
from redraw.model import make_model
from redraw.train import compute_loss

TINY = dict(arch="resnet18", drop_path=0.0, pretrained=False, K=4)


def test_baseline_and_stroke_model_outputs():
    x = torch.rand(2, 1, 64, 64)
    base = make_model(dict(TINY, strokes=False), 55).eval()
    out = base(x)
    assert set(out) == {"img", "fused", "logits"} and out["logits"].shape == (2, 55)
    full = make_model(dict(TINY, strokes=True), 55).eval()
    out = full(x)
    assert out["logits"].shape == (2, 55) and out["vec"].shape == (2, 55)
    assert torch.allclose(out["logits"], out["img"] + out["vec"]) and torch.equal(out["logits"], out["fused"])
    assert out["ctrl"].shape == (2, 4, 4, 2) and out["render"].shape == (2, 48, 48)
    assert out["ctrl"].min() >= 0 and out["ctrl"].max() <= 1
    assert "render" not in full(x, draw=False)


def test_stroke_model_trains_end_to_end():
    torch.manual_seed(0)
    model = make_model(dict(TINY, strokes=True), 5)
    x, y = torch.rand(3, 1, 64, 64), torch.tensor([0, 1, 2])
    cfg = dict(render_weight=1.0, head_weight=0.5)
    out = model(x, jitter=0.01)
    loss, parts = compute_loss(model, out, x, y, torch.nn.CrossEntropyLoss(), cfg)
    loss.backward()
    assert set(parts) == {"bce", "prec", "cover", "count"}
    for part in (model.decoder, model.head, model.backbone):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())


def test_no_render_variant_has_no_reconstruction_loss():
    model = make_model(dict(TINY, strokes=True), 5)
    x, y = torch.rand(2, 1, 64, 64), torch.tensor([0, 1])
    out = model(x, draw=False)
    loss, parts = compute_loss(model, out, x, y, torch.nn.CrossEntropyLoss(), dict(render_weight=0.0))
    assert parts == {} and torch.isfinite(loss)


def test_augment_keeps_shape_and_range():
    torch.manual_seed(0)
    x = torch.rand(8, 1, 32, 32)
    for preset in AUG_PRESETS.values():
        y = augment(x, preset)
        assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1


def test_views():
    x = torch.rand(2, 1, 32, 32)
    assert torch.equal(view(x, "id"), x)
    for v in TTA_VIEWS:
        assert view(x, v).shape == x.shape


def test_without_stroke_head_strokes_still_learn_from_the_ink():
    torch.manual_seed(0)
    model = make_model(dict(TINY, strokes=True, stroke_head=False), 5)
    x, y = torch.rand(2, 1, 64, 64), torch.tensor([0, 1])
    out = model(x)
    assert "vec" not in out and torch.equal(out["fused"], out["img"]) and "render" in out
    loss, parts = compute_loss(model, out, x, y, torch.nn.CrossEntropyLoss(), dict(render_weight=1.0))
    loss.backward()
    assert set(parts) == {"bce", "prec", "cover", "count"}
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())


def test_loss_is_float32_even_for_bfloat16_logits():
    y = torch.tensor([0, 1])
    out = {"fused": torch.randn(2, 5).bfloat16(), "img": torch.randn(2, 5).bfloat16()}
    loss, _ = compute_loss(None, out, torch.rand(2, 1, 8, 8), y, torch.nn.CrossEntropyLoss(label_smoothing=0.1), {})
    assert loss.dtype == torch.float32
