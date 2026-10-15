import torch
import torch.nn.functional as F

from redraw.model import make_model

TINY = dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=True, K=4, hdr=True, hdr_k=3)


def _model(seed=0, trained=False):
    torch.manual_seed(seed)
    model = make_model(TINY, 7)
    if trained:  # as after a few steps: the fitter's zero-initialised output layers have moved
        for head in (model.hdr.offset, model.hdr.affine, model.hdr.exist, model.hdr.wscale):
            torch.nn.init.normal_(head[-1].weight, std=0.02)
    return model


def grads(module):
    return [p.grad for p in module.parameters()]


def test_candidates_scores_and_output():
    model = _model().eval()
    x = torch.rand(4, 1, 64, 64)
    with torch.no_grad():
        out = model(x, ink=x)
    assert out["cand"].shape == (4, 3) and out["cost"].shape == (4, 3) and out["bonus"].shape == (4, 7)
    assert torch.allclose(out["logits"], out["fused"] + out["bonus"])
    # the bonus only touches the candidates and sums to zero over them
    off = torch.ones(4, 7, dtype=torch.bool).scatter(1, out["cand"], False)
    assert out["bonus"][off].abs().max() == 0 and out["bonus"].sum(1).abs().max() < 1e-5
    assert "cand" not in model(x)  # no ink, no hypothesis check


def test_true_class_is_always_a_candidate_in_training():
    model = _model().train()
    x, y = torch.rand(6, 1, 64, 64), torch.tensor([0, 1, 2, 3, 4, 5])
    out = model(x, ink=x, labels=y)
    assert (out["cand"] == y[:, None]).any(1).all()
    assert out["live"].sum(1).eq(1).all()


def test_fit_trains_fitter_and_own_templates_only():
    model = _model(trained=True).train()
    x, y = torch.rand(3, 1, 64, 64), torch.tensor([1, 2, 3])
    out = model(x, ink=x, labels=y)
    out["cost"].mean().backward()
    hdr = model.hdr
    assert hdr.offset[0].weight.grad.abs().sum() > 0 and hdr.cls_emb.weight.grad.abs().sum() > 0
    assert any(g is not None and g.abs().sum() > 0 for g in grads(model.backbone))
    touched = hdr.t_ctrl.grad.abs().flatten(1).sum(1) > 0
    assert set(torch.nonzero(touched).flatten().tolist()) <= {1, 2, 3}  # only the true classes
    assert model.beta.grad is None and model.gamma.grad is None


def test_discrimination_trains_templates_and_scalars_only():
    model = _model(trained=True).train()
    x, y = torch.rand(3, 1, 64, 64), torch.tensor([1, 2, 3])
    out = model(x, ink=x, labels=y)
    F.cross_entropy(out["verified"], y).backward()
    hdr = model.hdr
    assert hdr.t_ctrl.grad.abs().sum() > 0 and model.beta.grad is not None and model.gamma.grad is not None
    touched = set(torch.nonzero(hdr.t_ctrl.grad.abs().flatten(1).sum(1) > 0).flatten().tolist())
    assert touched <= set(out["cand"].flatten().tolist())  # never a class that was not a candidate
    for g in grads(hdr.blocks) + grads(hdr.offset) + grads(model.backbone) + grads(model.decoder):
        assert g is None or g.abs().sum() == 0  # the fitter and the features are not rewarded for this


def test_margin_trains_templates_only():
    from redraw.train import compute_loss
    model = _model(trained=True).train()
    x, y = torch.rand(3, 1, 64, 64), torch.tensor([1, 2, 3])
    out = model(x, ink=x, labels=y)
    cfg = dict(render_weight=0.0, head_weight=0.0, hdr_fit=0.0, hdr_disc=0.0, hdr_margin=1.0, hdr_m=10.0)
    loss, parts = compute_loss(model, out, x, y, torch.nn.CrossEntropyLoss(), cfg)
    model.zero_grad()
    parts["hdr_margin"].backward()
    assert parts["hdr_margin"] > 0 and model.hdr.t_ctrl.grad.abs().sum() > 0
    for g in grads(model.hdr.blocks) + grads(model.hdr.offset) + grads(model.backbone):
        assert g is None or g.abs().sum() == 0


def test_end_to_end_ablation_lets_every_loss_reach_the_fitter():
    torch.manual_seed(0)
    model = make_model(dict(TINY, honest=False), 7)
    for head in (model.hdr.offset, model.hdr.affine, model.hdr.exist, model.hdr.wscale):
        torch.nn.init.normal_(head[-1].weight, std=0.02)
    model.train()
    x, y = torch.rand(3, 1, 64, 64), torch.tensor([1, 2, 3])
    out = model(x, ink=x, labels=y)
    F.cross_entropy(out["verified"], y).backward()
    assert model.hdr.offset[0].weight.grad.abs().sum() > 0           # the fitter now learns to discriminate
    touched = set(torch.nonzero(model.hdr.t_ctrl.grad.abs().flatten(1).sum(1) > 0).flatten().tolist())
    assert touched == set(out["cand"].flatten().tolist())


def test_verify_eval_counts(tmp_path):
    from helpers import fake_zip
    from redraw.data import Store, build_cache, load_index
    from redraw.split import make_split
    from redraw.train import verify_eval
    fake_zip(tmp_path / "t.zip", classes=7)
    split_dir, _ = make_split(tmp_path / "t.zip", tmp_path / "data")
    index, labels, idx = load_index(split_dir)
    store = Store(build_cache(index.file.tolist(), 64, tmp_path / "cache", workers=1), "cpu", idx["train"])
    r = verify_eval(_model(), idx["test"], store, labels)
    assert set(r) == {"topk_recall", "cost_only", "fused", "final", "rescued", "broken"}
    assert 0 <= r["cost_only"] <= r["topk_recall"] <= 1
