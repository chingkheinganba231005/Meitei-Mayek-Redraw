import pytest
import torch

from helpers import fake_zip
from redraw.data import Store, build_cache, load_index
from redraw.erase import CHOOSERS, erase, erase_test, gradient_x_input, reader_importance, stroke_pixels
from redraw.model import make_model
from redraw.render import render
from redraw.split import make_split

# a horizontal stroke, a vertical stroke clear of it, and a third that is not written
CTRL = torch.tensor([[[[0.2, 0.3], [0.4, 0.3], [0.6, 0.3], [0.8, 0.3]],
                      [[0.7, 0.5], [0.7, 0.63], [0.7, 0.77], [0.7, 0.9]],
                      [[0.1, 0.8], [0.2, 0.8], [0.3, 0.8], [0.4, 0.8]]]])
WIDTH = torch.tensor([[0.06, 0.06, 0.06]])
EXIST = torch.tensor([[0.99, 0.99, 0.1]])


def px(own, x, y, size=64):
    return own[..., int(y * size), int(x * size)]


def test_every_pixel_belongs_to_at_most_one_written_stroke():
    own = stroke_pixels(CTRL, WIDTH, EXIST, 64)
    assert own.shape == (1, 3, 64, 64)
    assert own.sum(1).max() <= 1 and not own[0, 2].any()
    assert px(own[0, 0], 0.5, 0.3) and px(own[0, 1], 0.7, 0.7)
    assert not px(own[0], 0.05, 0.95).any()


def test_erasing_a_stroke_leaves_the_ink_of_the_others():
    img, _, _ = render(CTRL, WIDTH, EXIST, G=64)
    x = img.unsqueeze(1)
    rest, _, _ = render(CTRL, WIDTH, torch.tensor([[0.0, 0.99, 0.1]]), G=64)  # all but the first
    own = stroke_pixels(CTRL, WIDTH, EXIST, 64)
    left = erase(x, own, torch.tensor([0]))
    assert left[0, 0, int(0.3 * 64), int(0.5 * 64)] == 0
    assert torch.isclose(left.sum(), rest.sum(), rtol=0.02)


@pytest.fixture
def setup(tmp_path):
    fake_zip(tmp_path / "t.zip")
    split_dir, _ = make_split(tmp_path / "t.zip", tmp_path / "data")
    index, labels, idx = load_index(split_dir)
    store = Store(build_cache(index.file.tolist(), 64, tmp_path / "cache", workers=1), "cpu", idx["train"])
    torch.manual_seed(0)
    ours = make_model(dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=True, K=4, hdr=True,
                           hdr_k=2, hdr_M=1, hdr_S=4, route="honest"), 55).eval()
    torch.nn.init.normal_(ours.decoder.ctrl[-1].weight, std=0.5)  # strokes that differ between images
    torch.nn.init.constant_(ours.decoder.exist[-1].bias, 2.0)    # every stroke written
    plain = make_model(dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=False), 55).eval()
    return store, labels, idx, ours, plain


def test_reader_importance_only_for_written_strokes(setup):
    store, _, idx, ours, plain = setup
    x = store.fetch(idx["test"][:3], "gray")
    xn = store.normalize(x, "gray")
    with torch.no_grad():
        o = ours(xn, draw=False, ink=x)
        o["exist"][:, 1] = 0.2
        imp = reader_importance(ours, xn, o, o["logits"].argmax(1))
    assert imp.shape == (3, 4) and torch.isinf(imp[:, 1]).all() and torch.isfinite(imp[:, [0, 2, 3]]).all()
    with pytest.raises(ValueError):
        reader_importance(plain, xn, o, o["logits"].argmax(1))


def test_gradient_x_input_is_zero_on_paper(setup):
    store, _, idx, ours, plain = setup
    x = store.fetch(idx["test"][:2], "gray")
    for net in (ours, plain):
        gxi = gradient_x_input(net, x, store, torch.tensor([0, 1]))
        assert gxi.shape == x.shape and (gxi[x == 0] == 0).all() and gxi.abs().sum() > 0


def test_erase_test(setup):
    store, labels, idx, ours, plain = setup
    te = idx["test"]
    out = erase_test(ours, {"ours": ours, "plain": plain}, te, store, bs=4)
    n = len(te)
    assert (out["strokes"] == 4).all()
    assert set(out["erased"]) == set(CHOOSERS) and set(out["stroke"]) == set(out["ink_erased"])
    assert {"gradient x input, ours", "gradient x input, plain"} <= set(out["stroke"])
    for chooser in CHOOSERS:
        for name in ("ours", "plain"):
            assert out["erased"][chooser][name].shape == (n,)
    for v in out["ink_erased"].values():
        assert v.shape == (n,) and (v >= 0).all() and (v <= 1 + 1e-5).all()
    assert (out["stroke"]["reader, most"] != out["stroke"]["reader, least"]).any()
    # the network reads the intact image the same way as when it is not being explained
    with torch.no_grad():
        x = store.fetch(te, "gray")
        again = ours(store.normalize(x, "gray"), draw=False, ink=x)["logits"].argmax(1).numpy()
    assert (again == out["pred"]["ours"]).all()
