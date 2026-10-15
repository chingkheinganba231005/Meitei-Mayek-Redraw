import pytest
import torch

from helpers import fake_zip
from redraw import train as T
from redraw.data import Store, build_cache, load_index
from redraw.split import make_split


class Stop(Exception):
    pass


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "LOCAL", tmp_path / "local")
    monkeypatch.setattr(T, "DRIVE_EVERY", 1)
    fake_zip(tmp_path / "t.zip")
    split_dir, _ = make_split(tmp_path / "t.zip", tmp_path / "data")
    index, labels, idx = load_index(split_dir)
    store = Store(build_cache(index.file.tolist(), 64, tmp_path / "cache", workers=1), "cpu", idx["train"])
    return tmp_path, store, labels, idx


def test_stroke_model_training_resumes_after_a_crash(setup):
    tmp_path, store, labels, idx = setup
    cfg = dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=True, K=4, render_weight=1.0,
               epochs=3, lr=1e-3, wd=0.0, batch=8, aug="geometric")

    def crash_in_epoch_2(msg):
        if "epoch 2" in msg:
            raise Stop

    with pytest.raises(Stop):
        T.train("m", cfg, idx["train"], "dev", store, labels, tmp_path / "work", log=crash_in_epoch_2)
    logged = []
    model = T.train("m", cfg, idx["train"], "dev", store, labels, tmp_path / "work", val_idx=idx["val"],
                    log=logged.append)
    assert logged[0].endswith("resuming at epoch 2")
    ck = torch.load(tmp_path / "work" / "runs" / "m" / "dev" / "final.pt", weights_only=False)
    last = ck["history"][-1]
    assert [h["epoch"] for h in ck["history"]] == [1, 2, 3]
    assert {"bce", "prec", "cover", "count", "val_logits", "val_img", "val_vec", "parse_f"} <= set(last)

    probs = T.predict(model, idx["test"], cfg, store, views=("id", "s0.94"))
    assert set(probs) == {"logits", "fused", "img", "vec"} and probs["logits"].shape == (len(idx["test"]), 55)
    score = T.parse_eval(model, idx["test"], store)
    assert set(score) == {"precision", "recall", "f", "strokes"}
    preds = T.cached_preds(tmp_path / "work", "m", "test", lambda: probs)
    assert set(preds) == set(probs)


def test_baseline_training(setup):
    tmp_path, store, labels, idx = setup
    cfg = dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=False, epochs=1, lr=1e-3, wd=0.0,
               batch=8, aug="geometric")
    model = T.train("b", cfg, idx["train"], "dev", store, labels, tmp_path / "work", val_idx=idx["val"])
    assert set(T.predict(model, idx["test"], cfg, store)) == {"logits", "fused", "img"}
