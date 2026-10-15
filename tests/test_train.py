import pytest
import torch

from helpers import fake_zip
from redraw import train as T
from redraw.data import Store, build_cache, load_index
from redraw.model import make_model
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


def test_strokewarp_training(setup):
    tmp_path, store, labels, idx = setup
    cfg = dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=True, K=4, render_weight=1.0,
               hdr=True, hdr_k=2, hdr_M=1, hdr_S=4, epochs=2, lr=1e-3, wd=0.0, batch=8, aug="geometric",
               strokewarp=dict(p=1.0, warmup=0))
    model = T.train("sw", cfg, idx["train"], "dev", store, labels, tmp_path / "work", val_idx=idx["val"])
    ck = torch.load(tmp_path / "work" / "runs" / "sw" / "dev" / "final.pt", weights_only=False)
    assert {"bce", "hdr_fit", "hdr_margin", "hdr_disc", "val_logits"} <= set(ck["history"][-1])
    assert set(T.predict(model, idx["test"], cfg, store)) == {"logits", "fused", "img", "vec"}


def test_short_run_options_and_resized_input(setup):
    tmp_path, store, labels, idx = setup
    cfg = dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=False, epochs=3, lr=1e-3, wd=0.0,
               batch=8, aug="geometric", res=80, val_every=10**9, warmup_steps=2)
    model = T.train("short", cfg, idx["train"], "dev", store, labels, tmp_path / "work", val_idx=idx["val"])
    history = torch.load(tmp_path / "work" / "runs" / "short" / "dev" / "final.pt", weights_only=False)["history"]
    assert ["val_logits" in h for h in history] == [False, False, True]
    assert model.input_res == 80 and T.fetch(store, idx["val"][:2], 80).shape == (2, 1, 80, 80)
    seen = []
    hook = model.backbone.register_forward_pre_hook(lambda m, a: seen.append(a[0].shape[-1]))
    T.predict(model, idx["val"], cfg, store)
    hook.remove()
    assert seen and set(seen) == {80}


def test_test_time_redrawing(setup):
    tmp_path, store, labels, idx = setup
    torch.manual_seed(0)
    model = make_model(dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=True, K=4, hdr=True,
                            hdr_k=3, route="honest"), 55)
    probs = T.predict_ttr(model, idx["val"], store, n=2, views=("id", "s0.94"), heads=("logits", "img"))
    assert set(probs) == {"logits", "img"} and probs["logits"].shape == (len(idx["val"]), 55)
    assert abs(probs["logits"].sum(1) - 1).max() < 1e-4
    again = T.predict_ttr(model, idx["val"], store, n=2, views=("id", "s0.94"), heads=("logits", "img"))
    assert (probs["logits"] == again["logits"]).all()  # fixed draws: repeatable
    plain = T.predict_ttr(model, idx["val"], store, n=0)["logits"]
    assert abs(plain - T.predict(model, idx["val"], {}, store, heads=("logits",))["logits"]).max() < 1e-5
    base = make_model(dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=False), 55)
    with pytest.raises(ValueError):
        T.predict_ttr(base, idx["val"], store)


def test_sparse_checkpoints_and_logs_still_resume(setup, monkeypatch):
    tmp_path, store, labels, idx = setup
    cfg = dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=False, epochs=5, lr=1e-3, wd=0.0,
               batch=8, aug="geometric", ckpt_every=2, log_every=10, val_every=10**9)
    steps = len(idx["train"]) // 8
    calls = []
    real_augment = T.augment

    def crash_in_epoch_4(x, aug):
        calls.append(1)
        if len(calls) > 3 * steps:
            raise Stop
        return real_augment(x, aug)

    monkeypatch.setattr(T, "augment", crash_in_epoch_4)
    logged = []
    with pytest.raises(Stop):
        T.train("sparse", cfg, idx["train"], "dev", store, labels, tmp_path / "work", log=logged.append)
    assert [m.split()[2] for m in logged] == ["1"]  # only the first epoch is logged before the crash
    monkeypatch.setattr(T, "augment", real_augment)
    logged.clear()
    T.train("sparse", cfg, idx["train"], "dev", store, labels, tmp_path / "work", log=logged.append)
    assert logged[0].endswith("resuming at epoch 3")  # the last checkpoint was at epoch 2
    assert [m.split()[2] for m in logged[1:]] == ["3", "5"]  # the first epoch of this session, and the last
    history = torch.load(tmp_path / "work" / "runs" / "sparse" / "dev" / "final.pt", weights_only=False)["history"]
    assert [h["epoch"] for h in history] == [1, 2, 3, 4, 5]


def test_check_outputs(setup):
    tmp_path, store, labels, idx = setup
    torch.manual_seed(0)
    model = make_model(dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=True, K=4, hdr=True,
                            hdr_k=3, route="honest"), 55)
    out = T.check_outputs(model, idx["val"], store)
    n = len(idx["val"])
    assert out["cand"].shape == (n, 3) and out["energy"].shape == (n, 3) and out["probs"].shape == (n, 55)
    assert (out["pred"] == out["probs"].argmax(1)).all()
    assert all(p in c for p, c in zip(out["pred"], out["cand"]))  # the answer is always a candidate
