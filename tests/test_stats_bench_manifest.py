import json
import zipfile

import numpy as np
import torch

from redraw.bench import flops, latency, parameters, train_step_time
from redraw.explain import candidate_margins, check_transitions, refit, render_parses, representation_test
from redraw.manifest import environment, export_zip, file_sha256, split_hash, write_json
from redraw.model import make_model
from redraw.stats import paired_bootstrap, paired_runs, per_class, signed_rank_p

TINY_CHECK = dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=True, K=4, hdr=True, hdr_k=3,
                  hdr_M=1, hdr_S=4, route="honest")


def test_paired_runs_reproduces_the_paper():
    control = [0.97757, 0.97835, 0.97897]
    ours = [0.97897, 0.97944, 0.97827]
    s = paired_runs(control, ours)
    assert (s["wins"], s["losses"], s["ties"]) == (2, 1, 0) and abs(s["mean"] - 0.0006) < 1e-4
    # the 12 paired runs of the paper, in images of the 12,794 test images (ties: +-12, +-9, 13 and 13)
    images = np.array([18, 14, -9, 12, 30, -12, 11, 12, 17, 9, 13, 13])
    twelve = paired_runs(np.zeros(12), images / 12794)
    assert twelve["wins"] == 10 and abs(twelve["wilcoxon_exact_p"] - 0.0083) < 5e-4
    assert abs(signed_rank_p(images) - twelve["wilcoxon_exact_p"]) < 1e-12


def test_paired_bootstrap_and_per_class():
    rng = np.random.default_rng(0)
    a = rng.random(2000) < 0.9
    b = a.copy()
    b[:20] = True
    s = paired_bootstrap(a, b, n=2000)
    assert s["low"] <= s["difference"] <= s["high"] and s["only_a_right"] == 0
    p = per_class([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert p["recall"] == [0.5, 1.0] and p["support"] == [2, 2]


def test_check_bookkeeping():
    y = np.array([0, 1, 2, 3])
    fused, final = np.array([0, 2, 2, 1]), np.array([0, 1, 0, 2])
    t = check_transitions(fused, final, y)
    assert t == {"unchanged right": 1, "unchanged wrong": 0, "wrong -> right": 1, "right -> wrong": 1,
                 "wrong -> other wrong": 1, "images": 4}
    m = candidate_margins(np.array([[0, 1, 2], [3, 4, 5]]), np.array([[0.1, 0.3, 0.2], [0.2, 0.1, 0.3]]), np.array([0, 9]))
    assert m["in_candidates"].tolist() == [True, False] and abs(m["margin"][0] - 0.1) < 1e-9
    assert m["energy_pick_right"].tolist() == [True, False]


class Store:
    device = "cpu"

    def __init__(self, x):
        self.x = x

    def fetch(self, idx, channels):
        return self.x[torch.as_tensor(np.asarray(idx))]

    def normalize(self, x, channels):
        return (x - 0.1) / 0.3


def test_interventions_and_refit_run_on_a_small_network():
    torch.manual_seed(0)
    model = make_model(TINY_CHECK, 10).eval()
    for head in (model.decoder.ctrl, model.decoder.exist, model.hdr.offset, model.hdr.affine):
        torch.nn.init.normal_(head[-1].weight, std=0.05)
    torch.nn.init.constant_(model.decoder.exist[-1].bias, 2.0)
    store = Store(torch.rand(6, 1, 64, 64))
    r = representation_test(model, np.arange(6), store, bs=4)
    assert r["pred"].shape == (6,) and (r["strokes"] == 4).all() and np.isfinite(r["drop_most"]).all()
    assert (r["drop_most"] >= r["drop_least"] - 1e-5).mean() > 0.5
    img = render_parses(model, np.arange(6), store, R=32, bs=4)
    assert img.shape == (6, 32, 32) and img.dtype == np.uint8
    f = refit(model, store.x[:3], store, steps=15, lr=0.05)
    assert f["energy_refit"].shape == f["energy_fitter"].shape == (3, 3)
    assert (f["energy_refit"] <= f["energy_fitter"] + 1e-4).mean() > 0.6


def test_bench_on_cpu():
    model = make_model(TINY_CHECK, 10)
    p = parameters(model)
    assert p["total"] == sum(v.numel() for v in model.parameters()) and p["hypothesis check"] > 0
    assert flops(model, size=64) > 0
    assert latency(model, "cpu", size=64, iters=2, warmup=1)["median_ms"] > 0
    assert train_step_time(dict(TINY_CHECK, render_weight=1.0), "cpu", batch=2, size=64, steps=1, warmup=0,
                           num_classes=10)["seconds_per_step"] > 0


def test_manifest_tools(tmp_path):
    env = environment(".")
    assert {"torch", "device", "commit", "time"} <= set(env)
    for name in ("train.csv", "val.csv", "test.csv"):
        (tmp_path / name).write_text(name)
    h = split_hash(tmp_path)
    (tmp_path / "test.csv").write_text("changed")
    assert split_hash(tmp_path) != h
    write_json(tmp_path / "out" / "r.json", {"a": np.float32(1.5)})
    assert json.loads((tmp_path / "out" / "r.json").read_text()) == {"a": 1.5}
    (tmp_path / "out" / "w.pt").write_bytes(b"x")
    files = export_zip(tmp_path / "pack.zip", tmp_path / "out")
    assert files == ["r.json"] and zipfile.ZipFile(tmp_path / "pack.zip").namelist() == ["r.json"]
    assert len(file_sha256(tmp_path / "pack.zip")) == 64
