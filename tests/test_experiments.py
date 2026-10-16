import json

import numpy as np
import pytest
import torch

from redraw import experiments as X
from redraw.manifest import clean, write_json
from redraw.model import make_model
from redraw.train import cfg_conflicts, train


def test_metric_summary_counts_failures_and_skips_missing_values():
    m = {"f@1": [0.9, np.nan, 0.2, 0.8], "curves": [3, 0, 2, 4], "iou": [0.5, 0.5, 0.5, 0.5]}
    s = X.summarise_metrics(m)
    assert s["f@1"]["missing"] == 1 and s["f@1"]["mean"] == pytest.approx((0.9 + 0.2 + 0.8) / 3)
    assert s["failures"] == {"no_curve": 1, "f@1_below_0.5": 2, "images": 4}


def test_identical_parses_are_perfectly_stable():
    rng = np.random.default_rng(0)
    pts = rng.random((5, 4, 32, 2))
    on = np.ones((5, 4), bool)
    on[0, 3] = False
    c = X.compare_parses(pts, on, pts, on)
    assert c["union_chamfer_mean"] == pytest.approx(0) and c["matched_curve_mean"] == pytest.approx(0)
    assert c["same_stroke_count"] == 1.0
    reversed_pts = pts[:, :, ::-1]                       # same curves drawn the other way round
    assert X.compare_parses(pts, on, reversed_pts, on)["matched_curve_mean"] == pytest.approx(0)


def test_paired_tests_count_discordant_images():
    y = np.array([0, 1, 2, 0, 1, 2])
    right = np.eye(3)[y]
    wrong = np.eye(3)[(y + 1) % 3]
    P = {}
    for s in range(3):
        P[("control", s, "test_id")] = {"logits": np.where(np.arange(6)[:, None] < 1 + s, wrong, right)}
        P[("ours", s, "test_id")] = {"logits": right}
    t = X.paired_tests(P, y)
    assert t["test_id/seed1"]["control_only_right"] == 0 and t["test_id/seed1"]["ours_only_right"] == 2
    assert t["test_id/across_seeds"]["n"] == 3 and t["test_id/across_seeds"]["wins"] == 3
    assert "test_id/average_of_seeds_0-2" in t


def test_core_summary_pairs_each_seed_with_its_control():
    runs = {}
    for s, (c, o) in enumerate([(0.97, 0.98), (0.96, 0.97), (0.98, 0.98)]):
        runs[f"control_m/{s}"] = {"val_5view": {"logits": c}, "val_1view": {"logits": c}, "train_minutes": 10,
                                  "parameters": 1, "flops_1view": 1}
        runs[f"rt_both/{s}"] = {"val_5view": {"logits": o, "img": o - 0.01, "vec": o - 0.02},
                                "val_1view": {"logits": o}, "train_minutes": 20, "parameters": 2, "flops_1view": 2,
                                "rendered_ink_val": {"f": 0.9}}
    s = X.summarise_core({"runs": runs})
    d = s["rt_both"]["val_5view_vs_control"]
    assert d["n"] == 3 and d["wins"] == 2 and d["ties"] == 1
    assert s["rt_both"]["val_5view_vec"]["mean"] == pytest.approx(np.mean([0.96, 0.95, 0.96]))


def test_robustness_summary_is_paired_by_seed():
    conds = [("framed", "blur", 1.0), ("framed", "noise", 0.1)]
    R = {"framed/blur_1.0": {"control/0": 0.8, "ours/0": 0.82, "control/1": 0.7, "ours/1": 0.71},
         "framed/noise_0.1": {"control/0": 0.9, "ours/0": 0.88, "control/1": 0.9, "ours/1": 0.9}}
    s = X.summarise_robustness(R, conds)["framed"]
    assert s["minus_control_by_seed"]["framed/blur_1.0"]["ours"] == pytest.approx([0.02, 0.01])
    assert s["families"]["ours"]["networks"] == 2


def test_json_is_strict_and_numpy_safe(tmp_path):
    obj = {"a": np.float32(0.5), "b": np.int64(3), "c": float("nan"), "d": np.array([1.0, np.inf]), (1, 2): True}
    write_json(tmp_path / "x.json", obj)
    back = json.loads((tmp_path / "x.json").read_text())
    assert back == {"a": 0.5, "b": 3, "c": None, "d": [1.0, None], "(1, 2)": True}
    assert clean(np.bool_(True)) is True


def test_finished_run_is_not_reused_for_other_settings(tmp_path):
    assert cfg_conflicts({"lr": 1e-3, "seed": 0}, {"lr": 2e-3, "seed": 0, "new": 1}) == {"lr": (1e-3, 2e-3)}
    out = tmp_path / "runs" / "toy" / "dev"
    out.mkdir(parents=True)
    torch.save({"cfg": {"arch": "resnet18", "epochs": 2}, "model": {}}, out / "final.pt")
    class Store:
        device = "cpu"

    with pytest.raises(ValueError, match="other settings"):
        train("toy", {"arch": "resnet18", "epochs": 3}, np.arange(4), "dev", Store(), np.zeros(4, int), tmp_path)


def test_figure_energies_are_the_soft_minimum_of_the_template_energies():
    torch.manual_seed(0)
    cfg = dict(arch="resnet18", drop_path=0.0, pretrained=False, strokes=True, K=4, hdr=True, hdr_S=4, route="honest")
    model = make_model(cfg, 6).eval()

    class Store:
        def normalize(self, x, _):
            return x

    x = torch.rand(3, 1, 64, 64)
    o, cand, bent, energy, per = X.hypotheses(model, x, Store())
    tau = model.hdr.tau
    assert torch.allclose(energy, -tau * torch.logsumexp(-per / tau, dim=2), atol=1e-5)
    assert torch.equal(cand, o["cand"]) and per.shape == (3, cand.shape[1], model.hdr.M)
