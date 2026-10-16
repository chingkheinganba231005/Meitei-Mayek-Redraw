"""The revision experiments, one function each, called from the notebooks ``notebooks/rev*.ipynb``.

Every function can be interrupted and run again: networks are kept on Drive under ``work/runs``, saved
predictions under ``work/preds``, and results under ``work/revision`` (JSON written atomically, per-image
arrays as ``.npz``). What is already there is skipped. ``manifest.json`` records the data and split
hashes and the environment of every session; ``export`` packs everything small into one ZIP.

    setup                  data, split, cache, and the context every other function takes
    run_audit              gradient contract of every configuration, learned check scales, renderer on
                           real parses, per-class results, paired tests and check transitions (saved predictions)
    run_bench              parameters, FLOPs, latency, throughput, memory, training step time, preprocessing
    run_core               development split, validation: routing factorial, reader controls, simpler alternatives
    run_probes             what class information the frozen geometry and features hold (linear and MLP probes)
    run_extra_seeds        seeds 3 and 4 of the control and the method on all training images (test)
    run_recon              reconstruction against skeleton tracing and per-image optimisation, with budgets
    run_stability          parses of slightly moved images mapped back, and parses across seeds
    run_interventions      stroke erasure with controls, reader-input interventions, recognition from drawings
    run_checker            check transitions, energy margins, selective prediction, refitting the fitter
    run_coupled_seeds      seeds 1 and 2 of coupled routing
    run_robustness         raw and fixed-framing damage, texture and shape changes, crop failures
    run_dial               share of the drawing gradient let into the backbone (development split, validation)
    setup_second / run_second_dataset
                           the Devanagari Handwritten Character Dataset, same recipe and protocol
    run_figures            figures for the paper (readable labels, fixed selection rules)
    export                 run table, per-image table, every results file and the list of what is missing
"""

import json
import math
import os
import time
import zipfile
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import audit as audit_mod
from . import bench as bench_mod
from . import recon
from .analysis import mcnemar, rank_average, risk_at, risk_coverage
from .augment import TTA_VIEWS
from .data import Store, build_cache, load_index
from .erase import CHOOSERS, erase_test
from .explain import candidate_margins, check_transitions, refit, render_parses, representation_test
from .manifest import environment, export_zip, file_sha256, read_json, split_hash, write_json
from .model import CONFIGS, make_model
from .render import curve_points, parse_score, render
from .robustness import (CORRUPTIONS, FRAMED, SCAN_PX, TEXTURE_SHAPE, corrupt_file, corrupt_framed, crop_failures,
                         texture_shape)
from .split import URL, download, make_split, make_split_folders
from .stats import paired_bootstrap, paired_runs, per_class
from .train import amp_dtype, cached_preds, check_outputs, fetch, load_model, parse_eval, predict, train, verify_eval

REPO_ROOT = Path(__file__).resolve().parents[1]
TUMMHCD_SHA256 = "7637cb9ee3209b15b27d75b0e4dd03a433676a6ae9ddcd84b9511a79d6e9ce30"
SPLIT_SHA256 = "495f5e94e0dc4e7bdef7801f7b116baa1433c012a942ecf1227525dc2253f494"   # data/splits/*.csv
DHCD_URL = "https://archive.ics.uci.edu/static/public/389/devanagari+handwritten+character+dataset.zip"
SEEDS = (0, 1, 2)
PAIRS = [(44, 25), (46, 9), (11, 47), (33, 34)]
FINAL = {"control": "baseline_fp32", "ours": "redraw_hdr_hr", "coupled": "redraw_hdr"}
CORE = {  # development split (61,504 training images), validation (10,826); matched data and augmentation
    "control_m": "image-only control",
    "rt_none": "routing factorial: neither rule (coupled routing)",
    "rt_backbone": "routing factorial: drawing kept out of the backbone only",
    "rt_geometry": "routing factorial: reading cannot move the strokes only",
    "rt_both": "routing factorial: both rules (honest routing, no check)",
    "rd_geometry": "reader sees stroke geometry only",
    "rd_features": "reader sees backbone features sampled along the strokes only",
    "rd_grid": "reader samples backbone features along a fixed grid of curves",
    "rd_free": "learned stroke slots trained by classification, no rendering loss",
    "sequential": "decoder and reader trained on the frozen control (staged training)",
}


@dataclass
class Ctx:
    work: Path              # Drive folder with the networks (runs/) and saved predictions (preds/)
    out: Path               # results of the revision: JSON, per-image arrays, figures
    cache: Path             # local folder for the preprocessed images
    device: str
    fast: bool
    img: int
    index: object
    labels: np.ndarray
    idx: dict
    store: Store
    split_dir: Path
    num_classes: int
    names: list
    dataset: str = "TUMMHCD"
    meta: dict = field(default_factory=dict)

    def path(self, name):
        return self.out / f"{name}.json"

    def load(self, name):
        return read_json(self.path(name))

    def save(self, name, obj):
        write_json(self.path(name), obj)


# ------------------------------------------------------------------------------------------- set-up

def fast_cfg(cfg, fast):
    """FAST: a two-epoch resnet18 with few strokes, for a CPU smoke test of the code (no effect otherwise)."""
    if not fast:
        return dict(cfg)
    return dict(cfg, arch="resnet18", epochs=min(cfg.get("epochs", 40), 2), pretrained=False, batch=16, K=4, hdr_S=4)


def final_path(work, name, tag):
    return Path(work) / "runs" / name / tag / "final.pt"


def _speed(device):
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _free(device):
    if device == "cuda":
        torch.cuda.empty_cache()


def _find_zip(work, zip_path=None, name="TUMMHCD-TEST-TRAIN.zip", url=URL):
    zip_path = Path(zip_path) if zip_path else Path(work) / name
    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        return zip_path
    drive_root = Path("/content/drive/MyDrive")
    found = [p for d in ("*", "*/*", "*/*/*") for p in drive_root.glob(f"{d}/{name}") if zipfile.is_zipfile(p)]
    if found:
        return found[0]
    print(f"downloading {name} ...")
    download(url, zip_path)
    return zip_path


def _manifest(ctx, zip_path, summary, reference=None):
    m = read_json(ctx.out / "manifest.json")
    if m.get("data_file") != str(zip_path) or "data_sha256" not in m:
        m["data_file"], m["data_sha256"] = str(zip_path), file_sha256(zip_path)
    m.update(dataset=ctx.dataset, split_counts=summary, split_sha256=split_hash(ctx.split_dir),
             split_rule="validation = 15% of every training class, shuffled with seed 42 (redraw.split)",
             preprocessing=f"redraw.preprocess.preprocess at {ctx.img} px (Otsu mask for the crop and the "
                           "paper/ink levels; grey-level ink in [0, 1])")
    if reference:
        m["data_matches_paper"] = m["data_sha256"] == reference[0]
        m["split_matches_paper"] = m["split_sha256"] == reference[1]
    m.setdefault("sessions", []).append(environment(REPO_ROOT))
    write_json(ctx.out / "manifest.json", m)
    return m


def setup(work, data, cache, zip_path=None, fast=False, device=None, out_name="revision"):
    """Unpack and split TUMMHCD, cache the images and return the context. Checks the archive and split
    hashes against those of the paper (a mismatch is reported, not fatal)."""
    work, data, cache = Path(work), Path(data), Path(cache)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _speed(device)
    zp = _find_zip(work, zip_path)
    splits, summary = make_split(zp, data)
    if not fast:
        assert (summary["train"], summary["val"], summary["test"]) == (61504, 10826, 12794), summary
    index, labels, idx = load_index(splits)
    img = 64 if fast else 128
    store = Store(build_cache(index.file.tolist(), img, cache), device, idx["train"])
    from .charset import CLASSES
    names = [f"{c.id} {c.name}" for c in CLASSES][:summary["classes"]]
    ctx = Ctx(work, work / out_name, cache, device, fast, img, index, labels, idx, store, splits,
              summary["classes"], names)
    ctx.out.mkdir(parents=True, exist_ok=True)
    ctx.meta = _manifest(ctx, zp, summary, None if fast else (TUMMHCD_SHA256, SPLIT_SHA256))
    env = ctx.meta["sessions"][-1]
    print(f"{ctx.dataset} {summary} | {env['device']} | torch {env['torch']} | commit {str(env['commit'])[:10]}")
    if not fast:
        print("archive as in the paper:", ctx.meta["data_matches_paper"], "| split as in the paper:",
              ctx.meta["split_matches_paper"])
    return ctx


def load_final(ctx, name, seed, tag=None):
    tag = tag or f"full_s{seed}"
    return load_model(final_path(ctx.work, name, tag), fast_cfg(dict(CONFIGS[name], seed=seed), ctx.fast),
                      num_classes=ctx.num_classes, device=ctx.device)


def available(ctx, name, seeds=(0, 1, 2, 3, 4), tag="full_s{}"):
    return [s for s in seeds if final_path(ctx.work, name, tag.format(s)).exists()]


def saved_preds(ctx, stem, split):
    """Saved probabilities {head: (N, C)} of ``work/preds/<stem>_<split>.npz``, or None."""
    f = ctx.work / "preds" / f"{stem}_{split}.npz"
    if not f.exists():
        return None
    with np.load(f) as z:
        return {h: z[h].astype(np.float32) for h in z.files}


def _acc(p, y):
    return float((np.asarray(p).argmax(1) == y).mean())


def _mean(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def _sd(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.std(v, ddof=1)) if len(v) > 1 else None


def _npz_key(text):
    return "".join(c if c.isalnum() else "_" for c in text)


# ------------------------------------------------------------------------------------------- audit

AUDITED = ("redraw_hdr_hr", "redraw_hdr", "rt_none", "rt_backbone", "rt_geometry", "rt_both", "rd_geometry",
           "rd_features", "rd_grid", "rd_free", "sequential", "redraw_hdr_hr_g50")


def run_audit(ctx, configs=AUDITED, n_parse=2048):
    """Stage A checks that need the GPU or the saved predictions; nothing is trained."""
    R = ctx.load("audit")
    te, y = ctx.idx["test"], ctx.labels[ctx.idx["test"]]
    # 1. which loss trains which parameters, on the real architecture (random weights)
    contracts = R.setdefault("gradient_contract", {})
    for name in configs:
        if name not in contracts:
            table = audit_mod.audit(name, tiny=ctx.fast, size=64, batch=4, num_classes=ctx.num_classes)
            bad = audit_mod.mismatches(table, audit_mod.expected_for(CONFIGS[name]))
            contracts[name] = {"table": table, "markdown": audit_mod.markdown(table), "mismatches": bad}
            print(f"gradient contract {name}: {'as intended' if not bad else bad}")
            ctx.save("audit", R)
    # 2. the learned scales of the check: bonus_c = gamma (h_c - mean h), h_c = -beta E_c
    scales = R.setdefault("check_scales", {})
    for name in ("redraw_hdr_hr", "redraw_hdr"):
        for s in available(ctx, name):
            if f"{name}/{s}" not in scales:
                m = load_final(ctx, name, s)
                b, g = F.softplus(m.beta).item(), F.softplus(m.gamma).item()
                scales[f"{name}/{s}"] = {"beta": b, "gamma": g, "bonus_per_unit_energy": b * g}
                del m
    R["check_scales_note"] = ("beta and gamma pass through softplus, so both are positive and a lower energy "
                              "always raises a candidate's logit; only the product beta * gamma matters")
    ctx.save("audit", R)
    # 3. sampled against exact rendering on real parses; stroke widths, lengths and sample spacing
    if "renderer" not in R and available(ctx, "redraw_hdr_hr", (0,)):
        R["renderer"] = renderer_check(load_final(ctx, "redraw_hdr_hr", 0), te[:n_parse], ctx.store)
        _free(ctx.device)
        ctx.save("audit", R)
    # 4. from the saved test predictions: accuracy by head and view, per class, paired tests, check transitions
    P = {}
    for fam, name in FINAL.items():
        for s in range(5):
            for split in ("test_id", "test_tta"):
                p = saved_preds(ctx, f"{name}_full_s{s}", split)
                if p is not None:
                    P[(fam, s, split)] = p
    if P:
        R["accuracy"] = {f"{f}/{s}/{v}": {h: _acc(p, y) for h, p in pr.items()} for (f, s, v), pr in P.items()}
        R["errors"] = {f"{f}/{s}/{v}": int((pr["logits"].argmax(1) != y).sum()) for (f, s, v), pr in P.items()}
        R["per_class"] = {f"{f}/{s}/{v}": per_class(y, pr["logits"].argmax(1), ctx.num_classes)
                          for (f, s, v), pr in P.items()}
        R["paired_tests"] = paired_tests(P, y)
        R["paired_tests_note"] = ("McNemar and the paired bootstrap treat test images as the sample for two fixed "
                                  "trained networks; across_seeds treats training runs as the sample, conditional "
                                  "on this test set. Images, not writers, are resampled (no writer ids).")
        R["seed_average"] = {}
        for v in ("test_id", "test_tta"):
            for fam in FINAL:
                seeds = [s for s in range(5) if (fam, s, v) in P]
                if seeds:
                    avg = np.mean([P[(fam, s, v)]["logits"] for s in seeds], 0)
                    R["seed_average"][f"{fam}/{v}"] = {"seeds": seeds, "accuracy": _acc(avg, y),
                                                       "errors": int((avg.argmax(1) != y).sum())}
        R["check_transitions"] = {f"{f}/{s}/{v}": check_transitions(pr["fused"].argmax(1), pr["logits"].argmax(1), y)
                                  for (f, s, v), pr in P.items() if f != "control" and "fused" in pr}
        R["pair_errors"] = {f"{f}/{s}/{v}": {f"{a:03d}/{b:03d}": int((((y == a) & (pr["logits"].argmax(1) == b)) |
                                                                     ((y == b) & (pr["logits"].argmax(1) == a))).sum())
                                             for a, b in PAIRS} for (f, s, v), pr in P.items()}
    ctx.save("audit", R)
    return R


def paired_tests(P, y):
    """Control against the method from saved predictions P[(family, seed, split)]: McNemar and the paired
    bootstrap for every seed, for the average of seeds 0-2, and the across-seed comparison."""
    tests = {}
    for v in ("test_id", "test_tta"):
        for s in range(5):
            a, b = P.get(("control", s, v)), P.get(("ours", s, v))
            if a is None or b is None:
                continue
            ok_a, ok_b = a["logits"].argmax(1) == y, b["logits"].argmax(1) == y
            c_only, o_only, p = mcnemar(ok_a, ok_b)
            tests[f"{v}/seed{s}"] = {"control_only_right": c_only, "ours_only_right": o_only, "mcnemar_p": p,
                                     "paired_bootstrap_ours_minus_control": paired_bootstrap(ok_a, ok_b)}
        both = [s for s in range(3) if ("control", s, v) in P and ("ours", s, v) in P]
        if len(both) == 3:
            avg = {f: np.mean([P[(f, s, v)]["logits"] for s in both], 0).argmax(1) == y for f in ("control", "ours")}
            c_only, o_only, p = mcnemar(avg["control"], avg["ours"])
            tests[f"{v}/average_of_seeds_0-2"] = {
                "control_only_right": c_only, "ours_only_right": o_only, "mcnemar_p": p,
                "paired_bootstrap_ours_minus_control": paired_bootstrap(avg["control"], avg["ours"])}
        accs = [(_acc(P[("control", s, v)]["logits"], y), _acc(P[("ours", s, v)]["logits"], y))
                for s in range(5) if ("control", s, v) in P and ("ours", s, v) in P]
        if accs:
            tests[f"{v}/across_seeds"] = paired_runs(*zip(*accs))
    return tests


@torch.no_grad()
def renderer_check(model, idx, store):
    """Sampled (training) against exact (polyline) rendering on real parses, and the stroke statistics
    that decide whether sampling leaves gaps: a curve of length L has samples L / (T - 1) apart, and a
    stroke narrower than that spacing is drawn as a row of dots by the sampled renderer."""
    model.eval()
    rows = {"iou": [], "width": [], "spacing": [], "length": [], "f_sampled": [], "f_exact": []}
    for i in range(0, len(idx), 256):
        x = fetch(store, idx[i:i + 256])
        o = model(store.normalize(x, "gray"), draw=False)
        c, w, e = o["ctrl"].float(), o["width"].float(), o["exist"].float()
        a, _, pts = render(c, w, e, G=model.G, T=model.T)
        b, _, _ = render(c, w, e, G=model.G, T=model.T, exact=True)
        A, B = a > 0.5, b > 0.5
        union = (A | B).flatten(1).sum(1).float()
        iou = (A & B).flatten(1).sum(1) / union.clamp_min(1)
        rows["iou"].append(torch.where(union > 0, iou, torch.full_like(iou, float("nan"))).cpu())
        on = e > 0.5
        length = (pts[:, :, 1:] - pts[:, :, :-1]).norm(dim=-1).sum(-1)
        rows["width"].append(w[on].cpu())
        rows["length"].append(length[on].cpu())
        rows["spacing"].append((length / (model.T - 1))[on].cpu())
        target = F.interpolate(x, size=(model.G, model.G), mode="area").squeeze(1)
        rows["f_sampled"].append(parse_score(a, target)[2].cpu())
        rows["f_exact"].append(parse_score(b, target)[2].cpu())
    v = {k: torch.cat(t).numpy() for k, t in rows.items()}
    q = [5, 25, 50, 75, 95]
    return {"images": int(len(idx)), "canvas": model.G, "samples_per_curve": model.T,
            "iou_sampled_vs_exact_mean": float(np.nanmean(v["iou"])) if np.isfinite(v["iou"]).any() else None,
            "iou_sampled_vs_exact_p5": float(np.nanpercentile(v["iou"], 5)) if np.isfinite(v["iou"]).any() else None,
            "images_with_nothing_drawn": int(np.isnan(v["iou"]).sum()),
            "stroke_width_percentiles": dict(zip(map(str, q), np.percentile(v["width"], q).tolist())),
            "stroke_length_percentiles": dict(zip(map(str, q), np.percentile(v["length"], q).tolist())),
            "share_strokes_spacing_wider_than_width": float((v["spacing"] > v["width"]).mean()),
            "rendered_ink_f_sampled_mean": float(v["f_sampled"].mean()),
            "rendered_ink_f_exact_mean": float(v["f_exact"].mean()),
            "units": "canvas units (the glyph box is 1 x 1)"}


# ------------------------------------------------------------------------------------------- cost

def run_bench(ctx):
    R = ctx.load("bench")
    R["environment"] = environment(REPO_ROOT)
    for lab, name in FINAL.items():
        if lab in R.get("networks", {}):
            continue
        cfg = fast_cfg(dict(CONFIGS[name]), ctx.fast)
        m = make_model(dict(cfg, pretrained=False), ctx.num_classes, device=ctx.device).eval()
        size = ctx.img
        e = {"parameters": bench_mod.parameters(m), "flops_1view": bench_mod.flops(m, size, ctx.device)}
        e["flops_5view"] = 5 * e["flops_1view"]
        for views in (1, 5):
            e[f"latency_batch1_{views}view"] = bench_mod.latency(m, ctx.device, size, 1, views, iters=10 if ctx.fast else 100)
            e[f"throughput_{views}view"] = bench_mod.throughput(m, ctx.device, size, 16 if ctx.fast else 256, views,
                                                                iters=2 if ctx.fast else 20)
            e[f"peak_memory_mb_{views}view"] = bench_mod.peak_memory_mb(m, ctx.device, size, 256, views)
        del m
        _free(ctx.device)
        e["train_step"] = bench_mod.train_step_time(cfg, ctx.device, batch=8 if ctx.fast else 256, size=size,
                                                    steps=2 if ctx.fast else 20, num_classes=ctx.num_classes)
        R.setdefault("networks", {})[lab] = e
        _free(ctx.device)
        ctx.save("bench", R)
    if "preprocess_ms_per_scan" not in R:
        R["preprocess_ms_per_scan"] = bench_mod.preprocess_ms(ctx.index.file.to_numpy()[ctx.idx["test"]],
                                                              n=50 if ctx.fast else 500)
    nw = R["networks"]
    R["ours_over_control"] = {
        "train_step": nw["ours"]["train_step"]["seconds_per_step"] / nw["control"]["train_step"]["seconds_per_step"],
        "flops": nw["ours"]["flops_1view"] / nw["control"]["flops_1view"],
        "parameters": nw["ours"]["parameters"]["total"] / nw["control"]["parameters"]["total"],
        "latency_batch1_1view": nw["ours"]["latency_batch1_1view"]["median_ms"] / nw["control"]["latency_batch1_1view"]["median_ms"]}
    R["note"] = ("All timings in this file come from one session on the device named in 'environment'; FLOPs count "
                 "one multiply-add as 2 and include the stroke decoder, reader and the check (3 candidates).")
    ctx.save("bench", R)
    return R


# ------------------------------------------------------------------------------------------- core matrix

def _evaluate_val(ctx, name, cfg, tag, model):
    va = ctx.idx["val"]
    yv = ctx.labels[va]
    stem = f"{name}_{tag}"
    p1 = cached_preds(ctx.work, stem, "val_id", lambda: predict(model, va, cfg, ctx.store))
    p5 = cached_preds(ctx.work, stem, "val_tta", lambda: predict(model, va, cfg, ctx.store, views=TTA_VIEWS))
    e = {"val_1view": {h: _acc(v, yv) for h, v in p1.items()}, "val_5view": {h: _acc(v, yv) for h, v in p5.items()},
         "parameters": bench_mod.parameters(model)["total"], "flops_1view": bench_mod.flops(model, ctx.img, ctx.device)}
    if cfg.get("strokes"):
        e["rendered_ink_val"] = parse_eval(model, va[:4096], ctx.store)
    ck = torch.load(final_path(ctx.work, name, tag), map_location="cpu", weights_only=False)
    e["train_minutes"] = round(sum(h.get("sec", 0) for h in ck.get("history", [])) / 60, 1)
    e["meta"] = ck.get("meta", {})
    return e


def run_core(ctx, names=tuple(CORE), seeds=SEEDS, cosine_batches=8):
    """E1 and E2 on the development split, scored on validation. Seed-major: an interrupted run leaves the
    complete matrix of the first seeds. ``sequential`` starts from ``control_m`` of the same seed."""
    R = ctx.load("core")
    R["questions"] = CORE
    R["protocol"] = ("trained on the 61,504 development images, scored on the 10,826 validation images; the test set "
                     "is not used. Same data order and augmentation stream for every configuration of a seed "
                     "(matched_aug); drop-path and stroke jitter draws still differ between architectures.")
    for s in seeds:
        for name in names:
            key = f"{name}/{s}"
            if key in R.get("runs", {}):
                continue
            tag = f"rev_s{s}"
            cfg = fast_cfg(dict(CONFIGS[name], seed=s), ctx.fast)
            if name == "sequential":
                start = final_path(ctx.work, "control_m", tag)
                if not start.exists():
                    print(f"{key}: needs control_m/{s} first")
                    continue
                cfg["init_from"] = str(start)
            t0 = time.time()
            model = train(name, cfg, ctx.idx["train"], tag, ctx.store, ctx.labels, ctx.work, num_classes=ctx.num_classes)
            entry = _evaluate_val(ctx, name, cfg, tag, model)
            entry["wall_minutes"] = round((time.time() - t0) / 60, 1)
            if name == "rt_none" and cosine_batches:
                entry["gradient_conflict"] = gradient_conflict(ctx, model, cfg, cosine_batches)
            R.setdefault("runs", {})[key] = entry
            print(f"{key}: validation {100 * entry['val_5view']['logits']:.2f}% (5 views), "
                  f"{100 * entry['val_1view']['logits']:.2f}% (1 view)")
            ctx.save("core", R)
            del model
            _free(ctx.device)
    R["summary"] = summarise_core(R)
    ctx.save("core", R)
    return R


def gradient_conflict(ctx, model, cfg, batches=8, bs=64, seed=0):
    """Cosine between the drawing-loss and classification-loss gradients on the shared backbone (coupled
    routing; negative means the two losses pull the backbone in opposite directions). Measured on the
    trained network in evaluation mode, on training images without augmentation."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    names = {n for n, _ in model.named_parameters() if n.startswith("backbone.")}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(batches):
        idx = rng.choice(ctx.idx["train"], min(bs, len(ctx.idx["train"])), replace=False)
        x = fetch(ctx.store, idx)
        y = torch.as_tensor(ctx.labels[idx], device=ctx.device)
        g = {}
        for loss_name in ("drawing", "classification (fused)"):
            model.zero_grad(set_to_none=True)
            audit_mod.loss_terms(model, ctx.store.normalize(x, "gray"), y, cfg, ink=x)[loss_name].backward()
            g[loss_name] = {n: p.grad.detach().flatten().float().clone() for n, p in model.named_parameters()
                            if n in names and p.grad is not None}
        shared = [n for n in g["drawing"] if n in g["classification (fused)"]]   # the stages both losses reach
        a = torch.cat([g["drawing"][n] for n in shared])
        b = torch.cat([g["classification (fused)"][n] for n in shared])
        out.append({"cosine": float(F.cosine_similarity(a, b, dim=0)), "norm_drawing": float(a.norm()),
                    "norm_classification": float(b.norm()), "shared_parameters": int(a.numel())})
    model.zero_grad(set_to_none=True)
    cos = [o["cosine"] for o in out]
    return {"batches": out, "cosine_mean": float(np.mean(cos)), "cosine_sd": float(np.std(cos)),
            "note": ("final weights of the coupled network, evaluation mode, unaugmented training images; backbone "
                     "parameters that both losses reach (the drawing loss reaches the stages up to the 1/16 tap)")}


def summarise_core(R):
    runs = R.get("runs", {})
    summary = {}
    for name in CORE:
        seeds = sorted(int(k.split("/")[1]) for k in runs if k.split("/")[0] == name)
        if not seeds:
            continue
        e = {"seeds": seeds}
        for view in ("val_5view", "val_1view"):
            for head in ("logits", "img", "vec"):
                vals = [runs[f"{name}/{s}"][view].get(head) for s in seeds]
                if any(v is not None for v in vals):
                    e[f"{view}_{head}"] = {"mean": _mean(vals), "sd": _sd(vals), "runs": vals}
        e["train_minutes_mean"] = _mean([runs[f"{name}/{s}"]["train_minutes"] for s in seeds])
        e["parameters"] = runs[f"{name}/{seeds[0]}"]["parameters"]
        e["flops_1view"] = runs[f"{name}/{seeds[0]}"]["flops_1view"]
        ink = [runs[f"{name}/{s}"]["rendered_ink_val"]["f"] for s in seeds if "rendered_ink_val" in runs[f"{name}/{s}"]]
        if ink:
            e["rendered_ink_f_mean"] = _mean(ink)
        both = [s for s in seeds if f"control_m/{s}" in runs]
        if name != "control_m" and both:
            for view in ("val_5view", "val_1view"):
                e[f"{view}_vs_control"] = paired_runs([runs[f"control_m/{s}"][view]["logits"] for s in both],
                                                      [runs[f"{name}/{s}"][view]["logits"] for s in both])
        summary[name] = e
    return summary


# ------------------------------------------------------------------------------------------- probes

@torch.no_grad()
def _representations(ctx, model, idx, bs=512):
    """Frozen parse geometry (per stroke slot: 8 control coordinates, width, existence) and pooled backbone
    features of ``model`` for images ``idx``."""
    model.eval()
    amp = amp_dtype(ctx.device)
    geo, feat = [], []
    for i in range(0, len(idx), bs):
        x = fetch(ctx.store, idx[i:i + bs])
        xn = ctx.store.normalize(x, "gray")
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            f = model.backbone(xn)
            o = model(xn, draw=False) if getattr(model, "strokes", False) else None
        feat.append(f.float().cpu())
        if o is not None:
            geo.append(torch.cat([o["ctrl"].float().flatten(2), o["width"].float().unsqueeze(-1),
                                  o["exist"].float().unsqueeze(-1)], -1).flatten(1).cpu())
    return (torch.cat(geo) if geo else None), torch.cat(feat)


def train_probe(xtr, ytr, xte, num_classes, hidden=0, epochs=30, lr=1e-3, wd=1e-4, bs=512, device="cpu", seed=0):
    """A linear (hidden=0) or one-hidden-layer probe on standardised inputs, fixed recipe (nothing selected).
    Returns the predicted classes of ``xte``."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-6
    xtr, xte = ((xtr - mu) / sd).to(device), ((xte - mu) / sd).to(device)
    ytr = torch.as_tensor(ytr, device=device)
    d = xtr.shape[1]
    net = (torch.nn.Linear(d, num_classes) if not hidden else
           torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Dropout(0.1),
                               torch.nn.Linear(hidden, num_classes))).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * math.ceil(len(xtr) / bs))
    for _ in range(epochs):
        perm = torch.randperm(len(xtr), generator=g).to(device)
        for i in range(0, len(xtr), bs):
            j = perm[i:i + bs]
            loss = F.cross_entropy(net(xtr[j]), ytr[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    with torch.no_grad():
        return torch.cat([net(xte[i:i + 4096]).argmax(1) for i in range(0, len(xte), 4096)]).cpu().numpy()


def run_probes(ctx, seeds=(0,), epochs=30):
    """E1 (a): what class information the frozen representations of the final networks hold. Probes are
    trained on the 72,330 training images with a fixed recipe and scored on the test set."""
    R = ctx.load("probes")
    tr, te = ctx.idx["full"], ctx.idx["test"]
    y_tr, y_te = ctx.labels[tr], ctx.labels[te]
    for s in seeds:
        if f"seed{s}" in R or not (available(ctx, "redraw_hdr_hr", (s,)) and available(ctx, "baseline_fp32", (s,))):
            continue
        ours, ctrl = load_final(ctx, "redraw_hdr_hr", s), load_final(ctx, "baseline_fp32", s)
        g_tr, f_tr = _representations(ctx, ours, tr)
        g_te, f_te = _representations(ctx, ours, te)
        _, c_tr = _representations(ctx, ctrl, tr)
        _, c_te = _representations(ctx, ctrl, te)
        del ours, ctrl
        _free(ctx.device)
        reps = {"stroke geometry (ours)": (g_tr, g_te), "backbone features (ours)": (f_tr, f_te),
                "geometry + features (ours)": (torch.cat([g_tr, f_tr], 1), torch.cat([g_te, f_te], 1)),
                "backbone features (control)": (c_tr, c_te)}
        res = {}
        for rep, (a, b) in reps.items():
            for kind, hidden in (("linear", 0), ("mlp", 512)):
                pred = train_probe(a, y_tr, b, ctx.num_classes, hidden, 2 if ctx.fast else epochs, device=ctx.device)
                res[f"{rep}, {kind}"] = {"test_accuracy": float((pred == y_te).mean()), "dimensions": int(a.shape[1])}
        R[f"seed{s}"] = res
        R["note"] = ("probes on frozen outputs of the final networks: stroke geometry = for each of the K stroke slots "
                     "its 8 control coordinates (canvas units, global frame), width and existence probability")
        ctx.save("probes", R)
    return R


# ------------------------------------------------------------------------------------------- more seeds

def run_extra_seeds(ctx, seeds=(3, 4), names=("baseline_fp32", "redraw_hdr_hr")):
    """Seeds 3 and 4 of the control and the method on all 72,330 training images (five seeds in all, a
    number fixed before any of the new runs). Test predictions are saved like those of seeds 0-2."""
    R = ctx.load("extra_seeds")
    te, y = ctx.idx["test"], ctx.labels[ctx.idx["test"]]
    for s in seeds:
        for name in names:
            key = f"{name}/{s}"
            if key in R.get("runs", {}):
                continue
            cfg = fast_cfg(dict(CONFIGS[name], seed=s), ctx.fast)
            model = train(name, cfg, ctx.idx["full"], f"full_s{s}", ctx.store, ctx.labels, ctx.work,
                          num_classes=ctx.num_classes)
            stem = f"{name}_full_s{s}"
            p5 = cached_preds(ctx.work, stem, "test_tta", lambda: predict(model, te, cfg, ctx.store, views=TTA_VIEWS))
            p1 = cached_preds(ctx.work, stem, "test_id", lambda: predict(model, te, cfg, ctx.store))
            e = {"test_5view": {h: _acc(v, y) for h, v in p5.items()}, "test_1view": {h: _acc(v, y) for h, v in p1.items()}}
            if cfg.get("strokes"):
                e["rendered_ink_test"] = parse_eval(model, te, ctx.store)
            if cfg.get("hdr"):
                e["check_test"] = verify_eval(model, te, ctx.store, ctx.labels)
            R.setdefault("runs", {})[key] = e
            ctx.save("extra_seeds", R)
            del model
            _free(ctx.device)
    for view in ("test_tta", "test_id"):
        acc = {}
        for name in names:
            for s in range(5):
                p = saved_preds(ctx, f"{name}_full_s{s}", view)
                if p is not None:
                    acc[(name, s)] = _acc(p["logits"], y)
        both = [s for s in range(5) if (names[0], s) in acc and (names[1], s) in acc]
        if both:
            a, b = [acc[(names[0], s)] for s in both], [acc[(names[1], s)] for s in both]
            R[f"all_seeds_{view}"] = {"seeds": both, names[0]: a, names[1]: b,
                                      f"{names[0]}_mean": _mean(a), f"{names[0]}_sd": _sd(a),
                                      f"{names[1]}_mean": _mean(b), f"{names[1]}_sd": _sd(b),
                                      "paired": paired_runs(a, b)}
    ctx.save("extra_seeds", R)
    return R


# ------------------------------------------------------------------------------------------- reconstruction

def _skeleton_job(args):
    ink, tol = args
    return recon.skeleton_curves(ink, tol=tol)


def summarise_metrics(metrics):
    """Per-image metrics -> mean, sd, median and missing count of each, plus failure counts."""
    out = {}
    for k, v in metrics.items():
        v = np.asarray(v, float)
        if np.isfinite(v).any():
            out[k] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)), "median": float(np.nanmedian(v)),
                      "missing": int((~np.isfinite(v)).sum())}
    if "f@1" in metrics and "curves" in metrics:
        f1, curves = np.asarray(metrics["f@1"], float), np.asarray(metrics["curves"], float)
        out["failures"] = {"no_curve": int((curves == 0).sum()), "f@1_below_0.5": int((np.nan_to_num(f1) < 0.5).sum()),
                           "images": int(len(f1))}
    return out


def run_recon(ctx, seeds=SEEDS, R_eval=64, tols=(0.5, 1.0, 2.0, 3.0), Ks=(4, 8, 12, 16), steps=300, n=None,
              workers=None):
    """E4 on the test set, one evaluation canvas for every method (see ``recon``): our parses with all their
    strokes and with only the k most certain ones, skeleton tracing at four tolerances, and per-image
    optimisation of K strokes with our renderer and drawing loss. Per-image values are kept."""
    R = ctx.load("recon")
    te = ctx.idx["test"] if n is None else ctx.idx["test"][:n]
    per_image = ctx.out / "recon_per_image"
    per_image.mkdir(parents=True, exist_ok=True)
    timing = R.setdefault("seconds_per_image", {})

    def keep(key, metrics):
        np.savez_compressed(per_image / f"{_npz_key(key)}.npz", index=np.asarray(te),
                            **{_npz_key(k): np.asarray(v, np.float32) for k, v in metrics.items()})
        R.setdefault("methods", {})[key] = summarise_metrics(metrics)
        ctx.save("recon", R)

    for s in seeds:
        keys = [f"ours/{s}"] + [f"ours/{s}/top{k}" for k in (4, 6, 8, 10)]
        if all(k in R.get("methods", {}) for k in keys) or not available(ctx, "redraw_hdr_hr", (s,)):
            continue
        m = load_final(ctx, "redraw_hdr_hr", s)
        acc, seconds = {k: {} for k in keys}, 0.0
        with torch.no_grad():
            for i in range(0, len(te), 256):
                x = fetch(ctx.store, te[i:i + 256])
                t0 = time.perf_counter()
                o = m(ctx.store.normalize(x, "gray"), draw=False)
                if ctx.device == "cuda":
                    torch.cuda.synchronize()
                seconds += time.perf_counter() - t0
                c, w, e = o["ctrl"].float(), o["width"].float(), o["exist"].float()
                order = e.argsort(1, descending=True)
                variants = {f"ours/{s}": e}
                for k in (4, 6, 8, 10):
                    variants[f"ours/{s}/top{k}"] = e * torch.zeros_like(e).scatter(1, order[:, :k], 1.0)
                for key, ee in variants.items():
                    for name, v in recon.evaluate(c, w, ee, x, R=R_eval).items():
                        acc[key].setdefault(name, []).append(np.asarray(v))
        timing[f"ours/{s}"] = {"seconds": seconds / len(te), "device": ctx.device, "note": "whole network, batch 256"}
        for key in keys:
            keep(key, {k: np.concatenate(v) for k, v in acc[key].items()})
        del m
        _free(ctx.device)

    todo = [t for t in tols if f"skeleton/tol{t:g}" not in R.get("methods", {})]
    if todo:
        ink = np.concatenate([recon.ink_canvas(fetch(ctx.store, te[i:i + 1024]), R_eval).cpu().numpy()
                              for i in range(0, len(te), 1024)])
        for tol in todo:
            t0 = time.perf_counter()
            for i in range(min(200, len(te))):
                recon.skeleton_curves(ink[i], tol=tol)
            timing[f"skeleton/tol{tol:g}"] = {"seconds": (time.perf_counter() - t0) / min(200, len(te)),
                                              "device": "cpu, one process"}
            with get_context("fork").Pool(workers or os.cpu_count()) as pool:
                items = pool.map(_skeleton_job, [(ink[i], tol) for i in range(len(te))], chunksize=64)
            acc = {}
            for i in range(0, len(te), 256):
                c, w, e = recon.pad_strokes(items[i:i + 256])
                x = fetch(ctx.store, te[i:i + 256]).cpu()
                for name, v in recon.evaluate(c, w, e, x, R=R_eval).items():
                    acc.setdefault(name, []).append(np.asarray(v))
            keep(f"skeleton/tol{tol:g}", {k: np.concatenate(v) for k, v in acc.items()})
    for K in Ks:
        key = f"optimised/K{K}"
        if key in R.get("methods", {}):
            continue
        acc, seconds = {}, 0.0
        bs = 512 if ctx.device == "cuda" else 32
        n_steps = 2 if ctx.fast else steps
        for i in range(0, len(te), bs):
            x = fetch(ctx.store, te[i:i + bs])
            target = F.interpolate(x, size=(48, 48), mode="area")[:, 0]
            t0 = time.perf_counter()
            c, w, e, _ = recon.optimise_strokes(target, K=K, steps=n_steps)
            if ctx.device == "cuda":
                torch.cuda.synchronize()
            seconds += time.perf_counter() - t0
            for name, v in recon.evaluate(c, w, e, x, R=R_eval).items():
                acc.setdefault(name, []).append(np.asarray(v))
        timing[key] = {"seconds": seconds / len(te), "device": ctx.device, "note": f"{n_steps} Adam steps, batch {bs}"}
        keep(key, {k: np.concatenate(v) for k, v in acc.items()})
    R["evaluation"] = {"canvas_px": R_eval, "images": int(len(te)), "foreground": "value > 0.5 on both sides",
                       "ink": "128 px ink area-averaged to the canvas",
                       "drawing": "exact polyline distance, 64 samples per curve, soft edge of half a pixel",
                       "tolerances_px": list(recon.TOLS), "aggregation": "metrics per image, then mean / sd / median"}
    ctx.save("recon", R)
    return R


# ------------------------------------------------------------------------------------------- stability

TRANSFORMS = {"shift +2 px": dict(dx=2 / 128), "shift -2 px": dict(dx=-2 / 128), "rotate +5 deg": dict(angle=5.0),
              "rotate -5 deg": dict(angle=-5.0), "zoom 0.95": dict(scale=0.95), "zoom 1.05": dict(scale=1.05)}


@torch.no_grad()
def _parse(model, ctx, x):
    o = model(ctx.store.normalize(x, "gray"), draw=False)
    return o["ctrl"].float(), o["exist"].float()


def compare_parses(pa, oa, pb, ob):
    """Two sets of parses (N, K, T, 2) with existence masks (N, K), in canvas units -> summary in 128 px pixels."""
    ch = np.array([recon.union_chamfer(pa[i], oa[i], pb[i], ob[i]) for i in range(len(pa))]) * 128
    mc = [recon.match_curves(pa[i], oa[i], pb[i], ob[i]) for i in range(len(pa))]
    return {"union_chamfer_mean": float(np.nanmean(ch)), "union_chamfer_median": float(np.nanmedian(ch)),
            "matched_curve_mean": float(np.nanmean([a for a, _ in mc]) * 128),
            "stroke_count_difference_mean": float(np.mean([b for _, b in mc])),
            "same_stroke_count": float(np.mean([b == 0 for _, b in mc]))}


def run_stability(ctx, seeds=SEEDS, n=2000):
    """E5: parses of moved images, moved back, against the parse of the original (union of centrelines and
    one-to-one curve matching, both indifferent to stroke order and direction); then parses across seeds."""
    R = ctx.load("stability")
    te = ctx.idx["test"][:n]
    parses = {}
    for s in available(ctx, "redraw_hdr_hr", seeds):
        m = load_final(ctx, "redraw_hdr_hr", s)
        base_pts, base_on, moved = [], [], {t: ([], []) for t in TRANSFORMS}
        for i in range(0, len(te), 256):
            x = fetch(ctx.store, te[i:i + 256])
            c, e = _parse(m, ctx, x)
            base_pts.append(curve_points(c, 32).cpu().numpy())
            base_on.append((e > 0.5).cpu().numpy())
            for t, kw in TRANSFORMS.items():
                xm, theta = recon.affine_view(x, **kw)
                cm, em = _parse(m, ctx, xm)
                moved[t][0].append(recon.map_points(curve_points(cm, 32), theta).cpu().numpy())
                moved[t][1].append((em > 0.5).cpu().numpy())
        bp, bo = np.concatenate(base_pts), np.concatenate(base_on)
        parses[s] = (bp, bo)
        if str(s) not in R.get("transforms", {}):
            R.setdefault("transforms", {})[str(s)] = {
                t: compare_parses(bp, bo, np.concatenate(mp), np.concatenate(mo)) for t, (mp, mo) in moved.items()}
            ctx.save("stability", R)
        del m
        _free(ctx.device)
    ks = sorted(parses)
    if len(ks) > 1:
        R["across_seeds"] = {f"{a}-{b}": compare_parses(*parses[a], *parses[b]) for i, a in enumerate(ks) for b in ks[i + 1:]}
    R["images"] = int(len(te))
    R["units"] = "pixels of the 128 px input"
    ctx.save("stability", R)
    return R


# ------------------------------------------------------------------------------------------- interventions

def run_interventions(ctx, seeds=SEEDS, render_epochs=20):
    """E6: (1) erase the ink of one stroke, chosen six ways, and let whole networks read the edited image;
    (2) leave one stroke out of what the reader sees, image features fixed; (3) recognition from the
    rendered parses alone against the same classifier on the ink (information kept, not faithfulness)."""
    R = ctx.load("interventions")
    te, y = ctx.idx["test"], ctx.labels[ctx.idx["test"]]
    for s in seeds:
        if not (available(ctx, "redraw_hdr_hr", (s,)) and available(ctx, "baseline_fp32", (s,))):
            continue
        if f"erase/{s}" not in R:
            nets = {"ours": load_final(ctx, "redraw_hdr_hr", s), "control": load_final(ctx, "baseline_fp32", s)}
            e = erase_test(nets["ours"], nets, te, ctx.store, seed=s)
            valid = e["strokes"] >= 2
            changed = {c: {n: e["erased"][c][n] != e["pred"][n] for n in nets} for c in CHOOSERS}
            R[f"erase/{s}"] = {
                "images": int(valid.sum()),
                "changed": {c: {n: float(v[n][valid].mean()) for n in nets} for c, v in changed.items()},
                "accuracy_after": {c: {n: float((e["erased"][c][n] == y)[valid].mean()) for n in nets} for c in CHOOSERS},
                "ink_erased": {k: float(v[valid].mean()) for k, v in e["ink_erased"].items()},
                "intact_accuracy": {n: float((e["pred"][n] == y)[valid].mean()) for n in nets},
                "reader_most_vs_other_choice": {
                    c: {n: dict(zip(("only_other_changed", "only_most_changed", "mcnemar_p"),
                                    mcnemar(changed[c][n][valid], changed["reader, most"][n][valid]))) for n in nets}
                    for c in CHOOSERS if c != "reader, most"}}
            np.savez_compressed(ctx.out / f"erase_s{s}.npz", index=te, label=y, strokes=e["strokes"],
                                **{_npz_key(f"stroke {k}"): v for k, v in e["stroke"].items()},
                                **{_npz_key(f"ink {k}"): v for k, v in e["ink_erased"].items()},
                                **{_npz_key(f"erased {c} {n}"): e["erased"][c][n] for c in CHOOSERS for n in nets},
                                **{_npz_key(f"pred {n}"): e["pred"][n] for n in nets})
            ctx.save("interventions", R)
            del nets
            _free(ctx.device)
        if f"representation/{s}" not in R:
            m = load_final(ctx, "redraw_hdr_hr", s)
            r = representation_test(m, te, ctx.store, seed=s)
            valid = r["strokes"] >= 2
            R[f"representation/{s}"] = {
                "images": int(valid.sum()),
                "fused_changed": {k: float((r[f"pred_{k}"] != r["pred"])[valid].mean()) for k in ("most", "least", "random")},
                "reader_changed": {k: float((r[f"vec_pred_{k}"] != r["vec_pred"])[valid].mean())
                                   for k in ("most", "least", "random")},
                "fused_logprob_drop_mean": {k: float(r[f"drop_{k}"][valid].mean()) for k in ("most", "least", "random")}}
            np.savez_compressed(ctx.out / f"representation_s{s}.npz", index=te, **r)
            ctx.save("interventions", R)
            del m
            _free(ctx.device)
    if "from_drawings" not in R and available(ctx, "redraw_hdr_hr", (0,)):
        m = load_final(ctx, "redraw_hdr_hr", 0)
        drawn = render_parses(m, np.arange(len(ctx.labels)), ctx.store, R=ctx.img)
        del m
        _free(ctx.device)
        z = np.zeros_like(drawn)
        meta = np.zeros((len(drawn), 5), np.float32)
        out = {}
        for label, gray in (("drawings", drawn), ("ink", ctx.store.gray.cpu().numpy())):
            st = Store((gray, z, z, meta), ctx.device, ctx.idx["train"])
            cfg = fast_cfg(dict(CONFIGS["baseline"], epochs=render_epochs, matched_aug=True, seed=0), ctx.fast)
            net = train(f"from_{label}", cfg, ctx.idx["full"], "rev_s0", st, ctx.labels, ctx.work, num_classes=ctx.num_classes)
            out[label] = _acc(predict(net, te, cfg, st)["logits"], y)
            del net, st
            _free(ctx.device)
        R["from_drawings"] = {"test_accuracy_1view": out, "epochs": render_epochs,
                              "note": ("the same image classifier and recipe trained and tested on the rendered parses "
                                       "of the seed-0 network, or on the ink; measures class information kept by the "
                                       "drawing, not how the original network decides")}
        ctx.save("interventions", R)
    return R


# ------------------------------------------------------------------------------------------- checker

def run_checker(ctx, seeds=SEEDS, refit_images=1000, refit_steps=100):
    """E3: what the check changes, energy margins, selective prediction and the amortisation gap."""
    R = ctx.load("checker")
    te, y = ctx.idx["test"], ctx.labels[ctx.idx["test"]]
    ambiguous = np.isin(y, [c for pair in PAIRS for c in pair])
    for s in seeds:
        if f"ours/{s}" in R.get("per_seed", {}) or not available(ctx, "redraw_hdr_hr", (s,)):
            continue
        m = load_final(ctx, "redraw_hdr_hr", s)
        chk = check_outputs(m, te, ctx.store)
        fused = predict(m, te, fast_cfg(CONFIGS["redraw_hdr_hr"], ctx.fast), ctx.store, heads=("fused",))["fused"]
        del m
        _free(ctx.device)
        np.savez_compressed(ctx.out / f"check_s{s}.npz", index=te, label=y, fused_pred=fused.argmax(1), **chk)
        mg = candidate_margins(chk["cand"], chk["energy"], y)
        pred, ok = chk["pred"], chk["pred"] == y
        first_wrong_listed = (fused.argmax(1) != y) & mg["in_candidates"]
        probs = np.sort(chk["probs"], 1)
        in_cand = chk["cand"] == pred[:, None]
        e_pred = np.where(in_cand.any(1), (chk["energy"] * in_cand).sum(1), np.inf)
        e_rival = np.where(in_cand, np.inf, chk["energy"]).min(1)
        scores = {"softmax probability": probs[:, -1], "softmax margin": probs[:, -1] - probs[:, -2],
                  "energy margin": np.where(np.isfinite(e_pred), e_rival - e_pred, -np.inf)}
        scores["rank average (probability, energy margin)"] = rank_average(scores["softmax probability"],
                                                                         scores["energy margin"])

        def quant(v):
            v = v[np.isfinite(v)]
            return np.percentile(v, [5, 25, 50, 75, 95]).tolist() if len(v) else None

        R.setdefault("per_seed", {})[f"ours/{s}"] = {
            "accuracy_without_check_1view": float((fused.argmax(1) == y).mean()),
            "accuracy_with_check_1view": float(ok.mean()),
            "transitions": check_transitions(fused.argmax(1), pred, y),
            "candidate_recall": float(mg["in_candidates"].mean()),
            "energy_alone_right_among_candidates": float(mg["energy_pick_right"][mg["in_candidates"]].mean()),
            "first_choice_wrong_true_class_listed": int(first_wrong_listed.sum()),
            "of_those_right_after_check": int((first_wrong_listed & ok).sum()),
            "margin_quantiles": {"right": quant(mg["margin"][ok]), "wrong": quant(mg["margin"][~ok]),
                                 "confusable pairs": quant(mg["margin"][ambiguous])},
            "selective": {k: {"aurc": risk_coverage(v, ok)[2], **{f"risk@{c}": risk_at(v, ok, c) for c in (0.9, 0.95, 0.98)}}
                          for k, v in scores.items()}}
        R["note"] = ("energy-alone accuracy is among the three shortlisted candidates, not a 55-class classifier; "
                     "no calibrator is fitted")
        ctx.save("checker", R)
    dev = final_path(ctx.work, "redraw_hdr_hr", "dev")
    if "refit" not in R and dev.exists():
        m = load_model(dev, fast_cfg(dict(CONFIGS["redraw_hdr_hr"]), ctx.fast), num_classes=ctx.num_classes,
                       device=ctx.device)
        va = ctx.idx["val"][:refit_images]
        yv = ctx.labels[va]
        rows = {"cand": [], "energy_fitter": [], "energy_refit": []}
        for i in range(0, len(va), 100):
            r = refit(m, fetch(ctx.store, va[i:i + 100]), ctx.store, steps=2 if ctx.fast else refit_steps)
            for k in rows:
                rows[k].append(r[k])
        cand, e0, e1 = (np.concatenate(rows[k]) for k in ("cand", "energy_fitter", "energy_refit"))
        hit = cand == yv[:, None]
        listed = hit.any(1)
        r_ = np.arange(len(cand))
        pick0, pick1 = cand[r_, e0.argmin(1)] == yv, cand[r_, e1.argmin(1)] == yv
        np.savez_compressed(ctx.out / "refit_val.npz", index=va, label=yv, cand=cand, energy_fitter=e0, energy_refit=e1)
        R["refit"] = {"images": int(len(va)), "steps": refit_steps, "optimiser": "Adam, lr 0.05, same energy and bounds",
                      "network": "round-3 development network, validation images",
                      "energy_alone_right_fitter": float(pick0[listed].mean()) if listed.any() else None,
                      "energy_alone_right_refit": float(pick1[listed].mean()) if listed.any() else None,
                      "energy_drop_true_class_mean": float((e0 - e1)[hit].mean()) if hit.any() else None,
                      "energy_drop_rivals_mean": float((e0 - e1)[~hit].mean()),
                      "ranking_changed": float((e0.argmin(1) != e1.argmin(1))[listed].mean()) if listed.any() else None}
        ctx.save("checker", R)
        del m
        _free(ctx.device)
    return R


# ------------------------------------------------------------------------------------------- robustness

def run_coupled_seeds(ctx, seeds=(1, 2)):
    R = ctx.load("coupled_seeds")
    te, y = ctx.idx["test"], ctx.labels[ctx.idx["test"]]
    for s in seeds:
        key = f"redraw_hdr/{s}"
        if key in R:
            continue
        cfg = fast_cfg(dict(CONFIGS["redraw_hdr"], seed=s), ctx.fast)
        model = train("redraw_hdr", cfg, ctx.idx["full"], f"full_s{s}", ctx.store, ctx.labels, ctx.work,
                      num_classes=ctx.num_classes)
        stem = f"redraw_hdr_full_s{s}"
        p5 = cached_preds(ctx.work, stem, "test_tta", lambda: predict(model, te, cfg, ctx.store, views=TTA_VIEWS))
        p1 = cached_preds(ctx.work, stem, "test_id", lambda: predict(model, te, cfg, ctx.store))
        R[key] = {"test_5view": {h: _acc(v, y) for h, v in p5.items()}, "test_1view": {h: _acc(v, y) for h, v in p1.items()}}
        ctx.save("coupled_seeds", R)
        del model
        _free(ctx.device)
    return R


def conditions(raw=True, framed=True, texture=True):
    out = []
    if raw:
        out += [("raw", k, lv) for k, levels in CORRUPTIONS.items() for lv in levels]
    if framed:
        out += [("framed", k, lv) for k, levels in FRAMED.items() for lv in levels]
    if texture:
        out += [("texture_shape", k, lv) for k, levels in TEXTURE_SHAPE.items() for lv in levels]
    return out


def damaged_preds(ctx, model, idx, cond, kind, level, seed=0):
    """One-view classes read by ``model`` from damaged copies of images ``idx`` (the same draws for every network)."""
    amp = amp_dtype(ctx.device)
    fn = corrupt_framed if cond == "framed" else texture_shape
    out = []
    for i in range(0, len(idx), 512):
        x = fetch(ctx.store, idx[i:i + 512])
        xd = fn(x, kind, level, seed=seed * 100003 + i)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            o = model(ctx.store.normalize(xd, "gray"), draw=False, ink=xd)
        out.append(o["logits"].float().argmax(1).cpu().numpy())
    return np.concatenate(out)


def run_robustness(ctx, raw=True, framed=True, texture=True, seed=0, seeds=SEEDS):
    """E9 on the test set, one view: corruption of the raw scan before preprocessing (the whole pipeline;
    crop failures counted) and the same kinds of damage after preprocessing with the framing fixed, plus
    texture changes that keep the shape and shape changes that keep the texture. Per-image classes kept."""
    R = ctx.load("robustness")
    te, y = ctx.idx["test"], ctx.labels[ctx.idx["test"]]
    conds = conditions(raw, framed, texture)
    nets = {f"{fam}/{s}": (name, s) for fam, name in FINAL.items() for s in available(ctx, name, seeds)}
    clean = R.setdefault("clean_1view", {})
    per_image = ctx.out / "robustness_per_image"
    per_image.mkdir(parents=True, exist_ok=True)
    for lab, (name, s) in nets.items():
        missing = [c for c in conds if lab not in R.get(f"{c[0]}/{c[1]}_{c[2]}", {})]
        if not missing and lab in clean:
            continue
        cfg = fast_cfg(dict(CONFIGS[name], seed=s), ctx.fast)
        m = load_final(ctx, name, s)
        f = per_image / f"{_npz_key(lab)}.npz"
        P = dict(np.load(f)) if f.exists() else {}
        if lab not in clean:
            P["clean"] = predict(m, te, cfg, ctx.store, heads=("logits",))["logits"].argmax(1)
            clean[lab] = float((P["clean"] == y).mean())
        for cond, kind, level in missing:
            key = f"{cond}/{kind}_{level}"
            if cond == "raw":
                files = ctx.index.file.to_numpy()[te].tolist()
                arrays = build_cache(files, ctx.img, ctx.cache / "corrupt_test",
                                     fn=partial(corrupt_file, kind=kind, level=level), tag=f"{ctx.img}_{kind}_{level}")
                R.setdefault("crop_failures", {})[key] = crop_failures(arrays[3])
                st = Store(arrays, ctx.device, stats=ctx.store.stats, meta_norm=ctx.store.meta_norm)
                p = predict(m, np.arange(len(te)), cfg, st, heads=("logits",))["logits"].argmax(1)
                del st, arrays
            else:
                p = damaged_preds(ctx, m, te, cond, kind, level, seed)
            P[_npz_key(key)] = p
            R.setdefault(key, {})[lab] = float((p == y).mean())
            np.savez_compressed(f, index=te, label=y, **{k: v for k, v in P.items() if k not in ("index", "label")})
            ctx.save("robustness", R)
        del m
        _free(ctx.device)
    R["summary"] = summarise_robustness(R, conds)
    R["protocol"] = {"raw": "corrupt the scan, then preprocess (crop failure: empty Otsu mask)",
                     "framed": "corrupt the preprocessed 128 px ink with the framing fixed; lengths given in pixels of "
                               f"a {SCAN_PX} px scan are scaled to the canvas",
                     "texture_shape": "outline and stripes keep the shape; elastic keeps the texture",
                     "views": 1, "draws": "the same corruption draws for every network"}
    ctx.save("robustness", R)
    return R


def summarise_robustness(R, conds):
    out = {}
    for cond in ("raw", "framed", "texture_shape"):
        keys = [f"{c}/{k}_{lv}" for c, k, lv in conds if c == cond]
        if not keys:
            continue
        labs = sorted({lab for key in keys for lab in R.get(key, {})})
        means = {lab: _mean([R[key][lab] for key in keys]) for lab in labs if all(lab in R.get(k, {}) for k in keys)}
        fam = {}
        for f in FINAL:
            v = [means[lab] for lab in means if lab.startswith(f + "/")]
            if v:
                fam[f] = {"mean_over_conditions": _mean(v), "sd_over_seeds": _sd(v), "networks": len(v)}
        paired = {}
        for key in keys:
            row = R.get(key, {})
            paired[key] = {other: [row[f"{other}/{s}"] - row[f"control/{s}"] for s in range(5)
                                   if f"{other}/{s}" in row and f"control/{s}" in row] for other in ("ours", "coupled")}
        out[cond] = {"per_network_mean": means, "families": fam, "minus_control_by_seed": paired,
                     "note": "unweighted mean over the listed kinds and levels"}
    return out


def run_dial(ctx, levels=("redraw_hdr_hr", "redraw_hdr_hr_g25", "redraw_hdr_hr_g50", "redraw_hdr_hr_g100"),
             seeds=(0,)):
    """Optional: a validation-only sweep of the share of the drawing gradient that reaches the backbone (0,
    0.25, 0.5, 1) on the development split; clean validation, the stronger fixed-framing damage and the
    texture / shape changes. Share 0, seed 0 is the round-3 network ``runs/redraw_hdr_hr/dev`` if it exists."""
    R = ctx.load("dial")
    va, yv = ctx.idx["val"], ctx.labels[ctx.idx["val"]]
    conds = [("framed", k, lv[-1]) for k, lv in FRAMED.items()] + [("texture_shape", k, lv[0])
                                                                   for k, lv in TEXTURE_SHAPE.items()]
    for s in seeds:
        tag = "dev" if s == 0 else f"dev_s{s}"
        for name in levels:
            share = CONFIGS[name].get("draw_grad", 0.0)
            key = f"{share:g}/{s}"
            if key in R.get("runs", {}):
                continue
            cfg = fast_cfg(dict(CONFIGS[name], seed=s), ctx.fast)
            model = train(name, cfg, ctx.idx["train"], tag, ctx.store, ctx.labels, ctx.work, num_classes=ctx.num_classes)
            e = {"val_5view": _acc(predict(model, va, cfg, ctx.store, views=TTA_VIEWS)["logits"], yv),
                 "val_1view": _acc(predict(model, va, cfg, ctx.store)["logits"], yv),
                 "rendered_ink_val": parse_eval(model, va[:4096], ctx.store)}
            for cond, kind, lv in conds:
                e[f"{cond}/{kind}_{lv}"] = float((damaged_preds(ctx, model, va, cond, kind, lv) == yv).mean())
            R.setdefault("runs", {})[key] = e
            ctx.save("dial", R)
            del model
            _free(ctx.device)
    R["protocol"] = "development split, validation images only; one view for the damaged images"
    ctx.save("dial", R)
    return R


# ------------------------------------------------------------------------------------------- second dataset

def setup_second(ctx, data, cache, zip_path=None):
    """The Devanagari Handwritten Character Dataset (UCI 389: 46 classes, 78,200 training and 13,800 test
    images), with the same preprocessing and the same validation rule. Chosen before any result on it."""
    work = ctx.work / "dhcd"
    zp = _find_zip(work, zip_path, "dhcd.zip", DHCD_URL)
    splits, summary = make_split_folders(zp, data)
    index, labels, idx = load_index(splits)
    store = Store(build_cache(index.file.tolist(), ctx.img, cache), ctx.device, idx["train"])
    names = json.loads((splits / "classes.json").read_text())
    sub = Ctx(work, ctx.out / "dhcd", Path(cache), ctx.device, ctx.fast, ctx.img, index, labels, idx, store, splits,
              summary["classes"], names, dataset="DHCD")
    sub.out.mkdir(parents=True, exist_ok=True)
    sub.meta = _manifest(sub, zp, summary)
    print("DHCD", summary)
    return sub


def run_second_dataset(ctx2, seeds=SEEDS, coupled_seeds=SEEDS):
    """E7: the control, the method and coupled routing on the second dataset, recipe unchanged (only the
    number of classes differs); the test set is predicted once; stronger fixed-framing damage at one view."""
    R = ctx2.load("second_dataset")
    te, y = ctx2.idx["test"], ctx2.labels[ctx2.idx["test"]]
    R["dataset"], R["summary"] = ctx2.dataset, ctx2.meta.get("split_counts")
    plan = [("baseline_fp32", s) for s in seeds] + [("redraw_hdr_hr", s) for s in seeds] + \
           [("redraw_hdr", s) for s in coupled_seeds]
    for name, s in plan:
        key = f"{name}/{s}"
        if key in R.get("runs", {}):
            continue
        cfg = fast_cfg(dict(CONFIGS[name], seed=s), ctx2.fast)
        model = train(name, cfg, ctx2.idx["full"], f"full_s{s}", ctx2.store, ctx2.labels, ctx2.work,
                      num_classes=ctx2.num_classes)
        stem = f"{name}_full_s{s}"
        p5 = cached_preds(ctx2.work, stem, "test_tta", lambda: predict(model, te, cfg, ctx2.store, views=TTA_VIEWS))
        p1 = cached_preds(ctx2.work, stem, "test_id", lambda: predict(model, te, cfg, ctx2.store))
        e = {"test_5view": {h: _acc(v, y) for h, v in p5.items()}, "test_1view": {h: _acc(v, y) for h, v in p1.items()}}
        if cfg.get("strokes"):
            e["rendered_ink_test"] = parse_eval(model, te, ctx2.store)
        if cfg.get("hdr"):
            e["check_test"] = verify_eval(model, te, ctx2.store, ctx2.labels)
        for kind, levels in FRAMED.items():
            e[f"framed/{kind}_{levels[-1]}"] = float((damaged_preds(ctx2, model, te, "framed", kind, levels[-1]) == y).mean())
        R.setdefault("runs", {})[key] = e
        ctx2.save("second_dataset", R)
        del model
        _free(ctx2.device)
    runs = R.get("runs", {})
    for view in ("test_5view", "test_1view"):
        fam = {}
        for name in ("baseline_fp32", "redraw_hdr_hr", "redraw_hdr"):
            v = [runs[f"{name}/{s}"][view]["logits"] for s in range(5) if f"{name}/{s}" in runs]
            if v:
                fam[name] = {"runs": v, "mean": _mean(v), "sd": _sd(v)}
        R[f"families_{view}"] = fam
        for other in ("redraw_hdr_hr", "redraw_hdr"):
            both = [s for s in range(5) if f"baseline_fp32/{s}" in runs and f"{other}/{s}" in runs]
            if both:
                R[f"paired_{other}_vs_control_{view}"] = paired_runs(
                    [runs[f"baseline_fp32/{s}"][view]["logits"] for s in both], [runs[f"{other}/{s}"][view]["logits"] for s in both])
    tests = {}
    for s in seeds:
        a, b = saved_preds(ctx2, f"baseline_fp32_full_s{s}", "test_tta"), saved_preds(ctx2, f"redraw_hdr_hr_full_s{s}", "test_tta")
        if a is not None and b is not None:
            ok_a, ok_b = a["logits"].argmax(1) == y, b["logits"].argmax(1) == y
            c_only, o_only, p = mcnemar(ok_a, ok_b)
            tests[f"seed{s}"] = {"control_only_right": c_only, "ours_only_right": o_only, "mcnemar_p": p,
                                 "paired_bootstrap_ours_minus_control": paired_bootstrap(ok_a, ok_b)}
    R["paired_tests_5view"] = tests
    ctx2.save("second_dataset", R)
    return R


# ------------------------------------------------------------------------------------------- figures

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.bbox": "tight", "savefig.dpi": 300})
    return plt


COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
           "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94"]


def _save(plt, fig, path):
    for suffix in (".pdf", ".png"):
        fig.savefig(str(path) + suffix)
    plt.close(fig)


def _draw_strokes(ax, pts, width, exist, scale=60, thresh=0.5):
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    for k in np.argsort(-exist):
        if exist[k] > thresh:
            ax.plot(pts[k, :, 0], pts[k, :, 1], color=COLOURS[k % len(COLOURS)], lw=max(0.6, width[k] * scale),
                    solid_capstyle="round")


@torch.no_grad()
def hypotheses(model, x, store):
    """For images x: the network's output, the three candidates, the bent templates, the soft-minimum energy
    of each candidate (what re-ranks the classes) and the energy of each template alone (B, k, M)."""
    model.eval()
    xn = store.normalize(x, "gray")
    o = model(xn, draw=False, ink=x)
    _, inter = model.backbone.forward_intermediates(xn, indices=model.taps)
    mem, mpos = model.decoder(inter)[3:]
    cand = o["cand"]
    tmpl, fitted = model.hdr.fit(mem, mpos, cand)
    target = F.interpolate(x[:, :1].float(), size=(model.hdr_G, model.hdr_G), mode="area").squeeze(1)
    bent = model.hdr.bend(tmpl, fitted)
    energy = model.hdr.cost(bent, tmpl[2], fitted, target, G=model.hdr_G)
    per = torch.stack([model.hdr.cost(tuple(t[:, :, m:m + 1] for t in bent), tmpl[2][:, :, m:m + 1],
                                      {k: v[:, :, m:m + 1] for k, v in fitted.items()}, target, G=model.hdr_G)
                       for m in range(model.hdr.M)], -1)
    return o, cand, bent, energy, per


def figure_hypotheses(ctx, model, idx, path):
    """One image per row: the ink (true class and the class read), the parse, and for each of the three
    candidates its best-fitting template bent onto the ink, with the candidate's energy E (the soft minimum
    over its two templates, the number the check uses; lower explains the ink better; * = lowest)."""
    plt = _style()
    idx = np.asarray(idx)
    x = fetch(ctx.store, idx)
    o, cand, bent, energy, per = hypotheses(model, x, ctx.store)
    final = o["logits"].float().argmax(1).cpu().numpy()
    fused = o["fused"].float().argmax(1).cpu().numpy()
    ppts = curve_points(o["ctrl"].float(), 64).cpu().numpy()
    pw, pe = o["width"].float().cpu().numpy(), o["exist"].float().cpu().numpy()
    B, k, M, S = bent[0].shape[:4]
    bpts = curve_points(bent[0].float().reshape(B * k * M, S, 4, 2), 64).reshape(B, k, M, S, 64, 2).cpu().numpy()
    bw, be = bent[1].float().cpu().numpy(), torch.sigmoid(bent[2]).float().cpu().numpy()
    cand, energy, per = cand.cpu().numpy(), energy.cpu().numpy(), per.cpu().numpy()
    y = ctx.labels[idx]

    def name(c):
        return ctx.names[c] if c < len(ctx.names) else f"{c:03d}"

    fig, axes = plt.subplots(len(idx), 2 + k, figsize=(1.3 * (2 + k), 1.45 * len(idx)), squeeze=False)
    for r in range(len(idx)):
        ax = axes[r, 0]
        ax.imshow(1 - x[r, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"true {name(y[r])}\nread {name(final[r])}", color="black" if final[r] == y[r] else "#c00000")
        _draw_strokes(axes[r, 1], ppts[r], pw[r], pe[r])
        axes[r, 1].set_title(f"parse\n{int((pe[r] > 0.5).sum())} strokes")
        best = int(np.argmin(energy[r]))
        for j in range(k):
            t = int(np.argmin(per[r, j]))
            _draw_strokes(axes[r, 2 + j], bpts[r, j, t], bw[r, j, t], be[r, j, t], scale=40)
            axes[r, 2 + j].set_title(f"as {name(cand[r, j])}{' *' if j == best else ''}\nE = {energy[r, j]:.3f}",
                                     color=("#007a00" if cand[r, j] == y[r] else "#c00000") if j == best else "black")
    plt.tight_layout()
    _save(plt, fig, path)
    return {"index": idx.tolist(), "true": y.tolist(), "read": final.tolist(), "read_without_check": fused.tolist(),
            "candidates": cand.tolist(), "energy": energy.round(4).tolist(), "template_energy": per.round(4).tolist()}


def figure_reconstructions(ctx, model, groups, path):
    """Pairs of (ink, exact rendering of the parse), four pairs per row, for named groups of images."""
    plt = _style()
    rows = sum((len(v) + 3) // 4 for v in groups.values())
    fig, axes = plt.subplots(rows, 8, figsize=(8 * 0.9, 1.1 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    r = 0
    for label, idx in groups.items():
        if not len(idx):
            continue
        x = fetch(ctx.store, idx)
        with torch.no_grad():
            o = model(ctx.store.normalize(x, "gray"), draw=False, ink=x)
        img, _, _ = render(o["ctrl"].float(), o["width"].float(), o["exist"].float(), G=x.shape[-1], T=64, exact=True)
        pred = o["logits"].float().argmax(1).cpu().numpy()
        for i in range(len(idx)):
            rr, cc = r + i // 4, 2 * (i % 4)
            axes[rr, cc].imshow(1 - x[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[rr, cc].set_title(f"{ctx.labels[idx[i]]:03d} read {pred[i]:03d}", fontsize=8,
                                   color="black" if pred[i] == ctx.labels[idx[i]] else "#c00000")
            axes[rr, cc + 1].imshow(1 - img[i].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[rr, cc + 1].set_title("redrawn", fontsize=8)
        axes[r, 0].text(-0.12, 0.5, label, transform=axes[r, 0].transAxes, rotation=90, ha="right", va="center")
        r += (len(idx) + 3) // 4
    plt.tight_layout()
    _save(plt, fig, path)


def figure_neighbours(ctx, path, n_err=12, n_nb=3, seed=0):
    """Validation errors of the round-3 control (development network) and their nearest training images in
    its own feature space (cosine similarity), as in the diagnostics notebook, with readable labels."""
    f = final_path(ctx.work, "baseline_fp32", "dev")
    p = saved_preds(ctx, "baseline_fp32_dev", "val_tta")
    if not f.exists() or p is None:
        return None
    plt = _style()
    model = load_model(f, fast_cfg(dict(CONFIGS["baseline_fp32"]), ctx.fast), num_classes=ctx.num_classes,
                       device=ctx.device)
    tr, va = ctx.idx["train"], ctx.idx["val"]
    _, f_tr = _representations(ctx, model, tr)
    _, f_va = _representations(ctx, model, va)
    del model
    f_tr, f_va = F.normalize(f_tr, dim=1).to(ctx.device), F.normalize(f_va, dim=1).to(ctx.device)
    yv, pred = ctx.labels[va], p["logits"].argmax(1)
    err = np.flatnonzero(pred != yv)
    rng = np.random.default_rng(seed)
    show = np.concatenate([rng.permutation(err[np.isin(yv[err], [44, 25])])[:n_err // 2],
                           rng.permutation(err[~np.isin(yv[err], [44, 25])])[:n_err - n_err // 2]]).astype(int)
    if not len(show):
        return None
    sims, nbrs = (f_va[show] @ f_tr.T).topk(min(n_nb, len(tr)), dim=1)
    sims, nbrs = sims.cpu().numpy(), nbrs.cpu().numpy()
    cols, n_nb = 2, nbrs.shape[1]
    rows = (len(show) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * (1 + n_nb), figsize=(0.9 * cols * (1 + n_nb), 1.05 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, v in enumerate(show):
        r, c0 = i // cols, (i % cols) * (1 + n_nb)
        axes[r, c0].imshow(1 - fetch(ctx.store, va[v:v + 1])[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[r, c0].set_title(f"{yv[v]:03d} read {pred[v]:03d}")
        for j in range(n_nb):
            t = tr[nbrs[i, j]]
            axes[r, c0 + 1 + j].imshow(1 - fetch(ctx.store, [t])[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[r, c0 + 1 + j].set_title(f"{ctx.labels[t]:03d}  {sims[i, j]:.2f}",
                                          color="#c00000" if ctx.labels[t] != yv[v] else "black")
    plt.tight_layout()
    _save(plt, fig, path)
    return {"validation_index": [int(va[v]) for v in show], "true": yv[show].tolist(), "read": pred[show].tolist(),
            "neighbour_labels": ctx.labels[tr[nbrs]].tolist(), "similarity": sims.round(3).tolist()}


def figure_templates(ctx, model, path, cols=11):
    """Both learned templates of every class (strokes drawn with their width; faint = rarely written)."""
    plt = _style()
    ctrl, width, prior = (t.detach().float().cpu() for t in model.hdr.template_params())
    p = torch.sigmoid(prior).numpy()
    C, M, S = ctrl.shape[:3]
    pts = curve_points(ctrl.reshape(C * M, S, 4, 2), 64).reshape(C, M, S, 64, 2).numpy()
    rows = int(np.ceil(C / cols))
    fig, axes = plt.subplots(rows * M, cols, figsize=(cols * 0.72, rows * M * 0.78), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for c in range(C):
        r, k = divmod(c, cols)
        for m in range(M):
            ax = axes[r * M + m, k]
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_aspect("equal")
            for s in range(S):
                ax.plot(pts[c, m, s, :, 0], pts[c, m, s, :, 1], color=COLOURS[s % len(COLOURS)],
                        lw=max(0.5, float(width[c, m, s]) * 25), alpha=float(0.15 + 0.85 * p[c, m, s]),
                        solid_capstyle="round")
            if m == 0:
                ax.set_title(f"{c:03d}", fontsize=8)
    plt.tight_layout()
    _save(plt, fig, path)


def run_figures(ctx, n=6, seed=0):
    """Figures with fixed, seeded selection rules (saved with the figures): random test images; random errors
    of the seed-0 network; one image of each class of the confusable pairs; reconstructions (random, good
    redraw but wrong class, worst redraws); validation errors with their neighbours; all templates."""
    R = ctx.load("figures")
    out = ctx.out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    if not available(ctx, "redraw_hdr_hr", (0,)):
        return R
    te, y = ctx.idx["test"], ctx.labels[ctx.idx["test"]]
    rng = np.random.default_rng(seed)
    m = load_final(ctx, "redraw_hdr_hr", 0)
    p = saved_preds(ctx, "redraw_hdr_hr_full_s0", "test_id")
    if p is None:
        p = predict(m, te, fast_cfg(CONFIGS["redraw_hdr_hr"], ctx.fast), ctx.store, heads=("logits",))
    pred = p["logits"].argmax(1)
    wrong = np.flatnonzero(pred != y)
    sel = {"random": te[np.sort(rng.choice(len(te), min(n, len(te)), replace=False))],
           "errors": te[np.sort(rng.choice(wrong, min(n, len(wrong)), replace=False))] if len(wrong) else te[:0],
           "confusable pairs": np.asarray([te[rng.choice(np.flatnonzero(y == c))] for pair in PAIRS for c in pair
                                           if (y == c).any()][:2 * len(PAIRS)], int)}
    for label, idx in sel.items():
        if len(idx):
            R[f"hypotheses {label}"] = figure_hypotheses(ctx, m, idx, out / f"hypotheses_{_npz_key(label)}")
    rec = ctx.out / "recon_per_image" / "ours_0.npz"
    if rec.exists():
        with np.load(rec) as z:
            f1, index = z["f_1"], z["index"]
        pos = {int(v): i for i, v in enumerate(te)}
        ok = pred[[pos[int(v)] for v in index]] == ctx.labels[index]
        good_wrong = np.flatnonzero((np.nan_to_num(f1, nan=-1) >= np.nanmedian(f1)) & ~ok)
        worst = np.argsort(np.nan_to_num(f1, nan=-1))[:max(1, len(f1) // 50)]
        groups = {"random": index[np.sort(rng.choice(len(index), min(8, len(index)), replace=False))],
                  "good redraw,\nwrong class": index[np.sort(rng.choice(good_wrong, min(4, len(good_wrong)),
                                                                        replace=False))] if len(good_wrong) else index[:0],
                  "worst 2%\nredraws": index[np.sort(rng.choice(worst, min(8, len(worst)), replace=False))]}
        figure_reconstructions(ctx, m, groups, out / "reconstructions")
        R["reconstructions"] = {k.replace("\n", " "): [int(i) for i in v] for k, v in groups.items()}
    figure_templates(ctx, m, out / "templates")
    del m
    _free(ctx.device)
    nb = figure_neighbours(ctx, out / "errors_neighbours")
    if nb:
        R["neighbours"] = nb
    R["selection"] = ("numpy default_rng(0): 'random' = test images drawn uniformly; 'errors' = drawn uniformly from "
                      "the test images the seed-0 network gets wrong (one view); 'confusable pairs' = one random image "
                      "of each class of the four pairs; reconstructions: 'good redraw, wrong class' = rendered-ink F "
                      "at 1 px at or above the median and a wrong class; 'worst 2%' = lowest F at 1 px")
    ctx.save("figures", R)
    return R


# ------------------------------------------------------------------------------------------- export

def run_table(ctx):
    """One row per trained network on Drive: name, tag, seed, training time and device (where recorded),
    checkpoint hash. A small ``run.json`` is written next to each checkpoint so this is fast next time."""
    import pandas as pd
    rows = []
    for f in sorted((ctx.work / "runs").glob("*/*/final.pt")):
        side = f.with_name("run.json")
        info = read_json(side)
        stat = f.stat()
        if info.get("bytes") != stat.st_size or info.get("mtime") != int(stat.st_mtime):
            ck = torch.load(f, map_location="cpu", weights_only=False)
            hist = ck.get("history", [])
            info = {"name": f.parent.parent.name, "tag": f.parent.name, "seed": (ck.get("cfg") or {}).get("seed", 0),
                    "epochs": len(hist), "train_minutes": round(sum(h.get("sec", 0) for h in hist) / 60, 1),
                    "meta": ck.get("meta"), "cfg": ck.get("cfg"), "sha256": file_sha256(f),
                    "bytes": stat.st_size, "mtime": int(stat.st_mtime)}
            del ck
            write_json(side, info)
        rows.append({k: info.get(k) for k in ("name", "tag", "seed", "epochs", "train_minutes", "sha256")} |
                    {"device": (info.get("meta") or {}).get("device", "not recorded")})
    df = pd.DataFrame(rows)
    df.to_csv(ctx.out / "runs.csv", index=False)
    return df


def per_image_table(ctx):
    """Test images, one row each: file, class, and the classes read by every final network (one and five
    views, every head), with the candidates and energies of the check and reconstruction metrics if present."""
    import pandas as pd
    te, y = ctx.idx["test"], ctx.labels[ctx.idx["test"]]
    cols = {"index": te, "file": ctx.index.path.to_numpy()[te], "label": y}
    for fam, name in FINAL.items():
        for s in range(5):
            for view, short in (("test_id", "1v"), ("test_tta", "5v")):
                p = saved_preds(ctx, f"{name}_full_s{s}", view)
                if p is not None:
                    for h, v in p.items():
                        cols[f"{fam}{s}_{short}_{h}"] = v.argmax(1)
            chk = ctx.out / f"check_s{s}.npz"
            if fam == "ours" and chk.exists():
                with np.load(chk) as z:
                    for j in range(z["cand"].shape[1]):
                        cols[f"ours{s}_cand{j + 1}"] = z["cand"][:, j]
                        cols[f"ours{s}_energy{j + 1}"] = z["energy"][:, j].round(5)
    rec = ctx.out / "recon_per_image" / "ours_0.npz"
    if rec.exists():
        with np.load(rec) as z:
            if len(z["index"]) == len(te):
                for k in ("f_1", "iou", "centreline", "curves"):
                    if k in z.files:
                        cols[f"ours0_recon_{k}"] = z[k]
    df = pd.DataFrame(cols)
    (ctx.out / "per_image").mkdir(exist_ok=True)
    df.to_csv(ctx.out / "per_image" / "test_predictions.csv", index=False)
    return df


def release_predictions(ctx):
    """Final-output probabilities of every final network on the test set (one and five views), float16, with
    the test-set row index, for the public results archive (the paper's seed averages can be recomputed)."""
    out = ctx.out / "per_image" / "probabilities"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name in FINAL.values():
        for s in range(5):
            for view in ("test_id", "test_tta"):
                p = saved_preds(ctx, f"{name}_full_s{s}", view)
                if p is not None:
                    f = out / f"{name}_full_s{s}_{view}.npz"
                    np.savez_compressed(f, index=ctx.idx["test"], label=ctx.labels[ctx.idx["test"]],
                                        probabilities=p["logits"].astype(np.float16))
                    written.append(f.name)
    return written


EXPECTED = ("audit", "bench", "core", "probes", "extra_seeds", "recon", "stability", "interventions", "checker",
            "coupled_seeds", "robustness", "dial", "figures", "dhcd/second_dataset")


def export(ctx):
    """Run table, per-image table, every results file, the manifest and what is missing, in one ZIP under
    ``revision/exports`` (checkpoints stay on Drive; their hashes are in runs.csv)."""
    run_table(ctx)
    per_image_table(ctx)
    release_predictions(ctx)
    status = {name: ("present" if ctx.path(name).exists() else "missing") for name in EXPECTED}
    write_json(ctx.out / "status.json", {"files": status, "environment": environment(REPO_ROOT),
                                         "missing": [k for k, v in status.items() if v == "missing"]})
    (ctx.out / "README.txt").write_text(
        "Results of the revision experiments (redraw.experiments). JSON: one file per experiment; the docstrings\n"
        "of redraw/experiments.py say what each holds. *.npz: per-image arrays; 'index' is the row in\n"
        "data/splits (train, val, test in that order). runs.csv: every trained network with its checkpoint hash.\n"
        "per_image/test_predictions.csv: classes read by every final network; per_image/probabilities: their\n"
        "final-output probabilities (float16) with the test-set row index. manifest.json: data and split\n"
        "hashes and the environment of every session. status.json: experiments not yet run.\n")
    stamp = time.strftime("%Y%m%d_%H%M")
    zp = ctx.out / "exports" / f"revision_results_{stamp}.zip"
    files = export_zip(zp, ctx.out, patterns=("*.json", "*.csv", "*.md", "*.png", "*.pdf", "*.txt", "*.npz"),
                       exclude=("*.pt", "*.npy", "exports/*"))
    print(f"{len(files)} files -> {zp} ({zp.stat().st_size / 1e6:.1f} MB)")
    print("not yet run:", [k for k, v in status.items() if v == "missing"] or "nothing")
    return zp
