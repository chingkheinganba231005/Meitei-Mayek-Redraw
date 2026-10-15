"""Unpack TUMMHCD and write the train / validation / test split as CSV files.

    python -m redraw.split --zip TUMMHCD-TEST-TRAIN.zip --data-dir data
    python -m redraw.split --data-dir data          # download first

The archive only has train and test folders. Validation is carved out of
train: 15% of every class, shuffled with seed 42. This gives 61,504 / 10,826 /
12,794 images, the split every number in the paper uses. The CSV row order is
part of the split: saved predictions are stored in that order.
"""

import argparse
import json
import os
import random
import shutil
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://agnigarh.tezu.ernet.in/~sarat/TUMMHCD-TEST-TRAIN.zip"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        r = urllib.request.urlopen(req, timeout=300)
    except urllib.error.URLError as e:
        if not isinstance(e.reason, ssl.SSLError):
            raise
        # the university server's certificate has been broken before
        print(f"certificate check failed for {url} ({e.reason}); downloading without it")
        r = urllib.request.urlopen(req, context=ssl._create_unverified_context(), timeout=300)
    with r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    if not zipfile.is_zipfile(dest):
        dest.unlink()
        raise SystemExit(f"{url} did not return a zip file. Download it in a browser and pass --zip.")


def extract(zip_path, out_dir):
    out_dir = Path(out_dir)
    marker = out_dir / ".extracted"
    if marker.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    marker.write_text("ok")


def class_index(folder_name):
    return int(folder_name.split("_")[-1])  # "train_044" -> 44


def find_split_dirs(raw_dir):
    dirs = [p for p in Path(raw_dir).rglob("*") if p.is_dir()]
    train = [p for p in dirs if p.name.lower() == "tummhcdtrain"]
    test = [p for p in dirs if p.name.lower() == "tummhcdtest"]
    if not train:
        train = [p for p in dirs if any(c.is_dir() and c.name.startswith("train_") for c in p.iterdir())]
    if not test:
        test = [p for p in dirs if any(c.is_dir() and c.name.startswith("test_") for c in p.iterdir())]
    if not train or not test:
        raise SystemExit(f"could not find the TUMMHCDtrain / TUMMHCDtest folders under {raw_dir}")
    return train[0], test[0]


def collect(split_dir, prefix, rel_to):
    folders = sorted((p for p in Path(split_dir).iterdir() if p.is_dir() and p.name.startswith(prefix + "_")),
                     key=lambda p: class_index(p.name))
    rows = []
    for folder in folders:
        label = class_index(folder.name)
        for img in sorted(x for x in folder.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_EXTS):
            rows.append({"path": os.path.relpath(img, rel_to), "label": label,
                         "class_name": f"{label:03d}", "source_folder": folder.name})
    return rows


def make_split(zip_path, data_dir, val_ratio=0.15, seed=42):
    data_dir = Path(data_dir)
    extract(zip_path, data_dir / "raw")
    train_dir, test_dir = find_split_dirs(data_dir / "raw")

    split_dir = data_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_rows = collect(train_dir, "train", split_dir)
    test_rows = collect(test_dir, "test", split_dir)

    rng = random.Random(seed)
    train, val = [], []
    for label in sorted({r["label"] for r in train_rows}):
        rows = [r for r in train_rows if r["label"] == label]
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * val_ratio))
        val += rows[:n_val]
        train += rows[n_val:]

    for name, rows in (("train", train), ("val", val), ("test", test_rows)):
        pd.DataFrame(rows).to_csv(split_dir / f"{name}.csv", index=False)
    summary = {"classes": len({r["label"] for r in train_rows}), "train": len(train),
               "val": len(val), "test": len(test_rows)}
    (split_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return split_dir, summary


def _natural(name):
    """Sort key that puts character_2 before character_10."""
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _class_folders(d):
    return [c for c in Path(d).iterdir() if c.is_dir() and any(
        x.is_file() and x.suffix.lower() in IMAGE_EXTS for x in c.iterdir())]


def make_split_folders(zip_path, data_dir, val_ratio=0.15, seed=42):
    """The same split files for any archive with a train and a test folder (any case) of class folders.

    Labels follow the natural order of the training class folders; their names are kept in
    ``classes.json``. Validation is carved out of train as for TUMMHCD.
    """
    data_dir = Path(data_dir)
    extract(zip_path, data_dir / "raw")
    dirs = [p for p in (data_dir / "raw").rglob("*") if p.is_dir()]
    train_dirs = [p for p in dirs if p.name.lower() == "train" and _class_folders(p)]
    test_dirs = [p for p in dirs if p.name.lower() == "test" and _class_folders(p)]
    if not train_dirs or not test_dirs:
        raise SystemExit(f"could not find train and test folders of class folders under {data_dir / 'raw'}")
    train_dir, test_dir = train_dirs[0], test_dirs[0]
    names = sorted((c.name for c in _class_folders(train_dir)), key=_natural)
    label_of = {n: i for i, n in enumerate(names)}
    split_dir = data_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    def rows_of(root):
        rows = []
        for folder in sorted(_class_folders(root), key=lambda c: _natural(c.name)):
            if folder.name not in label_of:
                raise SystemExit(f"test class {folder.name} has no training folder")
            for img in sorted(x for x in folder.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS):
                rows.append({"path": os.path.relpath(img, split_dir), "label": label_of[folder.name],
                             "class_name": folder.name, "source_folder": folder.name})
        return rows

    train_rows, test_rows = rows_of(train_dir), rows_of(test_dir)
    rng = random.Random(seed)
    train, val = [], []
    for label in range(len(names)):
        rows = [r for r in train_rows if r["label"] == label]
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * val_ratio))
        val += rows[:n_val]
        train += rows[n_val:]
    for name, rows in (("train", train), ("val", val), ("test", test_rows)):
        pd.DataFrame(rows).to_csv(split_dir / f"{name}.csv", index=False)
    (split_dir / "classes.json").write_text(json.dumps(names, indent=1))
    summary = {"classes": len(names), "train": len(train), "val": len(val), "test": len(test_rows)}
    (split_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return split_dir, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--zip", help="an already downloaded TUMMHCD-TEST-TRAIN.zip")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    zip_path = Path(args.zip) if args.zip else Path(args.data_dir) / "TUMMHCD-TEST-TRAIN.zip"
    if not zip_path.exists():
        download(URL, zip_path)
    _, summary = make_split(zip_path, args.data_dir, args.val_ratio, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
