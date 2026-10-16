"""Records that make results traceable: environment, hashes, atomic writes and the results archive."""

import hashlib
import json
import math
import os
import platform
import subprocess
import time
import zipfile
from pathlib import Path


def file_sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def split_hash(split_dir):
    """One hash of the train / val / test CSV files (their order is part of the split)."""
    h = hashlib.sha256()
    for name in ("train.csv", "val.csv", "test.csv"):
        p = Path(split_dir) / name
        if p.exists():
            h.update(name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def git_commit(repo="."):
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def environment(repo="."):
    import numpy
    import torch
    try:
        import timm
        timm_v = timm.__version__
    except Exception:
        timm_v = None
    cuda = torch.cuda.is_available()
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(),
            "torch": torch.__version__, "timm": timm_v, "numpy": numpy.__version__,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if cuda else "cpu",
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if cuda else None,
            "commit": git_commit(repo)}


def clean(obj):
    """JSON-safe copy: numpy scalars and arrays to Python values, NaN and infinity to None, keys to str."""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path, obj):
    """Write JSON atomically: a crash leaves either the old file or the new one, never half of one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=1, allow_nan=False))
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)


def export_zip(out_zip, root, patterns=("*.json", "*.csv", "*.md", "*.png", "*.pdf", "*.svg", "*.txt"),
               exclude=("*.pt", "*.npy")):
    """Pack the small result files under ``root`` (not checkpoints or caches) into one archive."""
    root, out_zip = Path(root), Path(out_zip)
    files = sorted({p for pat in patterns for p in root.rglob(pat)
                    if p.is_file() and not any(p.match(e) for e in exclude) and p.resolve() != out_zip.resolve()})
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_zip.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(root))
    os.replace(tmp, out_zip)
    return [str(p.relative_to(root)) for p in files]
