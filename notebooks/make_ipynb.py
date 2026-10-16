"""Build the .ipynb notebooks from their .py sources (cells separated by "# %%" lines).

    python notebooks/make_ipynb.py notebooks/rev1_audit.py ...     # or no arguments: every rev*.py
"""

import json
import sys
from pathlib import Path


def cells(text):
    out, kind, lines = [], None, []

    def flush():
        while lines and not lines[-1].strip():
            lines.pop()
        if kind is not None and lines:
            src = [line[2:] if line.startswith("# ") else line.lstrip("#") for line in lines] if kind == "markdown" else lines
            out.append((kind, src))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        elif kind is not None:
            lines.append(line)
    flush()
    return out


def notebook(path):
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, src) in enumerate(cells(Path(path).read_text())):
        cell = {"cell_type": kind, "id": f"{Path(path).stem}-{i:02d}", "metadata": {},
                "source": [line + "\n" for line in src[:-1]] + src[-1:]}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    return nb


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(str(p) for p in Path(__file__).parent.glob("rev*.py"))
    for p in paths:
        out = Path(p).with_suffix(".ipynb")
        out.write_text(json.dumps(notebook(p), indent=1, ensure_ascii=False) + "\n")
        print(out)
