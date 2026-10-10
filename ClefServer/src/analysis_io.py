from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "runs" / "matrix"
BACKBONES = ("gemma", "qwen", "granite")


def backbone() -> str:
    bb = os.environ.get("CASE_BACKBONE", "")
    if bb not in BACKBONES:
        sys.exit("Set the backbone first:  setenv CASE_BACKBONE gemma   (or qwen)")
    return bb


def runs_dir() -> Path:
    bb = backbone()
    view = ROOT / "runs" / "view" / bb
    view.mkdir(parents=True, exist_ok=True)
    prefix = f"{bb}_"
    for src in MATRIX.iterdir():
        if not src.name.startswith(prefix) or ".ckpt." in src.name:
            continue
        name = src.name[len(prefix):].replace("_clef_", "_jev_")
        link = view / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(src, view))
    return view


def results_dir() -> Path:
    return ROOT / "results" / backbone()
