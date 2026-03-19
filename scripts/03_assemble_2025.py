from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class Cfg:
    out_root: str = "data/processed/BTCUSDT_2025_1d_kde_pdf_fast"
    months: List[str] = None

    def __post_init__(self):
        if self.months is None:
            self.months = [f"2025-{m:02d}" for m in range(1, 13)]


def main():
    cfg = Cfg()
    root = Path(cfg.out_root)

    pdfs_all = []
    closes_all = []
    times_all = []

    for m in cfg.months:
        d = root / m
        pdfs_all.append(np.load(d / f"pdfs_{m}.npy"))
        closes_all.append(np.load(d / f"closes_{m}.npy"))
        times_all.append(np.load(d / f"times_{m}.npy"))

    pdfs = np.concatenate(pdfs_all, axis=0)    # [T, Nx]
    closes = np.concatenate(closes_all, axis=0)
    times = np.concatenate(times_all, axis=0)

    np.save(root / "pdfs_2025.npy", pdfs)
    np.save(root / "closes_2025.npy", closes)
    np.save(root / "times_2025.npy", times)

    meta = {
        "months": cfg.months,
        "T_total": int(pdfs.shape[0]),
        "Nx": int(pdfs.shape[1]),
        "files": {
            "pdfs_2025": str(root / "pdfs_2025.npy"),
            "closes_2025": str(root / "closes_2025.npy"),
            "times_2025": str(root / "times_2025.npy"),
        }
    }
    with open(root / "meta_2025.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved:")
    print(root / "pdfs_2025.npy", pdfs.shape)
    print(root / "closes_2025.npy", closes.shape)
    print(root / "times_2025.npy", times.shape)


if __name__ == "__main__":
    main()