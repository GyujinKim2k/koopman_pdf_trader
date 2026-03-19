from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Cfg:
    symbol: str = "BTCUSDT"
    raw_root: str = "data/raw/binance_monthly/BTCUSDT"   # contains 2025-01/, ..., 2025-12/
    out_root: str = "data/processed/BTCUSDT_2025_1d_kde_pdf_fast"

    months: List[str] = None

    nx: int = 512
    x_min: float = -0.08
    x_max: float = +0.08
    bandwidth: float = 0.002
    empty_window: str = "carry"

    klines_name: str = "klines_30m.parquet"
    parts_glob: str = "agg_trades_part*.parquet"

    def __post_init__(self):
        if self.months is None:
            self.months = [f"2025-{m:02d}" for m in range(1, 13)]


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_parts(month_dir: Path, glob_pat: str) -> List[Path]:
    parts = sorted(month_dir.glob(glob_pat))
    if not parts:
        raise FileNotFoundError(f"No trade parts found in {month_dir} with glob={glob_pat}")
    return parts


def to_utc_naive_ns(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s, utc=True).dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")


def gaussian_kernel_1d(dx: float, bandwidth: float, truncate: float = 4.0) -> np.ndarray:
    sigma_bins = bandwidth / dx
    rad = int(np.ceil(truncate * sigma_bins))
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()
    return k.astype(np.float64)


def fft_convolve_same_rows(H: np.ndarray, k: np.ndarray) -> np.ndarray:
    T, Nx = H.shape
    Lk = len(k)
    nfft = 1 << int(np.ceil(np.log2(Nx + Lk - 1)))

    Hf = np.fft.rfft(H, n=nfft, axis=1)
    kpad = np.zeros(nfft, dtype=np.float64)
    kpad[:Lk] = k
    kf = np.fft.rfft(kpad)

    Y = np.fft.irfft(Hf * kf[None, :], n=nfft, axis=1)
    full = Y[:, : (Nx + Lk - 1)]
    start = (Lk - 1) // 2
    return full[:, start : start + Nx]


def build_one_month(cfg: Cfg, month: str):
    month_dir = Path(cfg.raw_root) / month
    out_dir = ensure_dir(Path(cfg.out_root) / month)

    kl = pd.read_parquet(month_dir / cfg.klines_name)
    open_time = to_utc_naive_ns(kl["open_time"])
    close = kl["close"].to_numpy(dtype=np.float64)
    T = len(open_time)

    edges = np.empty(T + 1, dtype="datetime64[ns]")
    edges[:T] = open_time
    edges[T] = open_time[-1] + np.timedelta64(30, "m")

    grid = np.linspace(cfg.x_min, cfg.x_max, cfg.nx, dtype=np.float64)
    dx = float(grid[1] - grid[0])

    H = np.zeros((T, cfg.nx), dtype=np.float64)
    parts = list_parts(month_dir, cfg.parts_glob)

    for pi, p in enumerate(parts):
        df = pd.read_parquet(p, columns=["timestamp", "price", "qty"])
        ts = to_utc_naive_ns(df["timestamp"])
        price = df["price"].to_numpy(dtype=np.float64)
        qty = df["qty"].to_numpy(dtype=np.float64)

        idx = np.searchsorted(edges, ts, side="right") - 1
        m = (idx >= 0) & (idx < T)
        if np.any(m):
            idx = idx[m]
            price = price[m]
            qty = qty[m]

            x = np.log(price / close[idx])
            x = np.clip(x, cfg.x_min, cfg.x_max)

            b = np.floor((x - cfg.x_min) / dx).astype(np.int64)
            b = np.clip(b, 0, cfg.nx - 1)

            order = np.argsort(idx)
            idxs = idx[order]
            bs = b[order]
            qs = qty[order]

            cut = np.flatnonzero(np.diff(idxs)) + 1
            starts = np.r_[0, cut]
            ends = np.r_[cut, len(idxs)]

            for s, e in zip(starts, ends):
                wi = int(idxs[s])
                H[wi] += np.bincount(bs[s:e], weights=qs[s:e], minlength=cfg.nx)

        if (pi + 1) % 10 == 0 or (pi + 1) == len(parts):
            print(f"[{month}] parts {pi+1}/{len(parts)}")

    k = gaussian_kernel_1d(dx=dx, bandwidth=cfg.bandwidth, truncate=4.0)
    Hs = fft_convolve_same_rows(H, k)

    pdfs = np.zeros((T, cfg.nx), dtype=np.float32)
    prev_pdf: Optional[np.ndarray] = None

    for i in range(T):
        f = Hs[i]
        s = float(f.sum())
        if s <= 0:
            if cfg.empty_window == "carry" and prev_pdf is not None:
                pdf = prev_pdf
            else:
                pdf = np.ones(cfg.nx, dtype=np.float32)
                pdf /= float(pdf.sum()) * dx
        else:
            pdf = (f / (s * dx)).astype(np.float32)
        pdfs[i] = pdf
        prev_pdf = pdf

    masses = pdfs.sum(axis=1) * dx
    print(f"[{month}] mass mean/min/max = {masses.mean():.6f}, {masses.min():.6f}, {masses.max():.6f}")

    np.save(out_dir / f"pdfs_{month}.npy", pdfs)
    np.save(out_dir / f"closes_{month}.npy", close.astype(np.float32))
    np.save(out_dir / f"times_{month}.npy", open_time.astype("datetime64[ns]"))
    np.save(out_dir / f"x_grid_{month}.npy", grid.astype(np.float32))

    meta = {
        "symbol": cfg.symbol,
        "month": month,
        "T": int(T),
        "nx": int(cfg.nx),
        "x_definition": "x = log(price / close_30m_window)",
        "x_min": float(cfg.x_min),
        "x_max": float(cfg.x_max),
        "dx": float(dx),
        "kde_implementation": "weighted histogram + Gaussian convolution (FFT)",
        "bandwidth": float(cfg.bandwidth),
        "empty_window": cfg.empty_window,
    }
    with open(out_dir / f"meta_{month}.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    cfg = Cfg()
    for m in cfg.months:
        print(f"\n===== Build {m} =====")
        build_one_month(cfg, m)

    print("\nAll months done.")


if __name__ == "__main__":
    main()