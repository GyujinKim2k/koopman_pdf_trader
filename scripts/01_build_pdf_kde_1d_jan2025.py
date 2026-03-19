from __future__ import annotations

import json
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Cfg:
    symbol: str = "BTCUSDT"
    month: str = "2025-01"

    month_dir: str = "data/raw/binance_monthly/BTCUSDT/2025-01"
    klines_name: str = "klines_30m.parquet"
    parts_glob: str = "agg_trades_part*.parquet"

    out_root: str = "data/processed/BTCUSDT_2025_1d_kde_pdf_fast"

    # x = log(price / close_window)
    nx: int = 512
    x_min: float = -0.08
    x_max: float = +0.08

    # KDE bandwidth (in x units)
    bandwidth: float = 0.002

    # empty window policy
    empty_window: str = "carry"  # carry or uniform


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
    """Convert a datetime series to tz-naive UTC numpy datetime64[ns]."""
    return pd.to_datetime(s, utc=True).dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")


def gaussian_kernel_1d(dx: float, bandwidth: float, truncate: float = 4.0) -> np.ndarray:
    """
    Discrete Gaussian kernel on grid spacing dx with std=bandwidth.
    Kernel sums to 1 (mass preserving).
    """
    sigma_bins = bandwidth / dx
    rad = int(np.ceil(truncate * sigma_bins))
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()
    return k.astype(np.float64)


def fft_convolve_same_rows(H: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Convolve each row of H with kernel k using FFT, returning 'same' length.
    Uses zero-padding; we post-normalize per row anyway, so boundary effects are ok.
    H: [T, Nx]
    """
    T, Nx = H.shape
    Lk = len(k)
    nfft = 1 << int(np.ceil(np.log2(Nx + Lk - 1)))

    Hf = np.fft.rfft(H, n=nfft, axis=1)
    kpad = np.zeros(nfft, dtype=np.float64)
    kpad[:Lk] = k
    kf = np.fft.rfft(kpad)

    Y = np.fft.irfft(Hf * kf[None, :], n=nfft, axis=1)

    # 'full' length is Nx+Lk-1, sitting at start of Y
    full = Y[:, : (Nx + Lk - 1)]

    # take 'same'
    start = (Lk - 1) // 2
    out = full[:, start : start + Nx]
    return out


def main():
    cfg = Cfg()

    month_dir = Path(cfg.month_dir)
    out_dir = ensure_dir(Path(cfg.out_root) / cfg.month)

    # --- Load klines to define 30m windows ---
    kl = pd.read_parquet(month_dir / cfg.klines_name)
    open_time = to_utc_naive_ns(kl["open_time"])
    close = kl["close"].to_numpy(dtype=np.float64)

    T = len(open_time)
    if T == 0:
        raise RuntimeError("No klines found.")

    # Define window edges for searchsorted
    # Window i covers [open_time[i], open_time[i]+30m)
    # We create edges = [t0, t1, ..., tT] where t_{i+1}=open_time[i]+30m for i=T-1? not exactly.
    # Simplest: edges are open_time plus final edge at last_open + 30m,
    # then assign idx = searchsorted(edges, ts, 'right')-1
    edges = np.empty(T + 1, dtype="datetime64[ns]")
    edges[:T] = open_time
    edges[T] = open_time[-1] + np.timedelta64(30, "m")

    # --- x grid ---
    grid = np.linspace(cfg.x_min, cfg.x_max, cfg.nx, dtype=np.float64)
    dx = float(grid[1] - grid[0])

    # histogram accumulator (volume mass per x-bin per window)
    H = np.zeros((T, cfg.nx), dtype=np.float64)

    parts = list_parts(month_dir, cfg.parts_glob)

    for pi, p in enumerate(parts):
        df = pd.read_parquet(p, columns=["timestamp", "price", "qty"])

        ts = to_utc_naive_ns(df["timestamp"])
        price = df["price"].to_numpy(dtype=np.float64)
        qty = df["qty"].to_numpy(dtype=np.float64)

        # window index for each trade
        idx = np.searchsorted(edges, ts, side="right") - 1
        m = (idx >= 0) & (idx < T)
        if not np.any(m):
            del df
            continue

        idx = idx[m]
        price = price[m]
        qty = qty[m]

        # x = log(price / close_window)
        close_i = close[idx]
        x = np.log(price / close_i)

        # clip to grid range
        x = np.clip(x, cfg.x_min, cfg.x_max)

        # bin index
        b = np.floor((x - cfg.x_min) / dx).astype(np.int64)
        b = np.clip(b, 0, cfg.nx - 1)

        # Fast accumulation: group by window, then bincount within each group
        order = np.argsort(idx)
        idx_sorted = idx[order]
        b_sorted = b[order]
        q_sorted = qty[order]

        # boundaries where window id changes
        cut = np.flatnonzero(np.diff(idx_sorted)) + 1
        starts = np.r_[0, cut]
        ends = np.r_[cut, len(idx_sorted)]

        for s, e in zip(starts, ends):
            wi = int(idx_sorted[s])
            H[wi] += np.bincount(b_sorted[s:e], weights=q_sorted[s:e], minlength=cfg.nx)

        del df
        gc.collect()

        if (pi + 1) % 5 == 0 or (pi + 1) == len(parts):
            print(f"[{cfg.month}] processed parts {pi+1}/{len(parts)}")

    # --- KDE-like smoothing (Gaussian convolution) ---
    k = gaussian_kernel_1d(dx=dx, bandwidth=cfg.bandwidth, truncate=4.0)
    Hs = fft_convolve_same_rows(H, k)

    # --- Convert volume-mass histogram to probability density and normalize ---
    pdfs = np.zeros((T, cfg.nx), dtype=np.float32)

    prev_pdf: Optional[np.ndarray] = None
    for i in range(T):
        f = Hs[i]
        if not np.isfinite(f).all():
            f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

        # Empty?
        if f.sum() <= 0:
            if cfg.empty_window == "carry" and prev_pdf is not None:
                pdf = prev_pdf
            else:
                pdf = np.ones(cfg.nx, dtype=np.float32)
                # continuous normalization: sum(pdf)*dx = 1
                pdf /= float(pdf.sum()) * dx
        else:
            # continuous normalization
            pdf = (f / (float(f.sum()) * dx)).astype(np.float32)

        pdfs[i] = pdf
        prev_pdf = pdf

    # Sanity check
    masses = pdfs.sum(axis=1) * dx
    print(f"[{cfg.month}] mass mean={masses.mean():.6f}  min={masses.min():.6f}  max={masses.max():.6f}")

    # Save outputs
    np.save(out_dir / f"pdfs_{cfg.month}.npy", pdfs)  # [T, Nx]
    np.save(out_dir / f"closes_{cfg.month}.npy", close.astype(np.float32))
    np.save(out_dir / f"times_{cfg.month}.npy", open_time.astype("datetime64[ns]"))
    np.save(out_dir / f"x_grid_{cfg.month}.npy", grid.astype(np.float32))

    meta = {
        "symbol": cfg.symbol,
        "month": cfg.month,
        "T": int(T),
        "nx": int(cfg.nx),
        "x_definition": "x = log(price / close_30m_window)",
        "x_min": float(cfg.x_min),
        "x_max": float(cfg.x_max),
        "dx": float(dx),
        "kde_implementation": "weighted histogram + Gaussian convolution (FFT)",
        "bandwidth": float(cfg.bandwidth),
        "empty_window": cfg.empty_window,
        "inputs": {
            "month_dir": str(month_dir),
            "klines": str(month_dir / cfg.klines_name),
            "parts_glob": cfg.parts_glob,
        },
        "outputs": {
            "pdfs": str(out_dir / f"pdfs_{cfg.month}.npy"),
            "closes": str(out_dir / f"closes_{cfg.month}.npy"),
            "times": str(out_dir / f"times_{cfg.month}.npy"),
            "x_grid": str(out_dir / f"x_grid_{cfg.month}.npy"),
        },
    }
    with open(out_dir / f"meta_{cfg.month}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()