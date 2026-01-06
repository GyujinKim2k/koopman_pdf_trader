from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------
@dataclass
class BuildCfg:
    symbol: str = "BTCUSDT"

    # inputs from Step 1 downloader
    agg_trades_path: str = "data/raw/binance/BTCUSDT/agg_trades.parquet"
    klines_30m_path: str = "data/raw/binance/BTCUSDT/klines_30m.parquet"

    # outputs
    out_dir: str = "data/processed/BTCUSDT_30m_pdf"

    # PDF grid
    nx: int = 64
    ny: int = 64

    # x-axis (log price) range from TRAIN quantiles
    x_q_lo: float = 0.001
    x_q_hi: float = 0.999

    # y-axis normalization: log1p + zscore (fit on TRAIN trades only)
    y_clip_sigma: float = 5.0  # clip normalized y to [-k, +k] before binning

    # histogram smoothing (optional; requires scipy). keep 0 for now.
    smoothing_sigma: float = 0.0

    # split by time (no leakage)
    train_frac: float = 0.70
    val_frac: float = 0.15  # test is the rest

    # if a 30m window has no trades, use previous pdf (more stable than uniform)
    empty_window: str = "carry"  # "carry" or "uniform"


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def fit_log1p_zscore(vol: np.ndarray) -> Tuple[float, float]:
    v = np.log1p(vol.astype(np.float64))
    mu = float(v.mean())
    sigma = float(v.std() + 1e-8)
    return mu, sigma


def transform_log1p_zscore(vol: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    v = np.log1p(vol.astype(np.float64))
    z = (v - mu) / sigma
    return z.astype(np.float32)


def make_edges_from_range(lo: float, hi: float, n_bins: int) -> np.ndarray:
    # n_bins bins -> n_bins+1 edges
    return np.linspace(lo, hi, n_bins + 1, dtype=np.float32)


def make_2d_pdf_hist(x: np.ndarray, y: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray,
                    smoothing_sigma: float = 0.0) -> np.ndarray:
    """
    Returns rho shape (ny, nx) sum=1
    Note histogram2d expects x then y, but we pass (y, x) to get [y_bins, x_bins].
    """
    H, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])  # y first -> rows
    rho = H.astype(np.float32)

    if smoothing_sigma > 0:
        # optional
        from scipy.ndimage import gaussian_filter
        rho = gaussian_filter(rho, sigma=smoothing_sigma)

    s = float(rho.sum())
    if s <= 0:
        return None  # caller handles empty window
    rho /= s
    return rho


def time_splits(T: int, train_frac: float, val_frac: float) -> Dict[str, Tuple[int, int]]:
    n_train = int(T * train_frac)
    n_val = int(T * val_frac)
    n_test = T - n_train - n_val
    assert n_train > 10 and n_val > 10 and n_test > 10, "Too few points for split."
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, T),
    }


# -----------------------------
# Main build
# -----------------------------
def main():
    cfg = BuildCfg()

    out_dir = ensure_dir(cfg.out_dir)

    # ---- load data from Step 1
    trades = pd.read_parquet(cfg.agg_trades_path)
    kl = pd.read_parquet(cfg.klines_30m_path)

    # standardize / ensure tz-aware
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    kl["open_time"] = pd.to_datetime(kl["open_time"], utc=True)
    kl["close_time"] = pd.to_datetime(kl["close_time"], utc=True)

    # we use base volume from aggTrades: qty
    if "qty" not in trades.columns:
        raise KeyError("agg_trades.parquet must contain 'qty' column (base volume).")

    trades = trades.sort_values("timestamp").reset_index(drop=True)
    kl = kl.sort_values("open_time").reset_index(drop=True)

    # ---- define the canonical 30m index using klines
    # We'll build PDFs for each kline open_time window [open_time, open_time + 30m)
    times = kl["open_time"].to_numpy()
    closes = kl["close"].to_numpy(dtype=np.float32)

    T = len(times)
    splits = time_splits(T, cfg.train_frac, cfg.val_frac)

    # ---- fit y normalizer on TRAIN trades only (by time)
    train_end_time = times[splits["train"][1] - 1] + np.timedelta64(30, "m")
    train_trades = trades[trades["timestamp"].to_numpy() < train_end_time]

    y_mu, y_sigma = fit_log1p_zscore(train_trades["qty"].to_numpy(dtype=np.float64))

    # ---- build x range from TRAIN trades only
    train_x = np.log(train_trades["price"].to_numpy(dtype=np.float64))
    x_lo = float(np.quantile(train_x, cfg.x_q_lo))
    x_hi = float(np.quantile(train_x, cfg.x_q_hi))

    # ---- define y range in normalized units (fixed)
    # We'll bin in [-k, +k] after clipping.
    y_lo, y_hi = -cfg.y_clip_sigma, cfg.y_clip_sigma

    x_edges = make_edges_from_range(x_lo, x_hi, cfg.nx)
    y_edges = make_edges_from_range(y_lo, y_hi, cfg.ny)

    # ---- pre-slice trades by time window efficiently
    trade_ts = trades["timestamp"].to_numpy()
    trade_price = trades["price"].to_numpy(dtype=np.float64)
    trade_qty = trades["qty"].to_numpy(dtype=np.float64)

    pdfs = np.zeros((T, 1, cfg.ny, cfg.nx), dtype=np.float32)

    # pointer walk over trades (fast)
    j0 = 0
    prev_rho: Optional[np.ndarray] = None

    for i in range(T):
        t0 = times[i]
        t1 = t0 + np.timedelta64(30, "m")

        # advance j0 to first trade >= t0
        while j0 < len(trade_ts) and trade_ts[j0] < t0:
            j0 += 1

        j1 = j0
        while j1 < len(trade_ts) and trade_ts[j1] < t1:
            j1 += 1

        if j1 <= j0:
            # empty window
            if cfg.empty_window == "carry" and prev_rho is not None:
                rho = prev_rho
            else:
                rho = np.ones((cfg.ny, cfg.nx), dtype=np.float32)
                rho /= rho.sum()
        else:
            x = np.log(trade_price[j0:j1])
            y = transform_log1p_zscore(trade_qty[j0:j1], y_mu, y_sigma)
            y = np.clip(y, y_lo, y_hi)

            # clip x softly to range (avoid out-of-range bins -> dropped mass)
            x = np.clip(x, x_lo, x_hi)

            rho_maybe = make_2d_pdf_hist(x, y, x_edges, y_edges, smoothing_sigma=cfg.smoothing_sigma)
            if rho_maybe is None:
                if cfg.empty_window == "carry" and prev_rho is not None:
                    rho = prev_rho
                else:
                    rho = np.ones((cfg.ny, cfg.nx), dtype=np.float32)
                    rho /= rho.sum()
            else:
                rho = rho_maybe

        pdfs[i, 0] = rho
        prev_rho = rho

    # ---- save
    np.save(out_dir / "pdfs.npy", pdfs)                   # [T,1,Ny,Nx]
    np.save(out_dir / "closes.npy", closes)               # [T]
    np.save(out_dir / "timestamps_open.npy", times.astype("datetime64[ns]"))

    meta: Dict[str, Any] = {
        "symbol": cfg.symbol,
        "T": int(T),
        "nx": cfg.nx,
        "ny": cfg.ny,
        "x_edges": {"lo": x_lo, "hi": x_hi, "q_lo": cfg.x_q_lo, "q_hi": cfg.x_q_hi},
        "y_norm": {"type": "log1p_zscore", "mu": y_mu, "sigma": y_sigma, "clip_sigma": cfg.y_clip_sigma},
        "split": splits,
        "empty_window": cfg.empty_window,
        "smoothing_sigma": cfg.smoothing_sigma,
        "source": {
            "agg_trades_path": cfg.agg_trades_path,
            "klines_30m_path": cfg.klines_30m_path,
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    np.save(out_dir / "x_edges.npy", x_edges)
    np.save(out_dir / "y_edges.npy", y_edges)

    print(f"Saved dataset to: {out_dir}")
    print(f"pdfs: {pdfs.shape}  closes: {closes.shape}")
    print("splits:", splits)
    print("x range:", x_lo, x_hi)
    print("y mu/sigma:", y_mu, y_sigma)


if __name__ == "__main__":
    main()