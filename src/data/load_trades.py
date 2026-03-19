from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class TradeColumns:
    ts: str = "timestamp"   # raw column name for timestamp
    price: str = "price"    # raw column name for price
    vol: str = "volume"     # raw column name for volume


@dataclass(frozen=True)
class LoadConfig:
    path: str
    file_type: Optional[Literal["parquet", "csv"]] = None
    timezone: str = "UTC"  # store/convert all timestamps to this tz (kept tz-aware)

    cols: TradeColumns = TradeColumns()

    # cleaning rules
    drop_duplicates: bool = True
    dedup_subset: tuple[str, ...] = ("timestamp", "price", "volume")
    min_price: float = 0.0
    min_volume: float = 0.0
    clip_price_quantiles: Optional[tuple[float, float]] = None  # e.g. (0.0001, 0.9999)
    clip_volume_quantiles: Optional[tuple[float, float]] = None


def _infer_file_type(path: str) -> str:
    p = Path(path)
    suf = p.suffix.lower().lstrip(".")
    if suf in ("parquet", "csv"):
        return suf
    raise ValueError(f"Cannot infer file type from suffix '{p.suffix}'. Please set file_type explicitly.")


def _read_trades(path: str, file_type: str) -> pd.DataFrame:
    if file_type == "parquet":
        return pd.read_parquet(path)
    if file_type == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file_type={file_type}")


def load_and_sanitize_trades(cfg: LoadConfig) -> pd.DataFrame:
    """
    Returns a clean trades dataframe with standardized columns:
      - timestamp (tz-aware, converted to cfg.timezone)
      - price (float64)
      - volume (float64)

    Notes:
      - No resampling here (Step 2).
      - Quantile clipping is optional; off by default.
    """
    file_type = cfg.file_type or _infer_file_type(cfg.path)
    df = _read_trades(cfg.path, file_type)

    # --- rename to standard names ---
    rename_map = {
        cfg.cols.ts: "timestamp",
        cfg.cols.price: "price",
        cfg.cols.vol: "volume",
    }
    missing = [c for c in rename_map.keys() if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing expected columns {missing}. "
            f"Available columns: {list(df.columns)[:50]}"
        )
    df = df.rename(columns=rename_map)[["timestamp", "price", "volume"]].copy()

    # --- timestamp parse ---
    # Accept int ms/us/ns, str, datetime already.
    ts = df["timestamp"]
    if np.issubdtype(ts.dtype, np.number):
        # Heuristic: decide unit by magnitude (ms vs us vs ns)
        med = float(np.nanmedian(ts.values))
        # ~1e12 ms (2020s), ~1e15 us, ~1e18 ns
        if med > 1e17:
            unit = "ns"
        elif med > 1e14:
            unit = "us"
        else:
            unit = "ms"
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit=unit, utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    df = df.dropna(subset=["timestamp"])
    # Convert to requested timezone (still tz-aware)
    df["timestamp"] = df["timestamp"].dt.tz_convert(cfg.timezone)

    # --- numeric cast ---
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["price", "volume"])

    # --- basic validity filters ---
    df = df[(df["price"] > cfg.min_price) & (df["volume"] > cfg.min_volume)]

    # --- optional quantile clipping (anti-outlier) ---
    if cfg.clip_price_quantiles is not None:
        lo, hi = cfg.clip_price_quantiles
        p_lo, p_hi = df["price"].quantile([lo, hi]).values
        df = df[(df["price"] >= p_lo) & (df["price"] <= p_hi)]

    if cfg.clip_volume_quantiles is not None:
        lo, hi = cfg.clip_volume_quantiles
        v_lo, v_hi = df["volume"].quantile([lo, hi]).values
        df = df[(df["volume"] >= v_lo) & (df["volume"] <= v_hi)]

    # --- sort & deduplicate ---
    df = df.sort_values("timestamp").reset_index(drop=True)
    if cfg.drop_duplicates:
        df = df.drop_duplicates(subset=list(cfg.dedup_subset), keep="last").reset_index(drop=True)

    return df