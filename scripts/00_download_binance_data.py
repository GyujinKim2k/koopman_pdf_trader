from __future__ import annotations

import time
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd


# -----------------------------
# Config
# -----------------------------
@dataclass
class BinanceConfig:
    base_url: str = "https://api.binance.com"   # spot REST
    symbol: str = "BTCUSDT"
    # time range in UTC ISO string e.g. "2024-01-01 00:00:00"
    start_utc: str = "2024-01-01 00:00:00"
    end_utc: str = "2024-02-01 00:00:00"

    out_dir: str = "data/raw/binance"
    sleep_s: float = 0.15           # gentle pacing; tune later
    timeout_s: int = 30
    max_retries: int = 8

    # aggTrades
    agg_limit: int = 1000           # docs say max 1000 for aggTrades page; safe default
    # klines
    kline_interval: str = "30m"
    kline_limit: int = 1000         # max 1000 for /api/v3/klines


def to_ms_utc(dt_str: str) -> int:
    # treat input as UTC
    ts = pd.Timestamp(dt_str, tz="UTC")
    return int(ts.value // 1_000_000)  # ns -> ms


def ms_to_iso(ms: int) -> str:
    return pd.Timestamp(ms, unit="ms", tz="UTC").isoformat()


# -----------------------------
# HTTP helper with retries
# -----------------------------
def _request_json(session: requests.Session, url: str, params: Dict[str, Any], cfg: BinanceConfig):
    last_err = None
    for attempt in range(cfg.max_retries):
        try:
            r = session.get(url, params=params, timeout=cfg.timeout_s)
            if r.status_code in (418, 429):
                # rate-limit / ban protection
                wait = min(60.0, (2 ** attempt) * 0.5)
                print(f"[rate-limit] {r.status_code} sleeping {wait:.1f}s, url={url}")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json(), r.headers
        except Exception as e:
            last_err = e
            wait = min(30.0, (2 ** attempt) * 0.5)
            print(f"[retry] attempt={attempt+1}/{cfg.max_retries} wait={wait:.1f}s err={e}")
            time.sleep(wait)

    raise RuntimeError(f"Request failed after retries. url={url} params={params} last_err={last_err}")


# -----------------------------
# Download aggTrades (for PDFs)
# -----------------------------
def fetch_agg_trades(cfg: BinanceConfig) -> pd.DataFrame:
    """
    Uses GET /api/v3/aggTrades with startTime/endTime paging by time.
    Binance returns aggregate trades with fields like:
      a (aggTradeId), p (price), q (qty), T (timestamp ms), m (isBuyerMaker)
    """
    url = f"{cfg.base_url}/api/v3/aggTrades"

    start_ms = to_ms_utc(cfg.start_utc)
    end_ms = to_ms_utc(cfg.end_utc)

    session = requests.Session()

    rows: List[Dict[str, Any]] = []
    cur_ms = start_ms

    while cur_ms < end_ms:
        params = {
            "symbol": cfg.symbol,
            "startTime": cur_ms,
            "endTime": end_ms,
            "limit": cfg.agg_limit,
        }

        data, headers = _request_json(session, url, params, cfg)
        if not data:
            # no more trades in range
            break

        rows.extend(data)

        # Advance cursor by last trade time + 1ms to avoid duplicates
        last_T = int(data[-1]["T"])
        next_ms = last_T + 1

        # Safety: if next_ms doesn't move, force a small jump
        if next_ms <= cur_ms:
            next_ms = cur_ms + 1

        cur_ms = next_ms

        used_w = headers.get("X-MBX-USED-WEIGHT-1M")
        if used_w:
            print(f"[aggTrades] got={len(data)} lastT={ms_to_iso(last_T)} used_weight_1m={used_w}")
        else:
            print(f"[aggTrades] got={len(data)} lastT={ms_to_iso(last_T)}")

        time.sleep(cfg.sleep_s)

        # If we keep hitting the 1000 cap frequently, it might indicate high activity.
        # This paging method still works because we move by last trade time.

    if not rows:
        return pd.DataFrame(columns=["timestamp", "price", "qty", "is_buyer_maker", "agg_id"])

    df = pd.DataFrame(rows)
    df = df.rename(
        columns={"T": "timestamp_ms", "p": "price", "q": "qty", "m": "is_buyer_maker", "a": "agg_id"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
    df = df[["timestamp", "price", "qty", "is_buyer_maker", "agg_id"]].sort_values("timestamp").reset_index(drop=True)
    return df


# -----------------------------
# Download 30m klines (for close prices)
# -----------------------------
def fetch_klines(cfg: BinanceConfig) -> pd.DataFrame:
    """
    GET /api/v3/klines supports startTime/endTime and max 1000 bars per request.
    We'll page forward by openTime.
    """
    url = f"{cfg.base_url}/api/v3/klines"

    start_ms = to_ms_utc(cfg.start_utc)
    end_ms = to_ms_utc(cfg.end_utc)

    session = requests.Session()

    rows: List[List[Any]] = []
    cur_ms = start_ms

    while cur_ms < end_ms:
        params = {
            "symbol": cfg.symbol,
            "interval": cfg.kline_interval,
            "startTime": cur_ms,
            "endTime": end_ms,
            "limit": cfg.kline_limit,
        }
        data, headers = _request_json(session, url, params, cfg)
        if not data:
            break
        rows.extend(data)

        last_open = int(data[-1][0])  # open time ms
        # move to next candle
        cur_ms = last_open + 1

        used_w = headers.get("X-MBX-USED-WEIGHT-1M")
        if used_w:
            print(f"[klines] got={len(data)} last_open={ms_to_iso(last_open)} used_weight_1m={used_w}")
        else:
            print(f"[klines] got={len(data)} last_open={ms_to_iso(last_open)}")

        time.sleep(cfg.sleep_s)

        if len(data) < cfg.kline_limit:
            # likely reached end
            break

    if not rows:
        return pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time"])

    cols = [
        "open_time_ms", "open", "high", "low", "close", "volume",
        "close_time_ms", "quote_asset_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df = df[["open_time","close_time","open","high","low","close","volume","num_trades"]].sort_values("open_time").reset_index(drop=True)
    return df


def main():
    cfg = BinanceConfig(
        symbol="BTCUSDT",
        start_utc="2025-01-01 00:00:00",
        end_utc="2026-01-01 00:00:00",
        out_dir="data/raw/binance",
    )

    out = Path(cfg.out_dir) / cfg.symbol
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading aggTrades...")
    trades = fetch_agg_trades(cfg)
    print("Downloading 30m klines...")
    klines = fetch_klines(cfg)

    trades_path = out / "agg_trades.parquet"
    klines_path = out / "klines_30m.parquet"

    trades.to_parquet(trades_path, index=False)
    klines.to_parquet(klines_path, index=False)

    meta = {
        "symbol": cfg.symbol,
        "start_utc": cfg.start_utc,
        "end_utc": cfg.end_utc,
        "rows_aggTrades": int(len(trades)),
        "rows_klines_30m": int(len(klines)),
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved:\n- {trades_path}\n- {klines_path}\n- {out/'meta.json'}")


if __name__ == "__main__":
    main()