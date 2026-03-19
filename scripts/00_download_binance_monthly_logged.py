from __future__ import annotations

import time
import json
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional

import requests
import pandas as pd

from src.utils.logging import setup_logger


@dataclass
class Cfg:
    base_url: str = "https://api.binance.com"
    symbol: str = "BTCUSDT"
    start_utc: str = "2025-01-01 00:00:00"
    end_utc: str = "2026-01-01 00:00:00"

    out_dir: str = "data/raw/binance_monthly"
    log_dir: str = "logs"

    sleep_s: float = 0.15
    timeout_s: int = 30
    max_retries: int = 8

    agg_limit: int = 1000
    kline_interval: str = "30m"
    kline_limit: int = 1000

    save_format: Literal["parquet", "csv"] = "parquet"

    # RAM safety: flush trades to disk every N rows
    flush_rows: int = 250_000
    resume: bool = True


def to_ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)


def month_starts(start_utc: str, end_utc: str) -> List[pd.Timestamp]:
    start = pd.Timestamp(start_utc, tz="UTC")
    end = pd.Timestamp(end_utc, tz="UTC")
    cur = start.normalize().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    out = []
    while cur < end:
        out.append(cur)
        cur = (cur + pd.offsets.MonthBegin(1))
    return out


def month_tag(ts: pd.Timestamp) -> str:
    return f"{ts.year:04d}-{ts.month:02d}"


def save_df(df: pd.DataFrame, path: Path, fmt: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def request_json(session: requests.Session, url: str, params: Dict[str, Any], cfg: Cfg, logger):
    last_err = None
    for attempt in range(cfg.max_retries):
        try:
            t0 = time.time()
            r = session.get(url, params=params, timeout=cfg.timeout_s)
            dt = time.time() - t0

            used_w = r.headers.get("X-MBX-USED-WEIGHT-1M")
            logger.info(f"HTTP {r.status_code} {url} dt={dt:.3f}s used_weight_1m={used_w}")

            if r.status_code in (418, 429):
                wait = min(60.0, (2 ** attempt) * 0.5)
                logger.warning(f"Rate limit {r.status_code}. Sleep {wait:.1f}s. params={params}")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json(), r.headers

        except Exception as e:
            last_err = e
            wait = min(30.0, (2 ** attempt) * 0.5)
            logger.exception(f"Request failed attempt={attempt+1}/{cfg.max_retries}. Sleep {wait:.1f}s. params={params}")
            time.sleep(wait)

    raise RuntimeError(f"Request failed after retries. last_err={last_err}")


def fetch_agg_trades_month(session, cfg: Cfg, logger, start_ms: int, end_ms: int, month_out: Path):
    """
    Fetch aggTrades and flush to disk in parts to prevent RAM blow-up.
    """
    url = f"{cfg.base_url}/api/v3/aggTrades"

    cur_ms = start_ms
    part = 0
    rows: List[Dict[str, Any]] = []
    total_rows = 0

    while cur_ms < end_ms:
        params = {"symbol": cfg.symbol, "startTime": cur_ms, "endTime": end_ms, "limit": cfg.agg_limit}
        data, _ = request_json(session, url, params, cfg, logger)
        if not data:
            break

        rows.extend(data)
        total_rows += len(data)

        last_T = int(data[-1]["T"])
        cur_ms = max(cur_ms + 1, last_T + 1)

        # flush if big
        if len(rows) >= cfg.flush_rows:
            df = pd.DataFrame(rows).rename(
                columns={"T": "timestamp_ms", "p": "price", "q": "qty", "m": "is_buyer_maker", "a": "agg_id"}
            )
            df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            df["price"] = df["price"].astype(float)
            df["qty"] = df["qty"].astype(float)
            df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
            df = df[["timestamp", "price", "qty", "is_buyer_maker", "agg_id"]].sort_values("timestamp").reset_index(drop=True)

            out_path = month_out / f"agg_trades_part{part:03d}.{ 'parquet' if cfg.save_format=='parquet' else 'csv' }"
            save_df(df, out_path, cfg.save_format)
            logger.info(f"[aggTrades] flushed part={part} rows={len(df)} -> {out_path.name}")

            part += 1
            rows.clear()
            del df
            gc.collect()

        logger.info(f"[aggTrades] progress total_rows={total_rows} cur_ms={cur_ms}")
        time.sleep(cfg.sleep_s)

    # final flush
    if rows:
        df = pd.DataFrame(rows).rename(
            columns={"T": "timestamp_ms", "p": "price", "q": "qty", "m": "is_buyer_maker", "a": "agg_id"}
        )
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df["price"] = df["price"].astype(float)
        df["qty"] = df["qty"].astype(float)
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
        df = df[["timestamp", "price", "qty", "is_buyer_maker", "agg_id"]].sort_values("timestamp").reset_index(drop=True)

        out_path = month_out / f"agg_trades_part{part:03d}.{ 'parquet' if cfg.save_format=='parquet' else 'csv' }"
        save_df(df, out_path, cfg.save_format)
        logger.info(f"[aggTrades] flushed FINAL part={part} rows={len(df)} -> {out_path.name}")

        rows.clear()
        del df
        gc.collect()

    return total_rows


def fetch_klines_month(session, cfg: Cfg, logger, start_ms: int, end_ms: int) -> int:
    url = f"{cfg.base_url}/api/v3/klines"
    cur_ms = start_ms
    rows: List[List[Any]] = []
    total = 0

    while cur_ms < end_ms:
        params = {"symbol": cfg.symbol, "interval": cfg.kline_interval, "startTime": cur_ms, "endTime": end_ms, "limit": cfg.kline_limit}
        data, _ = request_json(session, url, params, cfg, logger)
        if not data:
            break

        rows.extend(data)
        total += len(data)

        last_open = int(data[-1][0])
        cur_ms = max(cur_ms + 1, last_open + 1)

        logger.info(f"[klines] progress total_rows={total} last_open={last_open}")
        time.sleep(cfg.sleep_s)

        if len(data) < cfg.kline_limit:
            break

    if not rows:
        return 0

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
    cfg = Cfg()

    log_path = Path(cfg.log_dir) / f"binance_collector_{cfg.symbol}.log"
    logger = setup_logger(str(log_path), name=f"collector.{cfg.symbol}")
    logger.info("Starting Binance monthly collector")
    logger.info(f"cfg={cfg}")

    start = pd.Timestamp(cfg.start_utc, tz="UTC")
    end = pd.Timestamp(cfg.end_utc, tz="UTC")

    base_out = Path(cfg.out_dir) / cfg.symbol
    base_out.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    meta_year = {"symbol": cfg.symbol, "start_utc": cfg.start_utc, "end_utc": cfg.end_utc, "months": []}

    for ms in month_starts(cfg.start_utc, cfg.end_utc):
        m0 = ms
        m1 = ms + pd.offsets.MonthBegin(1)
        w0 = max(m0, start)
        w1 = min(m1, end)
        if w0 >= w1:
            continue

        tag = month_tag(w0)
        month_out = base_out / tag
        month_out.mkdir(parents=True, exist_ok=True)

        done_flag = month_out / "_DONE.json"
        if cfg.resume and done_flag.exists():
            logger.info(f"[resume] Month {tag} already done. Skipping.")
            continue

        logger.info(f"===== Month {tag} window {w0.isoformat()} -> {w1.isoformat()} =====")

        start_ms = to_ms(w0)
        end_ms = to_ms(w1)

        try:
            # 1) aggTrades in parts (RAM safe)
            n_trades = fetch_agg_trades_month(session, cfg, logger, start_ms, end_ms, month_out)

            # 2) klines (small enough to hold month in memory)
            kdf = fetch_klines_month(session, cfg, logger, start_ms, end_ms)
            kline_path = month_out / f"klines_30m.{ 'parquet' if cfg.save_format=='parquet' else 'csv' }"
            save_df(kdf, kline_path, cfg.save_format)

            month_meta = {
                "tag": tag,
                "window_start_utc": w0.isoformat(),
                "window_end_utc": w1.isoformat(),
                "aggTrades_rows": int(n_trades),
                "kline_rows": int(len(kdf)),
                "files": {
                    "aggTrades_parts_dir": str(month_out),
                    "klines_30m": str(kline_path),
                },
            }
            meta_year["months"].append(month_meta)

            with open(done_flag, "w") as f:
                json.dump(month_meta, f, indent=2)

            logger.info(f"[DONE] Month {tag} saved. trades={n_trades} klines={len(kdf)}")
            del kdf
            gc.collect()

        except Exception:
            logger.exception(f"[FAIL] Month {tag} crashed. You can resume later.")
            # do not write DONE flag
            break

    with open(base_out / "meta_year.json", "w") as f:
        json.dump(meta_year, f, indent=2)

    logger.info(f"Finished. meta_year={base_out/'meta_year.json'}")


if __name__ == "__main__":
    main()