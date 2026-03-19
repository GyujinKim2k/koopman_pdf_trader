from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Cfg:
    modes_dir: str = "runs/koopman_seq_gru_v1/modes"   # outputs from step 3.5
    use_left_right_A: bool = True  # if you saved A from left-right, set True; else it will compute from saved A

    # trading config
    decision_step: int = 1         # decide every 30m window
    hold_steps: int = 2            # 2 => hold 60m; 1 => hold 30m
    fee_bps: float = 7.5           # total cost per round-trip? We'll apply per trade event (enter/exit). Adjust.
    slippage_bps: float = 2.5      # additional per trade event

    # signals
    m_fast: int = 10               # number of fast modes to aggregate (modes 1..m_fast)
    zscore_lookback: int = 96      # 2 days = 96 windows

    # gating thresholds (z-scored)
    enter_slow_z: float = +0.3     # require slow amplitude strength
    enter_fast_z: float = -0.2     # require low fast energy (calm)
    exit_fast_z: float = +0.5      # exit if turbulence spikes


def zscore(x: np.ndarray, win: int) -> np.ndarray:
    """
    Rolling z-score with window win. First win-1 entries become nan.
    """
    out = np.full_like(x, np.nan, dtype=np.float64)
    if win < 5:
        return out
    csum = np.cumsum(np.r_[0.0, x])
    csum2 = np.cumsum(np.r_[0.0, x * x])
    for i in range(win - 1, len(x)):
        s = csum[i + 1] - csum[i + 1 - win]
        s2 = csum2[i + 1] - csum2[i + 1 - win]
        mu = s / win
        var = max(1e-12, s2 / win - mu * mu)
        out[i] = (x[i] - mu) / np.sqrt(var)
    return out


def main():
    cfg = Cfg()
    d = Path(cfg.modes_dir)

    eigvals = np.load(d / "eigvals.npy")              # complex (sorted)
    A = np.load(d / "A_amplitudes.npy")               # complex [T, d]
    closes = np.load(d / "closes_z.npy").astype(np.float64)
    times = np.load(d / "times_z.npy")

    T = len(closes)
    assert A.shape[0] == T, (A.shape, T)

    # --- build core signals ---
    a0 = np.abs(A[:, 0])                      # slow amplitude magnitude
    # fast energy from modes 1..m_fast (avoid mode 0)
    m = min(cfg.m_fast, A.shape[1] - 1)
    fastE = np.sum(np.abs(A[:, 1:1 + m]) ** 2, axis=1)

    # changes (momentum of regime/turbulence)
    da0 = np.r_[np.nan, np.diff(a0)]
    dfastE = np.r_[np.nan, np.diff(fastE)]

    # z-score signals
    a0_z = zscore(a0, cfg.zscore_lookback)
    fastE_z = zscore(fastE, cfg.zscore_lookback)
    dfastE_z = zscore(dfastE[np.isfinite(dfastE)].copy(), min(cfg.zscore_lookback, np.isfinite(dfastE).sum()))  # not used; keep simple

    # --- returns for backtest ---
    # close-to-close log returns
    r = np.r_[0.0, np.diff(np.log(closes))]  # per 30m

    # --- trading rule (buy/flat) ---
    pos = np.zeros(T, dtype=np.int8)  # 1 long, 0 flat
    in_pos = False
    hold_left = 0

    # costs per trade event (enter or exit)
    cost_per_event = (cfg.fee_bps + cfg.slippage_bps) * 1e-4

    # track trades
    trades = []  # (t, "BUY"/"SELL", close)

    for t in range(T):
        # only decide at decision steps
        if t % cfg.decision_step != 0:
            # still count down holding
            if in_pos and hold_left > 0:
                hold_left -= 1
            pos[t] = 1 if in_pos else 0
            continue

        # must have valid zscores
        if not np.isfinite(a0_z[t]) or not np.isfinite(fastE_z[t]):
            if in_pos and hold_left > 0:
                hold_left -= 1
            pos[t] = 1 if in_pos else 0
            continue

        # exit logic first
        if in_pos:
            # if turbulence spikes, exit early
            if fastE_z[t] >= cfg.exit_fast_z:
                in_pos = False
                hold_left = 0
                trades.append((t, "SELL", closes[t]))
            else:
                # continue holding countdown
                if hold_left > 0:
                    hold_left -= 1
                # if hold completed, go flat (or you can allow re-entry next step)
                if hold_left == 0:
                    in_pos = False
                    trades.append((t, "SELL", closes[t]))

        # entry logic
        if not in_pos:
            enter = (a0_z[t] >= cfg.enter_slow_z) and (fastE_z[t] <= cfg.enter_fast_z)
            if enter:
                in_pos = True
                hold_left = cfg.hold_steps
                trades.append((t, "BUY", closes[t]))

        pos[t] = 1 if in_pos else 0

    # --- PnL ---
    # strategy gross log return: sum(pos[t-1] * r[t]) (enter affects from next step)
    strat_r = pos[:-1] * r[1:]
    gross = np.sum(strat_r)

    # apply costs: each BUY and SELL is a trade event
    n_events = len(trades)
    cost = n_events * cost_per_event
    net = gross - cost

    # equity curve
    eq = np.cumsum(np.r_[0.0, strat_r])  # log equity gross
    eq_net = eq - np.linspace(0, cost, len(eq))  # distribute cost linearly for plotting-ish

    # summary
    print("---- Koopman gating backtest ----")
    print("T windows:", T)
    print("Hold steps:", cfg.hold_steps, "(30m each)")
    print("Trades (events):", n_events, " | round trips approx:", n_events // 2)
    print("Gross log return:", gross)
    print("Cost (log approx):", cost)
    print("Net log return:", net)
    print("Net simple return:", np.expm1(net))

    # Save artifacts
    out = d / "backtest"
    out.mkdir(exist_ok=True)

    np.save(out / "pos.npy", pos)
    np.save(out / "r.npy", r)
    np.save(out / "eq_log_gross.npy", eq)
    np.save(out / "eq_log_net.npy", eq_net)

    # Save trades as csv-like npy
    # (t, side, price) side encoded
    side = np.array([1 if s == "BUY" else -1 for _, s, _ in trades], dtype=np.int8)
    t_idx = np.array([t for t, _, _ in trades], dtype=np.int32)
    price = np.array([p for _, _, p in trades], dtype=np.float64)
    np.save(out / "trades_t.npy", t_idx)
    np.save(out / "trades_side.npy", side)
    np.save(out / "trades_price.npy", price)

    meta = {
        "hold_steps": cfg.hold_steps,
        "decision_step": cfg.decision_step,
        "fee_bps": cfg.fee_bps,
        "slippage_bps": cfg.slippage_bps,
        "m_fast": cfg.m_fast,
        "zscore_lookback": cfg.zscore_lookback,
        "thresholds": {
            "enter_slow_z": cfg.enter_slow_z,
            "enter_fast_z": cfg.enter_fast_z,
            "exit_fast_z": cfg.exit_fast_z
        },
        "results": {
            "gross_log_return": float(gross),
            "net_log_return": float(net),
            "net_simple_return": float(np.expm1(net)),
            "n_trade_events": int(n_events),
        }
    }
    import json
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved backtest to:", out)


if __name__ == "__main__":
    main()