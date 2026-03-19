from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import json


@dataclass
class Cfg:
    modes_dir: str = "runs/koopman_seq_gru_v1/modes"

    # time base
    step_minutes: int = 30

    # trading mechanics
    hold_min_steps: int = 12       # 12*30m = 6 hours minimum hold
    cooldown_steps: int = 12       # wait 6 hours after exit before re-enter
    decision_stride: int = 1       # can keep 1; turnover controlled by hold/cooldown

    # costs (per trade event: BUY or SELL)
    fee_bps: float = 7.5
    slippage_bps: float = 2.5

    # Koopman features
    m_fast: int = 10
    zlook: int = 96                # 2 days rolling zscore

    enter_a0_z = +0.1
    exit_a0_z  = -0.2
    enter_fast_z = +0.2
    exit_fast_z  = +1.0
    mom_enter_z  = +0.1

    # Trend trigger (simple momentum)
    mom_lookback_steps: int = 12   # 6 hours momentum
    mom_z_look: int = 96
    mom_enter_z: float = +0.4      # require positive momentum to enter long


def rolling_zscore(x: np.ndarray, win: int) -> np.ndarray:
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

    A = np.load(d / "A_amplitudes.npy")      # complex [T, d]
    closes = np.load(d / "closes_z.npy").astype(np.float64)
    times = np.load(d / "times_z.npy")

    T = len(closes)
    assert A.shape[0] == T

    # -------- Koopman signals --------
    a0 = np.abs(A[:, 0])  # slow amplitude magnitude
    m = min(cfg.m_fast, A.shape[1] - 1)
    fastE = np.sum(np.abs(A[:, 1:1 + m]) ** 2, axis=1)

    a0_z = rolling_zscore(a0, cfg.zlook)
    fast_z = rolling_zscore(fastE, cfg.zlook)

    # -------- Trend signal (momentum) --------
    logp = np.log(closes)
    mom = np.full(T, np.nan, dtype=np.float64)
    Lm = cfg.mom_lookback_steps
    mom[Lm:] = logp[Lm:] - logp[:-Lm]              # Lm-step log momentum
    mom_z = rolling_zscore(mom, cfg.mom_z_look)

    # returns per 30m
    r = np.r_[0.0, np.diff(logp)]

    # costs
    cost_event = (cfg.fee_bps + cfg.slippage_bps) * 1e-4

    # -------- Backtest state --------
    pos = np.zeros(T, dtype=np.int8)  # 1 long, 0 flat
    in_pos = False
    hold_left = 0
    cooldown = 0
    trades = []  # (t, side, price)

    def valid(t: int) -> bool:
        return np.isfinite(a0_z[t]) and np.isfinite(fast_z[t]) and np.isfinite(mom_z[t])

    for t in range(T):
        if t % cfg.decision_stride != 0:
            pos[t] = 1 if in_pos else 0
            continue

        # decrement timers
        if in_pos and hold_left > 0:
            hold_left -= 1
        if (not in_pos) and cooldown > 0:
            cooldown -= 1

        if not valid(t):
            pos[t] = 1 if in_pos else 0
            continue

        # Koopman regime status
        regime_enter_ok = (a0_z[t] >= cfg.enter_a0_z) and (fast_z[t] <= cfg.enter_fast_z)
        regime_stay_ok  = (a0_z[t] >= cfg.exit_a0_z)  and (fast_z[t] <= cfg.exit_fast_z)

        # Exit rules
        if in_pos:
            # enforce minimum holding; only allow exit after hold_left reaches 0
            if hold_left == 0:
                # exit if regime no longer ok
                if not regime_stay_ok:
                    in_pos = False
                    cooldown = cfg.cooldown_steps
                    trades.append((t, "SELL", closes[t]))
            pos[t] = 1 if in_pos else 0
            continue

        # Entry rules
        if (not in_pos) and cooldown == 0:
            trend_ok = (mom_z[t] >= cfg.mom_enter_z)
            if regime_enter_ok and trend_ok:
                in_pos = True
                hold_left = cfg.hold_min_steps
                trades.append((t, "BUY", closes[t]))

        pos[t] = 1 if in_pos else 0

    # -------- PnL --------
    strat_r = pos[:-1] * r[1:]  # position applies next step return
    gross = float(np.sum(strat_r))
    n_events = len(trades)
    cost = float(n_events * cost_event)
    net = gross - cost

    # buy-and-hold for reference
    bh = float(np.sum(r[1:]))

    # summary
    print("---- Koopman-gated trend backtest ----")
    print("T windows:", T)
    print("Min hold steps:", cfg.hold_min_steps, f"({cfg.hold_min_steps*cfg.step_minutes/60:.1f} hours)")
    print("Cooldown steps:", cfg.cooldown_steps, f"({cfg.cooldown_steps*cfg.step_minutes/60:.1f} hours)")
    print("Trade events:", n_events, "| round trips approx:", n_events // 2)
    print("Gross log return:", gross)
    print("Cost (log approx):", cost)
    print("Net log return:", net)
    print("Net simple return:", float(np.expm1(net)))
    print("Buy&Hold simple return:", float(np.expm1(bh)))

    out = d / "backtest_gated_trend"
    out.mkdir(exist_ok=True)

    np.save(out / "pos.npy", pos)
    np.save(out / "r.npy", r)
    np.save(out / "mom_z.npy", mom_z)
    np.save(out / "a0_z.npy", a0_z)
    np.save(out / "fast_z.npy", fast_z)

    # trades
    side = np.array([1 if s == "BUY" else -1 for _, s, _ in trades], dtype=np.int8)
    t_idx = np.array([t for t, _, _ in trades], dtype=np.int32)
    price = np.array([p for _, _, p in trades], dtype=np.float64)
    np.save(out / "trades_t.npy", t_idx)
    np.save(out / "trades_side.npy", side)
    np.save(out / "trades_price.npy", price)

    meta = {
        "cfg": cfg.__dict__,
        "results": {
            "gross_log_return": gross,
            "net_log_return": net,
            "net_simple_return": float(np.expm1(net)),
            "n_trade_events": int(n_events),
            "round_trips_est": int(n_events // 2),
            "buy_hold_simple_return": float(np.expm1(bh)),
        },
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved to:", out)


if __name__ == "__main__":
    main()