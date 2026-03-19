import numpy as np
from pathlib import Path

def rolling_zscore(x, win):
    out = np.full_like(x, np.nan, dtype=np.float64)
    csum = np.cumsum(np.r_[0.0, x])
    csum2 = np.cumsum(np.r_[0.0, x*x])
    for i in range(win-1, len(x)):
        s = csum[i+1] - csum[i+1-win]
        s2 = csum2[i+1] - csum2[i+1-win]
        mu = s / win
        var = max(1e-12, s2/win - mu*mu)
        out[i] = (x[i]-mu)/np.sqrt(var)
    return out

d = Path("runs/koopman_seq_gru_v1/modes")
A = np.load(d/"A_amplitudes.npy")
closes = np.load(d/"closes_z.npy").astype(np.float64)

T = len(closes)
a0 = np.abs(A[:,0])
m_fast = 10
fastE = np.sum(np.abs(A[:,1:1+m_fast])**2, axis=1)

zlook = 96
a0_z = rolling_zscore(a0, zlook)
fast_z = rolling_zscore(fastE, zlook)

logp = np.log(closes)
Lm = 12
mom = np.full(T, np.nan)
mom[Lm:] = logp[Lm:] - logp[:-Lm]
mom_z = rolling_zscore(mom, zlook)

valid = np.isfinite(a0_z) & np.isfinite(fast_z) & np.isfinite(mom_z)
print("valid fraction:", valid.mean())

def pct(x, p):
    return np.nanpercentile(x[valid], p)

print("a0_z percentiles:", {p: pct(a0_z, p) for p in [1,5,10,25,50,75,90,95,99]})
print("fast_z percentiles:", {p: pct(fast_z, p) for p in [1,5,10,25,50,75,90,95,99]})
print("mom_z percentiles:", {p: pct(mom_z, p) for p in [1,5,10,25,50,75,90,95,99]})

# check how often each condition holds for your current thresholds
cond_a0 = a0_z >= 0.6
cond_fast = fast_z <= -0.3
cond_mom = mom_z >= 0.4

print("hit a0_z>=0.6:", np.mean(cond_a0 & valid))
print("hit fast_z<=-0.3:", np.mean(cond_fast & valid))
print("hit mom_z>=0.4:", np.mean(cond_mom & valid))
print("hit all three:", np.mean(cond_a0 & cond_fast & cond_mom & valid))