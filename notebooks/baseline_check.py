import numpy as np

root = "data/processed/BTCUSDT_2025_1d_kde_pdf_fast"
pdfs = np.load(f"{root}/pdfs_2025.npy")
times = np.load(f"{root}/times_2025.npy")
xgrid = np.load(f"{root}/2025-01/x_grid_2025-01.npy")
dx = float(xgrid[1]-xgrid[0])

# dedup+sort
order = np.argsort(times)
times = times[order]; pdfs = pdfs[order]
uniq_times, uniq_idx = np.unique(times, return_index=True)
pdfs = pdfs[uniq_idx]

p = np.maximum(pdfs*dx, 0.0)
p = p / p.sum(axis=1, keepdims=True)

T = len(p)
train_frac, val_frac = 0.70, 0.15
n_train = int(T*train_frac)
n_val = int(T*val_frac)
val0, val1 = n_train, n_train+n_val
test0, test1 = val1, T

L, H = 48, 6

p_train = p[:n_train]
p_mean = p_train.mean(axis=0, keepdims=True)
p_mean /= p_mean.sum(axis=1, keepdims=True)

def CE(p_true, p_pred, eps=1e-12):
    p_pred = np.clip(p_pred, eps, 1.0)
    return float(-(p_true*np.log(p_pred)).sum(axis=1).mean())

def seq_baseline(idx0, idx1):
    # evaluate targets p_{t+1:t+H} where t ranges with enough future
    t_start = idx0 + (L-1)
    t_end = idx1 - H - 1
    ces = []
    for t in range(t_start, t_end+1):
        Y = p[t+1:t+1+H]                # [H,Nx]
        Yhat = np.repeat(p_mean, H, 0)  # [H,Nx]
        ces.append(CE(Y, Yhat))
    return float(np.mean(ces))

print("Baseline mean-PDF over H=6:")
print("val :", seq_baseline(val0, val1))
print("test:", seq_baseline(test0, test1))