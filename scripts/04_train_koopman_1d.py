from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Config
# -----------------------------
@dataclass
class Cfg:
    # data
    root: str = "data/processed/BTCUSDT_2025_1d_kde_pdf_fast"
    pdfs_path: str = "pdfs_2025.npy"
    times_path: str = "times_2025.npy"
    closes_path: str = "closes_2025.npy"
    xgrid_month_example: str = "2025-01/x_grid_2025-01.npy"  # for dx

    # split
    train_frac: float = 0.70
    val_frac: float = 0.15  # rest test

    # model
    nx: int = 512
    latent_dim: int = 64
    hidden: int = 256
    dropout: float = 0.05

    # training
    batch_size: int = 512
    num_workers: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 40
    grad_clip: float = 1.0
    patience: int = 6  # early stopping on val

    # misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    out_dir: str = "runs/koopman_1d_v1"


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dedup_and_sort(times: np.ndarray, pdfs: np.ndarray, closes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    times: datetime64[ns] array (may have duplicates / unsorted)
    pdfs: [T, Nx]
    closes: [T]
    Returns sorted unique by time.
    """
    # sort by time first
    order = np.argsort(times)
    times_s = times[order]
    pdfs_s = pdfs[order]
    closes_s = closes[order]

    # unique by time keeping first occurrence
    uniq_times, uniq_idx = np.unique(times_s, return_index=True)
    times_u = uniq_times
    pdfs_u = pdfs_s[uniq_idx]
    closes_u = closes_s[uniq_idx]
    return times_u, pdfs_u, closes_u


def split_indices(T: int, train_frac: float, val_frac: float) -> Dict[str, Tuple[int, int]]:
    n_train = int(T * train_frac)
    n_val = int(T * val_frac)
    n_test = T - n_train - n_val
    assert n_train > 10 and n_val > 10 and n_test > 10
    return {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, T)}


# -----------------------------
# Dataset: (p_t -> p_{t+1})
# -----------------------------
class KoopmanPairsDataset(Dataset):
    def __init__(self, probs: np.ndarray, idx0: int, idx1: int):
        """
        probs: [T, Nx] probabilities (sum=1)
        Use pairs (t -> t+1) within [idx0, idx1).
        """
        self.probs = probs
        self.idx0 = idx0
        self.idx1 = idx1
        # last usable t is idx1-2 because we need t+1
        self.N = max(0, (idx1 - idx0) - 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        t = self.idx0 + i
        x = self.probs[t].astype(np.float32)
        y = self.probs[t + 1].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


# -----------------------------
# Model: Encoder -> K -> Decoder
# -----------------------------
class MLPEncoder(nn.Module):
    def __init__(self, nx: int, hidden: int, latent_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(nx),
            nn.Linear(nx, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p)


class KoopmanLinear(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        # z_next = K z, implemented as Linear without bias
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.K(z)


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden: int, nx: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, nx),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.net(z)
        p = F.softmax(logits, dim=-1)
        return p


class KoopmanPDFModel(nn.Module):
    def __init__(self, nx: int, hidden: int, latent_dim: int, dropout: float):
        super().__init__()
        self.enc = MLPEncoder(nx, hidden, latent_dim, dropout)
        self.koop = KoopmanLinear(latent_dim)
        self.dec = MLPDecoder(latent_dim, hidden, nx, dropout)

    def forward(self, p_t: torch.Tensor) -> torch.Tensor:
        z = self.enc(p_t)
        z1 = self.koop(z)
        p1 = self.dec(z1)
        return p1


# -----------------------------
# Loss
# -----------------------------
def cross_entropy_probs(p_true: torch.Tensor, p_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    p_true, p_pred: [B, Nx], both probability vectors that sum to 1.
    CE = -sum p_true log p_pred
    """
    p_pred = torch.clamp(p_pred, eps, 1.0)
    return -(p_true * torch.log(p_pred)).sum(dim=-1).mean()


# -----------------------------
# Train / Eval
# -----------------------------
@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        yhat = model(x)
        loss = cross_entropy_probs(y, yhat)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def train():
    cfg = Cfg()
    set_seed(cfg.seed)
    out_dir = ensure_dir(cfg.out_dir)

    # --- load data ---
    root = Path(cfg.root)
    pdfs = np.load(root / cfg.pdfs_path)      # density [T,Nx]
    times = np.load(root / cfg.times_path)    # datetime64[ns]
    closes = np.load(root / cfg.closes_path)  # [T]

    # dx from xgrid (any month is fine)
    xgrid = np.load(root / cfg.xgrid_month_example)
    dx = float(xgrid[1] - xgrid[0])

    assert pdfs.shape[1] == cfg.nx, f"Expected Nx={cfg.nx}, got {pdfs.shape[1]}"

    # --- dedup + sort ---
    times_u, pdfs_u, closes_u = dedup_and_sort(times, pdfs, closes)
    T = len(times_u)

    # convert density -> probability mass
    probs = pdfs_u.astype(np.float64) * dx
    # tiny numeric cleanup
    probs = np.maximum(probs, 0.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    splits = split_indices(T, cfg.train_frac, cfg.val_frac)
    print("T:", T, "splits:", splits)

    # datasets (note: pairs reduce length by 1 inside each split)
    tr0, tr1 = splits["train"]
    va0, va1 = splits["val"]
    te0, te1 = splits["test"]

    ds_tr = KoopmanPairsDataset(probs, tr0, tr1)
    ds_va = KoopmanPairsDataset(probs, va0, va1)
    ds_te = KoopmanPairsDataset(probs, te0, te1)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True)

    # --- model ---
    model = KoopmanPDFModel(nx=cfg.nx, hidden=cfg.hidden, latent_dim=cfg.latent_dim, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_path = out_dir / "best.pt"
    bad = 0

    # save run meta
    run_meta = {
        "cfg": cfg.__dict__,
        "dx": dx,
        "T_raw": int(len(times)),
        "T_dedup": int(T),
        "splits": {k: [int(a), int(b)] for k, (a, b) in splits.items()},
        "data_files": {
            "pdfs": str(root / cfg.pdfs_path),
            "times": str(root / cfg.times_path),
            "closes": str(root / cfg.closes_path),
            "xgrid_example": str(root / cfg.xgrid_month_example),
        },
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # --- training loop ---
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        losses = []
        for x, y in dl_tr:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            yhat = model(x)
            loss = cross_entropy_probs(y, yhat)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            losses.append(loss.item())

        tr_loss = float(np.mean(losses)) if losses else float("nan")
        va_loss = eval_epoch(model, dl_va, cfg.device)

        print(f"epoch {epoch:03d} | train {tr_loss:.6f} | val {va_loss:.6f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            bad = 0
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "dx": dx}, best_path)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"Early stopping. Best val={best_val:.6f}")
                break

    # --- test ---
    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    te_loss = eval_epoch(model, dl_te, cfg.device)
    print(f"Test CE: {te_loss:.6f}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"best_val_ce": best_val, "test_ce": te_loss}, f, indent=2)

    print("Saved:", best_path)
    print("Run dir:", out_dir)


if __name__ == "__main__":
    train()