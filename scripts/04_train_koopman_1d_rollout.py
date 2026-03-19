from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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
    val_frac: float = 0.15

    # model
    nx: int = 512
    latent_dim: int = 64
    hidden: int = 256
    dropout: float = 0.05

    # rollout
    H: int = 6  # predict next H steps (30m each). H=6 => 3 hours
    w_recon: float = 1.0
    w_roll: float = 1.0

    # training
    batch_size: int = 256
    num_workers: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 60
    grad_clip: float = 1.0
    patience: int = 8

    # stabilize K
    spectral_clip: float = 1.0
    power_iters: int = 8  # for spectral norm estimate

    # misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    out_dir: str = "runs/koopman_1d_v2_rollout"


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


def dedup_and_sort(times: np.ndarray, pdfs: np.ndarray, closes: np.ndarray):
    order = np.argsort(times)
    times_s = times[order]
    pdfs_s = pdfs[order]
    closes_s = closes[order]
    uniq_times, uniq_idx = np.unique(times_s, return_index=True)
    return uniq_times, pdfs_s[uniq_idx], closes_s[uniq_idx]


def split_indices(T: int, train_frac: float, val_frac: float) -> Dict[str, Tuple[int, int]]:
    n_train = int(T * train_frac)
    n_val = int(T * val_frac)
    return {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, T)}


class KoopmanRolloutDataset(Dataset):
    """
    returns p_t and future sequence p_{t+1:t+H}
    """
    def __init__(self, probs: np.ndarray, idx0: int, idx1: int, H: int):
        self.probs = probs
        self.idx0 = idx0
        self.idx1 = idx1
        self.H = H
        # need up to t+H within range
        self.N = max(0, (idx1 - idx0) - H)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        t = self.idx0 + i
        x = self.probs[t].astype(np.float32)                       # [Nx]
        y = self.probs[t + 1 : t + 1 + self.H].astype(np.float32)  # [H, Nx]
        return torch.from_numpy(x), torch.from_numpy(y)


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
        return F.softmax(logits, dim=-1)


class KoopmanPDFModel(nn.Module):
    def __init__(self, nx: int, hidden: int, latent_dim: int, dropout: float):
        super().__init__()
        self.enc = MLPEncoder(nx, hidden, latent_dim, dropout)
        self.koop = KoopmanLinear(latent_dim)
        self.dec = MLPDecoder(latent_dim, hidden, nx, dropout)

    def encode(self, p: torch.Tensor) -> torch.Tensor:
        return self.enc(p)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def step(self, z: torch.Tensor) -> torch.Tensor:
        return self.koop(z)


def ce_probs(p_true: torch.Tensor, p_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p_pred = torch.clamp(p_pred, eps, 1.0)
    return -(p_true * torch.log(p_pred)).sum(dim=-1).mean()


@torch.no_grad()
def eval_epoch(model: KoopmanPDFModel, loader: DataLoader, device: str, H: int, w_recon: float, w_roll: float) -> float:
    model.eval()
    losses = []
    for p0, pfut in loader:
        p0 = p0.to(device)           # [B,Nx]
        pfut = pfut.to(device)       # [B,H,Nx]

        z0 = model.encode(p0)
        # recon loss
        p0_hat = model.decode(z0)
        loss_recon = ce_probs(p0, p0_hat)

        # rollout loss
        z = z0
        roll = 0.0
        for k in range(H):
            z = model.step(z)
            pk_hat = model.decode(z)
            roll = roll + ce_probs(pfut[:, k, :], pk_hat)
        loss_roll = roll / H

        loss = w_recon * loss_recon + w_roll * loss_roll
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def spectral_norm_estimate(W: torch.Tensor, iters: int = 8) -> torch.Tensor:
    """
    Power iteration estimate of spectral norm of matrix W (2D).
    """
    # W: [d,d]
    d = W.shape[0]
    v = torch.randn(d, device=W.device)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        v = W.T @ (W @ v)
        v = v / (v.norm() + 1e-12)
    # Rayleigh quotient sqrt(v^T W^T W v)
    sigma2 = (v @ (W.T @ (W @ v))).clamp(min=0.0)
    return torch.sqrt(sigma2 + 1e-12)


@torch.no_grad()
def project_K_spectral(model: KoopmanPDFModel, clip: float, iters: int):
    W = model.koop.K.weight  # [d,d]
    sig = spectral_norm_estimate(W, iters=iters)
    if sig > clip:
        model.koop.K.weight.copy_(W * (clip / sig))


def train():
    cfg = Cfg()
    set_seed(cfg.seed)
    out_dir = ensure_dir(cfg.out_dir)

    root = Path(cfg.root)
    pdfs = np.load(root / cfg.pdfs_path)      # density [T,Nx]
    times = np.load(root / cfg.times_path)
    closes = np.load(root / cfg.closes_path)

    xgrid = np.load(root / cfg.xgrid_month_example)
    dx = float(xgrid[1] - xgrid[0])

    times_u, pdfs_u, closes_u = dedup_and_sort(times, pdfs, closes)
    probs = pdfs_u.astype(np.float64) * dx
    probs = np.maximum(probs, 0.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    T = len(probs)
    splits = split_indices(T, cfg.train_frac, cfg.val_frac)
    print("T:", T, "splits:", splits)

    tr0, tr1 = splits["train"]
    va0, va1 = splits["val"]
    te0, te1 = splits["test"]

    ds_tr = KoopmanRolloutDataset(probs, tr0, tr1, cfg.H)
    ds_va = KoopmanRolloutDataset(probs, va0, va1, cfg.H)
    ds_te = KoopmanRolloutDataset(probs, te0, te1, cfg.H)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True)

    model = KoopmanPDFModel(cfg.nx, cfg.hidden, cfg.latent_dim, cfg.dropout).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_path = out_dir / "best.pt"
    bad = 0

    with open(out_dir / "run_meta.json", "w") as f:
        json.dump({"cfg": cfg.__dict__, "dx": dx, "splits": splits}, f, indent=2)

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        losses = []
        for p0, pfut in dl_tr:
            p0 = p0.to(cfg.device)      # [B,Nx]
            pfut = pfut.to(cfg.device)  # [B,H,Nx]

            z0 = model.encode(p0)

            # recon
            p0_hat = model.decode(z0)
            loss_recon = ce_probs(p0, p0_hat)

            # rollout
            z = z0
            roll = 0.0
            for k in range(cfg.H):
                z = model.step(z)
                pk_hat = model.decode(z)
                roll = roll + ce_probs(pfut[:, k, :], pk_hat)
            loss_roll = roll / cfg.H

            loss = cfg.w_recon * loss_recon + cfg.w_roll * loss_roll

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            # stabilize K
            project_K_spectral(model, cfg.spectral_clip, cfg.power_iters)

            losses.append(loss.item())

        tr_loss = float(np.mean(losses)) if losses else float("nan")
        va_loss = eval_epoch(model, dl_va, cfg.device, cfg.H, cfg.w_recon, cfg.w_roll)
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

    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    te_loss = eval_epoch(model, dl_te, cfg.device, cfg.H, cfg.w_recon, cfg.w_roll)
    print(f"Test total loss: {te_loss:.6f}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"best_val_total": best_val, "test_total": te_loss}, f, indent=2)

    print("Saved:", best_path)
    print("Run dir:", out_dir)


if __name__ == "__main__":
    train()