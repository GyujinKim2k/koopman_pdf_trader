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


# -----------------------------
# Config
# -----------------------------
@dataclass
class Cfg:
    # data
    root: str = "data/processed/BTCUSDT_2025_1d_kde_pdf_fast"
    pdfs_path: str = "pdfs_2025.npy"     # density [T,Nx]
    times_path: str = "times_2025.npy"
    closes_path: str = "closes_2025.npy"
    xgrid_month_example: str = "2025-01/x_grid_2025-01.npy"  # for dx

    # split
    train_frac: float = 0.70
    val_frac: float = 0.15

    # sequence lengths
    L: int = 48  # past 1 day
    H: int = 6   # predict next 3 hours

    # model dims
    nx: int = 512
    emb_dim: int = 128
    latent_dim: int = 64
    gru_layers: int = 1
    dropout: float = 0.05
    dec_hidden: int = 256

    # training
    batch_size: int = 256
    num_workers: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 60
    grad_clip: float = 1.0
    patience: int = 8

    # stabilize K (recommended)
    spectral_clip: float = 1.0
    power_iters: int = 8

    # misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    out_dir: str = "runs/koopman_seq_gru_v1"


# -----------------------------
# Utils
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


# -----------------------------
# Dataset: (past L -> future H)
# -----------------------------
class Seq2SeqPDFDataset(Dataset):
    """
    Returns:
      X: [L, Nx]   past probs
      Y: [H, Nx]   future probs
    """
    def __init__(self, probs: np.ndarray, idx0: int, idx1: int, L: int, H: int):
        self.probs = probs
        self.idx0 = idx0
        self.idx1 = idx1
        self.L = L
        self.H = H
        # Need t-L+1 >= idx0 and t+H < idx1
        self.t_start = idx0 + (L - 1)
        self.t_end = idx1 - H - 1
        self.N = max(0, self.t_end - self.t_start + 1)

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        t = self.t_start + i
        X = self.probs[t - self.L + 1 : t + 1].astype(np.float32)      # [L,Nx]
        Y = self.probs[t + 1 : t + 1 + self.H].astype(np.float32)      # [H,Nx]
        return torch.from_numpy(X), torch.from_numpy(Y)


# -----------------------------
# Model
# -----------------------------
class PDFEmbed(nn.Module):
    """Per-step embed: Nx -> emb_dim"""
    def __init__(self, nx: int, emb_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(nx),
            nn.Linear(nx, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,Nx] -> [B,L,emb]
        B, L, Nx = x.shape
        y = self.net(x.reshape(B * L, Nx))
        return y.reshape(B, L, -1)


class GRUEncoder(nn.Module):
    """Sequence encoder: embedded sequence -> latent z_t"""
    def __init__(self, emb_dim: int, latent_dim: int, layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=latent_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

    def forward(self, emb_seq: torch.Tensor) -> torch.Tensor:
        # emb_seq: [B,L,emb]
        _, h = self.gru(emb_seq)  # h: [layers, B, latent]
        return h[-1]              # [B, latent]


class KoopmanLinear(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.K(z)


class Decoder(nn.Module):
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


class KoopmanSeqModel(nn.Module):
    def __init__(self, nx: int, emb_dim: int, latent_dim: int, dec_hidden: int, gru_layers: int, dropout: float):
        super().__init__()
        self.embed = PDFEmbed(nx, emb_dim, dropout)
        self.enc = GRUEncoder(emb_dim, latent_dim, gru_layers, dropout)
        self.koop = KoopmanLinear(latent_dim)
        self.dec = Decoder(latent_dim, dec_hidden, nx, dropout)

    def forward(self, X: torch.Tensor, H: int) -> torch.Tensor:
        """
        X: [B,L,Nx]
        Returns preds: [B,H,Nx]
        """
        emb = self.embed(X)
        z = self.enc(emb)  # [B,latent]

        outs = []
        for _ in range(H):
            z = self.koop(z)
            p = self.dec(z)
            outs.append(p)
        return torch.stack(outs, dim=1)  # [B,H,Nx]


# -----------------------------
# Loss / Eval
# -----------------------------
def ce_probs(p_true: torch.Tensor, p_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p_pred = torch.clamp(p_pred, eps, 1.0)
    return -(p_true * torch.log(p_pred)).sum(dim=-1).mean()


def seq_ce(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # y_true/y_pred: [B,H,Nx]
    B, H, Nx = y_true.shape
    return ce_probs(y_true.reshape(B * H, Nx), y_pred.reshape(B * H, Nx))


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: str, H: int) -> float:
    model.eval()
    losses = []
    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)
        Yhat = model(X, H)
        loss = seq_ce(Y, Yhat)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def spectral_norm_estimate(W: torch.Tensor, iters: int = 8) -> torch.Tensor:
    d = W.shape[0]
    v = torch.randn(d, device=W.device)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        v = W.T @ (W @ v)
        v = v / (v.norm() + 1e-12)
    sigma2 = (v @ (W.T @ (W @ v))).clamp(min=0.0)
    return torch.sqrt(sigma2 + 1e-12)


@torch.no_grad()
def project_K_spectral(model: KoopmanSeqModel, clip: float, iters: int):
    W = model.koop.K.weight
    sig = spectral_norm_estimate(W, iters=iters)
    if sig > clip:
        model.koop.K.weight.copy_(W * (clip / sig))


# -----------------------------
# Train
# -----------------------------
def train():
    cfg = Cfg()
    set_seed(cfg.seed)
    out_dir = ensure_dir(cfg.out_dir)

    root = Path(cfg.root)
    pdfs = np.load(root / cfg.pdfs_path)      # density
    times = np.load(root / cfg.times_path)
    closes = np.load(root / cfg.closes_path)

    xgrid = np.load(root / cfg.xgrid_month_example)
    dx = float(xgrid[1] - xgrid[0])

    # dedup + sort
    times_u, pdfs_u, closes_u = dedup_and_sort(times, pdfs, closes)
    T = len(times_u)

    # density -> prob mass
    probs = pdfs_u.astype(np.float64) * dx
    probs = np.maximum(probs, 0.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    splits = split_indices(T, cfg.train_frac, cfg.val_frac)
    print("T:", T, "splits:", splits, "| L:", cfg.L, "H:", cfg.H)

    tr0, tr1 = splits["train"]
    va0, va1 = splits["val"]
    te0, te1 = splits["test"]

    ds_tr = Seq2SeqPDFDataset(probs, tr0, tr1, cfg.L, cfg.H)
    ds_va = Seq2SeqPDFDataset(probs, va0, va1, cfg.L, cfg.H)
    ds_te = Seq2SeqPDFDataset(probs, te0, te1, cfg.L, cfg.H)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True)

    model = KoopmanSeqModel(
        nx=cfg.nx, emb_dim=cfg.emb_dim, latent_dim=cfg.latent_dim,
        dec_hidden=cfg.dec_hidden, gru_layers=cfg.gru_layers, dropout=cfg.dropout
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_path = out_dir / "best.pt"
    bad = 0

    with open(out_dir / "run_meta.json", "w") as f:
        json.dump({"cfg": cfg.__dict__, "dx": dx, "splits": splits, "T": T}, f, indent=2)

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        losses = []
        for X, Y in dl_tr:
            X = X.to(cfg.device)
            Y = Y.to(cfg.device)

            Yhat = model(X, cfg.H)
            loss = seq_ce(Y, Yhat)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            project_K_spectral(model, cfg.spectral_clip, cfg.power_iters)

            losses.append(loss.item())

        tr_loss = float(np.mean(losses)) if losses else float("nan")
        va_loss = eval_epoch(model, dl_va, cfg.device, cfg.H)
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
    te_loss = eval_epoch(model, dl_te, cfg.device, cfg.H)
    print(f"Test CE (avg over H steps): {te_loss:.6f}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"best_val_ce": best_val, "test_ce": te_loss}, f, indent=2)

    print("Saved:", best_path)
    print("Run dir:", out_dir)


if __name__ == "__main__":
    train()