from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Config (match your training)
# -----------------------------
@dataclass
class Cfg:
    # data
    data_root: str = "data/processed/BTCUSDT_2025_1d_kde_pdf_fast"
    pdfs_path: str = "pdfs_2025.npy"     # density [T,Nx]
    times_path: str = "times_2025.npy"
    closes_path: str = "closes_2025.npy"
    xgrid_month_example: str = "2025-01/x_grid_2025-01.npy"  # for dx

    # model checkpoint
    ckpt_path: str = "runs/koopman_seq_gru_v1/best.pt"

    # sequence length
    L: int = 48

    # dims (must match training cfg)
    nx: int = 512
    emb_dim: int = 128
    latent_dim: int = 64
    gru_layers: int = 1
    dropout: float = 0.05
    dec_hidden: int = 256  # not needed for extraction but included for state_dict compatibility

    # compute
    batch_size: int = 1024
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # output
    out_dir: str = "runs/koopman_seq_gru_v1/modes"


# -----------------------------
# Utilities
# -----------------------------
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


# -----------------------------
# Model definition (same as training)
# -----------------------------
class PDFEmbed(nn.Module):
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
        # x: [B,L,Nx]
        B, L, Nx = x.shape
        y = self.net(x.reshape(B * L, Nx))
        return y.reshape(B, L, -1)


class GRUEncoder(nn.Module):
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
        _, h = self.gru(emb_seq)
        return h[-1]  # [B, latent]


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

    @torch.no_grad()
    def encode_sequence(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B,L,Nx] probabilities
        returns z_t: [B,latent]
        """
        emb = self.embed(X)
        z = self.enc(emb)
        return z


# -----------------------------
# Main
# -----------------------------
def build_probabilities(cfg: Cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    root = Path(cfg.data_root)
    pdfs = np.load(root / cfg.pdfs_path)      # density
    times = np.load(root / cfg.times_path)
    closes = np.load(root / cfg.closes_path)

    xgrid = np.load(root / cfg.xgrid_month_example)
    dx = float(xgrid[1] - xgrid[0])

    times_u, pdfs_u, closes_u = dedup_and_sort(times, pdfs, closes)

    probs = pdfs_u.astype(np.float64) * dx
    probs = np.maximum(probs, 0.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    return probs.astype(np.float32), times_u, closes_u.astype(np.float32), dx


def main():
    cfg = Cfg()
    out_dir = ensure_dir(cfg.out_dir)

    # --- load data ---
    probs, times_u, closes_u, dx = build_probabilities(cfg)
    T = probs.shape[0]
    print("T(dedup):", T, "Nx:", probs.shape[1], "dx:", dx)

    # valid t indices for z_t (need history length L)
    t0 = cfg.L - 1
    t1 = T - 1
    Nz = (t1 - t0 + 1)
    print("Latent states to compute:", Nz, f"(t={t0}..{t1})")

    # --- load model ---
    ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device)
    model = KoopmanSeqModel(
        nx=cfg.nx, emb_dim=cfg.emb_dim, latent_dim=cfg.latent_dim,
        dec_hidden=cfg.dec_hidden, gru_layers=cfg.gru_layers, dropout=cfg.dropout
    ).to(cfg.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- compute latent z_t in batches ---
    Z = np.zeros((Nz, cfg.latent_dim), dtype=np.float32)

    bs = cfg.batch_size
    with torch.no_grad():
        out_i = 0
        for start in range(t0, t1 + 1, bs):
            end = min(t1 + 1, start + bs)
            B = end - start

            # build batch X: [B,L,Nx]
            X = np.stack([probs[t - cfg.L + 1 : t + 1] for t in range(start, end)], axis=0)
            X_t = torch.from_numpy(X).to(cfg.device)

            z = model.encode_sequence(X_t).detach().cpu().numpy().astype(np.float32)
            Z[out_i : out_i + B] = z
            out_i += B

            if out_i % (10 * bs) == 0 or end == t1 + 1:
                print(f"encoded {out_i}/{Nz}")

    # times aligned to z_t
    times_z = times_u[t0 : t1 + 1]
    closes_z = closes_u[t0 : t1 + 1]

    # --- extract Koopman matrix K and eigendecompose ---
    K = model.koop.K.weight.detach().cpu().numpy().astype(np.float64)  # [d,d]
    eigvals, eigvecs = np.linalg.eig(K)  # eigvecs columns are eigenvectors

    # --- compute amplitudes a_t = V^{-1} z_t (complex) ---
    # Use solve for numerical stability: V * a = z
    V = eigvecs.astype(np.complex128)
    Zc = Z.astype(np.complex128)  # [Nz,d]
    A = np.zeros_like(Zc, dtype=np.complex128)
    for i in range(Nz):
        A[i] = np.linalg.solve(V, Zc[i])

    # sort modes by |eigval| descending (common for analysis)
    order = np.argsort(np.abs(eigvals))[::-1]
    eigvals_s = eigvals[order]
    eigvecs_s = eigvecs[:, order]
    A_s = A[:, order]

    # --- save ---
    np.save(out_dir / "K.npy", K.astype(np.float32))
    np.save(out_dir / "eigvals.npy", eigvals_s.astype(np.complex128))
    np.save(out_dir / "eigvecs.npy", eigvecs_s.astype(np.complex128))
    np.save(out_dir / "Z_latent.npy", Z.astype(np.float32))
    np.save(out_dir / "A_amplitudes.npy", A_s.astype(np.complex128))
    np.save(out_dir / "times_z.npy", times_z.astype("datetime64[ns]"))
    np.save(out_dir / "closes_z.npy", closes_z.astype(np.float32))

    meta = {
        "ckpt": cfg.ckpt_path,
        "T_dedup": int(T),
        "L": int(cfg.L),
        "Nz": int(Nz),
        "latent_dim": int(cfg.latent_dim),
        "sorting": "modes sorted by |eigval| descending",
        "files": {
            "K": str(out_dir / "K.npy"),
            "eigvals": str(out_dir / "eigvals.npy"),
            "eigvecs": str(out_dir / "eigvecs.npy"),
            "Z_latent": str(out_dir / "Z_latent.npy"),
            "A_amplitudes": str(out_dir / "A_amplitudes.npy"),
            "times_z": str(out_dir / "times_z.npy"),
            "closes_z": str(out_dir / "closes_z.npy"),
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved modes to:", out_dir)
    # quick summary
    print("Top 10 |eigval|:", np.abs(eigvals_s[:10]))


if __name__ == "__main__":
    main()