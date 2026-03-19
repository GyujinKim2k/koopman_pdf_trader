import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNNEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, latent_dim),
        )

    def forward(self, x):
        return self.net(x)

class SmallCNNDecoder(nn.Module):
    def __init__(self, latent_dim: int, ny: int, nx: int):
        super().__init__()
        self.ny, self.nx = ny, nx
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 32 * 8 * 8), nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 32, 8, 8)
        # upsample to target grid
        h = F.interpolate(h, size=(self.ny, self.nx), mode="bilinear", align_corners=False)
        logits = self.conv(h)  # [B,1,Ny,Nx]
        # enforce PDF via softmax over grid cells
        B = logits.size(0)
        pdf = F.softmax(logits.view(B, -1), dim=-1).view_as(logits)
        return pdf

class KoopmanPDF(nn.Module):
    def __init__(self, ny: int, nx: int, latent_dim: int):
        super().__init__()
        self.encoder = SmallCNNEncoder(latent_dim)
        self.decoder = SmallCNNDecoder(latent_dim, ny, nx)
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.normal_(self.K.weight, mean=0.0, std=0.01)

    def forward(self, rho_t):
        z_t = self.encoder(rho_t)
        z_tp1 = self.K(z_t)
        rho_hat = self.decoder(z_tp1)
        return rho_hat, z_t, z_tp1