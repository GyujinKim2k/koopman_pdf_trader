import torch

def kl_divergence(p_true, p_pred, eps=1e-12):
    # p_true, p_pred: [B,1,Ny,Nx], sums to 1
    p = torch.clamp(p_true, eps, 1.0)
    q = torch.clamp(p_pred, eps, 1.0)
    return (p * (p.log() - q.log())).sum(dim=(1,2,3)).mean()