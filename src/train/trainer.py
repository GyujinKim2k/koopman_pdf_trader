import torch
from torch.utils.data import DataLoader
from .losses import kl_divergence

def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_hat, _, _ = model(x)
        loss = kl_divergence(y, y_hat)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_hat, _, _ = model(x)
        loss = kl_divergence(y, y_hat)
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)