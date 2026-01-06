import torch
from torch.utils.data import Dataset

class PdfPairDataset(Dataset):
    def __init__(self, pdfs: torch.Tensor):
        """
        pdfs: [T, 1, Ny, Nx]
        """
        self.pdfs = pdfs

    def __len__(self):
        return self.pdfs.shape[0] - 1

    def __getitem__(self, idx):
        x = self.pdfs[idx]     # rho_t
        y = self.pdfs[idx + 1] # rho_{t+1}
        return x, y