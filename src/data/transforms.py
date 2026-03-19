import numpy as np

class Log1pZScore:
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, vol: np.ndarray):
        v = np.log1p(vol.astype(np.float64))
        self.mu = v.mean()
        self.sigma = v.std() + 1e-8
        return self

    def transform(self, vol: np.ndarray):
        v = np.log1p(vol.astype(np.float64))
        z = (v - self.mu) / self.sigma
        return z.astype(np.float32)

    def fit_transform(self, vol: np.ndarray):
        self.fit(vol)
        return self.transform(vol)