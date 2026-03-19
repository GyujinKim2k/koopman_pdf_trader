import numpy as np

def make_2d_pdf(x, y, x_edges, y_edges, smoothing_sigma: float = 0.0):
    """
    x: log(price) samples
    y: normalized volume samples
    returns rho (ny, nx) sum=1
    """
    H, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])  # note: y first
    rho = H.astype(np.float32)
    if smoothing_sigma > 0:
        # optional: simple separable gaussian smoothing
        from scipy.ndimage import gaussian_filter
        rho = gaussian_filter(rho, sigma=smoothing_sigma)
    s = rho.sum()
    if s <= 0:
        # empty window fallback: uniform or previous pdf handled upstream
        rho = np.ones_like(rho, dtype=np.float32)
        rho /= rho.sum()
        return rho
    rho /= s
    return rho