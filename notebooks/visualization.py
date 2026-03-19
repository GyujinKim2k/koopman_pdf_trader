import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----- paths -----
root = "data/processed/BTCUSDT_2025_1d_kde_pdf_fast"
month = "2025-01"

pdfs = np.load(f"{root}/{month}/pdfs_{month}.npy")      # [T, Nx]
times = np.load(f"{root}/{month}/times_{month}.npy")    # [T]
xgrid = np.load(f"{root}/{month}/x_grid_{month}.npy")   # [Nx]

# ----- cut x-range -----
x_min_vis, x_max_vis = -0.015, 0.015
mask = (xgrid >= x_min_vis) & (xgrid <= x_max_vis)

xg = xgrid[mask]
pdfs_cut = pdfs[:, mask]

# ----- downsample time for performance/clarity -----
stride = 72   # every 2 hours (30m * 4)
idx = np.arange(0, pdfs_cut.shape[0], stride)

pdfs_ds = pdfs_cut[idx]

# ----- mesh for surface -----
X, Z = np.meshgrid(xg, np.arange(len(idx)))
Y = pdfs_ds

# ----- plot -----
fig = plt.figure(figsize=(13, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    X, Z, Y,
    rstride=1, cstride=1,
    linewidth=0,
    antialiased=True,
    cmap="cividis"
)

ax.set_xlabel("x = log(price / close)")
ax.set_ylabel("time index (downsampled)")
ax.set_zlabel("probability density")

ax.set_title(f"Volume-weighted PDF surface ({month})\n"
             f"x ∈ [{x_min_vis}, {x_max_vis}], stride={stride}")

plt.savefig("plots/3d_pdf_test.png",dpi=200)
plt.show()

dx = float(xg[1] - xg[0])
mean_x = (pdfs_cut * xgrid[mask]).sum(axis=1) * dx

plt.figure(figsize=(10, 3))
plt.plot(mean_x, lw=1)
plt.axhline(0, color="k", ls="--", alpha=0.5)
plt.title(f"Mean relative price E[x] ({month})")
plt.ylabel("E[x]")
plt.xlabel("30m window index")
plt.tight_layout()
plt.savefig("plots/mean_relative_price.png",dpi=200)
plt.show()