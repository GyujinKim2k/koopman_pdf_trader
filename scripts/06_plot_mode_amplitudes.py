import numpy as np
import matplotlib.pyplot as plt

d = "runs/koopman_seq_gru_v1/modes"
V = np.load(f"{d}/eigvecs.npy")  # complex
condV = np.linalg.cond(V)
print("cond(V) =", condV)


import numpy as np

d = "runs/koopman_seq_gru_v1/modes"
K = np.load(f"{d}/K.npy").astype(np.float64)
Z = np.load(f"{d}/Z_latent.npy").astype(np.float64)  # [T,d]

eigvals, Vr = np.linalg.eig(K)        # right eigenvectors
eigvals_l, Vl = np.linalg.eig(K.T)    # left eigenvectors of K (columns)

# Match ordering (usually fine if K is diagonalizable; otherwise we match by eigvals proximity)
# We'll match by nearest eigvalue
order = np.argsort(np.abs(eigvals))[::-1]
eigvals = eigvals[order]
Vr = Vr[:, order]

# Build W^T from left eigenvectors:
# Vl columns are eigenvectors of K^T, so rows of W^T correspond to those eigenvectors transposed.
# We'll match left vectors to right eigenvalues by nearest value
W = np.zeros_like(Vr, dtype=np.complex128)
for j, lam in enumerate(eigvals):
    k = np.argmin(np.abs(eigvals_l - lam))
    w = Vl[:, k].astype(np.complex128)
    v = Vr[:, j].astype(np.complex128)
    # normalize so w^T v = 1
    s = (w.conj().T @ v)
    if np.abs(s) < 1e-12:
        s = 1.0
    W[:, j] = w / s

# amplitudes a_t = W^H z_t  (H = conjugate transpose)
A = (Z.astype(np.complex128) @ W.conj())  # [T,d]

print("Top |eigval|:", np.abs(eigvals[:10]))
print("Amplitude shape:", A.shape)

d = "runs/koopman_seq_gru_v1/modes"
eigvals = np.load(f"{d}/eigvals.npy")
A = np.load(f"{d}/A_amplitudes.npy")     # [Nz, d] complex
times = np.load(f"{d}/times_z.npy")

# pick top modes by |eigval| (already sorted). skip mode 0 if it dominates.
modes = [0, 1, 2, 3, 4]

plt.figure(figsize=(12, 5))
for j in modes:
    plt.plot(np.abs(A[:, j]), label=f"mode {j} |λ|={abs(eigvals[j]):.3f}")
plt.legend()
plt.title("Koopman mode amplitude magnitudes |a_t|")
plt.xlabel("time index")
plt.ylabel("|a_t|")
plt.tight_layout()
plt.savefig("plots/mode_amplitudes.png",dpi=200)
plt.show()