"""
validation_fixes.py -- addresses the three README caveats with evidence.
 1) EDMDc conditioning (max|eig| ~ 9e3): ridge / truncated-SVD regularisation + original-state error
 2) SINDy sparsity (8/8 or 176 active): threshold sweep + normalised library
 3) Kalman: error against CLEAN ground truth (X_drake.npy), not against the next noisy sample
Also: chronological 80/20 train/test split so errors are held-out, and all errors
are reported in ORIGINAL 12-state space so methods are comparable.
Run after drake_sim/circle_tracking.py and kalman_filter.py.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from edmdc import trig_dictionary
from sindy import build_library, stlsq
from kalman_filter import inject_noise, kalman_filter

D = os.path.join(os.path.dirname(__file__), "..", "data")
P = os.path.join(os.path.dirname(__file__), "..", "plots")
X  = np.load(f"{D}/X_drake.npy"); Xp = np.load(f"{D}/Xprime_drake.npy"); U = np.load(f"{D}/U_drake.npy")
N = X.shape[1]; ntr = int(0.8 * N)
tr, te = slice(0, ntr), slice(ntr, N)
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

def ridge_fit(Omega, Y, lam):
    # Y = G Omega ; G = Y Omega^T (Omega Omega^T + lam I)^-1  (columns standardised beforehand)
    return Y @ Omega.T @ np.linalg.inv(Omega @ Omega.T + lam * np.eye(Omega.shape[0]))

# ---------- 1. DMDc / EDMDc held-out, original-state error ----------
def dmdc(lam=0.0):
    Om = np.vstack([X[:, tr], U[:, tr]]); G = ridge_fit(Om, Xp[:, tr], lam) if lam else Xp[:, tr] @ np.linalg.pinv(Om)
    pred = lambda s: G[:, :12] @ X[:, s] + G[:, 12:] @ U[:, s]
    return G, rmse(pred(tr), Xp[:, tr]), rmse(pred(te), Xp[:, te])

def edmdc(lam=0.0):
    lift = lambda A_: trig_dictionary(A_, [3, 4, 5])
    Om = np.vstack([lift(X[:, tr]), U[:, tr]]); Y = lift(Xp[:, tr])
    G = ridge_fit(Om, Y, lam) if lam else Y @ np.linalg.pinv(Om)
    A = G[:, :18]
    pred = lambda s: (G[:, :18] @ lift(X[:, s]) + G[:, 18:] @ U[:, s])[:12]   # first 12 rows = original states
    return np.max(np.abs(np.linalg.eigvals(A))), rmse(pred(tr), Xp[:, tr]), rmse(pred(te), Xp[:, te])

G, tr_d, te_d = dmdc()
print(f"[DMDc ] max|eig|={np.max(np.abs(np.linalg.eigvals(G[:, :12]))):.4f}  train={tr_d:.3e}  test={te_d:.3e}")
lams = [0, 1e-8, 1e-6, 1e-4, 1e-2]
eig_curve = []
for lam in lams:
    e, a, b = edmdc(lam); eig_curve.append(e)
    print(f"[EDMDc] lambda={lam:<7g} max|eig|={e:12.4f} train={a:.3e} test={b:.3e}")

# ---------- 2. SINDy threshold sweep (discrete next-state fit on DELTA = x_{k+1}-x_k) ----------
Th, names = build_library(X[:, tr], U[:, tr], poly_order=1, include_trig_idx=[3, 4, 5])
Th_te, _ = build_library(X[:, te], U[:, te], poly_order=1, include_trig_idx=[3, 4, 5])
sc = np.linalg.norm(Th, axis=1, keepdims=True); sc[sc == 0] = 1
ths = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
act, tes = [], []
for t in ths:
    Xi = stlsq(Th / sc, Xp[:, tr] - X[:, tr], threshold=t)       # normalised library, residual form
    Xi = Xi / sc
    pred_te = X[:, te] + Xi.T @ Th_te
    act.append(int(np.sum(Xi != 0))); tes.append(rmse(pred_te, Xp[:, te]))
    print(f"[SINDy] thr={t:<7g} active={act[-1]:4d}/{Xi.size}  test RMSE={tes[-1]:.3e}")

# ---------- 3. Kalman vs clean truth ----------
Xn, Xpn = inject_noise(X, Xp, 0.01)
Om = np.vstack([Xn, U]); Gk = Xpn @ np.linalg.pinv(Om); A, B = Gk[:, :12], Gk[:, 12:]
xf = kalman_filter(Xn, U, A, B, 0.001 * np.eye(12), 0.01 * np.eye(12))
raw, fil = rmse(Xn, X), rmse(xf, X)
print(f"[Kalman] RMSE vs CLEAN truth: raw={raw:.6f} filtered={fil:.6f} improvement={(1-fil/raw)*100:.1f}%")
per = lambda a: np.sqrt(np.mean((a - X) ** 2, axis=1))
print("per-state raw     :", np.round(per(Xn), 4)); print("per-state filtered:", np.round(per(xf), 4))

# ---------- figure: the three diagnostics ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].semilogx([max(l, 1e-9) for l in lams], eig_curve, "o-"); ax[0].axhline(1, ls="--", c="k")
ax[0].set(xlabel=r"ridge $\lambda$", ylabel=r"max $|\lambda_i(A_\Psi)|$", title="EDMDc conditioning (log-x)"); ax[0].set_yscale("log")
ax[1].semilogx(ths, act, "o-", c="C1"); ax[1].set(xlabel="STLSQ threshold", ylabel="active coefficients", title="SINDy sparsity vs threshold")
ax[2].bar(["raw noisy", "Kalman"], [raw, fil], color=["C3", "C2"]); ax[2].set(ylabel="RMSE vs clean truth", title="Kalman (clean-truth metric)")
plt.tight_layout(); plt.savefig(f"{P}/validation_diagnostics.png", dpi=160)
print("saved plots/validation_diagnostics.png")
