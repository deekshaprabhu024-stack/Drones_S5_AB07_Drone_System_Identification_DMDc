"""
Kalman Filter on Noisy Drake Flight Data
===========================================
Pipeline: load clean Drake data -> inject sensor noise -> identify a plain
linear (A,B) via DMDc on the noisy data -> run a Kalman filter using that
(A,B) as the process model -> compare raw-noisy vs filtered error.

Uses CONSTANT Q/R (standard KF assumption) -- see docs/report_sections.md
for discussion of time-varying covariance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def inject_noise(X, Xprime, noise_std=0.01, seed=42):
    rng = np.random.default_rng(seed)
    X_noisy = X + rng.normal(0, noise_std, X.shape)
    Xprime_noisy = Xprime + rng.normal(0, noise_std, Xprime.shape)
    return X_noisy, Xprime_noisy


def identify_linear_model(X_noisy, Xprime_noisy, U):
    n = X_noisy.shape[0]
    Omega = np.vstack([X_noisy, U])
    G = Xprime_noisy @ np.linalg.pinv(Omega)
    A = G[:, :n]
    B = G[:, n:]
    return A, B


def kalman_filter(X_noisy, U, A, B, Q_proc, R_meas):
    n = X_noisy.shape[0]
    N = X_noisy.shape[1]
    x_est = np.zeros((n, N))
    P = np.eye(n)
    x_est[:, 0] = X_noisy[:, 0]

    for k in range(N - 1):
        x_pred = A @ x_est[:, k] + B @ U[:, k]
        P_pred = A @ P @ A.T + Q_proc

        z = X_noisy[:, k + 1]
        K_gain = P_pred @ np.linalg.inv(P_pred + R_meas)
        x_est[:, k + 1] = x_pred + K_gain @ (z - x_pred)
        P = (np.eye(n) - K_gain) @ P_pred

    return x_est


def main():
    X = np.load(os.path.join(DATA_DIR, "X_drake.npy"))
    Xprime = np.load(os.path.join(DATA_DIR, "Xprime_drake.npy"))
    U = np.load(os.path.join(DATA_DIR, "U_drake.npy"))

    X_noisy, Xprime_noisy = inject_noise(X, Xprime, noise_std=0.01)
    np.save(os.path.join(DATA_DIR, "X_drake_noisy.npy"), X_noisy)
    np.save(os.path.join(DATA_DIR, "Xprime_drake_noisy.npy"), Xprime_noisy)

    A, B = identify_linear_model(X_noisy, Xprime_noisy, U)

    n = X.shape[0]
    Q_proc = np.eye(n) * 0.001
    R_meas = np.eye(n) * 0.01

    x_filtered = kalman_filter(X_noisy, U, A, B, Q_proc, R_meas)
    np.save(os.path.join(DATA_DIR, "x_kalman_filtered.npy"), x_filtered)

    N_common = min(X_noisy.shape[1], Xprime_noisy.shape[1], x_filtered.shape[1])
    raw_err = np.sqrt(np.mean((X_noisy[:, :N_common] - Xprime_noisy[:, :N_common]) ** 2))
    filt_err = np.sqrt(np.mean((x_filtered[:, :N_common] - Xprime_noisy[:, :N_common]) ** 2))

    print(f"[Kalman] Raw noisy RMSE       = {raw_err:.6f}")
    print(f"[Kalman] Filtered RMSE        = {filt_err:.6f}")
    print(f"[Kalman] Improvement          = {(1 - filt_err/max(raw_err,1e-12))*100:.1f}%")
    print("[Kalman] NOTE: Q, R assumed CONSTANT. Real sensor noise can vary with "
          "flight regime -- an Extended/Adaptive KF would be needed for that. "
          "See docs/report_sections.md.")

    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.figure()
    plt.plot(X_noisy[0, :], label="x noisy (raw)", alpha=0.6)
    plt.plot(x_filtered[0, :], label="x Kalman-filtered")
    plt.xlabel("sample")
    plt.ylabel("x position (m)")
    plt.title("Kalman filter denoising (x-position)")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "kalman_result.png"))
    plt.close()
    print(f"[Kalman] Plot saved to plots/kalman_result.png")


if __name__ == "__main__":
    main()
