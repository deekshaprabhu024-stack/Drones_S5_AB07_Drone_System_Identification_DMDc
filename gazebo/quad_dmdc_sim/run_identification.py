"""
run_identification.py
----------------------
Loads data/collected_data.npz (produced by data_collection.py while
Gazebo was running), fits A and B via SVD-based DMDc, validates on a
held-out portion of the trajectory, and saves plots + the identified
model. This step has no dependency on ROS 2 / Gazebo and can be run
completely offline once data has been collected.

Run with:

    ros2 run quad_dmdc_sim run_identification
    # or, without ROS installed at all:
    python3 -m quad_dmdc_sim.run_identification
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from .dmdc import fit_dmdc, predict, rmse

DATA_DIR = "data"
MODEL_DIR = "models_out"
PLOT_DIR = "plots"

STATE_LABELS = ["x", "y", "z", "vx", "vy", "vz"]


def ensure_dirs():
    for d in (DATA_DIR, MODEL_DIR, PLOT_DIR):
        os.makedirs(d, exist_ok=True)


def main():
    ensure_dirs()

    data_path = os.path.join(DATA_DIR, "collected_data.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"'{data_path}' not found. Run data collection first:\n"
            "  ros2 launch quad_dmdc_sim sim.launch.py gui:=false\n"
            "  ros2 run quad_dmdc_sim data_collection"
        )

    npz = np.load(data_path)
    X, X_next, U = npz["X"], npz["X_next"], npz["U"]
    print(f"Loaded data: X{X.shape}, X_next{X_next.shape}, U{U.shape}")

    print("Fitting DMDc model via SVD...")
    A, B, s = fit_dmdc(X, X_next, U)
    print(f"A shape: {A.shape}, B shape: {B.shape}")

    np.savez(os.path.join(MODEL_DIR, "dmdc_model.npz"), A=A, B=B, singular_values=s)

    # ---- Validation on held-out second half of the trajectory ----
    split = X.shape[1] // 2
    x0 = X[:, split]
    U_test = U[:, split:]
    X_true = np.hstack([X[:, split:split + 1], X_next[:, split:]])

    X_pred = predict(A, B, x0, U_test)

    error = rmse(X_true, X_pred)
    print(f"Prediction RMSE over held-out trajectory: {error:.6f}")

    plot_group(X_true, X_pred, [0, 1, 2], "Position", "position.png")
    plot_group(X_true, X_pred, [3, 4, 5], "Velocity", "velocity.png")
    plot_singular_values(s)
    plot_eigenvalues(A)

    print(f"Plots saved in '{PLOT_DIR}/', model saved in '{MODEL_DIR}/'.")


def plot_group(X_true, X_pred, indices, title, filename):
    fig, axes = plt.subplots(len(indices), 1, figsize=(8, 8), sharex=True)
    for ax, i in zip(axes, indices):
        ax.plot(X_true[i, :], label="true")
        ax.plot(X_pred[i, :], "--", label="predicted")
        ax.set_ylabel(STATE_LABELS[i])
        ax.legend()
    axes[-1].set_xlabel("time step")
    fig.suptitle(f"{title}: true vs DMDc prediction")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename))
    plt.close(fig)


def plot_singular_values(s):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(s, "o-")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value (log scale)")
    ax.set_title("Singular values of Omega = [X; U]")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "singular_values.png"))
    plt.close(fig)


def plot_eigenvalues(A):
    eigvals = np.linalg.eigvals(A)
    fig, ax = plt.subplots(figsize=(5, 5))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8)
    ax.scatter(eigvals.real, eigvals.imag, c="tab:red")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("Eigenvalues of identified A (unit circle = discrete stability)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "eigenvalues.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
