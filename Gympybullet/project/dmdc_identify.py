import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


STATE_COLS = ["x", "y", "z", "vx", "vy", "vz"]
INPUT_COLS = ["rpm1", "rpm2", "rpm3", "rpm4"]


def fit_dmdc(X, X_next, U, rank=None):
    """
    Learns X_next ≈ A @ X + B @ U using SVD-based DMDc.
    X, X_next: shape (n_states, n_samples)
    U:          shape (n_inputs, n_samples)
    """
    omega = np.vstack((X, U))

    U_omega, singular_values, Vh_omega = np.linalg.svd(
        omega, full_matrices=False
    )

    if rank is None:
        rank = len(singular_values)

    U_r = U_omega[:, :rank]
    S_r = singular_values[:rank]
    V_r = Vh_omega[:rank, :].T

    G = X_next @ V_r @ np.diag(1.0 / S_r) @ U_r.T

    n_states = X.shape[0]
    A = G[:, :n_states]
    B = G[:, n_states:]

    return A, B, singular_values


def rollout(A, B, initial_state, U):
    """Predicts the entire state trajectory using learned A, B and logged inputs."""
    n_steps = U.shape[1]
    prediction = np.zeros((len(initial_state), n_steps + 1))
    prediction[:, 0] = initial_state

    for k in range(n_steps):
        prediction[:, k + 1] = A @ prediction[:, k] + B @ U[:, k]

    return prediction


def main():
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv("data/pid_dmdc_log.csv")

    # State rows = [x, y, z, vx, vy, vz], each column = one time snapshot
    states = df[STATE_COLS].to_numpy().T

    # Input rows = the four PID-generated motor RPM commands
    inputs = df[INPUT_COLS].to_numpy().T

    # Snapshot matrices: X = current state, X_next = one-step future state
    X = states[:, :-1]
    X_next = states[:, 1:]
    U = inputs[:, :-1]

    # Ten is the maximum meaningful rank here: 6 states + 4 inputs
    A, B, singular_values = fit_dmdc(X, X_next, U, rank=10)

    # One-step prediction: evaluates identification accuracy at each sample
    one_step = A @ X + B @ U

    # Multi-step prediction: starts once, then predicts repeatedly using the input sequence
    predicted = rollout(A, B, states[:, 0], U)
    actual = states

    one_step_rmse = np.sqrt(np.mean((one_step - X_next) ** 2, axis=1))
    rollout_rmse = np.sqrt(np.mean((predicted - actual) ** 2, axis=1))

    print("\nDMDc complete")
    print("State matrix shape:", states.shape)
    print("Input matrix shape:", inputs.shape)
    print("X shape:", X.shape)
    print("U shape:", U.shape)
    print("A shape:", A.shape)
    print("B shape:", B.shape)

    print("\nSingular values of Omega = [X; U]:")
    print(singular_values)

    print("\nOne-step RMSE [x, y, z, vx, vy, vz]:")
    print(one_step_rmse)

    print("\nRollout RMSE [x, y, z, vx, vy, vz]:")
    print(rollout_rmse)

    np.savez(
        "results/dmdc_model.npz",
        A=A,
        B=B,
        singular_values=singular_values,
        state_columns=np.array(STATE_COLS),
        input_columns=np.array(INPUT_COLS)
    )

    time = df["time"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    labels = ["x position (m)", "y position (m)", "z position (m)"]

    for i, ax in enumerate(axes):
        ax.plot(time, actual[i, :], label="PyBullet simulation", linewidth=2)
        ax.plot(time, predicted[i, :], "--", label="DMDc rollout", linewidth=1.8)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("3D Drone Position: PyBullet Simulation vs DMDc Prediction")
    plt.tight_layout()
    plt.savefig("results/dmdc_position_prediction.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    labels = ["vx (m/s)", "vy (m/s)", "vz (m/s)"]

    for i, ax in enumerate(axes):
        state_index = i + 3
        ax.plot(time, actual[state_index, :], label="PyBullet simulation", linewidth=2)
        ax.plot(time, predicted[state_index, :], "--", label="DMDc rollout", linewidth=1.8)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("3D Drone Velocity: PyBullet Simulation vs DMDc Prediction")
    plt.tight_layout()
    plt.savefig("results/dmdc_velocity_prediction.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.semilogy(np.arange(1, len(singular_values) + 1), singular_values, "o-")
    plt.xlabel("Singular-value index")
    plt.ylabel("Singular value (log scale)")
    plt.title("SVD of Combined Drone State and Motor-Input Data")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/dmdc_singular_values.png", dpi=200)
    plt.close()

    print("\nSaved:")
    print("- results/dmdc_model.npz")
    print("- results/dmdc_position_prediction.png")
    print("- results/dmdc_velocity_prediction.png")
    print("- results/dmdc_singular_values.png")


if __name__ == "__main__":
    main()
