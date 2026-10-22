import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lotka_volterra_rhs(state, alpha, beta, delta, gamma):
    prey, predator = state

    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator

    return np.array([dprey_dt, dpredator_dt])


def simulate_lotka_volterra(
    duration=30.0,
    dt=0.02,
    initial_state=np.array([20.0, 5.0]),
    alpha=1.0,
    beta=0.10,
    delta=0.075,
    gamma=1.5
):
    num_steps = int(duration / dt)
    time = np.arange(num_steps + 1) * dt

    states = np.zeros((num_steps + 1, 2))
    states[0] = initial_state

    # Fourth-order Runge-Kutta integration
    for k in range(num_steps):
        xk = states[k]

        k1 = lotka_volterra_rhs(xk, alpha, beta, delta, gamma)
        k2 = lotka_volterra_rhs(xk + 0.5 * dt * k1, alpha, beta, delta, gamma)
        k3 = lotka_volterra_rhs(xk + 0.5 * dt * k2, alpha, beta, delta, gamma)
        k4 = lotka_volterra_rhs(xk + dt * k3, alpha, beta, delta, gamma)

        states[k + 1] = xk + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Prevent tiny numerical negative values
        states[k + 1] = np.maximum(states[k + 1], 1e-8)

    return time, states


def fit_dmd(X, X_next, rank=None):
    U, singular_values, Vh = np.linalg.svd(X, full_matrices=False)

    if rank is None:
        rank = len(singular_values)

    U_r = U[:, :rank]
    S_r = singular_values[:rank]
    V_r = Vh[:rank, :].T

    A = X_next @ V_r @ np.diag(1.0 / S_r) @ U_r.T

    return A, singular_values


def rollout(A, initial_state, num_steps):
    prediction = np.zeros((len(initial_state), num_steps + 1))
    prediction[:, 0] = initial_state

    for k in range(num_steps):
        prediction[:, k + 1] = A @ prediction[:, k]

    return prediction


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Generate nonlinear Lotka–Volterra data
    time, states = simulate_lotka_volterra()

    df = pd.DataFrame({
        "time": time,
        "prey": states[:, 0],
        "predator": states[:, 1]
    })
    df.to_csv("data/lotka_volterra_data.csv", index=False)

    # 2. Form DMD snapshot matrices
    snapshots = states.T
    X = snapshots[:, :-1]
    X_next = snapshots[:, 1:]

    # 3. Identify best-fit linear DMD model
    A, singular_values = fit_dmd(X, X_next, rank=2)

    # 4. Evaluate one-step prediction
    one_step_prediction = A @ X
    one_step_rmse = np.sqrt(
        np.mean((one_step_prediction - X_next) ** 2, axis=1)
    )

    # 5. Evaluate full open-loop rollout
    predicted = rollout(A, snapshots[:, 0], X.shape[1])
    actual = snapshots
    rollout_rmse = np.sqrt(
        np.mean((predicted - actual) ** 2, axis=1)
    )

    print("\nLotka–Volterra DMD complete")
    print("State matrix shape:", snapshots.shape)
    print("X shape:", X.shape)
    print("A shape:", A.shape)

    print("\nLearned DMD matrix A:")
    print(A)

    print("\nSingular values:")
    print(singular_values)

    print("\nOne-step RMSE [prey, predator]:")
    print(one_step_rmse)

    print("\nRollout RMSE [prey, predator]:")
    print(rollout_rmse)

    np.savetxt(
        "results/lotka_volterra_A_matrix.csv",
        A,
        delimiter=",",
        header="A_matrix",
        comments=""
    )

    np.savez(
        "results/lotka_volterra_dmd_model.npz",
        A=A,
        singular_values=singular_values,
        state_columns=np.array(["prey", "predator"])
    )

    # Plot 1: actual population time series
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(time, actual[0], label="Prey", color="tab:blue", linewidth=2)
    axes[0].set_ylabel("Prey population")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time, actual[1], label="Predator", color="tab:red", linewidth=2)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Predator population")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Lotka–Volterra Nonlinear Predator–Prey Simulation")
    plt.tight_layout()
    plt.savefig("results/lotka_volterra_timeseries.png", dpi=200)
    plt.close()

    # Plot 2: phase portrait
    plt.figure(figsize=(7, 6))
    plt.plot(
        actual[0],
        actual[1],
        label="Nonlinear Lotka–Volterra trajectory",
        linewidth=2
    )
    plt.scatter(actual[0, 0], actual[1, 0], color="green", s=60, label="Initial state")
    plt.xlabel("Prey population")
    plt.ylabel("Predator population")
    plt.title("Lotka–Volterra Phase Portrait")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/lotka_volterra_phase_portrait.png", dpi=200)
    plt.close()

    # Plot 3: DMD rollout compared with nonlinear simulation
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(time, actual[0], label="Actual nonlinear simulation", linewidth=2)
    axes[0].plot(time, predicted[0], "--", label="DMD rollout", linewidth=1.8)
    axes[0].set_ylabel("Prey population")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time, actual[1], label="Actual nonlinear simulation", linewidth=2)
    axes[1].plot(time, predicted[1], "--", label="DMD rollout", linewidth=1.8)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Predator population")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Lotka–Volterra: Nonlinear Simulation vs Standard DMD Prediction")
    plt.tight_layout()
    plt.savefig("results/lotka_volterra_dmd_prediction.png", dpi=200)
    plt.close()

    with open("results/lotka_volterra_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Lotka–Volterra Standard DMD Metrics\n")
        f.write("===================================\n\n")
        f.write(f"One-step RMSE - prey: {one_step_rmse[0]:.6f}\n")
        f.write(f"One-step RMSE - predator: {one_step_rmse[1]:.6f}\n")
        f.write(f"Rollout RMSE - prey: {rollout_rmse[0]:.6f}\n")
        f.write(f"Rollout RMSE - predator: {rollout_rmse[1]:.6f}\n")

    print("\nSaved:")
    print("- data/lotka_volterra_data.csv")
    print("- results/lotka_volterra_A_matrix.csv")
    print("- results/lotka_volterra_dmd_model.npz")
    print("- results/lotka_volterra_timeseries.png")
    print("- results/lotka_volterra_phase_portrait.png")
    print("- results/lotka_volterra_dmd_prediction.png")
    print("- results/lotka_volterra_metrics.txt")


if __name__ == "__main__":
    main()