from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results_koopman"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# DMDc STATE / INPUT
# ============================================================

STATE_COLS = [
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "roll",
    "pitch",
    "yaw",
    "p",
    "q",
    "r",
]

INPUT_COLS = [
    "u1",
    "u2",
    "u3",
    "u4",
]

HOVER_THRUST = 9.81 / 4.0


# ============================================================
# DATA LOADING
# ============================================================

def load_data(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = ["time", *STATE_COLS, *INPUT_COLS]
    missing = [name for name in required if name not in df.columns]

    if missing:
        raise ValueError(
            f"{filename} is missing columns: {missing}"
        )

    df = df[required].copy()

    # Remove non-finite rows.
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # A manual recording can contain more than one recording segment.
    # Keep only the longest strictly increasing-time segment.
    time_values = df["time"].to_numpy(dtype=float)

    if len(time_values) > 1:
        breaks = np.where(np.diff(time_values) <= 0)[0]

        segments = []
        start = 0

        for index in breaks:
            segments.append(df.iloc[start:index + 1])
            start = index + 1

        segments.append(df.iloc[start:])

        df = max(segments, key=len).copy()

    states = df[STATE_COLS].to_numpy(dtype=float).T

    controls = (
        df[INPUT_COLS].to_numpy(dtype=float) - HOVER_THRUST
    ).T

    return df.reset_index(drop=True), states, controls


# ============================================================
# RIDGE DMDc
# ============================================================

def identify_dmdc(X, X_prime, U, ridge=1e-5):
    """
    Identify

        X' = A X + B U

    using normalized, ridge-regularized least squares.
    """

    # --------------------------------------------------------
    # Scaling factors
    # --------------------------------------------------------
    x_scale = np.std(X, axis=1, keepdims=True)
    x_scale = np.maximum(x_scale, 1e-8)

    u_scale = np.std(U, axis=1, keepdims=True)
    u_scale = np.maximum(u_scale, 1e-8)

    # Convert scale vectors into diagonal matrices.
    Dx = np.diagflat(x_scale.ravel())
    Du = np.diagflat(u_scale.ravel())

    # --------------------------------------------------------
    # Normalize data
    # --------------------------------------------------------
    X_scaled = X / x_scale
    X_prime_scaled = X_prime / x_scale
    U_scaled = U / u_scale

    # --------------------------------------------------------
    # Augmented matrix
    # --------------------------------------------------------
    Omega_scaled = np.vstack(
        (
            X_scaled,
            U_scaled,
        )
    )

    # --------------------------------------------------------
    # Ridge regression
    # --------------------------------------------------------
    gram = Omega_scaled @ Omega_scaled.T

    regularization = (
        ridge * np.eye(gram.shape[0])
    )

    G_scaled = (
        X_prime_scaled
        @ Omega_scaled.T
        @ np.linalg.solve(
            gram + regularization,
            np.eye(gram.shape[0]),
        )
    )

    state_dim = X.shape[0]

    A_scaled = G_scaled[:, :state_dim]
    B_scaled = G_scaled[:, state_dim:]

    # --------------------------------------------------------
    # Convert model back to physical coordinates
    #
    # X_scaled = Dx^-1 X
    # U_scaled = Du^-1 U
    #
    # Therefore:
    #
    # A = Dx A_scaled Dx^-1
    # B = Dx B_scaled Du^-1
    # --------------------------------------------------------
    Dx_inv = np.linalg.inv(Dx)
    Du_inv = np.linalg.inv(Du)

    A = (
        Dx
        @ A_scaled
        @ Dx_inv
    )

    B = (
        Dx
        @ B_scaled
        @ Du_inv
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    singular_values = np.linalg.svd(
        Omega_scaled,
        compute_uv=False,
    )

    numerical_rank = np.linalg.matrix_rank(
        Omega_scaled
    )

    return (
        A,
        B,
        Omega_scaled,
        singular_values,
        numerical_rank,
    )
    
def predict_dmdc(A, B, x0, U_sequence):
    predictions = [x0.copy()]
    x = x0.copy()

    for k in range(U_sequence.shape[1]):
        x = A @ x + B @ U_sequence[:, k]
        predictions.append(x.copy())

    return np.column_stack(predictions)


# ============================================================
# KOOPMAN / TRIGONOMETRIC DICTIONARY
# ============================================================

def lift_state(x):
    """
    Trigonometric Koopman dictionary.

    Physical state:
        [z_error, vz, roll, pitch, yaw, p, q, r]

    Lifted observables:
        physical state
        sin/cos of roll
        sin/cos of pitch
        sin/cos of yaw
    """

    roll = x[6]
    pitch = x[7]
    yaw = x[8]

    return np.concatenate(
        [
            x,
            np.array(
                [
                    np.sin(roll),
                    np.cos(roll),
                    np.sin(pitch),
                    np.cos(pitch),
                    np.sin(yaw),
                    np.cos(yaw),
                ],
                dtype=float,
            ),
        ]
    )


def build_lifted_snapshots(X, X_prime):
    Psi = np.column_stack(
        [lift_state(X[:, k]) for k in range(X.shape[1])]
    )

    Psi_prime = np.column_stack(
        [lift_state(X_prime[:, k]) for k in range(X_prime.shape[1])]
    )

    return Psi, Psi_prime


def identify_koopman(Psi, Psi_prime, U, ridge=1e-5):
    """
    Identify the controlled Koopman model

        Psi' = A_K Psi + B_K U

    using ridge-regularized least squares.

    A slightly stronger regularization is used here because
    the trigonometric lifted observables are highly correlated
    with the original angle states.
    """

    Omega = np.vstack((Psi, U))

    gram = Omega @ Omega.T
    regularization = ridge * np.eye(gram.shape[0])

    G = (
        Psi_prime
        @ Omega.T
        @ np.linalg.solve(
            gram + regularization,
            np.eye(gram.shape[0]),
        )
    )

    lifted_dim = Psi.shape[0]

    A_K = G[:, :lifted_dim]
    B_K = G[:, lifted_dim:]

    return A_K, B_K

def build_reconstruction_matrix(X, Psi):
    """
    Find C such that

        X ≈ C Psi

    so lifted predictions can be mapped back to physical states.
    """

    return (
        X
        @ np.linalg.pinv(Psi)
    )


def predict_koopman(A_K, B_K, C, x0, U_sequence):
    psi = lift_state(x0)

    predictions = [x0.copy()]

    for k in range(U_sequence.shape[1]):
        psi = (
            A_K @ psi
            + B_K @ U_sequence[:, k]
        )

        x = C @ psi
        predictions.append(x.copy())

    return np.column_stack(predictions)


# ============================================================
# METRICS
# ============================================================

def calculate_rmse(actual, predicted):
    n = min(actual.shape[1], predicted.shape[1])

    return np.sqrt(
        np.mean(
            (
                actual[:, :n]
                - predicted[:, :n]
            ) ** 2,
            axis=1,
        )
    )


def calculate_relative_error(actual, predicted):
    n = min(actual.shape[1], predicted.shape[1])

    numerator = np.linalg.norm(
        actual[:, :n] - predicted[:, :n],
        axis=1,
    )

    denominator = np.linalg.norm(
        actual[:, :n],
        axis=1,
    )

    denominator = np.maximum(denominator, 1e-12)

    return numerator / denominator

# ============================================================
# ONE-STEP VALIDATION DIAGNOSTICS
# ============================================================

def one_step_dmdc_prediction(A, B, X, U):
    return A @ X + B @ U


def one_step_koopman_prediction(A_K, B_K, C, X, U):
    predictions = []

    for k in range(X.shape[1]):
        psi = lift_state(X[:, k])

        psi_next = (
            A_K @ psi
            + B_K @ U[:, k]
        )

        x_next = C @ psi_next
        predictions.append(x_next)

    return np.column_stack(predictions)


def print_eigenvalue_diagnostics(A, A_K):
    dmdc_eigenvalues = np.linalg.eigvals(A)
    koopman_eigenvalues = np.linalg.eigvals(A_K)

    print()
    print("DMDc eigenvalues")
    print("----------------")
    for value in dmdc_eigenvalues:
        print(
            f"{value.real:+.6f}"
            f"{value.imag:+.6f}j"
            f" |lambda|={abs(value):.6f}"
        )

    print()
    print("Koopman eigenvalues")
    print("-------------------")
    for value in koopman_eigenvalues:
        print(
            f"{value.real:+.6f}"
            f"{value.imag:+.6f}j"
            f" |lambda|={abs(value):.6f}"
        )

    print()
    print(
        "Maximum DMDc eigenvalue magnitude : "
        f"{np.max(np.abs(dmdc_eigenvalues)):.6f}"
    )

    print(
        "Maximum Koopman eigenvalue magnitude : "
        f"{np.max(np.abs(koopman_eigenvalues)):.6f}"
    )


def print_condition_diagnostics(X, U, Psi):
    Omega_dmdc = np.vstack((X, U))
    Omega_koopman = np.vstack((Psi, U))

    print()
    print("Condition diagnostics")
    print("---------------------")

    print(
        "DMDc Omega condition number    : "
        f"{np.linalg.cond(Omega_dmdc):.6e}"
    )

    print(
        "Koopman Omega condition number : "
        f"{np.linalg.cond(Omega_koopman):.6e}"
    )

    print(
        "DMDc Omega numerical rank      : "
        f"{np.linalg.matrix_rank(Omega_dmdc)}"
    )

    print(
        "Koopman Omega numerical rank   : "
        f"{np.linalg.matrix_rank(Omega_koopman)}"
    )

# ============================================================
# PLOTS
# ============================================================

def save_validation_plots(
    time,
    actual,
    dmdc_prediction,
    koopman_prediction,
):
    n = min(
        len(time),
        actual.shape[1],
        dmdc_prediction.shape[1],
        koopman_prediction.shape[1],
    )

    for i, name in enumerate(STATE_COLS):
        plt.figure(figsize=(10, 4))

        plt.plot(
            time[:n],
            actual[i, :n],
            label="MuJoCo actual",
        )

        plt.plot(
            time[:n],
            dmdc_prediction[i, :n],
            "--",
            label="DMDc prediction",
        )

        plt.plot(
            time[:n],
            koopman_prediction[i, :n],
            ":",
            label="Koopman-DMDc prediction",
        )

        plt.xlabel("Time (s)")
        plt.ylabel(name)
        plt.title(f"Validation: {name}")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / f"validation_{name}.png",
            dpi=160,
        )

        plt.close()


def save_xy_trajectory_plot(
    validation_df,
):
    """
    Diagnostic only: the current recorded state does not contain
    x/y position, so this function is intentionally not called.
    The circular trajectory will be added later when the reference
    trajectory is introduced into the MuJoCo recorder.
    """
    return


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print("DMDc + KOOPMAN DEVELOPMENT PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load clean manual experiments
    # --------------------------------------------------------

    training_df, X_train, U_train = load_data(
        "circle_training.csv"
    )

    validation_df, X_validation, U_validation = load_data(
        "circle_validation.csv"
    )

    print()
    print("Training samples  :", len(training_df))
    print("Validation samples:", len(validation_df))

    # --------------------------------------------------------
    # Snapshot matrices
    # --------------------------------------------------------

    X = X_train[:, :-1]
    X_prime = X_train[:, 1:]
    U = U_train[:, :-1]

    print()
    print("Snapshot dimensions")
    print("-------------------")
    print("X       :", X.shape)
    print("X_prime :", X_prime.shape)
    print("U       :", U.shape)

    # --------------------------------------------------------
    # Standard DMDc
    # --------------------------------------------------------

    A, B, Omega, singular_values, omega_rank = identify_dmdc(
        X,
        X_prime,
        U,
    )

    print()
    print("Standard DMDc")
    print("-------------")
    print("Omega shape :", Omega.shape)
    print("Omega rank  :", omega_rank)

    # --------------------------------------------------------
    # Standard DMDc validation
    # --------------------------------------------------------

    U_val = U_validation[:, :-1]
    x0 = X_validation[:, 0]

    dmdc_prediction = predict_dmdc(
        A,
        B,
        x0,
        U_val,
    )

    # --------------------------------------------------------
    # Koopman lifting
    # --------------------------------------------------------

    Psi, Psi_prime = build_lifted_snapshots(
        X,
        X_prime,
    )

    print()
    print("Koopman lifting")
    print("---------------")
    print("Physical state dimension :", X.shape[0])
    print("Lifted dimension         :", Psi.shape[0])

    A_K, B_K = identify_koopman(
        Psi,
        Psi_prime,
        U,
    )

    C = build_reconstruction_matrix(
        X,
        Psi,
    )

    koopman_prediction = predict_koopman(
        A_K,
        B_K,
        C,
        x0,
        U_val,
    )

    # --------------------------------------------------------
    # Compare predictions
    # --------------------------------------------------------
    # --------------------------------------------------------
    # One-step validation
    # --------------------------------------------------------

    X_val_one_step = X_validation[:, :-1]
    X_val_next_actual = X_validation[:, 1:]
    U_val_one_step = U_validation[:, :-1]

    dmdc_one_step = one_step_dmdc_prediction(
        A,
        B,
        X_val_one_step,
        U_val_one_step,
    )

    koopman_one_step = one_step_koopman_prediction(
        A_K,
        B_K,
        C,
        X_val_one_step,
        U_val_one_step,
    )

    dmdc_one_step_rmse = calculate_rmse(
        X_val_next_actual,
        dmdc_one_step,
    )

    koopman_one_step_rmse = calculate_rmse(
        X_val_next_actual,
        koopman_one_step,
    )

    print()
    print("ONE-STEP VALIDATION RMSE")
    print("------------------------")

    for i, state_name in enumerate(STATE_COLS):
        print(
            f"{state_name:>10s} : "
            f"DMDc={dmdc_one_step_rmse[i]:.6f}, "
            f"Koopman={koopman_one_step_rmse[i]:.6f}"
        )

    print_eigenvalue_diagnostics(
        A,
        A_K,
    )

    print_condition_diagnostics(
        X,
        U,
        Psi,
    )

    # --------------------------------------------------------
    # SHORT-HORIZON ROLLOUT DIAGNOSTIC
    # --------------------------------------------------------

    print()
    print("SHORT-HORIZON ROLLOUT RMSE")
    print("--------------------------")

    horizons = [10, 50, 100, 500, 1000]

    for horizon in horizons:
        if horizon >= X_validation.shape[1]:
            continue

        U_short = U_validation[:, :horizon]
        actual_short = X_validation[:, :horizon + 1]

        dmdc_short = predict_dmdc(
            A,
            B,
            x0,
            U_short,
        )

        koopman_short = predict_koopman(
            A_K,
            B_K,
            C,
            x0,
            U_short,
        )

        dmdc_short_rmse = np.mean(
            calculate_rmse(
                actual_short,
                dmdc_short,
            )
        )

        koopman_short_rmse = np.mean(
            calculate_rmse(
                actual_short,
                koopman_short,
            )
        )

        print(
            f"Horizon {horizon:4d} steps : "
            f"DMDc={dmdc_short_rmse:.6f}, "
            f"Koopman={koopman_short_rmse:.6f}"
        )
        
        
    actual = X_validation[
        :,
        :dmdc_prediction.shape[1],
    ]

    dmdc_rmse = calculate_rmse(
        actual,
        dmdc_prediction,
    )

    koopman_rmse = calculate_rmse(
        actual,
        koopman_prediction,
    )

    dmdc_relative = calculate_relative_error(
        actual,
        dmdc_prediction,
    )

    koopman_relative = calculate_relative_error(
        actual,
        koopman_prediction,
    )

    comparison = pd.DataFrame(
        {
            "state": STATE_COLS,
            "dmdc_rmse": dmdc_rmse,
            "koopman_rmse": koopman_rmse,
            "dmdc_relative_error": dmdc_relative,
            "koopman_relative_error": koopman_relative,
        }
    )

    comparison.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False,
    )

    print()
    print("Validation comparison")
    print("---------------------")
    print(comparison.to_string(index=False))

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    validation_time = validation_df[
        "time"
    ].to_numpy(dtype=float)

    save_validation_plots(
        validation_time,
        actual,
        dmdc_prediction,
        koopman_prediction,
    )

    # --------------------------------------------------------
    # Save models
    # --------------------------------------------------------

    np.save(RESULTS_DIR / "A.npy", A)
    np.save(RESULTS_DIR / "B.npy", B)
    np.save(RESULTS_DIR / "A_K.npy", A_K)
    np.save(RESULTS_DIR / "B_K.npy", B_K)
    np.save(RESULTS_DIR / "C.npy", C)

    np.save(
        RESULTS_DIR / "omega_singular_values.npy",
        singular_values,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    with open(
        RESULTS_DIR / "model_summary.txt",
        "w",
    ) as f:
        f.write("DMDc + KOOPMAN QUADROTOR MODEL SUMMARY\n")
        f.write("=" * 55 + "\n\n")

        f.write(
            f"State dimension       : {X.shape[0]}\n"
        )
        f.write(
            f"Input dimension       : {U.shape[0]}\n"
        )
        f.write(
            f"DMDc Omega rank       : {omega_rank}\n"
        )
        f.write(
            f"Koopman lifted dim.   : {Psi.shape[0]}\n"
        )

        f.write("\nVALIDATION RMSE\n")
        f.write("-" * 35 + "\n")

        for row in comparison.itertuples():
            f.write(
                f"{row.state:>10s} : "
                f"DMDc={row.dmdc_rmse:.6f}, "
                f"Koopman={row.koopman_rmse:.6f}\n"
            )

    print()
    print("=" * 70)
    print("DMDc + KOOPMAN PIPELINE COMPLETE")
    print("=" * 70)
    print()
    print("Results saved in:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()
