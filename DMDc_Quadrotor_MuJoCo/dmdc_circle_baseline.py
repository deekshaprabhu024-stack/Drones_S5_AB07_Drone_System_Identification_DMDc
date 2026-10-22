from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"

RESULTS_DIR = ROOT / "results_dmdc_circle"

RESULTS_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# CIRCULAR QUADROTOR STATE
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


# Hover thrust used by the original DMDc pipeline.
HOVER_THRUST = 9.81 / 4.0


# ============================================================
# DATA LOADING
# ============================================================

def load_data(filename):
    """
    Load a circular-trajectory dataset.

    The DMDc state is:

        [x, y, z,
         vx, vy, vz,
         roll, pitch, yaw,
         p, q, r]

    The inputs are the actual motor commands expressed
    as deviations from hover thrust.
    """

    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required data file was not found:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "time",
        *STATE_COLS,
        *INPUT_COLS,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{filename} is missing required columns:\n"
            f"{missing}"
        )

    df = df[required_columns].copy()

    # --------------------------------------------------------
    # Remove non-finite samples
    # --------------------------------------------------------

    df = (
        df
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Check time
    # --------------------------------------------------------

    time = df["time"].to_numpy(
        dtype=float
    )

    if len(time) < 2:
        raise ValueError(
            f"{filename} does not contain enough samples."
        )

    dt = np.diff(time)

    nonpositive = np.where(
        dt <= 0
    )[0]

    if len(nonpositive) > 0:

        print()
        print(
            "WARNING:"
            f" {filename} contains "
            f"{len(nonpositive)} non-increasing "
            "time intervals."
        )

        # Keep the longest strictly increasing segment.
        segments = []

        start = 0

        for index in nonpositive:

            end = index + 1

            if end > start:
                segments.append(
                    df.iloc[start:end]
                )

            start = end

        if start < len(df):
            segments.append(
                df.iloc[start:]
            )

        if not segments:
            raise ValueError(
                f"Could not find a valid time segment "
                f"in {filename}."
            )

        df = max(
            segments,
            key=len
        ).copy()

        df.reset_index(
            drop=True,
            inplace=True
        )

    # --------------------------------------------------------
    # State matrix
    # --------------------------------------------------------

    states = (
        df[STATE_COLS]
        .to_numpy(
            dtype=float
        )
        .T
    )

    # --------------------------------------------------------
    # Control matrix
    #
    # Keep the exact convention from the original DMDc
    # implementation:
    #
    #     u = actual_motor_command - hover_thrust
    # --------------------------------------------------------

    controls = (
        df[INPUT_COLS]
        .to_numpy(
            dtype=float
        )
        -
        HOVER_THRUST
    ).T

    return (
        df.reset_index(drop=True),
        states,
        controls,
    )


# ============================================================
# AUTOMATIC ENERGY-BASED SVD RANK
# ============================================================

def energy_rank(
    singular_values,
    energy=0.999,
):
    """
    Select the smallest rank containing the requested
    fraction of the squared singular-value energy.

    This is the same rank-selection principle used by
    the original DMDc implementation.
    """

    singular_values = np.asarray(
        singular_values,
        dtype=float
    )

    if (
        singular_values.size == 0
        or singular_values[0] <= 0
    ):
        return 1

    total_energy = np.sum(
        singular_values ** 2
    )

    if total_energy <= 0:
        return 1

    cumulative_energy = (
        np.cumsum(
            singular_values ** 2
        )
        /
        total_energy
    )

    rank = (
        np.searchsorted(
            cumulative_energy,
            energy,
        )
        + 1
    )

    return int(rank)


# ============================================================
# ORIGINAL-STYLE DMDc IDENTIFICATION
# ============================================================

def identify_dmdc(
    X,
    X_prime,
    U,
    rank=None,
):
    """
    Identify the standard DMDc model

        X' = A X + B U

    using the same SVD/truncated-pseudoinverse
    methodology as the original working DMDc pipeline.

    IMPORTANT:

    There is deliberately NO:
        - feature scaling
        - ridge regression
        - normalization
        - Gram-matrix inversion

    here.

    The only change from the original implementation is
    that the state dimension is now 12 because the circular
    trajectory requires x and y position.
    """

    # --------------------------------------------------------
    # Augmented snapshot matrix
    #
    #             [ X ]
    # Omega =     [   ]
    #             [ U ]
    #
    # --------------------------------------------------------

    Omega = np.vstack(
        (
            X,
            U,
        )
    )

    # --------------------------------------------------------
    # SVD
    #
    # Omega = U Sigma V^T
    # --------------------------------------------------------

    U_omega, S, Vt = np.linalg.svd(
        Omega,
        full_matrices=False,
    )

    # --------------------------------------------------------
    # Automatic rank selection
    # --------------------------------------------------------

    if rank is None:

        rank = energy_rank(
            S,
            energy=0.999,
        )

    rank = max(
        1,
        min(
            rank,
            len(S),
        ),
    )

    # --------------------------------------------------------
    # Truncated SVD factors
    # --------------------------------------------------------

    U_r = (
        U_omega[
            :,
            :rank
        ]
    )

    S_r = S[
        :rank
    ]

    V_r = (
        Vt[
            :rank,
            :
        ].T
    )

    # --------------------------------------------------------
    # Truncated pseudoinverse
    #
    # Omega^+ = V_r Sigma_r^-1 U_r^T
    # --------------------------------------------------------

    S_inverse = np.diag(
        1.0 / S_r
    )

    # --------------------------------------------------------
    # G = X' Omega^+
    # --------------------------------------------------------

    G = (
        X_prime
        @ V_r
        @ S_inverse
        @ U_r.T
    )

    # --------------------------------------------------------
    # Separate A and B
    #
    # G = [ A  B ]
    # --------------------------------------------------------

    state_dim = X.shape[0]

    A = G[
        :,
        :state_dim
    ]

    B = G[
        :,
        state_dim:
    ]

    return (
        A,
        B,
        Omega,
        S,
        rank,
    )


# ============================================================
# DMDc ONE-STEP PREDICTION
# ============================================================

def predict_one_step(
    A,
    B,
    X,
    U,
):
    """
    Predict:

        x(k+1) = A x(k) + B u(k)

    for every available snapshot.
    """

    return (
        A @ X
        +
        B @ U
    )


# ============================================================
# DMDc RECURSIVE ROLLOUT
# ============================================================

def predict_dmdc(
    A,
    B,
    x0,
    U_sequence,
):
    """
    Recursive DMDc prediction.

    Starting from x0:

        x(k+1) = A x(k) + B u(k)
    """

    predictions = [
        x0.copy()
    ]

    x = x0.copy()

    for k in range(
        U_sequence.shape[1]
    ):

        x = (
            A @ x
            +
            B @ U_sequence[:, k]
        )

        predictions.append(
            x.copy()
        )

        # Stop immediately if the model has become
        # numerically invalid.
        if not np.all(
            np.isfinite(x)
        ):
            print()
            print(
                "WARNING: DMDc rollout became "
                "non-finite at step",
                k + 1,
            )
            break

    return np.column_stack(
        predictions
    )


# ============================================================
# RMSE
# ============================================================

def calculate_rmse(
    actual,
    predicted,
):
    """
    Per-state RMSE.
    """

    n = min(
        actual.shape[1],
        predicted.shape[1],
    )

    error = (
        actual[:, :n]
        -
        predicted[:, :n]
    )

    return np.sqrt(
        np.mean(
            error ** 2,
            axis=1,
        )
    )


# ============================================================
# RELATIVE ERROR
# ============================================================

def calculate_relative_error(
    actual,
    predicted,
):
    """
    Relative L2 error for each state.
    """

    n = min(
        actual.shape[1],
        predicted.shape[1],
    )

    numerator = np.linalg.norm(
        actual[:, :n]
        -
        predicted[:, :n],
        axis=1,
    )

    denominator = np.linalg.norm(
        actual[:, :n],
        axis=1,
    )

    denominator = np.maximum(
        denominator,
        1e-12,
    )

    return (
        numerator
        /
        denominator
    )


# ============================================================
# EIGENVALUE DIAGNOSTICS
# ============================================================

def print_eigenvalue_diagnostics(A):

    eigenvalues = np.linalg.eigvals(A)

    magnitudes = np.abs(
        eigenvalues
    )

    print()
    print("=" * 70)
    print("DMDc EIGENVALUES")
    print("=" * 70)

    for value in eigenvalues:

        print(
            f"{value.real:+.9f}"
            f"{value.imag:+.9f}j"
            f" |lambda|={abs(value):.9f}"
        )

    print()
    print(
        "Maximum eigenvalue magnitude : "
        f"{np.max(magnitudes):.9f}"
    )

    print(
        "Minimum eigenvalue magnitude : "
        f"{np.min(magnitudes):.9f}"
    )

    return eigenvalues


# ============================================================
# CONDITION DIAGNOSTICS
# ============================================================

def print_condition_diagnostics(
    Omega,
    singular_values,
    rank,
):

    print()
    print("=" * 70)
    print("DMDc NUMERICAL DIAGNOSTICS")
    print("=" * 70)

    print(
        "Omega shape                 : "
        f"{Omega.shape}"
    )

    print(
        "Full numerical rank         : "
        f"{np.linalg.matrix_rank(Omega)}"
    )

    print(
        "Selected SVD rank           : "
        f"{rank}"
    )

    print(
        "Condition number(Omega)     : "
        f"{np.linalg.cond(Omega):.6e}"
    )

    print()
    print("Singular values:")
    print(singular_values)

    print()
    print(
        "Retained squared-energy     : "
        f"{100.0 * np.sum(singular_values[:rank] ** 2) / np.sum(singular_values ** 2):.6f}%"
    )


# ============================================================
# SHORT-HORIZON ROLLOUT TEST
# ============================================================

def short_horizon_test(
    A,
    B,
    actual,
    controls,
    horizons,
):
    """
    Evaluate recursive prediction from the first state.

    The control sequence is taken from the actual experiment.
    """

    print()
    print("=" * 70)
    print("SHORT-HORIZON RECURSIVE ROLLOUT")
    print("=" * 70)

    print(
        f"{'Horizon':>12}"
        f"{'RMSE':>20}"
        f"{'Finite':>12}"
    )

    results = {}

    max_available = min(
        actual.shape[1] - 1,
        controls.shape[1],
    )

    for horizon in horizons:

        if horizon > max_available:
            continue

        prediction = predict_dmdc(
            A,
            B,
            actual[:, 0],
            controls[:, :horizon],
        )

        actual_segment = actual[
            :,
            :prediction.shape[1]
        ]

        rmse = calculate_rmse(
            actual_segment,
            prediction,
        )

        finite = np.all(
            np.isfinite(prediction)
        )

        if finite:
            overall_rmse = float(
                np.sqrt(
                    np.mean(
                        (
                            actual_segment
                            -
                            prediction
                        ) ** 2
                    )
                )
            )
        else:
            overall_rmse = np.inf

        results[horizon] = (
            overall_rmse
        )

        print(
            f"{horizon:>12}"
            f"{overall_rmse:>20.9f}"
            f"{str(finite):>12}"
        )

    return results


# ============================================================
# XY TRAJECTORY PLOT
# ============================================================

def save_xy_plot(
    actual,
    prediction,
):
    """
    Plot actual MuJoCo XY trajectory against
    recursive DMDc prediction.
    """

    n = min(
        actual.shape[1],
        prediction.shape[1],
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.plot(
        actual[0, :n],
        actual[1, :n],
        label="MuJoCo actual",
    )

    plt.plot(
        prediction[0, :n],
        prediction[1, :n],
        "--",
        label="DMDc prediction",
    )

    plt.xlabel(
        "x (m)"
    )

    plt.ylabel(
        "y (m)"
    )

    plt.title(
        "Circular Trajectory: MuJoCo vs DMDc"
    )

    plt.axis(
        "equal"
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "dmdc_xy_trajectory.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# STATE PREDICTION PLOTS
# ============================================================

def save_state_plots(
    time,
    actual,
    prediction,
):
    """
    Plot actual vs recursive DMDc prediction
    for every physical state.
    """

    n = min(
        len(time),
        actual.shape[1],
        prediction.shape[1],
    )

    for i, name in enumerate(
        STATE_COLS
    ):

        plt.figure(
            figsize=(10, 4)
        )

        plt.plot(
            time[:n],
            actual[i, :n],
            label="MuJoCo actual",
        )

        plt.plot(
            time[:n],
            prediction[i, :n],
            "--",
            label="DMDc prediction",
        )

        plt.xlabel(
            "Time (s)"
        )

        plt.ylabel(
            name
        )

        plt.title(
            f"DMDc validation: {name}"
        )

        plt.grid(
            True,
            alpha=0.25,
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR
            /
            f"dmdc_validation_{name}.png",
            dpi=160,
        )

        plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("CIRCULAR QUADROTOR — STANDARD DMDc BASELINE")
    print("=" * 70)

    print()
    print(
        "Identification method:"
    )
    print(
        "Original SVD + truncated pseudoinverse DMDc"
    )
    print(
        "No ridge regularization"
    )
    print(
        "No feature normalization"
    )

    # ========================================================
    # LOAD TRAINING DATA
    # ========================================================

    train_df, X_train, U_train = load_data(
        "circle_training.csv"
    )

    # ========================================================
    # LOAD VALIDATION DATA
    # ========================================================

    validation_df, X_validation, U_validation = load_data(
        "circle_validation.csv"
    )

    print()
    print("=" * 70)
    print("DATA")
    print("=" * 70)

    print(
        "Training samples   : "
        f"{len(train_df)}"
    )

    print(
        "Validation samples : "
        f"{len(validation_df)}"
    )

    # ========================================================
    # TRAINING SNAPSHOTS
    #
    # X      = [x0 x1 ... xN-1]
    # X'     = [x1 x2 ... xN]
    # U      = [u0 u1 ... uN-1]
    #
    # ========================================================

    X = X_train[:, :-1]

    X_prime = X_train[:, 1:]

    U = U_train[:, :-1]

    print()
    print("=" * 70)
    print("SNAPSHOT DIMENSIONS")
    print("=" * 70)

    print(
        "X       : "
        f"{X.shape}"
    )

    print(
        "X_prime : "
        f"{X_prime.shape}"
    )

    print(
        "U       : "
        f"{U.shape}"
    )

    # ========================================================
    # IDENTIFY DMDc
    # ========================================================

    (
        A,
        B,
        Omega,
        singular_values,
        omega_rank,
    ) = identify_dmdc(
        X,
        X_prime,
        U,
    )

    print()
    print("=" * 70)
    print("DMDc MODEL")
    print("=" * 70)

    print(
        "State dimension : "
        f"{X.shape[0]}"
    )

    print(
        "Input dimension : "
        f"{U.shape[0]}"
    )

    print(
        "Omega shape     : "
        f"{Omega.shape}"
    )

    print(
        "Selected rank   : "
        f"{omega_rank}"
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    print_condition_diagnostics(
        Omega,
        singular_values,
        omega_rank,
    )

    eigenvalues = (
        print_eigenvalue_diagnostics(
            A
        )
    )

    # ========================================================
    # ONE-STEP VALIDATION
    # ========================================================

    X_validation_snapshots = (
        X_validation[:, :-1]
    )

    X_validation_next = (
        X_validation[:, 1:]
    )

    U_validation_snapshots = (
        U_validation[:, :-1]
    )

    one_step_prediction = (
        predict_one_step(
            A,
            B,
            X_validation_snapshots,
            U_validation_snapshots,
        )
    )

    one_step_rmse = calculate_rmse(
        X_validation_next,
        one_step_prediction,
    )

    print()
    print("=" * 70)
    print("ONE-STEP VALIDATION RMSE")
    print("=" * 70)

    for name, error in zip(
        STATE_COLS,
        one_step_rmse,
    ):

        print(
            f"{name:>8} : "
            f"{error:.9f}"
        )

    # ========================================================
    # FULL RECURSIVE VALIDATION
    # ========================================================

    validation_prediction = predict_dmdc(
        A,
        B,
        X_validation[:, 0],
        U_validation[:, :-1],
    )

    validation_actual = (
        X_validation[
            :,
            :validation_prediction.shape[1]
        ]
    )

    full_rmse = calculate_rmse(
        validation_actual,
        validation_prediction,
    )

    relative_error = (
        calculate_relative_error(
            validation_actual,
            validation_prediction,
        )
    )

    print()
    print("=" * 70)
    print("FULL RECURSIVE VALIDATION")
    print("=" * 70)

    print(
        f"{'State':>10}"
        f"{'RMSE':>20}"
        f"{'Relative error':>20}"
    )

    for name, rmse, rel in zip(
        STATE_COLS,
        full_rmse,
        relative_error,
    ):

        print(
            f"{name:>10}"
            f"{rmse:>20.9f}"
            f"{rel:>20.9f}"
        )

    # ========================================================
    # SHORT-HORIZON TEST
    # ========================================================

    short_horizon_test(
        A,
        B,
        X_validation,
        U_validation,
        horizons=[
            10,
            50,
            100,
            500,
            1000,
            2500,
            5000,
            10000,
        ],
    )

    # ========================================================
    # SAVE MATRICES
    # ========================================================

    np.save(
        RESULTS_DIR / "A_dmdc.npy",
        A,
    )

    np.save(
        RESULTS_DIR / "B_dmdc.npy",
        B,
    )

    np.save(
        RESULTS_DIR / "dmdc_eigenvalues.npy",
        eigenvalues,
    )

    np.save(
        RESULTS_DIR / "singular_values.npy",
        singular_values,
    )

    # ========================================================
    # SAVE XY PLOT
    # ========================================================

    save_xy_plot(
        validation_actual,
        validation_prediction,
    )

    # ========================================================
    # SAVE STATE PLOTS
    # ========================================================

    save_state_plots(
        validation_df["time"].to_numpy(
            dtype=float
        ),
        validation_actual,
        validation_prediction,
    )

    # ========================================================
    # SAVE SUMMARY CSV
    # ========================================================

    summary = pd.DataFrame(
        {
            "state": STATE_COLS,
            "one_step_rmse": one_step_rmse,
            "full_recursive_rmse": full_rmse,
            "relative_error": relative_error,
        }
    )

    summary.to_csv(
        RESULTS_DIR / "dmdc_validation_summary.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        f"Results directory:\n{RESULTS_DIR}"
    )

    print()
    print("DMDc baseline completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()