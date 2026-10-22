from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import eigvals


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"

RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# DMDc STATE
# ============================================================

STATE_COLS = [
    "z_error",
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
# LOAD DATA
# ============================================================

def load_data(filename):

    path = DATA_DIR / filename

    df = pd.read_csv(path)

    states = (
        df[STATE_COLS]
        .to_numpy(
            dtype=float
        )
        .T
    )

    # DMDc works around the hover equilibrium,
    # so use motor deviations from hover.
    controls = (
        df[INPUT_COLS]
        .to_numpy(
            dtype=float
        )
        - HOVER_THRUST
    ).T

    return (
        df,
        states,
        controls,
    )


# ============================================================
# AUTOMATIC SVD RANK
# ============================================================

def energy_rank(
    singular_values,
    energy=0.999,
):

    if (
        len(singular_values) == 0
        or singular_values[0] <= 0
    ):
        return 1

    energy_curve = (
        np.cumsum(
            singular_values ** 2
        )
        /
        np.sum(
            singular_values ** 2
        )
    )

    rank = (
        np.searchsorted(
            energy_curve,
            energy,
        )
        + 1
    )

    return int(rank)


# ============================================================
# DMDc IDENTIFICATION
# ============================================================

def identify_dmdc(
    X,
    X_prime,
    U,
    rank=None,
):

    # --------------------------------------------------------
    # Augmented state-input matrix
    #
    # Ω = [ X ]
    #     [ U ]
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
    # Ω = U Σ Vᵀ
    # --------------------------------------------------------

    U_omega, S, Vt = np.linalg.svd(
        Omega,
        full_matrices=False,
    )

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
    # Ω† = V Σ† Uᵀ
    # --------------------------------------------------------

    S_inverse = np.diag(
        1.0 / S_r
    )

    # --------------------------------------------------------
    # G = X' Ω†
    # --------------------------------------------------------

    G = (
        X_prime
        @ V_r
        @ S_inverse
        @ U_r.T
    )

    # --------------------------------------------------------
    # G = [ A B ]
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
        U_omega,
    )


# ============================================================
# REDUCED-ORDER MODEL
# ============================================================

def build_reduced_model(
    A,
    B,
    X_prime,
    rom_rank=8,
):
    """
    Build the reduced-order DMDc model.

    First:
        X' = U_hat Sigma_hat V_hat^T

    Then:
        A_tilde = U_hat^T A U_hat
        B_tilde = U_hat^T B

    The reduced state is:

        x_tilde = U_hat^T x

    """

    state_dim = A.shape[0]

    # --------------------------------------------------------
    # SECOND SVD
    #
    # X' = U_hat Sigma_hat V_hat^T
    # --------------------------------------------------------

    U_y, S_y, Vt_y = np.linalg.svd(
        X_prime,
        full_matrices=False,
    )

    # --------------------------------------------------------
    # Select reduced-order rank
    # --------------------------------------------------------

    rom_rank = max(
        1,
        min(
            rom_rank,
            U_y.shape[1],
            state_dim,
        ),
    )

    # Dominant state basis
    U_hat = (
        U_y[
            :,
            :rom_rank
        ]
    )

    # --------------------------------------------------------
    # REDUCED A
    #
    # A_tilde = U_hat^T A U_hat
    # --------------------------------------------------------

    A_tilde = (
        U_hat.T
        @ A
        @ U_hat
    )

    # --------------------------------------------------------
    # REDUCED B
    #
    # B_tilde = U_hat^T B
    # --------------------------------------------------------

    B_tilde = (
        U_hat.T
        @ B
    )

    return (
        U_hat,
        A_tilde,
        B_tilde,
        S_y,
        rom_rank,
    )

# ============================================================
# REDUCED MODEL PREDICTION
# ============================================================

def predict_reduced_model(
    A_tilde,
    B_tilde,
    basis,
    x0,
    U_sequence,
):

    # Reduced initial state.
    z = (
        basis.T
        @ x0
    )

    predictions = [
        x0.copy()
    ]

    for k in range(
        U_sequence.shape[1]
    ):

        # Reduced dynamics:
        #
        # z(k+1) =
        # A_tilde z(k)
        # +
        # B_tilde u(k)

        z = (
            A_tilde @ z
            +
            B_tilde
            @ U_sequence[:, k]
        )

        # Back to full state.
        x = (
            basis @ z
        )

        predictions.append(
            x
        )

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

    n = min(
        actual.shape[1],
        predicted.shape[1],
    )

    return np.sqrt(
        np.mean(
            (
                actual[:, :n]
                -
                predicted[:, :n]
            ) ** 2,
            axis=1,
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DMDc SYSTEM IDENTIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    train_df, X_train, U_train = load_data(
        "manual_training.csv"
    )

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    validation_df, X_validation, U_validation = load_data(
        "manual_validation.csv"
    )

    # --------------------------------------------------------
    # Snapshot matrices
    #
    # X  = [x0 x1 ... x(n-1)]
    #
    # X' = [x1 x2 ... xn]
    #
    # U  = [u0 u1 ... u(n-1)]
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
    # DMDc
    # --------------------------------------------------------

    (
        A,
        B,
        Omega,
        singular_values,
        omega_rank,
        U_omega,
    ) = identify_dmdc(
        X,
        X_prime,
        U,
    )

    print()
    print("Omega shape:", Omega.shape)

    print()
    print("Omega singular values:")
    print(singular_values)
    

    print()
    print(
        "Selected Omega rank:",
        omega_rank,
    )

    # --------------------------------------------------------
    # Print A
    # --------------------------------------------------------

    print()
    print("================================")
    print("IDENTIFIED A")
    print("================================")

    print(A)

    # --------------------------------------------------------
    # Print B
    # --------------------------------------------------------

    print()
    print("================================")
    print("IDENTIFIED B")
    print("================================")

    print(B)

    # --------------------------------------------------------
    # SECOND SVD / REDUCED MODEL
    # --------------------------------------------------------
    (
        basis,
        A_tilde,
        B_tilde,
        state_singular_values,
        rom_rank,
    ) = build_reduced_model(
        A,
        B,
        X_prime,
        rom_rank=8,
    )
    
    print()
    print("================================")
    print("REDUCED-ORDER MODEL")
    print("================================")

    print(
        "Reduced rank:",
        rom_rank,
    )

    print()
    print("State singular values:")
    print(
        state_singular_values
    )
    
    # --------------------------------------------------------
    # SINGULAR VALUE ENERGY
    # --------------------------------------------------------

    energy = (
        np.cumsum(state_singular_values ** 2)
        /
        np.sum(state_singular_values ** 2)
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(state_singular_values) + 1),
        energy,
        marker="o",
    )

    plt.axhline(
        0.99,
        linestyle="--",
        label="99% energy",
    )

    plt.axvline(
        rom_rank,
        linestyle="--",
        label=f"Selected rank = {rom_rank}",
    )

    plt.xlabel("Number of modes")
    plt.ylabel("Cumulative energy")
    plt.title("ROM Singular-Value Energy")
    plt.ylim(0, 1.05)
    plt.grid(
        True,
        alpha=0.25,
    )
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "singular_value_energy.png",
        dpi=160,
    )

    plt.close()

    print()
    print(
        f"Energy retained by ROM rank {rom_rank}: "
        f"{energy[rom_rank - 1] * 100:.2f}%"
    )

    print()
    print("A_tilde:")
    print(A_tilde)

    print()
    print("B_tilde:")
    print(B_tilde)

    # --------------------------------------------------------
    # VALIDATION PREDICTION
    # --------------------------------------------------------

    U_val = (
        U_validation[:, :-1]
    )

    x0 = (
        X_validation[:, 0]
    )

    X_prediction = (
        predict_reduced_model(
            A_tilde,
            B_tilde,
            basis,
            x0,
            U_val,
        )
    )

    actual = (
        X_validation[
            :,
            :X_prediction.shape[1]
        ]
    )

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    errors = calculate_rmse(
        actual,
        X_prediction,
    )

    print()
    print("================================")
    print("VALIDATION RMSE")
    print("================================")
    for name, error in zip(
        STATE_COLS,
        errors,
    ):
        print(
            f"{name:>10s} : "
            f"{error:.6f}"
        )
    # --------------------------------------------------------
    # RMSE SUMMARY
    # --------------------------------------------------------

    rmse_df = pd.DataFrame({
        "state": STATE_COLS,
        "rmse": errors,
    })

    rmse_df.to_csv(
        RESULTS_DIR / "validation_rmse.csv",
        index=False,
    )

    plt.figure(figsize=(9, 5))

    plt.bar(
        STATE_COLS,
        errors,
    )

    plt.xlabel("State")
    plt.ylabel("RMSE")
    plt.title("DMDc Reduced-Order Model Validation RMSE")
    plt.xticks(rotation=30)
    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "validation_rmse.png",
        dpi=160,
    )

    plt.close()
    
    # --------------------------------------------------------
    # EIGENVALUES
    # --------------------------------------------------------

    eigenvalues = eigvals(
        A_tilde
    )

    print()
    print("================================")
    print("REDUCED MODEL EIGENVALUES")
    print("================================")

    for value in eigenvalues:

        print(
            f"{value.real:+.5f}"
            f"{value.imag:+.5f}j"
            f"   |lambda| = "
            f"{abs(value):.5f}"
        )
    
    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_path = RESULTS_DIR / "model_summary.txt"

    with open(summary_path, "w") as f:

        f.write("DMDc QUADROTOR MODEL SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"State dimension      : {A.shape[0]}\n")
        f.write(f"Input dimension      : {B.shape[1]}\n")
        f.write(f"DMDc Omega rank      : {omega_rank}\n")
        f.write(f"ROM rank             : {rom_rank}\n")

        retained_energy = (
            np.sum(
                state_singular_values[:rom_rank] ** 2
            )
            /
            np.sum(
                state_singular_values ** 2
            )
        )

        f.write(
            f"ROM energy retained  : "
            f"{retained_energy * 100:.2f}%\n"
        )

        f.write("\nVALIDATION RMSE\n")
        f.write("-" * 30 + "\n")

        for name, error in zip(
            STATE_COLS,
            errors,
        ):
            f.write(
                f"{name:>10s} : "
                f"{error:.6f}\n"
            )

        f.write("\nREDUCED MODEL EIGENVALUES\n")
        f.write("-" * 30 + "\n")

        for value in eigenvalues:
            f.write(
                f"{value.real:+.6f}"
                f"{value.imag:+.6f}j"
                f" |lambda|={abs(value):.6f}\n"
            )

    # --------------------------------------------------------
    # SAVE MATRICES
    # --------------------------------------------------------

    np.save(
        RESULTS_DIR / "A.npy",
        A,
    )

    np.save(
        RESULTS_DIR / "B.npy",
        B,
    )

    np.save(
        RESULTS_DIR / "basis.npy",
        basis,
    )

    np.save(
        RESULTS_DIR / "A_tilde.npy",
        A_tilde,
    )

    np.save(
        RESULTS_DIR / "B_tilde.npy",
        B_tilde,
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    validation_time = (
        validation_df[
            "time"
        ].to_numpy()
    )

    for i, name in enumerate(
        STATE_COLS
    ):

        plt.figure(
            figsize=(9, 4)
        )

        plt.plot(
            validation_time[
                :actual.shape[1]
            ],
            actual[i],
            label="MuJoCo actual",
        )

        plt.plot(
            validation_time[
                :X_prediction.shape[1]
            ],
            X_prediction[i],
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
            / f"validation_{name}.png",
            dpi=160,
        )

        plt.close()

    # --------------------------------------------------------
    # EIGENVALUE PLOT
    # --------------------------------------------------------

    theta = np.linspace(
        0,
        2.0 * np.pi,
        400,
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.plot(
        np.cos(theta),
        np.sin(theta),
        label="Unit circle",
    )

    plt.scatter(
        eigenvalues.real,
        eigenvalues.imag,
        s=50,
        label="DMDc eigenvalues",
    )

    plt.axhline(
        0,
        linewidth=0.8,
    )

    plt.axvline(
        0,
        linewidth=0.8,
    )

    plt.xlabel(
        "Real"
    )

    plt.ylabel(
        "Imaginary"
    )

    plt.title(
        "Reduced DMDc eigenvalues"
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
        RESULTS_DIR
        / "eigenvalues.png",
        dpi=160,
    )

    plt.close()

    print()
    print("=" * 70)
    print("DMDc PIPELINE COMPLETE")
    print("=" * 70)

    print()
    print(
        "Results saved in:"
    )

    print(
        RESULTS_DIR
    )


if __name__ == "__main__":
    main()