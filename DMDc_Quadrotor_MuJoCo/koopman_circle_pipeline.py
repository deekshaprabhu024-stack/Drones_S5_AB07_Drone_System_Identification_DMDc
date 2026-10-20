import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CIRCULAR QUADROTOR — KOOPMAN-DMDc
#
# Physical state:
#   [x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r]
#
# Lifted state:
#   12 physical states
#   + sin/cos(roll)
#   + sin/cos(pitch)
#   + sin/cos(yaw)
#
# Total lifted dimension = 18
#
# Model:
#
#   Psi_{k+1} = A_K Psi_k + B_K U_k
#
# Identification:
#   SAME SVD / truncated-pseudoinverse methodology
#   as the frozen standard DMDc baseline.
#
# No ridge regression.
# No normalization.
# No Gram-matrix inversion.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = os.path.join(
    "data",
    "circle_training.csv"
)

VAL_FILE = os.path.join(
    "data",
    "circle_validation.csv"
)

RESULTS_DIR = "results_koopman_circle"

STATE_COLS = [
    "x", "y", "z",
    "vx", "vy", "vz",
    "roll", "pitch", "yaw",
    "p", "q", "r"
]

INPUT_COLS = [
    "u1", "u2", "u3", "u4"
]

REF_COLS = [
    "x_ref",
    "y_ref",
    "z_ref"
]

ENERGY_THRESHOLD = 0.999

HOVER_THRUST = 9.81 / 4.0


# ============================================================
# DATA LOADING
# ============================================================

def load_data(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nFile not found:\n{path}"
        )

    df = pd.read_csv(path)

    states = df[
        STATE_COLS
    ].to_numpy(
        dtype=float
    )

    inputs_raw = df[
        INPUT_COLS
    ].to_numpy(
        dtype=float
    )

    # Same input convention as original DMDc:
    # subtract hover thrust from each motor command.
    inputs = (
        inputs_raw
        - HOVER_THRUST
    )

    # IMPORTANT:
    #
    # CSV:
    #   rows    = time samples
    #   columns = state variables
    #
    # Model convention:
    #   rows    = state variables
    #   columns = time samples
    #
    states = states.T
    inputs = inputs.T

    return (
        df,
        states,
        inputs
    )


# ============================================================
# ENERGY-BASED SVD RANK
# ============================================================

def energy_rank(
    singular_values,
    threshold=ENERGY_THRESHOLD
):

    squared = singular_values ** 2

    cumulative = np.cumsum(
        squared
    )

    total = cumulative[-1]

    energy = (
        cumulative / total
    )

    rank = (
        np.searchsorted(
            energy,
            threshold
        )
        + 1
    )

    return (
        rank,
        energy
    )


# ============================================================
# TRIGONOMETRIC LIFT
# ============================================================

def lift_state(x):

    """
    Input:
        x shape = (12, N)

    Output:
        Psi shape = (18, N)

    Ordering:

        0   x
        1   y
        2   z
        3   vx
        4   vy
        5   vz
        6   roll
        7   pitch
        8   yaw
        9   p
        10  q
        11  r

        12  sin(roll)
        13  cos(roll)

        14  sin(pitch)
        15  cos(pitch)

        16  sin(yaw)
        17  cos(yaw)
    """

    if x.ndim != 2:

        raise ValueError(
            f"lift_state expected a 2D array, "
            f"got shape {x.shape}"
        )

    if x.shape[0] != 12:

        raise ValueError(
            f"lift_state expected 12 states, "
            f"got {x.shape[0]}"
        )

    roll = x[6, :]
    pitch = x[7, :]
    yaw = x[8, :]

    Psi = np.vstack([
        x,

        np.sin(roll),
        np.cos(roll),

        np.sin(pitch),
        np.cos(pitch),

        np.sin(yaw),
        np.cos(yaw)
    ])

    return Psi


# ============================================================
# BUILD PHYSICAL SNAPSHOTS
# ============================================================

def build_snapshots(
    states,
    inputs
):

    if states.shape[0] != 12:

        raise ValueError(
            f"Expected states shape "
            f"(12,N), got {states.shape}"
        )

    if inputs.shape[0] != 4:

        raise ValueError(
            f"Expected inputs shape "
            f"(4,N), got {inputs.shape}"
        )

    if states.shape[1] != inputs.shape[1]:

        raise ValueError(
            "States and inputs have different "
            f"numbers of samples: "
            f"{states.shape[1]} vs "
            f"{inputs.shape[1]}"
        )

    X = states[:, :-1]

    X_prime = states[:, 1:]

    U = inputs[:, :-1]

    return (
        X,
        X_prime,
        U
    )


# ============================================================
# BUILD LIFTED SNAPSHOTS
# ============================================================

def build_lifted_snapshots(
    states,
    inputs
):

    (
        X,
        X_prime,
        U
    ) = build_snapshots(
        states,
        inputs
    )

    Psi = lift_state(
        X
    )

    Psi_prime = lift_state(
        X_prime
    )

    # --------------------------------------------------------
    # HARD SHAPE CHECK
    # --------------------------------------------------------

    assert X.shape[0] == 12
    assert X_prime.shape[0] == 12

    assert Psi.shape[0] == 18
    assert Psi_prime.shape[0] == 18

    assert U.shape[0] == 4

    assert (
        X.shape[1]
        ==
        X_prime.shape[1]
        ==
        Psi.shape[1]
        ==
        Psi_prime.shape[1]
        ==
        U.shape[1]
    )

    return (
        X,
        X_prime,
        Psi,
        Psi_prime,
        U
    )


# ============================================================
# KOOPMAN-DMDc IDENTIFICATION
# ============================================================

def identify_koopman(
    Psi,
    Psi_prime,
    U
):

    """
    Identify:

        Psi' = A_K Psi + B_K U

    using the SAME SVD methodology as
    standard DMDc.
    """

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    assert Psi.shape[0] == 18
    assert Psi_prime.shape[0] == 18
    assert U.shape[0] == 4

    assert (
        Psi.shape[1]
        ==
        Psi_prime.shape[1]
        ==
        U.shape[1]
    )

    # --------------------------------------------------------
    # Augmented snapshot matrix
    # --------------------------------------------------------

    Omega = np.vstack([
        Psi,
        U
    ])

    print("\n" + "=" * 70)
    print("KOOPMAN-DMDc IDENTIFICATION")
    print("=" * 70)

    print(
        f"Psi        shape = {Psi.shape}"
    )

    print(
        f"Psi'       shape = {Psi_prime.shape}"
    )

    print(
        f"U          shape = {U.shape}"
    )

    print(
        f"Omega      shape = {Omega.shape}"
    )

    # --------------------------------------------------------
    # SVD
    # --------------------------------------------------------

    U_svd, S, Vh = np.linalg.svd(
        Omega,
        full_matrices=False
    )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    rank, energy = energy_rank(
        S
    )

    full_rank = np.linalg.matrix_rank(
        Omega
    )

    if S[-1] > 0:

        condition_number = (
            S[0] / S[-1]
        )

    else:

        condition_number = np.inf

    print(
        f"\nFull numerical rank : "
        f"{full_rank}"
    )

    print(
        f"Selected rank       : "
        f"{rank}"
    )

    print(
        f"Condition number    : "
        f"{condition_number:.12e}"
    )

    print(
        "\nSingular values:"
    )

    for i, s in enumerate(S):

        print(
            f"  {i+1:2d}: "
            f"{s:.12e}"
        )

    retained_energy = (
        energy[rank - 1]
    )

    print(
        f"\nRetained squared-energy: "
        f"{retained_energy * 100:.9f}%"
    )

    # --------------------------------------------------------
    # Truncated SVD
    # --------------------------------------------------------

    U_r = U_svd[:, :rank]

    S_r = S[:rank]

    V_r = Vh.conj().T[:, :rank]

    # --------------------------------------------------------
    # Truncated pseudoinverse
    # --------------------------------------------------------

    S_inv = np.diag(
        1.0 / S_r
    )

    Omega_pinv = (
        V_r
        @ S_inv
        @ U_r.T
    )

    # --------------------------------------------------------
    # System matrix
    # --------------------------------------------------------

    G = (
        Psi_prime
        @ Omega_pinv
    )

    n_lifted = Psi.shape[0]
    n_inputs = U.shape[0]

    A_K = G[
        :,
        :n_lifted
    ]

    B_K = G[
        :,
        n_lifted:n_lifted + n_inputs
    ]

    return (
        A_K,
        B_K,
        S,
        rank,
        condition_number,
        retained_energy
    )


# ============================================================
# RECONSTRUCTION MATRIX
# ============================================================

def build_reconstruction_matrix(
    states,
    lifted_states
):

    """
    Find C such that:

        X ≈ C Psi
    """

    assert states.shape[0] == 12

    assert lifted_states.shape[0] == 18

    C = (
        states
        @ np.linalg.pinv(
            lifted_states
        )
    )

    assert C.shape == (
        12,
        18
    )

    return C


# ============================================================
# KOOPMAN EIGENVALUES
# ============================================================

def eigenvalue_report(
    A_K
):

    eigenvalues = np.linalg.eigvals(
        A_K
    )

    magnitudes = np.abs(
        eigenvalues
    )

    order = np.argsort(
        magnitudes
    )[::-1]

    print("\n" + "=" * 70)
    print("KOOPMAN EIGENVALUES")
    print("=" * 70)

    for idx in order:

        lam = eigenvalues[idx]

        print(
            f"{lam.real:.9f}"
            f"{lam.imag:+.9f}j"
            f"    |lambda|={abs(lam):.9f}"
        )

    print(
        f"\nMaximum |lambda| = "
        f"{np.max(magnitudes):.12f}"
    )

    return eigenvalues


# ============================================================
# KOOPMAN ONE-STEP PREDICTION
# ============================================================

def koopman_one_step(
    A_K,
    B_K,
    C,
    states,
    inputs
):

    Psi = lift_state(
        states
    )

    Psi_pred = (
        A_K @ Psi
        +
        B_K @ inputs
    )

    X_pred = (
        C @ Psi_pred
    )

    return X_pred


# ============================================================
# RECURSIVE KOOPMAN ROLLOUT
# ============================================================

def koopman_rollout(
    A_K,
    B_K,
    C,
    initial_state,
    inputs
):

    n_steps = inputs.shape[1]

    n_states = (
        initial_state.shape[0]
    )

    predictions = np.zeros(
        (
            n_states,
            n_steps + 1
        )
    )

    predictions[:, 0] = (
        initial_state
    )

    Psi = lift_state(
        initial_state.reshape(
            -1,
            1
        )
    )[:, 0]

    for k in range(n_steps):

        Psi = (
            A_K @ Psi
            +
            B_K @ inputs[:, k]
        )

        predictions[
            :,
            k + 1
        ] = (
            C @ Psi
        )

        if not np.all(
            np.isfinite(
                predictions[
                    :,
                    k + 1
                ]
            )
        ):

            print(
                f"\nWARNING: Non-finite value "
                f"at rollout step {k + 1}"
            )

            predictions[
                :,
                k + 1:
            ] = np.nan

            break

    return predictions


# ============================================================
# RMSE
# ============================================================

def rmse_per_state(
    actual,
    predicted
):

    errors = (
        actual
        -
        predicted
    )

    return np.sqrt(
        np.nanmean(
            errors ** 2,
            axis=1
        )
    )


def relative_rmse(
    actual,
    predicted
):

    errors = (
        actual
        -
        predicted
    )

    rmse = np.sqrt(
        np.nanmean(
            errors ** 2,
            axis=1
        )
    )

    scale = np.std(
        actual,
        axis=1
    )

    scale[
        scale < 1e-12
    ] = 1.0

    return (
        rmse / scale
    )


# ============================================================
# ONE-STEP VALIDATION
# ============================================================

def run_one_step_validation(
    A_K,
    B_K,
    C,
    states,
    inputs,
    name
):

    print("\n" + "=" * 70)
    print(
        f"{name} — ONE-STEP KOOPMAN VALIDATION"
    )
    print("=" * 70)

    X = states[:, :-1]

    actual = states[:, 1:]

    U = inputs[:, :-1]

    predicted = koopman_one_step(
        A_K,
        B_K,
        C,
        X,
        U
    )

    errors = rmse_per_state(
        actual,
        predicted
    )

    relative = relative_rmse(
        actual,
        predicted
    )

    for i, state in enumerate(
        STATE_COLS
    ):

        print(
            f"{state:5s}: "
            f"RMSE={errors[i]:.9f}    "
            f"relative={relative[i]:.9f}"
        )

    total = np.sqrt(
        np.nanmean(
            (
                actual
                -
                predicted
            ) ** 2
        )
    )

    print(
        f"\nOverall one-step RMSE = "
        f"{total:.12f}"
    )

    return (
        predicted,
        errors
    )


# ============================================================
# RECURSIVE VALIDATION
# ============================================================

def run_recursive_validation(
    A_K,
    B_K,
    C,
    states,
    inputs,
    name
):

    print("\n" + "=" * 70)
    print(
        f"{name} — FULL RECURSIVE VALIDATION"
    )
    print("=" * 70)

    # Use exactly the same convention as
    # the DMDc baseline:
    #
    # initial state = first recorded state
    # inputs        = u_0 ... u_{N-2}
    #
    predicted = koopman_rollout(
        A_K,
        B_K,
        C,
        states[:, 0],
        inputs[:, :-1]
    )

    actual = states[:, :predicted.shape[1]]

    errors = rmse_per_state(
        actual,
        predicted
    )

    relative = relative_rmse(
        actual,
        predicted
    )

    for i, state in enumerate(
        STATE_COLS
    ):

        print(
            f"{state:5s}: "
            f"RMSE={errors[i]:.9f}    "
            f"relative={relative[i]:.9f}"
        )

    total = np.sqrt(
        np.nanmean(
            (
                actual
                -
                predicted
            ) ** 2
        )
    )

    print(
        f"\nOverall recursive RMSE = "
        f"{total:.12f}"
    )

    return (
        predicted,
        errors
    )


# ============================================================
# SHORT-HORIZON TEST
# ============================================================

def short_horizon_test(
    A_K,
    B_K,
    C,
    states,
    inputs
):

    print("\n" + "=" * 70)
    print(
        "KOOPMAN SHORT-HORIZON TEST"
    )
    print("=" * 70)

    horizons = [
        10,
        50,
        100,
        500,
        1000,
        2500,
        5000,
        10000
    ]

    max_available = (
        inputs.shape[1]
        -
        1
    )

    for horizon in horizons:

        if horizon > max_available:

            continue

        pred = koopman_rollout(
            A_K,
            B_K,
            C,
            states[:, 0],
            inputs[:, :horizon]
        )

        actual = (
            states[
                :,
                :horizon + 1
            ]
        )

        error = np.sqrt(
            np.nanmean(
                (
                    actual
                    -
                    pred
                ) ** 2
            )
        )

        finite = np.all(
            np.isfinite(pred)
        )

        print(
            f"{horizon:5d} steps : "
            f"RMSE={error:.9f}    "
            f"finite={finite}"
        )


# ============================================================
# PLOT XY
# ============================================================

def plot_xy(
    actual,
    predicted,
    reference,
    save_path
):

    plt.figure(
        figsize=(9, 8)
    )

    plt.plot(
        actual[0],
        actual[1],
        label="MuJoCo actual"
    )

    plt.plot(
        predicted[0],
        predicted[1],
        label="Koopman recursive"
    )

    plt.plot(
        reference[0],
        reference[1],
        "--",
        label="Reference"
    )

    plt.xlabel("x")

    plt.ylabel("y")

    plt.title(
        "Circular Quadrotor — XY Trajectory"
    )

    plt.axis("equal")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()


# ============================================================
# PLOT STATES
# ============================================================

def plot_states(
    actual,
    predicted,
    time,
    save_dir
):

    n = min(
        actual.shape[1],
        predicted.shape[1],
        len(time)
    )

    time = time[:n]

    actual = actual[:, :n]

    predicted = predicted[:, :n]

    for i, state in enumerate(
        STATE_COLS
    ):

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            time,
            actual[i],
            label="MuJoCo actual"
        )

        plt.plot(
            time,
            predicted[i],
            label="Koopman recursive"
        )

        plt.xlabel(
            "Time (s)"
        )

        plt.ylabel(
            state
        )

        plt.title(
            f"Koopman vs MuJoCo — {state}"
        )

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                f"{state}.png"
            ),
            dpi=200
        )

        plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    A_K,
    B_K,
    C,
    eigenvalues,
    singular_values,
    rank,
    condition_number,
    retained_energy,
    directory
):

    np.save(
        os.path.join(
            directory,
            "A_K.npy"
        ),
        A_K
    )

    np.save(
        os.path.join(
            directory,
            "B_K.npy"
        ),
        B_K
    )

    np.save(
        os.path.join(
            directory,
            "C.npy"
        ),
        C
    )

    np.save(
        os.path.join(
            directory,
            "koopman_eigenvalues.npy"
        ),
        eigenvalues
    )

    np.save(
        os.path.join(
            directory,
            "singular_values.npy"
        ),
        singular_values
    )

    with open(
        os.path.join(
            directory,
            "summary.txt"
        ),
        "w"
    ) as f:

        f.write(
            "CIRCULAR QUADROTOR — KOOPMAN-DMDc\n"
        )

        f.write(
            "====================================\n\n"
        )

        f.write(
            f"Physical dimension: 12\n"
        )

        f.write(
            f"Lifted dimension: 18\n"
        )

        f.write(
            f"Input dimension: 4\n"
        )

        f.write(
            f"Selected SVD rank: "
            f"{rank}\n"
        )

        f.write(
            f"Condition number: "
            f"{condition_number:.12e}\n"
        )

        f.write(
            f"Retained energy: "
            f"{retained_energy * 100:.12f}%\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CIRCULAR QUADROTOR — KOOPMAN-DMDc"
    )
    print("=" * 70)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD TRAINING DATA
    # --------------------------------------------------------

    (
        train_df,
        train_states,
        train_inputs
    ) = load_data(
        TRAIN_FILE
    )

    # --------------------------------------------------------
    # LOAD VALIDATION DATA
    # --------------------------------------------------------

    (
        val_df,
        val_states,
        val_inputs
    ) = load_data(
        VAL_FILE
    )

    print(
        f"\nTraining rows   : "
        f"{len(train_df)}"
    )

    print(
        f"Validation rows : "
        f"{len(val_df)}"
    )

    # --------------------------------------------------------
    # VERIFY ORIENTATIONS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ORIENTATION CHECK")
    print("=" * 70)

    print(
        f"Training states : "
        f"{train_states.shape}"
    )

    print(
        f"Training inputs : "
        f"{train_inputs.shape}"
    )

    print(
        f"Validation states : "
        f"{val_states.shape}"
    )

    print(
        f"Validation inputs : "
        f"{val_inputs.shape}"
    )

    assert train_states.shape[0] == 12
    assert train_inputs.shape[0] == 4

    assert val_states.shape[0] == 12
    assert val_inputs.shape[0] == 4

    # --------------------------------------------------------
    # DATASET IDENTITY
    # --------------------------------------------------------

    same_dataset = np.array_equal(
        train_df.to_numpy(),
        val_df.to_numpy()
    )

    print(
        f"\nTraining and validation identical: "
        f"{same_dataset}"
    )

    if same_dataset:

        print(
            "\nWARNING:"
        )

        print(
            "The validation file is numerically "
            "identical to the training file."
        )

        print(
            "Results are therefore "
            "same-trajectory validation, "
            "NOT independent generalization."
        )

    # --------------------------------------------------------
    # BUILD LIFTED TRAINING SNAPSHOTS
    # --------------------------------------------------------

    (
        X_train,
        X_prime_train,
        Psi_train,
        Psi_prime_train,
        U_train
    ) = build_lifted_snapshots(
        train_states,
        train_inputs
    )

    print("\n" + "=" * 70)
    print("DATA DIMENSIONS")
    print("=" * 70)

    print(
        f"Physical state dimension : "
        f"{X_train.shape[0]}"
    )

    print(
        f"Lifted state dimension   : "
        f"{Psi_train.shape[0]}"
    )

    print(
        f"Input dimension          : "
        f"{U_train.shape[0]}"
    )

    print(
        f"Snapshot count           : "
        f"{X_train.shape[1]}"
    )

    # --------------------------------------------------------
    # IDENTIFY KOOPMAN MODEL
    # --------------------------------------------------------

    (
        A_K,
        B_K,
        singular_values,
        rank,
        condition_number,
        retained_energy
    ) = identify_koopman(
        Psi_train,
        Psi_prime_train,
        U_train
    )

    # --------------------------------------------------------
    # RECONSTRUCTION MATRIX
    # --------------------------------------------------------

    C = build_reconstruction_matrix(
        X_train,
        Psi_train
    )

    print("\n" + "=" * 70)
    print("MODEL DIMENSIONS")
    print("=" * 70)

    print(
        f"A_K : {A_K.shape}"
    )

    print(
        f"B_K : {B_K.shape}"
    )

    print(
        f"C   : {C.shape}"
    )

    # --------------------------------------------------------
    # EIGENVALUES
    # --------------------------------------------------------

    eigenvalues = eigenvalue_report(
        A_K
    )

    # --------------------------------------------------------
    # TRAINING ONE-STEP
    # --------------------------------------------------------

    train_one_step_pred, _ = (
        run_one_step_validation(
            A_K,
            B_K,
            C,
            train_states,
            train_inputs,
            "TRAINING"
        )
    )

    # --------------------------------------------------------
    # VALIDATION ONE-STEP
    # --------------------------------------------------------

    val_one_step_pred, _ = (
        run_one_step_validation(
            A_K,
            B_K,
            C,
            val_states,
            val_inputs,
            "VALIDATION"
        )
    )

    # --------------------------------------------------------
    # FULL RECURSIVE VALIDATION
    # --------------------------------------------------------

    (
        val_recursive_pred,
        _
    ) = run_recursive_validation(
        A_K,
        B_K,
        C,
        val_states,
        val_inputs,
        "VALIDATION"
    )

    # --------------------------------------------------------
    # SHORT HORIZON TEST
    # --------------------------------------------------------

    short_horizon_test(
        A_K,
        B_K,
        C,
        val_states,
        val_inputs
    )

    # --------------------------------------------------------
    # XY PLOT
    # --------------------------------------------------------

    reference_xy = val_df[
        [
            "x_ref",
            "y_ref"
        ]
    ].to_numpy(
        dtype=float
    ).T

    plot_xy(
        val_states,
        val_recursive_pred,
        reference_xy,
        os.path.join(
            RESULTS_DIR,
            "xy_trajectory.png"
        )
    )

    # --------------------------------------------------------
    # STATE PLOTS
    # --------------------------------------------------------

    plot_states(
        val_states,
        val_recursive_pred,
        val_df[
            "time"
        ].to_numpy(),
        RESULTS_DIR
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    save_results(
        A_K,
        B_K,
        C,
        eigenvalues,
        singular_values,
        rank,
        condition_number,
        retained_energy,
        RESULTS_DIR
    )

    print("\n" + "=" * 70)
    print(
        "KOOPMAN PIPELINE COMPLETE"
    )
    print("=" * 70)

    print(
        f"\nResults saved to:"
        f"\n  {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()