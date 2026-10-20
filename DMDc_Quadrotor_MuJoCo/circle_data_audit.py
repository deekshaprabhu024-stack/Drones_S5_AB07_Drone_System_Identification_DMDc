import os
import numpy as np
import pandas as pd

# ============================================================
# CIRCULAR QUADROTOR — DATASET VALIDITY AUDIT
# ============================================================

TRAIN_FILE = os.path.join("data", "circle_training.csv")
VAL_FILE   = os.path.join("data", "circle_validation.csv")

STATE_COLS = [
    "x", "y", "z",
    "vx", "vy", "vz",
    "roll", "pitch", "yaw",
    "p", "q", "r"
]

REF_COLS = [
    "x_ref", "y_ref", "z_ref",
    "vx_ref", "vy_ref", "vz_ref"
]

INPUT_COLS = ["u1", "u2", "u3", "u4"]


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nFile not found:\n{path}\n"
            f"\nMake sure the CSV is located at this path."
        )

    df = pd.read_csv(path)

    print(f"\nLoaded: {path}")
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    return df


def check_columns(df, name):
    required = ["time"] + STATE_COLS + REF_COLS + INPUT_COLS

    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"\n{name}: MISSING COLUMNS")
        for c in missing:
            print("  ", c)
        return False

    print(f"\n{name}: all required columns present.")
    return True


def dataset_summary(df, name):
    print("\n" + "=" * 70)
    print(f"{name} SUMMARY")
    print("=" * 70)

    print("\nTime:")
    print(f"  Start : {df['time'].iloc[0]:.9f}")
    print(f"  End   : {df['time'].iloc[-1]:.9f}")
    print(f"  Span  : {df['time'].iloc[-1] - df['time'].iloc[0]:.9f}")

    dt = np.diff(df["time"].to_numpy())

    print("\nTime step:")
    print(f"  Mean : {np.mean(dt):.12f}")
    print(f"  Min  : {np.min(dt):.12f}")
    print(f"  Max  : {np.max(dt):.12f}")

    print("\nMissing values:")
    missing = df.isna().sum().sum()
    print(f"  Total missing entries: {missing}")

    print("\nState ranges:")

    for col in STATE_COLS:
        arr = df[col].to_numpy(dtype=float)

        print(
            f"  {col:5s}: "
            f"min={np.min(arr): .8f}, "
            f"max={np.max(arr): .8f}, "
            f"mean={np.mean(arr): .8f}, "
            f"std={np.std(arr): .8f}"
        )

    print("\nInput ranges:")

    for col in INPUT_COLS:
        arr = df[col].to_numpy(dtype=float)

        print(
            f"  {col:3s}: "
            f"min={np.min(arr): .8f}, "
            f"max={np.max(arr): .8f}, "
            f"mean={np.mean(arr): .8f}, "
            f"std={np.std(arr): .8f}"
        )


def check_dt(df, name):
    dt = np.diff(df["time"].to_numpy(dtype=float))

    expected = 0.002

    print("\n" + "-" * 70)
    print(f"{name} DT CHECK")
    print("-" * 70)

    print(f"Expected dt : {expected}")
    print(f"Mean dt     : {np.mean(dt):.12f}")
    print(f"Min dt      : {np.min(dt):.12f}")
    print(f"Max dt      : {np.max(dt):.12f}")

    max_error = np.max(np.abs(dt - expected))

    print(f"Maximum dt error: {max_error:.12e}")

    if max_error < 1e-10:
        print("RESULT: PASS — constant 0.002 s sampling.")
    else:
        print("RESULT: WARNING — timestep is not exactly 0.002 s.")


def compare_datasets(train, val):
    print("\n" + "=" * 70)
    print("TRAINING vs VALIDATION COMPARISON")
    print("=" * 70)

    if list(train.columns) != list(val.columns):
        print("\nColumn ordering differs.")
        print("Train columns:")
        print(train.columns.tolist())
        print("\nValidation columns:")
        print(val.columns.tolist())

    print(f"\nSame number of rows: {len(train) == len(val)}")
    print(f"Same number of columns: {len(train.columns) == len(val.columns)}")

    common_cols = [
        c for c in train.columns
        if c in val.columns
    ]

    train_common = train[common_cols]
    val_common = val[common_cols]

    if train_common.shape == val_common.shape:

        a = train_common.to_numpy(dtype=float)
        b = val_common.to_numpy(dtype=float)

        exact_equal = np.array_equal(a, b)

        max_abs_difference = np.max(np.abs(a - b))
        mean_abs_difference = np.mean(np.abs(a - b))

        print("\nNumerical comparison:")
        print(f"  Exact equality       : {exact_equal}")
        print(f"  Maximum abs diff     : {max_abs_difference:.12e}")
        print(f"  Mean absolute diff   : {mean_abs_difference:.12e}")

        if exact_equal:
            print(
                "\nRESULT: WARNING — training and validation files "
                "are byte-level numerically identical."
            )
            print(
                "They cannot be treated as independent trajectories."
            )
        elif max_abs_difference < 1e-12:
            print(
                "\nRESULT: WARNING — datasets are effectively identical "
                "within numerical precision."
            )
        else:
            print(
                "\nRESULT: PASS — datasets contain numerical differences."
            )

    # Compare trajectory portions independently
    for col in STATE_COLS:
        a = train[col].to_numpy(dtype=float)
        b = val[col].to_numpy(dtype=float)

        if len(a) == len(b):
            diff = a - b

            print(
                f"\n{col}: "
                f"max_diff={np.max(np.abs(diff)):.8e}, "
                f"RMSE={np.sqrt(np.mean(diff**2)):.8e}"
            )


def trajectory_geometry(df, name):
    print("\n" + "=" * 70)
    print(f"{name} CIRCULAR TRAJECTORY GEOMETRY")
    print("=" * 70)

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    x_ref = df["x_ref"].to_numpy(dtype=float)
    y_ref = df["y_ref"].to_numpy(dtype=float)

    # Estimate circle center from actual XY trajectory
    cx = np.mean(x)
    cy = np.mean(y)

    radius = np.sqrt((x - cx)**2 + (y - cy)**2)

    print("\nEstimated actual trajectory center:")
    print(f"  cx = {cx:.8f}")
    print(f"  cy = {cy:.8f}")

    print("\nActual XY radius:")
    print(f"  mean = {np.mean(radius):.8f}")
    print(f"  std  = {np.std(radius):.8f}")
    print(f"  min  = {np.min(radius):.8f}")
    print(f"  max  = {np.max(radius):.8f}")

    # Reference trajectory geometry
    ref_cx = np.mean(x_ref)
    ref_cy = np.mean(y_ref)

    ref_radius = np.sqrt(
        (x_ref - ref_cx)**2 +
        (y_ref - ref_cy)**2
    )

    print("\nReference XY trajectory center:")
    print(f"  cx = {ref_cx:.8f}")
    print(f"  cy = {ref_cy:.8f}")

    print("\nReference XY radius:")
    print(f"  mean = {np.mean(ref_radius):.8f}")
    print(f"  std  = {np.std(ref_radius):.8f}")
    print(f"  min  = {np.min(ref_radius):.8f}")
    print(f"  max  = {np.max(ref_radius):.8f}")

    # Actual tracking error
    tracking_error = np.sqrt(
        (x - x_ref)**2 +
        (y - y_ref)**2
    )

    print("\nActual XY tracking error:")
    print(f"  RMSE = {np.sqrt(np.mean(tracking_error**2)):.8f}")
    print(f"  Mean = {np.mean(tracking_error):.8f}")
    print(f"  Max  = {np.max(tracking_error):.8f}")


def angular_progress(df, name):
    print("\n" + "=" * 70)
    print(f"{name} ANGULAR PROGRESS")
    print("=" * 70)

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    cx = np.mean(x)
    cy = np.mean(y)

    theta = np.unwrap(np.arctan2(y - cy, x - cx))

    total_angle = theta[-1] - theta[0]

    print(f"\nInitial angle : {theta[0]:.8f} rad")
    print(f"Final angle   : {theta[-1]:.8f} rad")
    print(f"Total change  : {total_angle:.8f} rad")
    print(f"Total rotations: {total_angle / (2*np.pi):.8f}")

    if abs(total_angle) > 2*np.pi:
        print("RESULT: Multiple circular revolutions detected.")
    elif abs(total_angle) > np.pi:
        print("RESULT: Significant circular motion detected.")
    else:
        print("RESULT: Less than one half-turn of angular progress.")


def sample_points(df, name):
    print("\n" + "=" * 70)
    print(f"{name} SAMPLE POINTS")
    print("=" * 70)

    indices = [
        0,
        len(df) // 4,
        len(df) // 2,
        3 * len(df) // 4,
        len(df) - 1
    ]

    for i in indices:
        row = df.iloc[i]

        print(
            f"\nindex={i:5d}, "
            f"time={row['time']:.6f}"
        )

        print(
            f"  actual    = "
            f"({row['x']:.6f}, "
            f"{row['y']:.6f}, "
            f"{row['z']:.6f})"
        )

        print(
            f"  reference = "
            f"({row['x_ref']:.6f}, "
            f"{row['y_ref']:.6f}, "
            f"{row['z_ref']:.6f})"
        )


def main():

    print("=" * 70)
    print("CIRCULAR QUADROTOR — DATA VALIDITY AUDIT")
    print("=" * 70)

    train = load_csv(TRAIN_FILE)
    val = load_csv(VAL_FILE)

    train_ok = check_columns(train, "TRAINING")
    val_ok = check_columns(val, "VALIDATION")

    if not train_ok or not val_ok:
        print("\nERROR: Required columns are missing.")
        return

    dataset_summary(train, "TRAINING")
    dataset_summary(val, "VALIDATION")

    check_dt(train, "TRAINING")
    check_dt(val, "VALIDATION")

    compare_datasets(train, val)

    trajectory_geometry(train, "TRAINING")
    trajectory_geometry(val, "VALIDATION")

    angular_progress(train, "TRAINING")
    angular_progress(val, "VALIDATION")

    sample_points(train, "TRAINING")
    sample_points(val, "VALIDATION")

    print("\n" + "=" * 70)
    print("FINAL AUDIT")
    print("=" * 70)

    if len(train) == len(val):
        print(f"Training samples   : {len(train)}")
        print(f"Validation samples : {len(val)}")

    print("\nAudit complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()