import matplotlib

# IMPORTANT: WSL/headless plotting
matplotlib.use("Agg")

import csv
import os
import numpy as np
import matplotlib.pyplot as plt


DATA = "../processed/flight08_final.csv"
RESULTS = "../results/final"
OUT = "../results/final/plots"

os.makedirs(OUT, exist_ok=True)


# =========================================================
# LOAD DMDc RESULTS
# =========================================================

true = np.loadtxt(
    f"{RESULTS}/true.csv",
    delimiter=","
)

pred = np.loadtxt(
    f"{RESULTS}/pred.csv",
    delimiter=","
)


# =========================================================
# RECONSTRUCT TRAINING NORMALIZATION
# =========================================================

rows = list(
    csv.DictReader(
        open(DATA, newline="")
    )
)

X = np.array([
    [
        float(r["q0"]),
        float(r["q1"]),
        float(r["q2"]),
        float(r["q3"]),
        float(r["wx"]),
        float(r["wy"]),
        float(r["wz"])
    ]
    for r in rows
])

split = int(0.70 * len(X))

X_train = X[:split]

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

X_std[X_std < 1e-12] = 1.0


# Normalized -> physical

true_phys = true * X_std + X_mean
pred_phys = pred * X_std + X_mean


# =========================================================
# QUATERNION -> EULER
# =========================================================

def quat_to_euler(q):

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    # Normalize
    norm = np.sqrt(
        q0*q0 + q1*q1 + q2*q2 + q3*q3
    )

    q0 = q0 / norm
    q1 = q1 / norm
    q2 = q2 / norm
    q3 = q3 / norm

    roll = np.arctan2(
        2*(q0*q1 + q2*q3),
        1 - 2*(q1*q1 + q2*q2)
    )

    pitch_arg = 2*(q0*q2 - q3*q1)

    pitch_arg = np.clip(
        pitch_arg,
        -1.0,
        1.0
    )

    pitch = np.arcsin(pitch_arg)

    yaw = np.arctan2(
        2*(q0*q3 + q1*q2),
        1 - 2*(q2*q2 + q3*q3)
    )

    return (
        np.rad2deg(roll),
        np.rad2deg(pitch),
        np.rad2deg(yaw)
    )


true_roll, true_pitch, true_yaw = quat_to_euler(
    true_phys[:, :4]
)

pred_roll, pred_pitch, pred_yaw = quat_to_euler(
    pred_phys[:, :4]
)


# Unwrap for visualisation only

true_yaw = np.rad2deg(
    np.unwrap(
        np.deg2rad(true_yaw)
    )
)

pred_yaw = np.rad2deg(
    np.unwrap(
        np.deg2rad(pred_yaw)
    )
)


# =========================================================
# PLOT 1: ACTUAL VS PREDICTED ATTITUDE
# =========================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 8),
    dpi=150,
    sharex=True
)

curves = [
    ("Roll", true_roll, pred_roll),
    ("Pitch", true_pitch, pred_pitch),
    ("Yaw", true_yaw, pred_yaw)
]

for ax, (name, actual, predicted) in zip(
    axes,
    curves
):

    ax.plot(
        actual,
        linewidth=1.5,
        label="Actual"
    )

    ax.plot(
        predicted,
        linestyle="--",
        linewidth=1.2,
        label="DMDc Prediction"
    )

    ax.set_ylabel(
        f"{name} (deg)"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        loc="upper right"
    )


axes[-1].set_xlabel(
    "Test sample"
)

fig.suptitle(
    "Quadcopter Attitude: Actual vs DMDc Prediction"
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

fig.savefig(
    f"{OUT}/actual_vs_predicted_attitude.png",
    dpi=150,
    facecolor="white"
)

plt.close(fig)


# =========================================================
# PLOT 2: ACTUAL VS PREDICTED ANGULAR RATES
# =========================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 8),
    dpi=150,
    sharex=True
)

curves = [
    ("wx", true_phys[:, 4], pred_phys[:, 4]),
    ("wy", true_phys[:, 5], pred_phys[:, 5]),
    ("wz", true_phys[:, 6], pred_phys[:, 6])
]

for ax, (name, actual, predicted) in zip(
    axes,
    curves
):

    ax.plot(
        actual,
        linewidth=1.5,
        label="Actual"
    )

    ax.plot(
        predicted,
        linestyle="--",
        linewidth=1.2,
        label="DMDc Prediction"
    )

    ax.set_ylabel(
        f"{name} (rad/s)"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        loc="upper right"
    )


axes[-1].set_xlabel(
    "Test sample"
)

fig.suptitle(
    "Angular Rates: Actual vs DMDc Prediction"
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

fig.savefig(
    f"{OUT}/actual_vs_predicted_rates.png",
    dpi=150,
    facecolor="white"
)

plt.close(fig)


# =========================================================
# PLOT 3: ATTITUDE ERRORS
# =========================================================

roll_error = pred_roll - true_roll
pitch_error = pred_pitch - true_pitch
yaw_error = pred_yaw - true_yaw


fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 8),
    dpi=150,
    sharex=True
)

errors = [
    ("Roll error", roll_error),
    ("Pitch error", pitch_error),
    ("Yaw error", yaw_error)
]

for ax, (name, error) in zip(
    axes,
    errors
):

    ax.plot(
        error,
        linewidth=1.4
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_ylabel(
        f"{name} (deg)"
    )

    ax.grid(
        True,
        alpha=0.25
    )


axes[-1].set_xlabel(
    "Test sample"
)

fig.suptitle(
    "DMDc Attitude Prediction Error"
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

fig.savefig(
    f"{OUT}/attitude_error.png",
    dpi=150,
    facecolor="white"
)

plt.close(fig)


# =========================================================
# PLOT 4: QUATERNION COMPONENTS
# =========================================================

fig, axes = plt.subplots(
    4,
    1,
    figsize=(12, 10),
    dpi=150,
    sharex=True
)

for i, name in enumerate(
    ["q0", "q1", "q2", "q3"]
):

    axes[i].plot(
        true_phys[:, i],
        linewidth=1.5,
        label="Actual"
    )

    axes[i].plot(
        pred_phys[:, i],
        linestyle="--",
        linewidth=1.2,
        label="DMDc"
    )

    axes[i].set_ylabel(name)
    axes[i].grid(True, alpha=0.25)
    axes[i].legend(loc="upper right")


axes[-1].set_xlabel(
    "Test sample"
)

fig.suptitle(
    "Quaternion State: Actual vs DMDc"
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

fig.savefig(
    f"{OUT}/quaternion_components.png",
    dpi=150,
    facecolor="white"
)

plt.close(fig)


# =========================================================
# PLOT 5: 3D ATTITUDE TRAJECTORY
# =========================================================

fig = plt.figure(
    figsize=(10, 8),
    dpi=150
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot(
    true_roll,
    true_pitch,
    true_yaw,
    linewidth=1.5,
    label="Actual"
)

ax.plot(
    pred_roll,
    pred_pitch,
    pred_yaw,
    linestyle="--",
    linewidth=1.2,
    label="DMDc"
)

ax.set_xlabel("Roll (deg)")
ax.set_ylabel("Pitch (deg)")
ax.set_zlabel("Yaw (deg)")

ax.set_title(
    "Attitude-Space Trajectory"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    f"{OUT}/attitude_3d.png",
    dpi=150,
    facecolor="white"
)

plt.close(fig)


# =========================================================
# VERIFY FILES
# =========================================================

print()
print("Plots generated:")

for filename in sorted(os.listdir(OUT)):

    path = os.path.join(
        OUT,
        filename
    )

    size = os.path.getsize(path)

    print(
        f"{filename}: {size} bytes"
    )