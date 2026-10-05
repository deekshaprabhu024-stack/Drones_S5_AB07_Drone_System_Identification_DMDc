import csv
import numpy as np
import matplotlib.pyplot as plt


INPUT = "../processed/flight08_final.csv"
RESULTS = "../results/final"

import os
os.makedirs(RESULTS, exist_ok=True)


# =========================================================
# LOAD
# =========================================================

rows = list(csv.DictReader(open(INPUT, newline="")))

N = len(rows)

print("Samples:", N)


# =========================================================
# STATE
#
# x = [q0 q1 q2 q3 wx wy wz]
# =========================================================

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


# =========================================================
# MOTOR INPUT
# =========================================================

M = np.array([
    [
        float(r["motor1"]),
        float(r["motor2"]),
        float(r["motor3"]),
        float(r["motor4"])
    ]
    for r in rows
])


# =========================================================
# MOTOR MIXING
#
# Coordinate transformation:
#
# u0 = collective
# u1-u3 = differential actuation coordinates
#
# We do NOT claim exact physical roll/pitch/yaw signs
# until motor numbering is verified.
# =========================================================

T = np.array([
    [1,  1,  1,  1],
    [1, -1,  1, -1],
    [1,  1, -1, -1],
    [1, -1, -1,  1]
], dtype=float) / 2.0

U = M @ T.T


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

split = int(0.70 * N)

X_train_raw = X[:split]
X_test_raw = X[split:]

U_train_raw = U[:split]
U_test_raw = U[split:]


# =========================================================
# NORMALIZE USING TRAIN DATA ONLY
# =========================================================

X_mean = X_train_raw.mean(axis=0)
X_std = X_train_raw.std(axis=0)

U_mean = U_train_raw.mean(axis=0)
U_std = U_train_raw.std(axis=0)

X_std[X_std < 1e-12] = 1
U_std[U_std < 1e-12] = 1


X_train = (X_train_raw - X_mean) / X_std
X_test = (X_test_raw - X_mean) / X_std

U_train = (U_train_raw - U_mean) / U_std
U_test = (U_test_raw - U_mean) / U_std


# =========================================================
# DMDc
#
# x(k+1) = A x(k) + B u(k)
# =========================================================

X1 = X_train[:-1].T
X2 = X_train[1:].T
U1 = U_train[:-1].T

Omega = np.vstack([
    X1,
    U1
])

AB = X2 @ np.linalg.pinv(
    Omega,
    rcond=1e-8
)

n = 7

A = AB[:, :n]
B = AB[:, n:]


# =========================================================
# ONE-STEP VALIDATION
# =========================================================

pred = []

for k in range(len(X_test) - 1):

    x = X_test[k]
    u = U_test[k]

    pred.append(
        A @ x + B @ u
    )

pred = np.asarray(pred)

true = X_test[1:]


# =========================================================
# QUATERNION NORMALIZATION
#
# Convert predicted normalized state back to physical
# units, normalize quaternion, then return to normalized
# coordinates.
# =========================================================

for k in range(len(pred)):

    physical = pred[k] * X_std + X_mean

    q = physical[:4]
    norm = np.linalg.norm(q)

    if norm > 1e-12:
        physical[:4] = q / norm

    pred[k] = (
        physical - X_mean
    ) / X_std


# =========================================================
# METRICS
# =========================================================

names = [
    "q0",
    "q1",
    "q2",
    "q3",
    "wx",
    "wy",
    "wz"
]

print()
print("=" * 65)
print("ONE-STEP DMDc VALIDATION")
print("=" * 65)

for i, name in enumerate(names):

    error = true[:, i] - pred[:, i]

    rmse = np.sqrt(np.mean(error ** 2))

    ss_res = np.sum(error ** 2)

    ss_tot = np.sum(
        (true[:, i] - np.mean(true[:, i])) ** 2
    )

    r2 = 1.0 - ss_res / ss_tot

    print(
        f"{name:>3}: "
        f"RMSE={rmse:.6f} | "
        f"R2={r2:.6f}"
    )


# =========================================================
# ANGULAR-RATE R2
#
# More useful than quaternion-component R2 because
# angular rates are direct physical states.
# =========================================================

print()
print("=" * 65)
print("ANGULAR-RATE R2")
print("=" * 65)

for i, name in enumerate(["wx", "wy", "wz"], start=4):

    error = true[:, i] - pred[:, i]

    ss_res = np.sum(error ** 2)

    ss_tot = np.sum(
        (true[:, i] - np.mean(true[:, i])) ** 2
    )

    r2 = 1.0 - ss_res / ss_tot

    rmse = np.sqrt(np.mean(error ** 2)) * X_std[i]

    print(
        f"{name}: "
        f"RMSE={rmse:.6f} rad/s | "
        f"R2={r2:.6f}"
    )


# =========================================================
# QUATERNION ORIENTATION ERROR
# =========================================================

q_true = (
    true[:, :4] * X_std[:4]
    + X_mean[:4]
)

q_pred = (
    pred[:, :4] * X_std[:4]
    + X_mean[:4]
)

q_true /= np.linalg.norm(
    q_true,
    axis=1,
    keepdims=True
)

q_pred /= np.linalg.norm(
    q_pred,
    axis=1,
    keepdims=True
)

dots = np.sum(
    q_true * q_pred,
    axis=1
)

dots = np.clip(
    np.abs(dots),
    0,
    1
)

orientation_error_rad = (
    2 * np.arccos(dots)
)

orientation_error_deg = np.rad2deg(
    orientation_error_rad
)

orientation_rmse = np.sqrt(
    np.mean(
        orientation_error_deg ** 2
    )
)

orientation_mean = np.mean(
    orientation_error_deg
)

orientation_max = np.max(
    orientation_error_deg
)

print()
print("=" * 65)
print("ORIENTATION METRICS")
print("=" * 65)

print(
    f"Quaternion orientation RMSE : "
    f"{orientation_rmse:.4f} deg"
)

print(
    f"Mean orientation error      : "
    f"{orientation_mean:.4f} deg"
)

print(
    f"Maximum orientation error   : "
    f"{orientation_max:.4f} deg"
)



# =========================================================
# EIGENVALUES
# =========================================================

eigvals = np.linalg.eigvals(A)

print()
print("Eigenvalues:")

for e in eigvals:
    print(e)

rho = np.max(np.abs(eigvals))

print("\nSpectral radius:", rho)


# =========================================================
# SAVE MATRICES
# =========================================================

np.savetxt(
    f"{RESULTS}/A.csv",
    A,
    delimiter=","
)

np.savetxt(
    f"{RESULTS}/B.csv",
    B,
    delimiter=","
)

np.savetxt(
    f"{RESULTS}/true.csv",
    true,
    delimiter=","
)

np.savetxt(
    f"{RESULTS}/pred.csv",
    pred,
    delimiter=","
)


# =========================================================
# PLOT QUATERNION COMPONENTS
# =========================================================

fig, ax = plt.subplots(figsize=(12, 6))

for i, name in enumerate(["q0","q1","q2","q3"]):

    ax.plot(
        true[:, i],
        label=f"{name} measured"
    )

    ax.plot(
        pred[:, i],
        "--",
        label=f"{name} DMDc"
    )

ax.set_title(
    "DMDc One-Step Quaternion Prediction"
)

ax.set_xlabel("Test sample")
ax.set_ylabel("Normalized state")

ax.legend(
    ncol=2
)

fig.tight_layout()

fig.savefig(
    f"{RESULTS}/quaternion_prediction.png",
    dpi=200
)

plt.close(fig)


# =========================================================
# PLOT ANGULAR RATES
# =========================================================

fig, ax = plt.subplots(figsize=(12, 6))

for i, name in enumerate(["wx","wy","wz"], start=4):

    ax.plot(
        true[:, i],
        label=f"{name} measured"
    )

    ax.plot(
        pred[:, i],
        "--",
        label=f"{name} DMDc"
    )

ax.set_title(
    "DMDc One-Step Angular Rate Prediction"
)

ax.set_xlabel("Test sample")
ax.set_ylabel("Normalized angular rate")

ax.legend()

fig.tight_layout()

fig.savefig(
    f"{RESULTS}/angular_rate_prediction.png",
    dpi=200
)

plt.close(fig)


# =========================================================
# ORIENTATION ERROR
# =========================================================

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    orientation_error_deg
)

ax.set_title(
    "DMDc Quaternion Orientation Prediction Error"
)

ax.set_xlabel("Test sample")
ax.set_ylabel("Angular error (deg)")

fig.tight_layout()

fig.savefig(
    f"{RESULTS}/orientation_error.png",
    dpi=200
)

plt.close(fig)


# =========================================================
# INPUTS
# =========================================================

fig, ax = plt.subplots(
    figsize=(12, 5)
)

for i in range(4):

    ax.plot(
        U[:, i],
        label=f"u{i}"
    )

ax.set_title(
    "DMDc Control Input Coordinates"
)

ax.set_xlabel("Sample")
ax.set_ylabel("Input")

ax.legend()

fig.tight_layout()

fig.savefig(
    f"{RESULTS}/inputs.png",
    dpi=200
)

plt.close(fig)


print()
print("Saved final results to:", RESULTS)