"""
Meshgrid Vector Field Visualization
=======================================
This is the strongest visual "proof" you can show a reviewer: a grid of
arrows showing which direction the system moves at every point in state
space (a "vector field" or "phase portrait"), for the TRUE dynamics and the
IDENTIFIED (SINDy/EDMDc) dynamics side by side. If your identification is
good, the two arrow fields should look nearly identical.

Part A: Lotka-Volterra (2D state -- easy to grid and has known ground truth)
Part B: Drake drone, roll-pitch slice (12D state -- we fix all other states
         at hover/zero and grid just two of them, to get a readable 2D plot)

HOW A MESHGRID WORKS (explain this if asked):
np.meshgrid takes two 1D arrays (e.g. 50 roll values, 50 pitch values) and
builds two 2D arrays representing every (roll, pitch) COMBINATION -- a grid
of points covering the whole 2D plane. At each grid point, we evaluate the
dynamics (how fast roll/pitch change) and draw a small arrow. The collection
of arrows is the "vector field" -- it shows the system's behavior
EVERYWHERE, not just along one single trajectory.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from sindy import fit_sindy
from edmdc import trig_dictionary, fit_edmdc

PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

ALPHA, BETA, DELTA, GAMMA = 1.1, 0.4, 0.1, 0.4


# ======================================================================
# PART A: Lotka-Volterra -- true vs. SINDy-identified vector field
# ======================================================================
def lv_true_vector_field(X, Y):
    """True Lotka-Volterra dynamics (no control, for a clean field plot)."""
    dX = ALPHA * X - BETA * X * Y
    dY = DELTA * X * Y - GAMMA * Y
    return dX, dY


def lv_identified_vector_field(X, Y, Xi, names):
    """
    Evaluate the SINDy-identified equation at every grid point.
    Xi, names come from sindy.fit_sindy on Lotka-Volterra data (derivative mode).
    """
    # names order from build_library: "1", "x0", "x1", "u0", "u1", "x0*x0", "x0*x1", "x1*x1"
    # Build each term's value at every grid point, matching that order.
    N_pts = X.shape
    term_values = {
        "1": np.ones(N_pts),
        "x0": X,
        "x1": Y,
        "u0": np.zeros(N_pts),   # no control input for this field plot
        "u1": np.zeros(N_pts),
        "x0*x0": X * X,
        "x0*x1": X * Y,
        "x1*x1": Y * Y,
    }
    dX = np.zeros(N_pts)
    dY = np.zeros(N_pts)
    for feat_idx, name in enumerate(names):
        if name not in term_values:
            continue
        dX += Xi[feat_idx, 0] * term_values[name]
        dY += Xi[feat_idx, 1] * term_values[name]
    return dX, dY


def run_lotka_volterra_meshgrid():
    print("=" * 70)
    print("PART A: Lotka-Volterra meshgrid vector field (true vs SINDy)")
    print("=" * 70)

    # Re-generate the same LV data and SINDy fit as lotka_volterra.py,
    # so this plot is consistent with your other reported results.
    from scipy.integrate import solve_ivp

    def u_func(t):
        return 0.05 * np.sin(0.3 * t), 0.05 * np.cos(0.2 * t)

    def lv_dynamics(t, state, u_func):
        x, y = state
        u1, u2 = u_func(t)
        dx = ALPHA * x - BETA * x * y + u1
        dy = DELTA * x * y - GAMMA * y + u2
        return [dx, dy]

    T, dt = 50.0, 0.05
    t_eval = np.arange(0, T, dt)
    sol = solve_ivp(lv_dynamics, [0, T], [10.0, 5.0], t_eval=t_eval, args=(u_func,))
    X_clean = sol.y
    rng = np.random.default_rng(0)
    X_noisy = X_clean + rng.normal(0, 0.02, X_clean.shape)
    U = np.array([u_func(t) for t in t_eval]).T

    X_data = X_noisy[:, :-1]
    Xprime_data = X_noisy[:, 1:]
    U_data = U[:, :-1]

    Xi, Theta, names, rmse, n_terms = fit_sindy(
        X_data, Xprime_data, U_data, poly_order=2, threshold=0.02, dt=dt
    )

    # ---- Build the meshgrid ----
    x_range = np.linspace(0, 20, 25)   # prey population range
    y_range = np.linspace(0, 15, 25)   # predator population range
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    dX_true, dY_true = lv_true_vector_field(X_grid, Y_grid)
    dX_ident, dY_ident = lv_identified_vector_field(X_grid, Y_grid, Xi, names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].quiver(X_grid, Y_grid, dX_true, dY_true, color="black", alpha=0.7)
    axes[0].plot(X_clean[0, :], X_clean[1, :], color="red", linewidth=1.5, label="actual trajectory")
    axes[0].set_xlabel("prey population (x)")
    axes[0].set_ylabel("predator population (y)")
    axes[0].set_title("TRUE dynamics (ground-truth equations)")
    axes[0].legend()

    axes[1].quiver(X_grid, Y_grid, dX_ident, dY_ident, color="blue", alpha=0.7)
    axes[1].plot(X_clean[0, :], X_clean[1, :], color="red", linewidth=1.5, label="actual trajectory")
    axes[1].set_xlabel("prey population (x)")
    axes[1].set_ylabel("predator population (y)")
    axes[1].set_title("SINDy-IDENTIFIED dynamics (learned from noisy data)")
    axes[1].legend()

    plt.suptitle("Meshgrid vector field: every arrow = direction+speed of motion at that point")
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "meshgrid_lotka_volterra.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")
    print("How to read it: arrows point in the direction the system moves from that")
    print("point; arrow LENGTH = speed. If the two panels look alike, SINDy learned")
    print("the true dynamics well, not just memorized the one red trajectory.")


# ======================================================================
# PART B: Drake drone -- roll/pitch slice of the EDMDc-identified field
# ======================================================================
def run_drake_meshgrid():
    print("\n" + "=" * 70)
    print("PART B: Drake roll-pitch meshgrid vector field (EDMDc-identified)")
    print("=" * 70)

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    X_path = os.path.join(DATA_DIR, "X_drake.npy")
    if not os.path.exists(X_path):
        print("SKIPPED: data/X_drake.npy not found. Run drake_sim/circle_tracking.py "
              "first, then re-run this script to generate the drone meshgrid plot.")
        return

    X = np.load(X_path)
    Xprime = np.load(os.path.join(DATA_DIR, "Xprime_drake.npy"))
    U = np.load(os.path.join(DATA_DIR, "U_drake.npy"))

    lift = lambda Xin: trig_dictionary(Xin, angle_indices=[3, 4, 5])
    A, B, rmse, eigs = fit_edmdc(X, Xprime, U, lift)

    # Build a roll-pitch grid, all other states held at hover/zero
    roll_range = np.linspace(-0.3, 0.3, 20)
    pitch_range = np.linspace(-0.3, 0.3, 20)
    Roll, Pitch = np.meshgrid(roll_range, pitch_range)

    d_roll = np.zeros_like(Roll)
    d_pitch = np.zeros_like(Pitch)
    u_hover_guess = np.mean(U, axis=1)  # use the average observed hover input

    for i in range(Roll.shape[0]):
        for j in range(Roll.shape[1]):
            x_point = np.zeros(12)
            x_point[3] = Roll[i, j]
            x_point[4] = Pitch[i, j]
            psi = lift(x_point.reshape(-1, 1)).flatten()
            x_next_lifted = A @ psi + B @ u_hover_guess
            # roll, pitch are the 4th and 5th raw-state entries of the lifted vector
            d_roll[i, j] = x_next_lifted[3] - Roll[i, j]
            d_pitch[i, j] = x_next_lifted[4] - Pitch[i, j]

    plt.figure(figsize=(7, 6))
    plt.quiver(Roll, Pitch, d_roll, d_pitch, color="purple", alpha=0.8)
    plt.xlabel("roll (rad)")
    plt.ylabel("pitch (rad)")
    plt.title("EDMDc-identified attitude dynamics\n(roll-pitch slice, other states at hover)")
    out_path = os.path.join(PLOT_DIR, "meshgrid_drake_rollpitch.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")
    print("How to read it: arrows show how the IDENTIFIED model predicts roll/pitch")
    print("will change from each starting angle. Arrows pointing back toward (0,0)")
    print("indicate the identified model correctly captured stabilizing behavior.")


if __name__ == "__main__":
    run_lotka_volterra_meshgrid()
    run_drake_meshgrid()
