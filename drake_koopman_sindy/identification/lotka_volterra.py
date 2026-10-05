"""
Lotka-Volterra Testbed — Validate EDMDc and SINDy on a Known Nonlinear System
=================================================================================
Addresses the faculty's note: "Lotka Volterra to make it more similar to real
world noise and condition."

Why this matters: before trusting EDMDc/SINDy on real Drake flight data, we
validate the pipeline on a system with KNOWN ground-truth nonlinear dynamics
(predator-prey). If the methods can't recover known dynamics here, they won't
work reliably on noisy real data either -- same "safety net" philosophy as the
team's Week-3 synthetic-data gate for plain DMDc.

The classic Lotka-Volterra system is autonomous (no control input). We add a
harvesting/stocking control term to make it a genuine SINDYc/EDMDc problem
("...with Control"), which is a standard, well-established extension in
ecological control literature.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import sys
sys.path.insert(0, os.path.dirname(__file__))
from edmdc import trig_dictionary, poly_dictionary, fit_edmdc, fit_plain_dmdc
from sindy import fit_sindy, print_equations

PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Lotka-Volterra with control (harvesting/stocking term)
#   dx/dt = alpha*x - beta*x*y + u1   (prey)
#   dy/dt = delta*x*y - gamma*y + u2  (predator)
# ----------------------------------------------------------------------
ALPHA, BETA, DELTA, GAMMA = 1.1, 0.4, 0.1, 0.4


def lv_dynamics(t, state, u_func):
    x, y = state
    u1, u2 = u_func(t)
    dx = ALPHA * x - BETA * x * y + u1
    dy = DELTA * x * y - GAMMA * y + u2
    return [dx, dy]


def generate_lv_data(T=50.0, dt=0.05, noise_std=0.02, seed=0):
    rng = np.random.default_rng(seed)
    t_eval = np.arange(0, T, dt)

    def u_func(t):
        # small random-ish harvesting/stocking control, smoothed
        return 0.05 * np.sin(0.3 * t), 0.05 * np.cos(0.2 * t)

    sol = solve_ivp(lv_dynamics, [0, T], [10.0, 5.0], t_eval=t_eval, args=(u_func,))
    X_clean = sol.y  # (2, N)
    U = np.array([u_func(t) for t in t_eval]).T  # (2, N)

    X_noisy = X_clean + rng.normal(0, noise_std, X_clean.shape)
    return X_noisy, U, t_eval


def main():
    X_full, U_full, t_eval = generate_lv_data()
    X = X_full[:, :-1]
    Xprime = X_full[:, 1:]
    U = U_full[:, :-1]

    print("=" * 60)
    print("Lotka-Volterra testbed: plain DMDc vs EDMDc (trig+poly) vs SINDy")
    print("=" * 60)

    # Plain DMDc baseline
    A_dmdc, B_dmdc, rmse_dmdc, eigs_dmdc = fit_plain_dmdc(X, Xprime, U)
    print(f"Plain DMDc        : one-step RMSE = {rmse_dmdc:.4f}")

    # EDMDc with trig dictionary only
    lift_trig = lambda Xin: trig_dictionary(Xin, angle_indices=[0, 1])
    A_t, B_t, rmse_trig, eigs_t = fit_edmdc(X, Xprime, U, lift_trig)
    print(f"EDMDc (trig only) : one-step RMSE = {rmse_trig:.4f}")

    # EDMDc with trig + polynomial (x*y) dictionary -- should close the gap,
    # since Lotka-Volterra's true nonlinearity IS the x*y product term
    lift_full = lambda Xin: poly_dictionary(trig_dictionary(Xin, [0, 1]), [(0, 1)])
    A_f, B_f, rmse_full, eigs_f = fit_edmdc(X, Xprime, U, lift_full)
    print(f"EDMDc (trig+poly) : one-step RMSE = {rmse_full:.4f}  "
          f"<- includes x*y term, matches true LV nonlinearity")

    # SINDy -- fit against the DERIVATIVE (dt=0.05, matching generate_lv_data),
    # not next-state directly. This is the classic SINDy formulation and is
    # what lets it recover true, interpretable ODE coefficients. Threshold is
    # small relative to the true parameter scale (alpha=1.1, beta=0.4, etc.)
    DT = 0.05
    Xi, Theta, names, rmse_sindy, n_terms = fit_sindy(
        X, Xprime, U, poly_order=2, threshold=0.02, dt=DT
    )
    print(f"SINDy              : one-step RMSE = {rmse_sindy:.4f}, "
          f"active terms = {n_terms}/{len(names)}")
    print("SINDy identified equations (continuous-time, dx/dt form):")
    print_equations(Xi, names, state_labels=["prey(x)", "predator(y)"])

    print("\nExpected ground truth form:")
    print("  dx/dt = 1.1*x - 0.4*x*y + u1")
    print("  dy/dt = 0.1*x*y - 0.4*y + u2")
    print("(Compare SINDy's identified equation above to this -- a close match")
    print(" is a strong, concrete validation result for your report.)")

    # Plot phase portrait
    plt.figure()
    plt.plot(X_full[0, :], X_full[1, :])
    plt.xlabel("prey population (x)")
    plt.ylabel("predator population (y)")
    plt.title("Lotka-Volterra phase portrait (noisy, with control)")
    plt.savefig(os.path.join(PLOT_DIR, "lotka_volterra_phase.png"))
    plt.close()

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Method':<20}{'One-step RMSE':<18}")
    print("-" * 38)
    print(f"{'Plain DMDc':<20}{rmse_dmdc:<18.4f}")
    print(f"{'EDMDc (trig)':<20}{rmse_trig:<18.4f}")
    print(f"{'EDMDc (trig+poly)':<20}{rmse_full:<18.4f}")
    print(f"{'SINDy':<20}{rmse_sindy:<18.4f}")
    print("=" * 60)
    print("Plot saved to plots/lotka_volterra_phase.png")


if __name__ == "__main__":
    main()
