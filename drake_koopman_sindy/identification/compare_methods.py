"""
Compare Methods on Real Drake Circle-Tracking Data
======================================================
Runs plain DMDc (team's existing baseline method), EDMDc (Koopman/trig lift),
and SINDy side-by-side on the SAME Drake flight data, so you have one clean
comparison table/plot for your report.

Run this AFTER drake_sim/circle_tracking.py has produced data/X_drake.npy etc.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from edmdc import trig_dictionary, fit_edmdc, fit_plain_dmdc
from sindy import fit_sindy, print_equations

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

STATE_LABELS = ["x", "y", "z", "roll", "pitch", "yaw",
                "vx", "vy", "vz", "wx", "wy", "wz"]


def main():
    X = np.load(os.path.join(DATA_DIR, "X_drake.npy"))
    Xprime = np.load(os.path.join(DATA_DIR, "Xprime_drake.npy"))
    U = np.load(os.path.join(DATA_DIR, "U_drake.npy"))

    print("=" * 70)
    print(f"Drake circle data: X{X.shape}, Xprime{Xprime.shape}, U{U.shape}")
    print("=" * 70)

    results = {}

    # 1. Plain DMDc (team's existing baseline method, for apples-to-apples comparison)
    A_dmdc, B_dmdc, rmse_dmdc, eigs_dmdc = fit_plain_dmdc(X, Xprime, U)
    results["Plain DMDc"] = rmse_dmdc
    print(f"\n[1] Plain DMDc      : RMSE={rmse_dmdc:.6f} | max|eig|={np.max(np.abs(eigs_dmdc)):.4f}")

    # 2. EDMDc with trig dictionary on angle states (roll=3, pitch=4, yaw=5)
    lift = lambda Xin: trig_dictionary(Xin, angle_indices=[3, 4, 5])
    A_edmd, B_edmd, rmse_edmd, eigs_edmd = fit_edmdc(X, Xprime, U, lift)
    results["EDMDc (trig)"] = rmse_edmd
    print(f"[2] EDMDc (trig)    : RMSE={rmse_edmd:.6f} | max|eig|={np.max(np.abs(eigs_edmd)):.4f} "
          f"| lifted dim={A_edmd.shape[0]}")

    # 3. SINDy -- discrete-map mode (dt=None), i.e. fits directly against
    # next-state, matching the project's x_{k+1}=Ax+Bu convention for a fair
    # apples-to-apples comparison with DMDc/EDMDc on this dataset.
    Xi, Theta, names, rmse_sindy, n_terms = fit_sindy(
        X, Xprime, U, poly_order=1, include_trig_idx=[3, 4, 5], threshold=0.03, dt=None
    )
    results["SINDy"] = rmse_sindy
    print(f"[3] SINDy           : RMSE={rmse_sindy:.6f} | active terms={n_terms}/{len(names)}")

    print("\nSINDy identified equations (first few states, discrete next-state form):")
    print_equations(Xi, names, state_labels=STATE_LABELS, max_terms_per_eq=5, derivative_form=False)

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Method':<20}{'One-step RMSE':<18}{'Notes'}")
    print("-" * 70)
    print(f"{'Plain DMDc':<20}{rmse_dmdc:<18.6f}{'Team baseline, no lifting'}")
    print(f"{'EDMDc (trig)':<20}{rmse_edmd:<18.6f}{'Koopman lift, dense operator'}")
    print(f"{'SINDy':<20}{rmse_sindy:<18.6f}{'Sparse symbolic equation'}")
    print("=" * 70)

    # Bar chart
    plt.figure()
    plt.bar(results.keys(), results.values())
    plt.ylabel("One-step prediction RMSE")
    plt.title("Identification method comparison (Drake circle data)")
    plt.savefig(os.path.join(PLOT_DIR, "method_comparison.png"))
    plt.close()
    print("\nComparison bar chart saved to plots/method_comparison.png")


if __name__ == "__main__":
    main()
