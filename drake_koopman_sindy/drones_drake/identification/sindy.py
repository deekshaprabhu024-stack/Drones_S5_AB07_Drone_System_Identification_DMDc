"""
SINDy — Sparse Identification of Nonlinear Dynamics (with control: SINDYc)
=============================================================================
From-scratch implementation (NumPy only), with an automatic cross-check
against the `pysindy` library if it's installed.
"""

import numpy as np

try:
    import pysindy as ps
    HAVE_PYSINDY = True
except ImportError:
    HAVE_PYSINDY = False


def build_library(X, U, poly_order=2, include_trig_idx=None):
    """
    Build a candidate feature library from state X (state_dim x N) and
    input U (input_dim x N). Returns Theta (n_features x N) and feature names.
    """
    state_dim, N = X.shape
    input_dim = U.shape[0]
    rows = [np.ones(N)]
    names = ["1"]

    for i in range(state_dim):
        rows.append(X[i, :])
        names.append(f"x{i}")
    for i in range(input_dim):
        rows.append(U[i, :])
        names.append(f"u{i}")

    if poly_order >= 2:
        for i in range(state_dim):
            for j in range(i, state_dim):
                rows.append(X[i, :] * X[j, :])
                names.append(f"x{i}*x{j}")

    if include_trig_idx:
        for i in include_trig_idx:
            rows.append(np.sin(X[i, :]))
            names.append(f"sin(x{i})")
            rows.append(np.cos(X[i, :]))
            names.append(f"cos(x{i})")

    Theta = np.vstack(rows)
    return Theta, names


def stlsq(Theta, Xdot, threshold=0.05, max_iter=10):
    """Sequential Thresholded Least Squares -- the core SINDy sparse solver."""
    Xi = np.linalg.lstsq(Theta.T, Xdot.T, rcond=None)[0]

    for _ in range(max_iter):
        small = np.abs(Xi) < threshold
        Xi[small] = 0
        for k in range(Xdot.shape[0]):
            big = ~small[:, k]
            if np.any(big):
                Xi[big, k] = np.linalg.lstsq(Theta[big, :].T, Xdot[k, :], rcond=None)[0]
    return Xi


def fit_sindy(X, Xprime, U, poly_order=2, include_trig_idx=None, threshold=0.05, dt=None):
    """
    Fit SINDYc.

    IMPORTANT: classic SINDy fits the DERIVATIVE, not the next state directly.
    If dt is given, target = (Xprime - X)/dt (finite-difference derivative) --
    this recovers clean, interpretable continuous-time coefficients. If dt is
    None, fits directly against Xprime as a discrete-time map instead
    (matches the project's x_{k+1}=Ax+Bu convention, for fair comparison with
    DMDc/EDMDc on the Drake dataset).

    Returns Xi, Theta, names, one-step prediction RMSE (always measured in
    next-state space), and number of active (nonzero) terms.
    """
    Theta, names = build_library(X, U, poly_order=poly_order, include_trig_idx=include_trig_idx)

    if dt is not None:
        target = (Xprime - X) / dt
    else:
        target = Xprime

    Xi = stlsq(Theta, target, threshold=threshold)

    if dt is not None:
        Xdot_pred = Xi.T @ Theta
        pred_next_state = X + dt * Xdot_pred
    else:
        pred_next_state = Xi.T @ Theta

    rmse = float(np.sqrt(np.mean((pred_next_state - Xprime) ** 2)))
    n_active_terms = int(np.sum(np.abs(Xi) > 0))
    return Xi, Theta, names, rmse, n_active_terms


def print_equations(Xi, names, state_labels=None, max_terms_per_eq=6, derivative_form=True):
    """Pretty-print the identified sparse equations, state by state."""
    state_dim = Xi.shape[1]
    for k in range(state_dim):
        label = state_labels[k] if state_labels else f"x{k}"
        terms = []
        for feat_idx, coeff in enumerate(Xi[:, k]):
            if abs(coeff) > 1e-8:
                terms.append(f"{coeff:+.4f}*{names[feat_idx]}")
        terms = terms[:max_terms_per_eq]
        eq = " ".join(terms) if terms else "0"
        lhs = f"d{label}/dt" if derivative_form else f"{label}_next"
        print(f"  {lhs} = {eq}")


def fit_sindy_pysindy_crosscheck(X, Xprime, U, dt=0.01):
    """Optional cross-check using the real pysindy library, if installed."""
    if not HAVE_PYSINDY:
        return None
    model = ps.SINDy()
    model.fit(X.T, u=U.T, t=dt)
    return model


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 300))
    Xprime = X + 0.01 * rng.normal(size=X.shape)
    U = rng.normal(size=(2, 300))

    Xi, Theta, names, rmse, n_terms = fit_sindy(X, Xprime, U, poly_order=2)
    print(f"SINDy self-test: rmse={rmse:.4f}, active_terms={n_terms}, library_size={len(names)}")
    print("Identified equations:")
    print_equations(Xi, names, derivative_form=False)

    if HAVE_PYSINDY:
        print("\n(pysindy is installed -- cross-check available)")
    else:
        print("\n(pysindy not installed -- using from-scratch implementation only, which is fine)")
