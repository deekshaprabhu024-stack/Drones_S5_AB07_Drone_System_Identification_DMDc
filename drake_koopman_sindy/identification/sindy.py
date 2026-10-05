"""
SINDy — Sparse Identification of Nonlinear Dynamics (with control: SINDYc)
=============================================================================
From-scratch implementation (NumPy only, no extra install needed), with an
automatic cross-check against the `pysindy` library if it's installed.

Where this differs from DMDc/EDMDc:
- DMDc/EDMDc fit a DENSE linear operator (possibly in a lifted space).
- SINDy instead builds a large LIBRARY of candidate nonlinear terms
  (polynomials, trig functions, products, etc.), then uses SPARSE regression
  (Sequential Thresholded Least Squares, STLSQ) to find the FEWEST terms that
  explain the observed derivative/next-state data. The result is a compact,
  human-readable symbolic equation rather than a dense matrix.

This is the method your faculty's "SINDy" note is pointing at, and it's a
genuinely different philosophy from DMDc/EDMDc -- worth presenting as a
head-to-head comparison, not just a mention.
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
    input U (input_dim x N).

    Includes:
      - constant term
      - raw state and input terms
      - polynomial terms up to poly_order (pairwise products, squares)
      - optionally sin/cos of specified state indices (include_trig_idx)

    Returns the library matrix Theta, shape (n_features, N), and a list of
    human-readable names for each feature (for printing the final equation).
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

    Theta = np.vstack(rows)  # (n_features, N)
    return Theta, names


def stlsq(Theta, Xdot, threshold=0.05, max_iter=10):
    """
    Sequential Thresholded Least Squares -- the core SINDy sparse solver.

    Theta : (n_features, N) library matrix
    Xdot  : (state_dim, N) target (next state or derivative)
    threshold : coefficients below this magnitude are zeroed each iteration

    Returns Xi: (n_features, state_dim) sparse coefficient matrix, such that
    Xdot ~= Xi.T @ Theta
    """
    # Solve least squares: Xdot.T = Theta.T @ Xi   =>  Xi = pinv(Theta.T) @ Xdot.T
    Xi = np.linalg.lstsq(Theta.T, Xdot.T, rcond=None)[0]  # (n_features, state_dim)

    for _ in range(max_iter):
        small = np.abs(Xi) < threshold
        Xi[small] = 0
        for k in range(Xdot.shape[0]):
            big = ~small[:, k]
            if np.any(big):
                Xi[big, k] = np.linalg.lstsq(Theta[big, :].T, Xdot[k, :], rcond=None)[0]
    return Xi


def fit_sindy(X, Xprime, U, poly_order=2, include_trig_idx=None, threshold=0.05,
              dt=None):
    """
    Fit SINDYc.

    IMPORTANT: classic SINDy fits the DERIVATIVE, not the next state directly.
    If dt is given, the target becomes Xdot = (Xprime - X) / dt (finite-difference
    derivative), and the returned Xi gives a genuine continuous-time ODE --
    this is what recovers clean, interpretable coefficients (e.g. matching
    true Lotka-Volterra parameters). If dt is None, fits directly against
    Xprime as a discrete-time sparse map instead (consistent with the project's
    x_{k+1}=Ax+Bu convention) -- use this mode for the Drake comparison, where
    we want apples-to-apples with DMDc/EDMDc's discrete-time formulation.

    Returns Xi (sparse coefficients), Theta library, feature names, one-step
    prediction RMSE (always measured in NEXT-STATE space for fair comparison
    with DMDc/EDMDc, regardless of which mode was used to fit), and dt (None
    if discrete-map mode was used).
    """
    Theta, names = build_library(X, U, poly_order=poly_order, include_trig_idx=include_trig_idx)

    if dt is not None:
        target = (Xprime - X) / dt
    else:
        target = Xprime

    Xi = stlsq(Theta, target, threshold=threshold)

    if dt is not None:
        # Predict next state via one Euler step using the identified derivative
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
    """
    Optional cross-check using the actual pysindy library, if installed.
    Returns None if pysindy isn't available (caller should handle gracefully).
    """
    if not HAVE_PYSINDY:
        return None
    model = ps.SINDy()
    # pysindy expects (N, state_dim) shaped arrays
    model.fit(X.T, u=U.T, t=dt)
    return model


if __name__ == "__main__":
    # Self-test on random data
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 300))
    Xprime = X + 0.01 * rng.normal(size=X.shape)
    U = rng.normal(size=(2, 300))

    Xi, Theta, names, rmse, n_terms = fit_sindy(X, Xprime, U, poly_order=2)
    print(f"SINDy self-test: rmse={rmse:.4f}, active_terms={n_terms}, library_size={len(names)}")
    print("Identified equations:")
    print_equations(Xi, names)

    if HAVE_PYSINDY:
        print("\n(pysindy is installed -- cross-check available via fit_sindy_pysindy_crosscheck)")
    else:
        print("\n(pysindy not installed -- using from-scratch implementation only, which is fine)")
