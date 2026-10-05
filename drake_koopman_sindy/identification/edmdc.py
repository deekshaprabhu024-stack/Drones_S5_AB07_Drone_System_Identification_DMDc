"""
EDMDc — Extended Dynamic Mode Decomposition with Control (Koopman-lifted DMDc)
================================================================================
Extends the team's existing plain-DMDc approach by lifting the state through a
dictionary of nonlinear observable functions BEFORE doing the exact same
Omega = [Psi(X); U], G = X'Omega^+ regression the team's README already defines.

This is the direct, minimal-structural-change extension from DMDc -> Koopman/EDMD
that the faculty's review-1 notes ("mapping to higher dimension, Koopman") ask for.
"""

import numpy as np


def trig_dictionary(X, angle_indices):
    """
    Lift state X (shape: state_dim x N) by appending sin/cos of the given
    angle indices. Returns the lifted state Psi(X), shape: (state_dim + 2*len(angle_indices)) x N.
    """
    angles = X[angle_indices, :]
    return np.vstack([X, np.sin(angles), np.cos(angles)])


def poly_dictionary(X, pair_indices):
    """
    Lift state X by appending pairwise products x_i * x_j for each (i, j) in
    pair_indices. Useful when the true nonlinearity is multiplicative
    (e.g. Lotka-Volterra's x*y coupling term), which a trig dictionary alone
    cannot represent.
    """
    extra = [X[i, :] * X[j, :] for (i, j) in pair_indices]
    return np.vstack([X] + extra) if extra else X


def fit_edmdc(X, Xprime, U, lift_fn):
    """
    Fit an EDMDc model: Psi(x_{k+1}) ~= A Psi(x_k) + B u_k

    Parameters
    ----------
    X, Xprime : (state_dim, N) arrays of current/next states
    U         : (input_dim, N) array of control inputs
    lift_fn   : function X -> Psi(X), the dictionary lift

    Returns
    -------
    A, B      : identified operators in the LIFTED space
    rmse      : one-step prediction RMSE in the lifted space
    eigs      : eigenvalues of A (stability / mode info)
    """
    Psi_X = lift_fn(X)
    Psi_Xp = lift_fn(Xprime)
    Omega = np.vstack([Psi_X, U])
    G = Psi_Xp @ np.linalg.pinv(Omega)

    n_lifted = Psi_X.shape[0]
    A = G[:, :n_lifted]
    B = G[:, n_lifted:]

    pred = A @ Psi_X + B @ U
    rmse = float(np.sqrt(np.mean((pred - Psi_Xp) ** 2)))
    eigs = np.linalg.eigvals(A)
    return A, B, rmse, eigs


def fit_plain_dmdc(X, Xprime, U):
    """The team's existing baseline method (no lifting) -- used for comparison."""
    Omega = np.vstack([X, U])
    G = Xprime @ np.linalg.pinv(Omega)
    n = X.shape[0]
    A = G[:, :n]
    B = G[:, n:]
    pred = A @ X + B @ U
    rmse = float(np.sqrt(np.mean((pred - Xprime) ** 2)))
    eigs = np.linalg.eigvals(A)
    return A, B, rmse, eigs


if __name__ == "__main__":
    # Quick self-test on random data (not meant as a real result, just confirms no crashes)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 500))
    Xprime = X + 0.01 * rng.normal(size=X.shape)
    U = rng.normal(size=(4, 500))

    angle_idx = [3, 4, 5]  # roll, pitch, yaw
    lift = lambda Xin: trig_dictionary(Xin, angle_idx)

    A, B, rmse, eigs = fit_edmdc(X, Xprime, U, lift)
    print("EDMDc self-test: A", A.shape, "B", B.shape, "rmse", rmse)

    A2, B2, rmse2, eigs2 = fit_plain_dmdc(X, Xprime, U)
    print("Plain DMDc self-test: A", A2.shape, "B", B2.shape, "rmse", rmse2)
