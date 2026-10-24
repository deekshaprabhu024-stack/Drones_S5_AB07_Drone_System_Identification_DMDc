"""
dmdc.py
-------
SVD-based Dynamic Mode Decomposition with control (DMDc).

Identifies a discrete-time linear model

    x(k+1) = A x(k) + B u(k)

from collected data matrices X, X_next, U.
"""

import numpy as np


def fit_dmdc(X, X_next, U, rank=None):
    """Fit A and B using SVD-based DMDc."""

    n = X.shape[0]
    m = U.shape[0]

    # Stack state and control data:
    #
    # Omega = [ X
    #           U ]
    #
    #Omega has dimension (n+m) x N
    Omega = np.vstack([X, U])

    # Singular Value Decomposition
    U_svd, s, Vt = np.linalg.svd(
        Omega,
        full_matrices=False
    )

    # Choose numerical rank automatically
    if rank is None:
        tolerance = (
            max(Omega.shape)
            * np.finfo(float).eps
            * s[0]
        )

        rank = np.sum(s > tolerance)

    rank = int(max(1, min(rank, len(s))))

    # Truncated SVD
    U_r = U_svd[:, :rank]
    s_r = s[:rank]
    V_r = Vt[:rank, :].T

    # Safe inverse of singular values.
    # Prevents very small singular values from
    # producing extremely large numerical values.
    S_inv = np.diag(
        1.0 / np.maximum(s_r, 1e-10)
    )

    # Identify the combined matrix:
    #
    # G = [A B]
    #
    # X_next = G [X; U]
    G = X_next @ V_r @ S_inv @ U_r.T

    # Separate A and B
    A = G[:, :n]
    B = G[:, n:n + m]

    return A, B, s


def predict(A, B, x0, U):
    """Roll the identified linear model forward."""

    n = A.shape[0]
    N = U.shape[1]

    X_pred = np.zeros((n, N + 1))
    X_pred[:, 0] = x0

    for k in range(N):
        X_pred[:, k + 1] = (
            A @ X_pred[:, k]
            + B @ U[:, k]
        )

    return X_pred


def rmse(X_true, X_pred):
    """Root-mean-square error between state arrays."""

    return float(
        np.sqrt(
            np.mean(
                (X_true - X_pred) ** 2
            )
        )
    )