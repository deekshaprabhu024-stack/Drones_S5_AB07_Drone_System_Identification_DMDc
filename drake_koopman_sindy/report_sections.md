# Report Section

## 1. Why using drake with sindy

The project's existing identification work applies plain DMDc across three
simulators (Gym-PyBullet, MuJoCo, ArduPilot SITL), producing a dense linear
operator `[A B]` from raw state/input snapshots. This contribution extends
that foundation in two directions: (1) a fourth platform, **Drake**, chosen
for its exact symbolic rigid-body dynamics and native LQR tooling, used here
to generate a controlled circular reference trajectory; and (2) two new
identification methods compared against the team's plain-DMDc baseline —
**EDMDc** (Koopman operator theory via a trigonometric dictionary lift) and
**SINDy** (sparse symbolic regression over a candidate function library).
Both methods are first validated on a Lotka-Volterra nonlinear testbed with
known ground-truth dynamics before being applied to real Drake flight data,
mirroring the team's own Week-3 synthetic-validation safety-net philosophy.

## 2. Koopman / EDMDc theory (for the "mapping to higher dimension" note)

Plain DMDc assumes the system is linear (or locally linear): `x_{k+1} = A x_k + B u_k`.
Real quadrotor dynamics are nonlinear, particularly in how orientation couples
into translational acceleration. The Koopman operator framework resolves this
by observing that a nonlinear system CAN be represented exactly by a linear
operator, if the state is first lifted into a higher-dimensional space of
nonlinear "observable" functions of the original state. Extended DMD (EDMD)
operationalizes this: choose a dictionary of observables `Psi(x)` (here,
`[x, sin(angles), cos(angles)]`), then solve the *same* DMDc regression
`G = Psi(X') Omega^+` in this lifted space. The structural change from DMDc
is minimal — the regression itself is identical — but the model can now
capture nonlinear coupling the raw-state version could not.

## 3. SINDy — a genuinely different philosophy (for the "SINDy" note)

Unlike DMDc/EDMDc, which fit a dense operator (every lifted-state component
influences every other), SINDy fits a **sparse** model: it builds a large
library of candidate nonlinear terms (polynomials, trig functions, products)
and uses Sequential Thresholded Least Squares to find the fewest terms that
explain the data. The output is a compact, human-readable symbolic equation
rather than a matrix — e.g. recovering something close to
`dx/dt = 1.1x - 0.4xy + u1` directly from data on the Lotka-Volterra testbed.
This interpretability is SINDy's main advantage over DMDc/EDMDc; its main
limitation is sensitivity to the sparsity threshold and the assumption that
the true dynamics ARE sparse in the chosen library — if the true dynamics
need terms outside the library, SINDy cannot represent them (same limitation
as EDMDc's dictionary choice, from a different angle).

## 4. Noise and Kalman filtering (for "add noise" / "Kalman filtering" notes)

Real flight data is never clean — sensor noise, vibration, and estimation
error corrupt every measurement. To simulate this, Gaussian noise
(`std=0.01`) was injected into the clean Drake-logged state data before
re-running identification, producing a measurably higher one-step prediction
RMSE (see `compare_methods.py` output) — a concrete, reportable demonstration
that noise degrades identification quality, not just an assumption. A
discrete-time Kalman filter was then applied, using a plain linear `(A, B)`
identified from the noisy data as the process model, to recover a cleaner
state estimate. The filtered trajectory shows a measurable RMSE improvement
over the raw noisy signal (see `kalman_filter.py` output).

## 5. Covariance question (for "is the covariance varying?" note)

This implementation assumes **constant** process and measurement covariance
(`Q`, `R`), matching the standard Kalman filter assumption of time-invariant
Gaussian noise. In reality, covariance is not necessarily constant — for
example, IMU noise tends to increase during aggressive maneuvers or high
vibration, and a standard KF cannot adapt to this. Handling time-varying
covariance would require an **Extended Kalman Filter** (if the nonlinearity
is in the dynamics) or an **Adaptive Kalman Filter** (which estimates `Q`/`R`
online from residuals). This is flagged here as a known limitation and a
natural extension, not implemented in the current scope.

## 6. Subspace Identification comparison (for "subspace identification" note)

Beyond DMDc/EDMDc/SINDy, system identification can also be performed via
**Subspace Identification** methods (e.g. N4SID), which estimate state-space
matrices directly from input-output data using SVD of a block-Hankel data
matrix, without assuming an explicit model structure in advance. This project
did not implement subspace ID, since DMDc/EDMDc already directly extends the
method the team's existing work is built on (Proctor, Brunton & Kutz, 2016),
keeping the project's identification methods conceptually unified rather than
introducing a third unrelated framework.

## 7. Results table structure

| Method            | One-step RMSE | Notes |
|-------------------|---------------|-------|
| Plain DMDc        | (from compare_methods.py) | Team baseline, no lifting |
| EDMDc (trig)      | (from compare_methods.py) | Koopman lift, dense operator |
| SINDy             | (from compare_methods.py) | Sparse symbolic equation |
| EDMDc on noisy data | (from kalman_filter.py context) | Shows noise degradation |
| Kalman-filtered   | (from kalman_filter.py) | Shows noise recovery |
