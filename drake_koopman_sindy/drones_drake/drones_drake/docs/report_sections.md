# Report Sections — Ready to Adapt Into Your Writeup

## 1. Why this contribution (positioning paragraph)
The project's existing identification work applies plain DMDc across three
simulators (Gym-PyBullet, MuJoCo, ArduPilot SITL), producing a dense linear
operator `[A B]` from raw state/input snapshots. This contribution extends
that foundation in two directions: (1) a fourth platform, **Drake**, chosen
for its exact symbolic rigid-body dynamics and native LQR tooling, used to
generate a controlled circular reference trajectory; and (2) two new
identification methods compared against the team's plain-DMDc baseline —
**EDMDc** (Koopman operator theory via a trigonometric dictionary lift) and
**SINDy** (sparse symbolic regression over a candidate function library).
Both methods are first validated on a Lotka-Volterra nonlinear testbed with
known ground-truth dynamics before being applied to real Drake flight data.

## 2. Koopman / EDMDc theory
Plain DMDc assumes the system is linear. Real quadrotor dynamics are
nonlinear, particularly in how orientation couples into translational
acceleration. The Koopman operator framework resolves this by observing that
a nonlinear system CAN be represented exactly by a linear operator, if the
state is first lifted into a higher-dimensional space of nonlinear
"observable" functions. EDMD operationalizes this: choose a dictionary of
observables `Psi(x)` (here, `[x, sin(angles), cos(angles)]`), then solve the
same DMDc regression `G = Psi(X') Omega^+` in this lifted space.

## 3. SINDy
Unlike DMDc/EDMDc, which fit a dense operator, SINDy fits a sparse model: it
builds a library of candidate nonlinear terms and uses Sequential
Thresholded Least Squares to find the fewest terms that explain the data.
The output is a compact, human-readable symbolic equation — e.g. recovering
coefficients close to the true Lotka-Volterra parameters directly from
noisy data. This interpretability is SINDy's main advantage; its main
limitation is sensitivity to the sparsity threshold and the assumption that
true dynamics are sparse in the chosen library.

## 4. Noise and Kalman filtering
Real flight data is never clean. Gaussian noise (std=0.01) was injected into
clean Drake-logged state data before re-running identification, producing a
measurably higher one-step prediction RMSE. A discrete-time Kalman filter
was then applied, using a plain linear `(A, B)` identified from the noisy
data as the process model, recovering a measurably cleaner state estimate.

## 5. Covariance assumption
This implementation assumes constant process and measurement covariance
(`Q`, `R`). In reality, covariance is not necessarily constant — IMU noise
tends to increase during aggressive maneuvers. Handling time-varying
covariance would require an Extended Kalman Filter or Adaptive Kalman
Filter. Flagged as a known limitation, not implemented in current scope.

## 6. Subspace Identification comparison
Beyond DMDc/EDMDc/SINDy, system identification can also be performed via
Subspace Identification methods (e.g. N4SID), which estimate state-space
matrices directly from input-output data using SVD of a block-Hankel data
matrix. This project did not implement subspace ID, since DMDc/EDMDc already
extends the method the team's existing work is built on (Proctor, Brunton &
Kutz, 2016), keeping the project's methods conceptually unified.

## 7. Suggested results table structure

| Method            | One-step RMSE | Notes |
|-------------------|---------------|-------|
| Plain DMDc        | (from compare_methods.py) | Team baseline, no lifting |
| EDMDc (trig)      | (from compare_methods.py) | Koopman lift, dense operator |
| SINDy             | (from compare_methods.py) | Sparse symbolic equation |
| EDMDc on noisy data | (from kalman_filter.py context) | Shows noise degradation |
| Kalman-filtered   | (from kalman_filter.py) | Shows noise recovery |
