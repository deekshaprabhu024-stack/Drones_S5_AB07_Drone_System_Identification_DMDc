# Presentation Guide — Detailed Explanation of Every Output

This document explains, in full detail, every concept, every script, and every
graph produced — including what each axis means, what each visual element
represents, and what conclusion to draw. Written for walking a reviewer
through the work point by point.

---

## PART 1 — Concepts, explained in depth

### 1.1 System Identification (the umbrella topic)
The general problem: given recorded data of a system's inputs and resulting
states, find a mathematical model that predicts future states — without
deriving that model from first-principles physics. All the methods below
(DMDc, EDMDc, SINDy) are different approaches to this same problem.

### 1.2 DMDc (Dynamic Mode Decomposition with control) — the team's baseline
Assumes the system is **linear**: `x_{k+1} = A x_k + B u_k`. Given many
recorded (state, input, next-state) snapshots, stack them into matrices and
solve for `A` and `B` via least-squares regression (`G = X' Ω⁺`, where
`Ω = [X; U]` stacks state and input together). `A`'s **eigenvalues** describe
the system's natural modes — the "dynamic modes" the method is named after.
An eigenvalue's magnitude tells you about stability: `|λ| ≈ 1` means
marginally stable (oscillates without growing or decaying), `|λ| < 1` means
decaying, `|λ| > 1` means growing/unstable.

### 1.3 Koopman Operator Theory / EDMDc
**Core idea**: a nonlinear system CAN be represented *exactly* by a linear
operator — if you first transform ("lift") the state into a higher-dimensional
space of nonlinear functions of the original state (called "observables").
This higher-dimensional space is where the Koopman operator acts linearly,
even though the original system is nonlinear.

**EDMDc** (Extended DMD with control) operationalizes this: choose a
dictionary of observable functions `Ψ(x)` — here, `[x, sin(angles), cos(angles)]`
— then run the *exact same* DMDc regression in this lifted space:
`Ψ(x_{k+1}) ≈ A Ψ(x_k) + B u_k`. The math doesn't change; only the input to
it does.

**Why trig functions specifically?** Angles are periodic and their effect on
motion is nonlinear (e.g., a drone's sideways acceleration depends on
`sin(roll)`, not `roll` itself). Including `sin`/`cos` as extra lifted-state
components lets a *linear* operator capture this relationship correctly.

### 1.4 SINDy (Sparse Identification of Nonlinear Dynamics)
A fundamentally different strategy from DMDc/EDMDc. Instead of fitting a
dense operator, SINDy:
1. Builds a large **library** of candidate terms (constants, raw states,
   products of states, trig functions) — more terms than you expect to need.
2. Uses **Sequential Thresholded Least Squares (STLSQ)**: fit a least-squares
   solution, zero out any coefficient below a threshold, refit using only the
   surviving terms, repeat until stable.
3. The result is a **sparse** equation — only a handful of terms survive —
   giving a human-readable symbolic model instead of a dense matrix.

**Key implementation detail** (important if asked): SINDy classically fits
the **derivative** `dx/dt`, approximated as `(x_{k+1} - x_k)/dt`, not the
next-state directly. Fitting next-state directly makes true coefficients
artificially small (scaled by `dt`) and get wrongly zeroed by the sparsity
threshold — we hit this bug during development and fixed it by switching to
derivative-mode fitting for the Lotka-Volterra validation.

### 1.5 Kalman Filter
A recursive algorithm for estimating a system's true state from **noisy**
measurements, given a model of how the state evolves (`A`, `B`) and
statistical assumptions about the noise. Two steps each timestep:
- **Predict**: use the model to guess the next state (`x_pred = A x + B u`)
  and how uncertain that guess is (`P_pred`).
- **Update**: blend the prediction with the new noisy measurement, weighted
  by the **Kalman gain** (`K`) — which automatically balances "trust the
  model" vs. "trust the measurement" based on their relative uncertainties
  (`Q` = process noise covariance, `R` = measurement noise covariance).

**Our assumption**: `Q` and `R` are **constant** over time — the standard
textbook assumption. In reality, sensor noise can vary (e.g., more vibration
during aggressive maneuvers), which a constant-covariance filter can't adapt
to — we flag this explicitly as a scoped limitation, not an oversight.

### 1.6 Lotka-Volterra (the validation testbed)
A classic nonlinear ODE pair modeling predator-prey population dynamics:
```
dx/dt = αx − βxy     (prey: grows on its own, shrinks from predation)
dy/dt = δxy − γy      (predator: grows from predation, shrinks on its own)
```
We added a control input (harvesting/stocking term) to make it a genuine
"...with control" problem. **Why use this at all for a drone project?**
Because we know the exact ground-truth equations, so we can objectively
check whether EDMDc/SINDy recover something close to the truth — a test we
can't run on real/simulated drone data, where we don't independently know
the "true" equations.

### 1.7 Meshgrid Vector Fields
`np.meshgrid` takes two 1D coordinate arrays (e.g., 25 x-values, 25 y-values)
and produces two 2D arrays covering every (x,y) combination — a grid of
points spanning the plane. At each grid point, we evaluate the dynamics
(how fast the state changes from there) and draw a small arrow. The result
is a **vector field**: it shows system behavior at *every* point in the
state space, not just along one recorded trajectory — letting you visually
compare the TRUE dynamics against the IDENTIFIED dynamics over the whole
space at once.

---

## PART 2 — Every graph, explained axis by axis

### 2.1 `plots/lotka_volterra_phase.png`
- **X-axis**: prey population (arbitrary units, starts at 10)
- **Y-axis**: predator population (arbitrary units, starts at 5)
- **The curve**: a single trajectory over 50 simulated time units — spirals
  inward/outward depending on the added control input (pure Lotka-Volterra
  without control would trace a closed loop).
- **What to say**: "This shows the simulated, noisy predator-prey data we
  used to test our identification pipeline before trusting it on drone data."

### 2.2 `plots/meshgrid_lotka_volterra.png` (two side-by-side panels)
- **X-axis (both panels)**: prey population, gridded from 0 to 20
- **Y-axis (both panels)**: predator population, gridded from 0 to 15
- **Arrows**: at each grid point, the arrow shows the **direction** the
  system would move from that point (arrow direction) and **how fast**
  (arrow length — longer = faster change)
- **Red line (both panels)**: the one actual recorded trajectory, overlaid
  for reference
- **Left panel**: computed from the TRUE, known Lotka-Volterra equations
- **Right panel**: computed from the equation SINDy *discovered* from noisy
  data alone
- **What to say**: "If our identification method is accurate, the arrows in
  both panels should point the same way at every location — not just along
  the one red path we happened to record. You can see the two fields are
  visually almost identical, which is strong evidence SINDy learned the true
  underlying dynamics, not just memorized one trajectory."
- **How to interpret differences**: any region where the arrows diverge
  noticeably between panels is a region the identification is less confident
  about — typically areas the red trajectory never passed through (since the
  model only learns from data it has seen).

### 2.3 `plots/drake_circle_result.png`
- **X-axis**: drone's x-position (meters)
- **Y-axis**: drone's y-position (meters)
- **The curve**: the drone's actual flown path over 30 seconds, viewed from
  above (bird's-eye view)
- **What to say**: "This confirms our LQR controller successfully tracked
  the circular reference trajectory — radius 0.5 meters, as specified."
- **What a good result looks like**: a closed, round loop near radius 0.5m.
  Spiraling inward/outward would indicate the controller isn't tracking
  well; this should not happen here since the method is already validated.

### 2.4 Live Meshcat 3D view (during `circle_tracking.py`)
- Not a saved plot — a **live, interactive 3D rendering** in your browser.
- Shows the actual quadrotor model flying the circular path in real time,
  using Drake's `QuadrotorGeometry` (the same visual component Drake's own
  official demo uses).
- **What to say**: "This is a live physics simulation, not a canned
  animation — you're watching Drake numerically integrate the quadrotor's
  equations of motion and our LQR controller respond to them, frame by
  frame, in real time."

### 2.5 `plots/method_comparison.png`
- **X-axis**: method name (Plain DMDc, EDMDc trig, SINDy) — categorical, not
  numeric
- **Y-axis**: one-step prediction RMSE (Root Mean Squared Error) — **lower
  bar = better**, since RMSE measures average prediction error
- **What to say**: "This is our core comparison result: we ran all three
  identification methods on the exact same recorded drone flight data, so
  it's a fair, apples-to-apples comparison. [State which bar is shortest and
  by how much, once you've run it on your machine — gives this a crucial,
  specific number to finish your sentence with.]"

### 2.6 `plots/meshgrid_drake_rollpitch.png` (if Drake available)
- **X-axis**: roll angle (radians), ranging roughly −0.3 to +0.3
- **Y-axis**: pitch angle (radians), same range
- **Arrows**: show how the EDMDc-**identified** model predicts roll and
  pitch will change, starting from each point, with all other states
  (position, velocity, yaw) held at their hover values
- **What to say**: "This slices our full 12-dimensional identified model
  down to just the roll-pitch plane, so we can visualize it the same way we
  did for Lotka-Volterra. Arrows pointing back toward the center (0,0)
  confirm the identified model correctly captured that the controller
  stabilizes tilt back to level flight — exactly the physically correct
  behavior."

### 2.7 `plots/kalman_result.png`
- **X-axis**: sample number (time step index, not seconds directly)
- **Y-axis**: drone's x-position (meters)
- **Two lines**:
  - *noisy (raw)*: the x-position after we deliberately injected Gaussian
    sensor noise (std = 0.01m) — jagged/jittery
  - *Kalman-filtered*: the same signal after running it through our Kalman
    filter — should look smoother, closer to the true underlying motion
- **What to say**: "This demonstrates our noise-handling pipeline end to
  end: we simulate realistic sensor imperfection, then recover a cleaner
  estimate using the Kalman filter. The filtered line visibly tracks the
  true circular motion more smoothly than the raw noisy signal."

---

## PART 3 — Anticipated questions and how to answer them

**Q: Why trig functions specifically for the EDMDc dictionary, and not
something else?**
A: Because the nonlinearity in attitude dynamics is fundamentally
trigonometric — forces and accelerations depend on sine/cosine of tilt
angles, not the angles themselves. This is the standard, physically
motivated choice in the Koopman/EDMD literature for rotational systems.

**Q: Why does SINDy do well on Lotka-Volterra but you only showed next-state
discrete form for the Drake comparison — why not derivative form there too?**
A: For the head-to-head comparison with DMDc/EDMDc, we kept SINDy in the same
discrete-time `x_{k+1}` convention the rest of the project uses, for a fair
apples-to-apples comparison. For the Lotka-Volterra validation specifically,
we used the classic derivative-based SINDy formulation because that's what
correctly recovers interpretable, physically meaningful coefficients.

**Q: Is your Kalman filter assuming noise never changes?**
A: Yes, explicitly — we assume constant process/measurement covariance
(`Q`/`R`), which is the standard Kalman filter assumption. We flag this as a
known limitation; handling time-varying noise would require an Extended or
Adaptive Kalman Filter, which we scoped as future work rather than
implementing, given the project timeline.

**Q: Why Drake and not a 4th copy of MuJoCo/PyBullet?**
A: Drake gives exact, symbolic rigid-body dynamics rather than a purely
numerical simulator, and ships native LQR tooling — making it a natural fit
for a project centered on control-theoretic identification methods, and it
lets us visually confirm correctness by comparing identified-vs-true vector
fields directly, which the other platforms don't make as straightforward.
