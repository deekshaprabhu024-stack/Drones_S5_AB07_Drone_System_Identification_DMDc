<p align="center">
  <img src="https://github.com/user-attachments/assets/060f7774-a73f-4132-9413-36887ed09cfa" 
  alt="Amrita Vishwa Vidyapeetham" width="430">
</p>

# Drone System Identification Using DMDc

## Team Members

| Name | Roll Number | Email |
|---|---|---|
| Deeksha Prabhu | CB.SC.U4AIE24015 | deekshaprabhu024@gmail.com |
| Shivanandana | CB.SC.U4AIE24049 | shivanandana83@gmail.com |
| K. Supriya | CB.SC.U4AIE24025 | ksupriya2430@gmail.com |
| V R Sridevi | CB.SC.U4AIE24166 | sridevirajansri@gmail.com |
| Diya Prakash | CB.SC.U4AIE24111 | diyapnair07@gmail.com |

---

## Abstract

This project addresses the problem of data-driven system identification for quadrotor dynamics using **Dynamic Mode Decomposition with Control (DMDc)**. Quadrotors are inherently nonlinear, underactuated, and dynamically coupled systems, which makes deriving a fully analytical model both tedious and often imprecise for practical prediction tasks. Rather than pursuing a first-principles derivation, this project takes a data-driven approach: flight data — consisting of states and control inputs — is collected from simulation, and DMDc is used to approximate the underlying dynamics with a linear discrete-time state-space model of the form

$$x(k+1) = Ax(k) + Bu(k)$$

where $A$ represents the influence of the current system state on the next state, and $B$ represents the influence of the applied control input on the next state.

To evaluate the generality of this approach, the same DMDc-based identification workflow is applied across **three different quadrotor environments**: Gym-PyBullet, MuJoCo, and ArduPilot Software-In-The-Loop (SITL). Each environment generates flight data through a different mechanism and uses a different state representation, but all three follow an identical downstream pipeline:

$$\text{Data generation} \rightarrow \text{State/control collection} \rightarrow X, X_{\text{next}}, U \text{ construction} \rightarrow \text{SVD-based DMDc} \rightarrow \text{Identification of } A, B \rightarrow \text{Prediction and validation}$$

It should be noted that this project identifies the system dynamics only; closed-loop control design (e.g., LQR) is **not** implemented as part of this work and is discussed only as future work.

---

## Introduction

A **quadrotor** is an unmanned aerial vehicle propelled by four rotors, typically arranged in a symmetric cross or "X" configuration. By independently varying the speed of each rotor, a quadrotor can control its vertical thrust as well as its roll, pitch, and yaw moments, allowing it to translate and rotate freely in three-dimensional space.

Modelling quadrotor dynamics accurately is difficult for several reasons:

- The system is **nonlinear**, particularly in how orientation and translational motion interact.
- The rotational and translational dynamics are **coupled** — a change in orientation (e.g., pitch) directly affects translational acceleration.
- The full state of a quadrotor spans **six degrees of freedom (6-DOF)**:
  - **Position** — location in 3D space
  - **Velocity** — rate of change of position
  - **Orientation** — roll, pitch, and yaw (attitude)
  - **Angular velocity** — rate of change of orientation

Traditionally, quadrotor dynamics are derived analytically using Newton-Euler or Lagrangian mechanics, which requires precise knowledge of physical parameters (mass, inertia, drag coefficients, motor thrust curves, etc.) and often makes simplifying assumptions to remain tractable.

**Data-driven system identification** offers an alternative: instead of deriving the dynamics from physics, the model is learned directly from observed input-output data. This is particularly useful when:

- Physical parameters are unknown, hard to measure, or vary across platforms.
- The underlying dynamics are too complex to derive analytically with sufficient accuracy.
- A model is needed quickly across multiple platforms/simulators without re-deriving equations of motion for each.

---

## Project Objective

- Generate quadrotor flight data across multiple simulation environments.
- Record system states and corresponding control inputs during flight.
- Construct the data matrices $X$, $X_{\text{next}}$, and $U$ from recorded flight data.
- Apply **Dynamic Mode Decomposition with Control (DMDc)** to this data.
- Use **Singular Value Decomposition (SVD)** for numerically stable computation.
- Identify the system matrices $A$ and $B$.
- Construct a discrete-time linear state-space approximation of the quadrotor dynamics.
- Use the identified model to predict future drone behaviour.
- Validate the learned model against unseen flight data.

---

## Quadrotor Dynamics

The full dynamic state of a quadrotor can be described using:

**Position:**
$$[x, y, z]$$

**Velocity:**
$$[v_x, v_y, v_z]$$

**Orientation:**
$$[\text{Roll}, \text{Pitch}, \text{Yaw}]$$

**Angular velocity:**
$$[p, q, r]$$

Not every simulation environment records or exposes all of these quantities in the same form. Depending on the internal representation and sensor/logging capabilities of each simulator, different subsets or parameterizations (e.g., Euler angles vs. quaternions) of this full state are used. This project does **not** force a common state vector across environments — each environment's natural state representation is preserved.

---

## State Representation

State representation is simulator-dependent.

### Gym-PyBullet

$$x = [x, y, z, v_x, v_y, v_z]^T$$

Six states:
- $x$ position
- $y$ position
- $z$ position
- $x$ velocity
- $y$ velocity
- $z$ velocity

### MuJoCo

Uses an attitude-based state representation:

$$x = [z_{\text{error}}, v_z, \text{roll}, \text{pitch}, \text{yaw}, p, q, r]^T$$

Eight states.

### ArduPilot

Uses flight-log-based attitude information, including quaternion/attitude data and angular-rate information extracted from DataFlash logs.

---

## Control Input

The quadrotor is actuated via its four motors. In general form:

$$u = [u_1, u_2, u_3, u_4]^T$$

These four components represent the individual motor commands. Depending on the simulation environment, these values may represent:

- Motor **RPM**
- Normalized **actuator commands**
- Raw **motor output** signals

---

## Methodology

This section describes the complete end-to-end workflow used to go from raw flight data to a validated, identified dynamical model.

### 1. Data Generation

In each environment, the drone is simulated (or, in ArduPilot's case, flown in SITL) while control inputs are applied and the resulting states are recorded. Each collected sample takes the form of a state-input-nextstate triplet:

$$(x_k, u_k, x_{k+1})$$

The mechanism for generating this data differs across environments, as described below.

#### Gym-PyBullet Implementation

- Simulated using the **PyBullet** physics engine.
- A dedicated drone environment provides physics stepping and state feedback.
- Flight is generated using **PID-based control**.
- Target positions/trajectories are generated to drive the PID controller.
- Motor commands are issued as **RPM values**.
- States and inputs are recorded at each simulation step.

Flight sequence used for data generation:

$$\text{Takeoff} \rightarrow \text{Hover} \rightarrow \text{Move along X} \rightarrow \text{Move along Y} \rightarrow \text{Return} \rightarrow \text{Land}$$

- **State:** $[x, y, z, v_x, v_y, v_z]$
- **Input:** four motor RPM values

#### MuJoCo Implementation

- Simulated using the **MuJoCo** physics engine with a six-degree-of-freedom quadrotor model.
- Movement is driven using manually specified commands.
- An attitude controller converts these high-level commands into individual motor inputs.
- State and motor data are collected throughout simulation.

- **State:** $[z_{\text{error}}, v_z, \text{roll}, \text{pitch}, \text{yaw}, p, q, r]$
- **Input:** four motor commands

Following data collection, this environment is used to demonstrate:
- DMDc-based identification
- Prediction over held-out trajectories
- Validation against unseen flight data
- Eigenvalue analysis of the identified $A$ matrix

*(Note: LQR control is not implemented in this environment or elsewhere in this project.)*

#### ArduPilot Implementation

- Flight data is generated using **ArduPilot Software-In-The-Loop (SITL)**.
- Data is extracted from **DataFlash logs** produced during SITL flight.
- Extracted signals are processed and time-synchronized.
- State and motor/control information are collected from the synchronized logs.

Pipeline:

$$\text{DataFlash logs} \rightarrow \text{Signal extraction} \rightarrow \text{Preprocessing} \rightarrow X, X_{\text{next}}, U \rightarrow \text{DMDc}$$

- **State:** quaternion/attitude information and angular rates
- **Input:** motor output information

---

## DMDc Theory

Dynamic Mode Decomposition with Control begins from the assumption of an underlying discrete-time linear (or locally linear) dynamical system:

$$x(k+1) = Ax(k) + Bu(k)$$

- $A$ captures the effect of the **current state** on the **next state**.
- $B$ captures the effect of the **control input** on the **next state**.

Given a dataset of consecutive state and input snapshots, this relationship can be written in matrix form as:

$$X_{\text{next}} = AX + BU$$

To solve for $A$ and $B$ simultaneously, the state and input snapshots are stacked into a single augmented matrix:

$$\Omega = \begin{bmatrix} X \\ U \end{bmatrix}$$

and the combined operator is defined as:

$$G = \begin{bmatrix} A & B \end{bmatrix}$$

so that:

$$X_{\text{next}} \approx G\Omega$$

DMDc then solves for $G$ using the Moore-Penrose pseudoinverse of $\Omega$:

$$G = X_{\text{next}} \Omega^{\dagger}$$

---

## Singular Value Decomposition (SVD)

Rather than computing the pseudoinverse of $\Omega$ directly (which can be numerically unstable, especially for ill-conditioned or high-dimensional data), DMDc uses the **Singular Value Decomposition**:

$$\Omega = U \Sigma V^T$$

where:

- $U$ — matrix of **left singular vectors**
- $\Sigma$ — diagonal matrix of **singular values**
- $V$ — matrix of **right singular vectors**

SVD is used because it provides:

- **Numerical stability** when computing pseudoinverses
- A well-conditioned way to compute the **pseudoinverse**
- The ability to perform **rank selection** (retaining only the most significant modes)
- **Dimensionality reduction**, discarding directions dominated by noise or negligible energy

A **truncated SVD** retains only the top $r$ singular values/vectors:

$$\Omega \approx U_r \Sigma_r V_r^T$$

---

## DMDc Derivation

Starting from the truncated SVD of the augmented data matrix:

$$\Omega \approx U_r \Sigma_r V_r^T$$

the pseudoinverse of $\Omega$ can be written as:

$$\Omega^{\dagger} = V_r \Sigma_r^{-1} U_r^T$$

Substituting into the expression for $G$:

$$G = X_{\text{next}} V_r \Sigma_r^{-1} U_r^T$$

Since:

$$G = \begin{bmatrix} A & B \end{bmatrix}$$

the resulting matrix $G$ is split column-wise (according to the number of state dimensions vs. input dimensions used to build $\Omega$) to recover $A$ and $B$ individually.

---

## Prediction

Once $A$ and $B$ have been identified, the model can be used to predict future states.

**One-step prediction:**

$$\hat{x}(k+1) = A\hat{x}(k) + Bu(k)$$

**Multi-step prediction:** the predicted output at each step is fed back in as the input state for the next prediction step, i.e., $\hat{x}(k)$ is replaced by the model's own previous prediction rather than the true measured state. This allows the model to be rolled forward over a full trajectory, but also means prediction errors can **accumulate** over successive steps, since each prediction depends on the accuracy of the one before it.

---

## Validation

- **Training data** — used to compute $A$ and $B$ via DMDc.
- **Validation data** — a separate, unseen trajectory used to test the learned model's predictive accuracy.

Validation is performed by comparing:

$$\text{Actual trajectory} \quad \text{vs.} \quad \text{DMDc-predicted trajectory}$$

---

## Error Metric

Model accuracy is quantified using **Root Mean Squared Error (RMSE)**:

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}$$

where $y_i$ is the actual (measured) value, $\hat{y}_i$ is the predicted value, and $N$ is the number of samples. A lower RMSE indicates that the predicted trajectory more closely matches the actual trajectory.

---

## Comparison Across Environments

| Aspect | Gym-PyBullet | MuJoCo | ArduPilot |
|---|---|---|---|
| **Environment** | PyBullet physics simulation | MuJoCo physics simulation | ArduPilot SITL |
| **Data source** | Simulated flight episodes | Simulated flight episodes | DataFlash flight logs |
| **Controller / data generation** | PID-based control | Manual commands + attitude controller | Native ArduPilot flight controller |
| **State representation** | $[x,y,z,v_x,v_y,v_z]$ | $[z_{\text{error}},v_z,\text{roll},\text{pitch},\text{yaw},p,q,r]$ | Quaternion/attitude + angular rates |
| **Control input** | 4 motor RPM values | 4 motor commands | Motor output information |
| **DMDc usage** | Identification of $A$, $B$ | Identification of $A$, $B$ | Identification of $A$, $B$ |
| **Validation method** | Held-out trajectory comparison | Held-out trajectory + eigenvalue analysis | Held-out log segment comparison |

---

## Limitations

- Quadrotor dynamics are fundamentally **nonlinear**, while DMDc produces a **linear approximation** of the system.
- The quality of the identified model is directly dependent on the **quality and diversity of the collected data**.
- Models identified purely in simulation are subject to a **simulation-to-real gap** and may not transfer directly to physical hardware.
- **Limited or narrow trajectories** used during data collection may reduce the model's ability to generalize to maneuvers outside the training distribution.

---

## Future Work

- Collecting a broader range of flight trajectories to improve model generalization.
- Validating the identified model against **real drone** flight data.
- Exploring **nonlinear system identification** techniques.
- Improving **model order selection** (rank truncation) methodology.
- Using the identified dynamics for **model-based control design**.
- Exploring **LQR/MPC** as candidate controllers built on top of the identified model, as future work.

---

## Conclusion

This project demonstrates a data-driven framework for learning quadrotor dynamics using Dynamic Mode Decomposition with Control (DMDc). The same identification methodology — data generation, construction of $X$, $X_{\text{next}}$, and $U$, SVD-based computation of $G = [A \; B]$, and validation against held-out data — is applied consistently across three different quadrotor environments: Gym-PyBullet, MuJoCo, and ArduPilot SITL. The resulting matrices $A$ and $B$ provide a compact, discrete-time, linear mathematical representation of quadrotor dynamics that can be used for prediction and, in future work, for model-based control design.

---

## Results

Results will be added later by the team.

---

## References

Base paper and other relevant study materials will be added later.
