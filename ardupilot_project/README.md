# Quaternion-Based System Identification of a Quadcopter Attitude Dynamics via DMDc

> **Project Report — Drones / UAV System Identification**

---

## Amrita Vishwa Vidyapeetham

<!-- Replace with the official Amrita logo -->
<!-- ![Amrita Vishwa Vidyapeetham](path/to/amrita_logo.png) -->

### Team Members

| Name | Roll No. | Email |
|---|---|---|
| [Member 1] | [Roll No.] | [Email] |
| [Member 2] | [Roll No.] | [Email] |
| [Member 3] | [Roll No.] | [Email] |
| [Member 4] | [Roll No.] | [Email] |

---

# Abstract

This project investigates data-driven system identification of quadcopter attitude dynamics using **Dynamic Mode Decomposition with Control (DMDc)**. The objective is to obtain a low-order discrete-time state-space representation directly from flight data collected using an ArduPilot Software-In-The-Loop (SITL) simulation.

The identified model is represented as

\[
x_{k+1} = A x_k + B u_k,
\]

where the state contains quaternion attitude and body angular-rate information, while the input contains transformed quadcopter motor commands.

The final implementation uses ArduPilot DataFlash flight logs containing attitude (`ATT`), angular-rate (`RATE`), and motor-output (`RCOU`) measurements. These streams are synchronized onto a common timestamp grid before constructing the DMDc state and input matrices.

The present stage of the project focuses on **system identification and one-step-ahead validation**. Data-driven LQR control and a second independent controlled flight are planned as the subsequent stage.

> **Current status:** DMDc identification, matrix estimation, one-step validation, and visualization have been implemented. Controller design and second-flight experimental validation remain future work.

---

# 1. Introduction

Quadrotor dynamics are nonlinear, coupled, and strongly dependent on the interaction between multiple actuators. Obtaining an accurate analytical model requires knowledge of vehicle parameters such as mass, inertia, motor characteristics, aerodynamic effects, and actuator dynamics.

A data-driven alternative is to identify a dynamical model directly from measured trajectories and control inputs.

**Dynamic Mode Decomposition with Control (DMDc)** extends Dynamic Mode Decomposition to systems affected by external actuation. Instead of modelling only the state evolution, DMDc explicitly incorporates the control input and estimates an input-output state-space representation from measured snapshots. Proctor, Brunton, and Kutz introduced DMDc as a method for separating the underlying dynamics of an actuated system from the effect of control inputs. [1]

The objective of this project is therefore:

1. Collect quadcopter flight data using ArduPilot SITL.
2. Extract attitude, angular-rate, and motor-output measurements.
3. Construct a quaternion-based attitude state.
4. Transform the four motor outputs into collective/differential input coordinates.
5. Estimate the matrices \(A\) and \(B\) using DMDc.
6. Validate the identified model using held-out flight data.
7. Evaluate the predicted attitude using quaternion orientation error.
8. Investigate the identified model as a basis for future data-driven LQR control.

---

# 2. Project Objective

The target system is a quadcopter attitude subsystem described by the discrete-time model

\[
x_{k+1} = A x_k + B u_k.
\]

The project specifically investigates whether the dominant attitude dynamics of the simulated quadcopter can be approximated using a low-order linear model identified only from flight data.

The current state definition is

\[
x_k =
\begin{bmatrix}
q_0 &
q_1 &
q_2 &
q_3 &
\omega_x &
\omega_y &
\omega_z
\end{bmatrix}^{T},
\]

where

- \(q_0,q_1,q_2,q_3\) are quaternion components,
- \(\omega_x,\omega_y,\omega_z\) are body angular rates.

The control input consists of four transformed motor coordinates.

---

# 3. Background: Dynamic Mode Decomposition with Control

## 3.1 Dynamic Mode Decomposition

Given sequential state measurements,

\[
X_1 =
\begin{bmatrix}
x_1 & x_2 & \cdots & x_{N-1}
\end{bmatrix},
\]

and

\[
X_2 =
\begin{bmatrix}
x_2 & x_3 & \cdots & x_N
\end{bmatrix},
\]

standard DMD attempts to identify a linear operator

\[
X_2 \approx A X_1.
\]

This provides a data-driven approximation to the local dynamics of the measured system.

---

## 3.2 DMD with Control

For an actuated system, the state evolution depends on both the current state and the applied input:

\[
x_{k+1}=A x_k+B u_k.
\]

Collecting the state and input snapshots gives

\[
\Omega =
\begin{bmatrix}
X_1\\
U_1
\end{bmatrix},
\]

where

\[
U_1 =
\begin{bmatrix}
u_1 & u_2 & \cdots & u_{N-1}
\end{bmatrix}.
\]

The data equation becomes

\[
X_2 =
\begin{bmatrix}
A & B
\end{bmatrix}
\Omega.
\]

The combined matrix is estimated using the Moore--Penrose pseudoinverse:

\[
\boxed{
\begin{bmatrix}
A & B
\end{bmatrix}
=
X_2\Omega^\dagger
}
\]

where \(\Omega^\dagger\) denotes the Moore--Penrose pseudoinverse.

Thus,

\[
\boxed{
x_{k+1}=A x_k+B u_k
}
\]

is the identified discrete-time model.

The approach follows the formulation introduced by Proctor, Brunton, and Kutz. [1]

---

# 4. Experimental Platform

## 4.1 ArduPilot SITL

The experiment was performed using **ArduPilot Software-In-The-Loop (SITL)**.

The simulated vehicle used in the experiment was an ArduCopter quadrotor.

The SITL environment provides simulated:

- attitude measurements,
- angular velocity measurements,
- motor outputs,
- GPS/position information,
- battery information,
- DataFlash flight logs.

The experiment was performed entirely in simulation, allowing repeatable flight experiments without dependence on physical hardware.

---

# 5. Flight Data Acquisition

The flight data is stored by ArduPilot in DataFlash `.BIN` log files.

The final experiment used a flight log corresponding to approximately:

- **2933 synchronized samples**
- **approximately 293.2 seconds of recorded data**
- **approximately 0.1 s sampling interval**
- approximately **10 Hz effective sampling**

The final raw experiment was extracted from the ArduPilot DataFlash log.

The relevant DataFlash message types were:

| Message | Information used |
|---|---|
| `ATT` | Roll, pitch, yaw |
| `RATE` | Body angular rates |
| `RCOU` | Motor output commands |

---

# 6. Data Extraction and Synchronization

The final extraction pipeline is implemented in:

```text
scripts/extract_final.py
