# Drone Flight Simulation and DMDc System Identification

This project implements a 3D drone simulation using the **gym-pybullet-drones** platform, performs PID-controlled flight experiments, and applies **SVD-based Dynamic Mode Decomposition with control (DMDc)** to identify a linear state-space model of the drone dynamics.

## Simulation Framework

The 3D drone physics simulation uses the open-source [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones) framework and PyBullet. This repository contains our custom PID flight experiment, data-logging pipeline, and SVD-based DMDc system-identification code.

## Project Structure

```text
AB07/
├── data/
│   └── pid_dmdc_log.csv              # Controlled flight trajectory data
├── project/
│   ├── run_pid_dmdc.py               # PID controller simulation with data logging
│   └── dmdc_identify.py              # SVD-based DMDc model identification
├── results/
│   ├── dmdc_model.npz                # Identified DMDc matrices (A, B)
│   ├── dmdc_position_prediction.png  # Position prediction vs ground truth
│   ├── dmdc_velocity_prediction.png  # Velocity prediction vs ground truth
│   └── dmdc_singular_values.png      # Singular value spectrum for model order selection
├── README.md
└── requirements.txt
```

## What We Did

### 1. PID-Controlled Drone Simulation

We implemented a PID controller to stabilize a quadrotor drone in the PyBullet physics environment. The controller tracks reference trajectories in 3D space while logging state and control inputs.

**Key features:**
- 3D position and velocity control using PID feedback
- Real-time physics simulation with PyBullet
- Flight data logging (position, velocity, control inputs) saved to `data/pid_dmdc_log.csv`

### 2. DMDc System Identification

Using the logged flight data, we applied **SVD-based Dynamic Mode Decomposition with control (DMDc)** to identify a linear state-space model of the form:

\[
x_{k+1} = A x_k + B u_k
\]

where \(x_k\) is the state vector (position and velocity) and \(u_k\) is the control input vector.

**Key features:**
- Data-driven model identification from flight experiments
- SVD truncation for model order selection
- Validation of identified model against held-out test data

## Results

The DMDc model successfully captures the dominant dynamics of the drone. The identified model matrices are saved in `results/dmdc_model.npz` and can be loaded for further analysis or control design.

Prediction accuracy is visualized in:
- `dmdc_position_prediction.png` — Position prediction vs ground truth
- `dmdc_velocity_prediction.png` — Velocity prediction vs ground truth
- `dmdc_singular_values.png` — Singular value spectrum for model order selection

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Run PID Simulation

```bash
python project/run_pid_dmdc.py
```

This generates flight data in `data/pid_dmdc_log.csv`.

### Identify DMDc Model

```bash
python project/dmdc_identify.py
```

This loads the flight data, identifies the DMDc model, and saves results in `results/`.

## License

This project is for educational purposes. The simulation framework uses the open-source [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones) library.