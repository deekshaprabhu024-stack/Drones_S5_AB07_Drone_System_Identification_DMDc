"""
data_collection.py
-------------------
Flies the Gazebo quadrotor through a takeoff -> hover -> move -> return
-> land sequence using the cascaded PID controller, injecting a small
deterministic excitation signal (as in the original PyBullet version)
so DMDc can identify the effect of the control input, and logs
(x_k, u_k, x_{k+1}) triplets to data/collected_data.npz.

Requires the simulation to already be running:

    ros2 launch quad_dmdc_sim sim.launch.py gui:=false

Run with:

    ros2 run quad_dmdc_sim data_collection
"""

import os

import numpy as np
import rclpy

from .gazebo_env import GazeboDroneEnv
from .pid_controller import PIDController


def collect_data(env, num_steps=6000, seed=0):

    rng = np.random.default_rng(seed)

    # Hover speed such that 4 * k_f * omega^2 = mass * g
    mass = 0.9
    gravity = 9.81
    k_f = 8.54858e-06
    hover_omega = np.sqrt((mass * gravity / 4.0) / k_f)

    controller = PIDController(
        hover_omega=hover_omega,
        dt=env.dt,
        mass=mass,
        gravity=gravity,
        motor_constant=k_f,
    )

    X_list, X_next_list, U_list = [], [], []

    env.wait_for_odometry()
    state = env.get_state()

    targets = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.5, 0.0, 1.2]),
        np.array([0.0, 0.5, 1.2]),
        np.array([-0.5, 0.0, 1.0]),
        np.array([0.0, 0.0, 0.8]),
        np.array([0.0, 0.0, 0.15]),
    ]
    target_duration = 900

    for step in range(num_steps):

        target_index = min(step // target_duration, len(targets) - 1)
        target = targets[target_index]

        position = state[:3]
        velocity = state[3:6]
        attitude = state[6:9]
        angular_velocity = state[9:12]

        omega_cmd = controller.compute(
            position=position,
            velocity=velocity,
            target=target,
            attitude=attitude,
            angular_velocity=angular_velocity,
        )

        # Small deterministic excitation on two motors, same idea as the
        # original PyBullet data collection script.
        excitation = 15.0 * np.sin(2.0 * np.pi * step / 80.0)
        omega_cmd = omega_cmd.copy()
        omega_cmd[0] += excitation
        omega_cmd[2] -= excitation
        omega_cmd = np.clip(omega_cmd, 0.0, env.max_omega)

        next_state, _, terminated, _, _ = env.step(omega_cmd)

        # DMDc uses only the 6 translational states. The controller still
        # receives the full 12-state Gazebo estimate above.
        x_k = state[:6]
        x_next = next_state[:6]

        if np.all(np.isfinite(x_k)) and np.all(np.isfinite(x_next)):
            X_list.append(x_k)
            U_list.append(omega_cmd)
            X_next_list.append(x_next)

        state = next_state

        if terminated:
            print(f"[data_collection] step {step}: drone became unstable, "
                  f"skipping to next waypoint segment.")
            state = env.get_state()

        if step % 500 == 0:
            print(f"[data_collection] step {step}/{num_steps}, "
                  f"pos={np.round(position, 2)}, target={target}")

    X = np.asarray(X_list, dtype=float).T
    X_next = np.asarray(X_next_list, dtype=float).T
    U = np.asarray(U_list, dtype=float).T

    return X, X_next, U


def main():
    rclpy.init()

    env = GazeboDroneEnv(dt=0.02)

    try:
        X, X_next, U = collect_data(env, num_steps=6000)
    finally:
        env.close()
        rclpy.shutdown()

    os.makedirs("data", exist_ok=True)
    np.savez("data/collected_data.npz", X=X, X_next=X_next, U=U)

    print(f"Collected {X.shape[1]} valid samples.")
    print("X shape:", X.shape)
    print("X_next shape:", X_next.shape)
    print("U shape:", U.shape)


if __name__ == "__main__":
    main()
