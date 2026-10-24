"""
simulate.py
-----------
Flies a fixed waypoint sequence in the running Gazebo simulation using
the PID controller, printing progress - useful for sanity-checking the
model/controller before running full data collection. Watch it in the
Gazebo GUI.

Requires:

    ros2 launch quad_dmdc_sim sim.launch.py

Run with:

    ros2 run quad_dmdc_sim fly_manual
"""

import numpy as np
import rclpy

from .gazebo_env import GazeboDroneEnv
from .pid_controller import PIDController


def reached_target(position, velocity, target):
    position_error = np.linalg.norm(target - position)
    speed = np.linalg.norm(velocity)
    return position_error < 0.08 and speed < 0.15


def main():
    rclpy.init()

    env = GazeboDroneEnv(dt=0.02)
    env.wait_for_odometry()

    mass, gravity, k_f = 0.9, 9.81, 8.54858e-06
    hover_omega = np.sqrt((mass * gravity / 4.0) / k_f)

    controller = PIDController(
        hover_omega=hover_omega, dt=env.dt, mass=mass,
        gravity=gravity, motor_constant=k_f,
    )

    targets = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.5, 0.0, 1.0]),
        np.array([0.5, 0.5, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 0.10]),
    ]

    state = env.get_state()

    try:
        for target_number, target in enumerate(targets, start=1):
            print(f"\n--- Target {target_number}: {target} ---")
            reached_count = 0

            for step in range(1500):
                position, velocity = state[:3], state[3:6]
                attitude, angular_velocity = state[6:9], state[9:12]

                omega_cmd = controller.compute(
                    position=position, velocity=velocity, target=target,
                    attitude=attitude, angular_velocity=angular_velocity,
                )

                state, _, terminated, _, _ = env.step(omega_cmd)

                if terminated:
                    print("Drone became unstable.")
                    return

                if reached_target(state[:3], state[3:6], target):
                    reached_count += 1
                else:
                    reached_count = 0

                if reached_count >= 40:
                    print(f"Reached target {target_number} after {step + 1} steps. "
                          f"Position: {np.round(state[:3], 3)}")
                    break
            else:
                print(f"Warning: target {target_number} not fully reached.")

        print("\n--- CONTROLLED LANDING ---")
        landing_target = np.array([state[0], state[1], 0.12])
        for step in range(500):
            position, velocity = state[:3], state[3:6]
            attitude, angular_velocity = state[6:9], state[9:12]
            omega_cmd = controller.compute(
                position=position, velocity=velocity, target=landing_target,
                attitude=attitude, angular_velocity=angular_velocity,
            )
            state, _, terminated, _, _ = env.step(omega_cmd)
            if state[2] <= 0.08 and np.linalg.norm(state[3:6]) < 0.2:
                print("Drone reached landing height.")
                break
            if terminated:
                break
        # Stop motors only after the controlled descent is complete.
        env.send_action(np.zeros(4))
        print("Landing complete.")

    finally:
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
