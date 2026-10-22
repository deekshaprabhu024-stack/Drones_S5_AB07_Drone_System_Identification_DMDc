import os
import time
import argparse
import numpy as np
import pandas as pd

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync


def reference_position(t):
    """Creates deliberate position steps so motor inputs vary for DMDc."""
    if t < 2.0:
        return np.array([0.0, 0.0, 0.30])
    elif t < 5.0:
        return np.array([0.0, 0.0, 0.70])
    elif t < 8.0:
        return np.array([0.30, 0.0, 0.70])
    elif t < 11.0:
        return np.array([0.30, 0.0, 0.40])
    else:
        return np.array([0.0, 0.0, 0.40])


def run(gui=True, duration=14.0, output_csv="data/pid_dmdc_log.csv"):
    os.makedirs("data", exist_ok=True)

    drone_model = DroneModel.CF2X
    physics = Physics.PYB
    sim_freq_hz = 240
    control_freq_hz = 48
    aggregate = int(sim_freq_hz / control_freq_hz)

    env = CtrlAviary(
        drone_model=drone_model,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 0.10]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        physics=physics,
        pyb_freq=sim_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=False,
        obstacles=False
    )

    controller = DSLPIDControl(drone_model=drone_model)
    obs, info = env.reset(seed=42)

    action = np.zeros((1, 4))
    rows = []
    start = time.time()
    num_steps = int(duration * control_freq_hz)

    print("Running single-drone PID experiment...")
    print("Logging position, velocity, attitude, angular rate, and four motor RPM commands.")

    for i in range(num_steps):
        t = i / control_freq_hz

        state = obs[0]
        target = reference_position(t)

        rpm, _, _ = controller.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=target,
            target_rpy=np.zeros(3)
        )
        action[0, :] = rpm

        obs, reward, terminated, truncated, info = env.step(action)
        next_state = obs[0]

        rows.append({
            "time": t,
            "target_x": target[0],
            "target_y": target[1],
            "target_z": target[2],
            "x": next_state[0],
            "y": next_state[1],
            "z": next_state[2],
            "qx": next_state[3],
            "qy": next_state[4],
            "qz": next_state[5],
            "qw": next_state[6],
            "vx": next_state[10],
            "vy": next_state[11],
            "vz": next_state[12],
            "wx": next_state[13],
            "wy": next_state[14],
            "wz": next_state[15],
            "rpm1": rpm[0],
            "rpm2": rpm[1],
            "rpm3": rpm[2],
            "rpm4": rpm[3]
        })

        if gui:
            env.render()
            sync(i, start, env.CTRL_TIMESTEP)

        if terminated or truncated:
            obs, info = env.reset(seed=42)

    env.close()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} samples to: {output_csv}")
    print(df.head())
    print(df.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Open PyBullet's 3D GUI")
    parser.add_argument("--duration", type=float, default=14.0, help="Simulation duration in seconds")
    parser.add_argument("--output", type=str, default="data/pid_dmdc_log.csv", help="Output CSV path")
    args = parser.parse_args()

    run(gui=args.gui, duration=args.duration, output_csv=args.output)
