# manual_control.py

import time
from pathlib import Path

import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(__file__).with_name("quadrotor.xml")
DATA_DIR = Path(__file__).with_name("data")
DATA_DIR.mkdir(exist_ok=True)

MANUAL_TRAINING_PATH = DATA_DIR / "manual_training.csv"

DT = 0.002

MASS = 1.0
G = 9.81
HOVER_THRUST = MASS * G / 4.0

MOTOR_MIN = 0.0
MOTOR_MAX = 8.0

Z_REF = 1.0


# ============================================================
# MANUAL CONTROL SETTINGS
# ============================================================

MAX_ROLL = np.deg2rad(8.0)
MAX_PITCH = np.deg2rad(8.0)

YAW_STEP = np.deg2rad(5.0)
THROTTLE_STEP = 0.025
ATTITUDE_STEP = np.deg2rad(1.0)


# ============================================================
# USER COMMAND STATE
# ============================================================

target_roll = 0.0
target_pitch = 0.0
target_yaw = 0.0

throttle_offset = 0.0


# ============================================================
# QUATERNION -> EULER
# ============================================================

def quat_to_euler(q):

    w, x, y, z = q

    sinr_cosp = 2.0 * (
        w * x + y * z
    )

    cosr_cosp = 1.0 - 2.0 * (
        x * x + y * y
    )

    roll = np.arctan2(
        sinr_cosp,
        cosr_cosp,
    )

    sinp = 2.0 * (
        w * y - z * x
    )

    if abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (
            np.pi / 2.0
        )
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (
        w * z + x * y
    )

    cosy_cosp = 1.0 - 2.0 * (
        y * y + z * z
    )

    yaw = np.arctan2(
        siny_cosp,
        cosy_cosp,
    )

    return np.array(
        [roll, pitch, yaw],
        dtype=float,
    )


# ============================================================
# ANGLE WRAPPING
# ============================================================

def wrap_angle(angle):

    return (
        angle + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


# ============================================================
# DMDc STATE
# ============================================================

def get_state(data):

    roll, pitch, yaw = quat_to_euler(
        data.qpos[3:7]
    )

    return np.array(
        [
            data.qpos[2] - Z_REF,
            data.qvel[2],
            roll,
            pitch,
            wrap_angle(yaw),
            data.qvel[3],
            data.qvel[4],
            data.qvel[5],
        ],
        dtype=float,
    )


# ============================================================
# KEYBOARD CALLBACK
# ============================================================

def key_callback(keycode):

    global target_roll
    global target_pitch
    global target_yaw
    global throttle_offset

    try:
        key = chr(keycode).lower()
    except ValueError:
        return

    if key == "w":

        throttle_offset += THROTTLE_STEP

    elif key == "s":

        throttle_offset -= THROTTLE_STEP

    elif key == "a":

        target_roll += ATTITUDE_STEP

    elif key == "d":

        target_roll -= ATTITUDE_STEP

    elif key == "i":

        target_pitch += ATTITUDE_STEP

    elif key == "k":

        target_pitch -= ATTITUDE_STEP

    elif key == "j":

        target_yaw -= YAW_STEP

    elif key == "l":

        target_yaw += YAW_STEP

    elif key == "r":

        target_roll = 0.0
        target_pitch = 0.0
        target_yaw = 0.0
        throttle_offset = 0.0

        print()
        print("RESET")
        print("Returning to level hover.")

    elif key == "p":

        print()
        print(
            f"Target roll  : "
            f"{np.rad2deg(target_roll):+.2f} deg"
        )

        print(
            f"Target pitch : "
            f"{np.rad2deg(target_pitch):+.2f} deg"
        )

        print(
            f"Target yaw   : "
            f"{np.rad2deg(target_yaw):+.2f} deg"
        )

        print(
            f"Throttle offset: "
            f"{throttle_offset:+.3f}"
        )

    target_roll = np.clip(
        target_roll,
        -MAX_ROLL,
        MAX_ROLL,
    )

    target_pitch = np.clip(
        target_pitch,
        -MAX_PITCH,
        MAX_PITCH,
    )

    throttle_offset = np.clip(
        throttle_offset,
        -1.5,
        1.5,
    )


# ============================================================
# STABILIZED MANUAL CONTROLLER
# ============================================================

def stabilized_controller(data):

    global target_roll
    global target_pitch
    global target_yaw
    global throttle_offset

    roll, pitch, yaw = quat_to_euler(
        data.qpos[3:7]
    )

    p = data.qvel[3]
    q = data.qvel[4]
    r = data.qvel[5]

    base = (
        HOVER_THRUST
        + throttle_offset
    )

    roll_error = (
        target_roll - roll
    )

    pitch_error = (
        target_pitch - pitch
    )

    yaw_error = wrap_angle(
        target_yaw - yaw
    )

    roll_command = np.clip(
        1.0 * roll_error
        - 0.20 * p,
        -0.35,
        0.35,
    )

    pitch_command = np.clip(
        1.0 * pitch_error
        - 0.20 * q,
        -0.35,
        0.35,
    )

    yaw_command = np.clip(
        0.30 * yaw_error
        - 0.08 * r,
        -0.20,
        0.20,
    )

    motor_1 = (
        base
        - pitch_command
        + yaw_command
    )

    motor_2 = (
        base
        - roll_command
        - yaw_command
    )

    motor_3 = (
        base
        + pitch_command
        + yaw_command
    )

    motor_4 = (
        base
        + roll_command
        - yaw_command
    )

    u = np.array(
        [
            motor_1,
            motor_2,
            motor_3,
            motor_4,
        ],
        dtype=float,
    )

    return np.clip(
        u,
        MOTOR_MIN,
        MOTOR_MAX,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STABILIZED MANUAL MUJOCO QUADROTOR")
    print("=" * 60)

    print()
    print("Controls")
    print("--------")
    print("W / S : increase / decrease thrust")
    print("A / D : roll left / right")
    print("I / K : pitch forward / backward")
    print("J / L : yaw left / right")
    print("R     : return to level hover")
    print("P     : print current targets")
    print()
    print("Maximum roll/pitch command: +/- 8 degrees")
    print()
    print("Click the MuJoCo window before using the keyboard.")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)

    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = Z_REF

    data.qpos[3] = 1.0
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    records = []

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    ) as viewer:

        print()
        print("MuJoCo viewer started.")
        print("Starting in stabilized level hover.")
        print()

        while viewer.is_running():

            step_start = time.perf_counter()

            # Calculate motor commands
            u = stabilized_controller(
                data
            )

            # Apply motor commands
            data.ctrl[:] = u

            # Advance physics
            mujoco.mj_step(
                model,
                data,
            )

            # Read state AFTER physics step
            state = get_state(
                data
            )

            # Record state + ACTUAL applied motor inputs
            records.append(
                [
                    data.time,
                    *state,
                    *data.ctrl.copy(),
                ]
            )

            # Follow quadrotor
            with viewer.lock():

                viewer.cam.lookat[:] = (
                    data.qpos[0:3]
                )

            viewer.sync()

            # Real-time timing
            elapsed = (
                time.perf_counter()
                - step_start
            )

            remaining = DT - elapsed

            if remaining > 0:

                time.sleep(
                    remaining
                )

    # ========================================================
    # SAVE MANUAL TRAINING DATA
    # ========================================================

    columns = [
        "time",
        "z_error",
        "vz",
        "roll",
        "pitch",
        "yaw",
        "p",
        "q",
        "r",
        "u1",
        "u2",
        "u3",
        "u4",
    ]

    manual_training = pd.DataFrame(
        records,
        columns=columns,
    )

    manual_training.to_csv(
        MANUAL_TRAINING_PATH,
        index=False,
    )

    print()
    print("=" * 60)
    print("MANUAL EXPERIMENT COMPLETE")
    print("=" * 60)
    print(
        f"Samples recorded: "
        f"{len(manual_training)}"
    )
    print(
        f"Saved to: "
        f"{MANUAL_TRAINING_PATH}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()