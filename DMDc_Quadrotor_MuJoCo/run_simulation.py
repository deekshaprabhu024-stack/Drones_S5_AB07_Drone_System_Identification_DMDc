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

DT = 0.002

MASS = 1.0
G = 9.81

HOVER_THRUST = MASS * G / 4.0

Z_REF = 1.0

SIMULATION_TIME = 12.0

MOTOR_MIN = 0.0
MOTOR_MAX = 8.0


# ============================================================
# QUATERNION -> EULER
# ============================================================

def quat_to_euler(q):
    """
    Convert MuJoCo quaternion [w, x, y, z]
    to roll, pitch, yaw.
    """

    w, x, y, z = q

    # Roll
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)

    roll = np.arctan2(
        sinr_cosp,
        cosr_cosp,
    )

    # Pitch
    sinp = 2.0 * (w * y - z * x)

    if abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    # Yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

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
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# ============================================================
# DMDc STATE
# ============================================================

def get_state(data):
    """
    State used by our DMDc model:

    x =
    [
        z_error,
        vertical_velocity,
        roll,
        pitch,
        yaw,
        roll_rate,
        pitch_rate,
        yaw_rate
    ]
    """

    z = data.qpos[2]

    roll, pitch, yaw = quat_to_euler(
        data.qpos[3:7]
    )

    vz = data.qvel[2]

    p = data.qvel[3]
    q = data.qvel[4]
    r = data.qvel[5]

    return np.array(
        [
            z - Z_REF,
            vz,
            roll,
            pitch,
            wrap_angle(yaw),
            p,
            q,
            r,
        ],
        dtype=float,
    )


# ============================================================
# HOVER / POSITION CONTROLLER
# ============================================================

class HoverController:

    def __init__(self):
        self.integral_z = 0.0

    def reset(self):
        self.integral_z = 0.0

    def command(self, data, dt):

        # ----------------------------------------------------
        # Position
        # ----------------------------------------------------

        x = data.qpos[0]
        y = data.qpos[1]
        z = data.qpos[2]

        vx = data.qvel[0]
        vy = data.qvel[1]
        vz = data.qvel[2]

        # ----------------------------------------------------
        # Orientation
        # ----------------------------------------------------

        roll, pitch, yaw = quat_to_euler(
            data.qpos[3:7]
        )

        p = data.qvel[3]
        q = data.qvel[4]
        r = data.qvel[5]

        # ----------------------------------------------------
        # Horizontal position controller
        # ----------------------------------------------------

        x_error = -x
        y_error = -y

        ax_command = (
            1.5 * x_error
            - 0.9 * vx
        )

        ay_command = (
            1.5 * y_error
            - 0.9 * vy
        )

        desired_pitch = np.clip(
            ax_command / G,
            -0.25,
            0.25,
        )

        desired_roll = np.clip(
            -ay_command / G,
            -0.25,
            0.25,
        )

        # ----------------------------------------------------
        # Vertical controller
        # ----------------------------------------------------

        z_error = Z_REF - z

        self.integral_z += (
            z_error * dt
        )

        self.integral_z = np.clip(
            self.integral_z,
            -0.5,
            0.5,
        )

        az_command = (
            8.0 * z_error
            - 3.5 * vz
            + 0.8 * self.integral_z
        )

        total_thrust = MASS * (
            G + az_command
        )

        total_thrust = np.clip(
            total_thrust,
            0.0,
            4.0 * MOTOR_MAX,
        )

        # ----------------------------------------------------
        # Attitude controller
        # ----------------------------------------------------

        roll_command = np.clip(
            0.35 * (
                desired_roll - roll
            )
            - 0.10 * p,
            -0.8,
            0.8,
        )

        pitch_command = np.clip(
            0.35 * (
                desired_pitch - pitch
            )
            - 0.10 * q,
            -0.8,
            0.8,
        )

        yaw_error = wrap_angle(
            -yaw
        )

        yaw_command = np.clip(
            0.18 * yaw_error
            - 0.05 * r,
            -0.6,
            0.6,
        )

        # ----------------------------------------------------
        # QUADROTOR MOTOR MIXING
        # ----------------------------------------------------

        base = total_thrust / 4.0

        # Motor layout:
        #
        #             M3 (-X)
        #
        #               |
        #
        # M2 (-Y) ---- BODY ---- M4 (+Y)
        #
        #               |
        #
        #             M1 (+X)
        #
        # The actual signs below match the actuator
        # locations defined in quadrotor.xml.

        u = np.array(
            [
                base - pitch_command + yaw_command,
                base - roll_command - yaw_command,
                base + pitch_command + yaw_command,
                base + roll_command - yaw_command,
            ],
            dtype=float,
        )

        return np.clip(
            u,
            MOTOR_MIN,
            MOTOR_MAX,
        )


# ============================================================
# EXCITATION SIGNAL
# ============================================================

def excitation(t, trial):
    """
    Small deterministic multisine excitation.

    This is important for DMDc because a perfectly
    stationary hover does not contain enough information
    to identify the input-output dynamics.
    """

    phase = (
        0.0
        if trial == 0
        else 0.7
    )

    e = np.array(
        [
            0.22 * np.sin(
                2.0 * np.pi * 0.7 * t
                + phase
            )
            + 0.10 * np.sin(
                2.0 * np.pi * 1.3 * t
            ),

            0.18 * np.sin(
                2.0 * np.pi * 0.9 * t
                + 1.2
                + phase
            )
            + 0.08 * np.sin(
                2.0 * np.pi * 1.6 * t
            ),

            0.22 * np.sin(
                2.0 * np.pi * 0.8 * t
                + 2.1
                + phase
            )
            + 0.10 * np.sin(
                2.0 * np.pi * 1.1 * t
            ),

            0.18 * np.sin(
                2.0 * np.pi * 1.0 * t
                + 2.8
                + phase
            )
            + 0.08 * np.sin(
                2.0 * np.pi * 1.7 * t
            ),
        ],
        dtype=float,
    )

    # Smoothly turn excitation on and off.
    ramp_up = np.clip(
        t / 1.0,
        0.0,
        1.0,
    )

    ramp_down = np.clip(
        (SIMULATION_TIME - t) / 1.0,
        0.0,
        1.0,
    )

    ramp = (
        ramp_up
        * ramp_down
    )

    return ramp * e


# ============================================================
# RESET
# ============================================================

def reset_data(model, data):

    mujoco.mj_resetData(
        model,
        data,
    )

    data.qpos[:] = model.qpos0

    # Start 1 meter above the floor.
    data.qpos[0:3] = np.array(
        [0.0, 0.0, Z_REF]
    )

    # Identity quaternion.
    data.qpos[3:7] = np.array(
        [1.0, 0.0, 0.0, 0.0]
    )

    data.qvel[:] = 0.0

    data.ctrl[:] = HOVER_THRUST

    mujoco.mj_forward(
        model,
        data,
    )


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_trial(
    trial,
    duration=SIMULATION_TIME,
    show_viewer=False,
):

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)

    reset_data(
        model,
        data,
    )

    controller = HoverController()

    records = []

    wall_start = time.perf_counter()

    viewer_context = None

    if show_viewer:
        viewer_context = (
            mujoco.viewer.launch_passive(
                model,
                data,
            )
        )

    try:

        while data.time < duration:

            t = float(data.time)

            # ------------------------------------------------
            # Baseline controller
            # ------------------------------------------------

            u_base = controller.command(
                data,
                DT,
            )

            # ------------------------------------------------
            # Inject excitation
            # ------------------------------------------------

            u = (
                u_base
                + excitation(
                    t,
                    trial,
                )
            )

            # ------------------------------------------------
            # Apply motor commands
            # ------------------------------------------------

            data.ctrl[:] = np.clip(
                u,
                MOTOR_MIN,
                MOTOR_MAX,
            )

            # ------------------------------------------------
            # Advance MuJoCo physics
            # ------------------------------------------------

            mujoco.mj_step(
                model,
                data,
            )

            # ------------------------------------------------
            # Read state
            # ------------------------------------------------

            state = get_state(
                data
            )

            # ------------------------------------------------
            # Save sample
            # ------------------------------------------------

            records.append(
                [
                    data.time,
                    *state,
                    *data.ctrl.copy(),
                ]
            )

            # ------------------------------------------------
            # Viewer synchronization
            # ------------------------------------------------

            if viewer_context is not None:

            # ------------------------------------------------
            # Keep camera centered on the quadrotor
            # ------------------------------------------------

                viewer_context.cam.lookat[:] = data.qpos[0:3]

                viewer_context.sync()

                elapsed = (
                    time.perf_counter()
                    - wall_start
                )

                if elapsed < data.time:
                    time.sleep(
                        data.time
                        - elapsed
                    )

    finally:

        if viewer_context is not None:
            viewer_context.close()

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

    return pd.DataFrame(
        records,
        columns=columns,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("DMDc QUADROTOR SIMULATION")
    print("=" * 60)

    print()
    print("Running training experiment...")

    training = run_trial(
        trial=0,
        duration=SIMULATION_TIME,
        show_viewer=False,
    )

    training_path = (
        DATA_DIR
        / "training.csv"
    )

    training.to_csv(
        training_path,
        index=False,
    )

    print(
        f"Training samples: "
        f"{len(training)}"
    )

    print()
    print(
        "Training data saved to:"
    )
    print(training_path)

    print()
    print(
        "Running validation experiment."
    )
    print(
        "A MuJoCo window will open."
    )

    validation = run_trial(
        trial=1,
        duration=SIMULATION_TIME,
        show_viewer=True,
    )

    validation_path = (
        DATA_DIR
        / "validation.csv"
    )

    validation.to_csv(
        validation_path,
        index=False,
    )

    print()
    print(
        f"Validation samples: "
        f"{len(validation)}"
    )

    print()
    print(
        "Validation data saved to:"
    )
    print(validation_path)

    print()
    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)