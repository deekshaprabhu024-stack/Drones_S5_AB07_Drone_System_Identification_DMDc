# manual_control_koopman.py

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

TRAINING_PATH = DATA_DIR / "circle_training.csv"
VALIDATION_PATH = DATA_DIR / "circle_validation.csv"

DT = 0.002

MASS = 1.0
G = 9.81

HOVER_THRUST = MASS * G / 4.0

MOTOR_MIN = 0.0
MOTOR_MAX = 8.0


# ============================================================
# FLIGHT SEQUENCE
# ============================================================

TAKEOFF_TIME = 5.0
CIRCLE_TIME = 20.0
LAND_TIME = 5.0

TOTAL_TIME = (
    TAKEOFF_TIME
    + CIRCLE_TIME
    + LAND_TIME
)


# ============================================================
# CIRCLE
# ============================================================

CIRCLE_RADIUS = 1.0

CIRCLE_PERIOD = 10.0

OMEGA = (
    2.0 * np.pi
    / CIRCLE_PERIOD
)

FLIGHT_ALTITUDE = 1.5


# ============================================================
# POSITION CONTROLLER
# ============================================================

KP_X = 2.0
KD_X = 1.5

KP_Y = 2.0
KD_Y = 1.5

KP_Z = 6.0
KD_Z = 3.5


# ============================================================
# ATTITUDE CONTROLLER
# ============================================================

KP_ROLL = 4.0
KD_ROLL = 0.8

KP_PITCH = 4.0
KD_PITCH = 0.8

KP_YAW = 1.0
KD_YAW = 0.15

MAX_ROLL = np.deg2rad(10.0)
MAX_PITCH = np.deg2rad(10.0)


# ============================================================
# QUATERNION -> EULER
# ============================================================

def quat_to_euler(q):

    w, x, y, z = q

    sinr = 2.0 * (
        w * x
        + y * z
    )

    cosr = 1.0 - 2.0 * (
        x * x
        + y * y
    )

    roll = np.arctan2(
        sinr,
        cosr,
    )

    sinp = 2.0 * (
        w * y
        - z * x
    )

    pitch = np.arcsin(
        np.clip(
            sinp,
            -1.0,
            1.0,
        )
    )

    siny = 2.0 * (
        w * z
        + x * y
    )

    cosy = 1.0 - 2.0 * (
        y * y
        + z * z
    )

    yaw = np.arctan2(
        siny,
        cosy,
    )

    return (
        roll,
        pitch,
        yaw,
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
# CIRCLE PHASE
# ============================================================

def get_circle_phase(mode):

    if mode == "T":
        return 0.0

    if mode == "V":
        return np.pi

    raise ValueError(
        f"Invalid mode: {mode}"
    )


# ============================================================
# FLIGHT REFERENCE
# ============================================================

def flight_reference(t, mode):

    phase = get_circle_phase(mode)

    # --------------------------------------------------------
    # TAKEOFF
    # --------------------------------------------------------

    if t < TAKEOFF_TIME:

        alpha = (
            t / TAKEOFF_TIME
        )

        # Smooth altitude ramp
        alpha = (
            3.0 * alpha ** 2
            - 2.0 * alpha ** 3
        )

        x_ref = (
            CIRCLE_RADIUS
            * np.cos(phase)
        )

        y_ref = (
            CIRCLE_RADIUS
            * np.sin(phase)
        )

        vx_ref = 0.0
        vy_ref = 0.0

        z_ref = (
            0.15
            + (
                FLIGHT_ALTITUDE
                - 0.15
            ) * alpha
        )

        vz_ref = 0.0

        return (
            x_ref,
            y_ref,
            z_ref,
            vx_ref,
            vy_ref,
            vz_ref,
        )

    # --------------------------------------------------------
    # CIRCULAR FLIGHT
    # --------------------------------------------------------

    if t < (
        TAKEOFF_TIME
        + CIRCLE_TIME
    ):

        tau = (
            t
            - TAKEOFF_TIME
        )

        theta = (
            OMEGA * tau
            + phase
        )

        x_ref = (
            CIRCLE_RADIUS
            * np.cos(theta)
        )

        y_ref = (
            CIRCLE_RADIUS
            * np.sin(theta)
        )

        vx_ref = (
            -CIRCLE_RADIUS
            * OMEGA
            * np.sin(theta)
        )

        vy_ref = (
            CIRCLE_RADIUS
            * OMEGA
            * np.cos(theta)
        )

        z_ref = FLIGHT_ALTITUDE

        vz_ref = 0.0

        return (
            x_ref,
            y_ref,
            z_ref,
            vx_ref,
            vy_ref,
            vz_ref,
        )

    # --------------------------------------------------------
    # LAND
    # --------------------------------------------------------

    tau = (
        t
        - TAKEOFF_TIME
        - CIRCLE_TIME
    )

    alpha = np.clip(
        tau / LAND_TIME,
        0.0,
        1.0,
    )

    alpha = (
        3.0 * alpha ** 2
        - 2.0 * alpha ** 3
    )

    # After exactly two circles, the vehicle
    # returns to the phase starting point.
    x_ref = (
        CIRCLE_RADIUS
        * np.cos(phase)
    )

    y_ref = (
        CIRCLE_RADIUS
        * np.sin(phase)
    )

    vx_ref = 0.0
    vy_ref = 0.0

    z_ref = (
        FLIGHT_ALTITUDE
        + (
            0.15
            - FLIGHT_ALTITUDE
        ) * alpha
    )

    vz_ref = 0.0

    return (
        x_ref,
        y_ref,
        z_ref,
        vx_ref,
        vy_ref,
        vz_ref,
    )


# ============================================================
# CONTROLLER
# ============================================================

def controller(data, mode):

    roll, pitch, yaw = quat_to_euler(
        data.qpos[3:7]
    )

    p = data.qvel[3]
    q = data.qvel[4]
    r = data.qvel[5]

    x = data.qpos[0]
    y = data.qpos[1]
    z = data.qpos[2]

    vx = data.qvel[0]
    vy = data.qvel[1]
    vz = data.qvel[2]

    (
        x_ref,
        y_ref,
        z_ref,
        vx_ref,
        vy_ref,
        vz_ref,
    ) = flight_reference(
        data.time,
        mode
    )

    # --------------------------------------------------------
    # Position errors
    # --------------------------------------------------------

    ex = x_ref - x
    ey = y_ref - y
    ez = z_ref - z

    evx = vx_ref - vx
    evy = vy_ref - vy
    evz = vz_ref - vz

    # --------------------------------------------------------
    # Desired horizontal acceleration
    # --------------------------------------------------------

    ax = (
        KP_X * ex
        + KD_X * evx
    )

    ay = (
        KP_Y * ey
        + KD_Y * evy
    )

    # Limit horizontal acceleration
    ax = np.clip(
        ax,
        -1.5,
        1.5,
    )

    ay = np.clip(
        ay,
        -1.5,
        1.5,
    )

    # --------------------------------------------------------
    # Convert acceleration to attitude
    #
    # Small-angle approximation:
    #
    #     pitch ~= ax / g
    #     roll  ~= -ay / g
    # --------------------------------------------------------

    target_pitch = np.clip(
        ax / G,
        -MAX_PITCH,
        MAX_PITCH,
    )

    target_roll = np.clip(
        -ay / G,
        -MAX_ROLL,
        MAX_ROLL,
    )

    # --------------------------------------------------------
    # Vertical control
    # --------------------------------------------------------

    vertical_acceleration = (
        KP_Z * ez
        + KD_Z * evz
    )

    vertical_acceleration = np.clip(
        vertical_acceleration,
        -5.0,
        5.0,
    )

    total_thrust = MASS * (
        G
        + vertical_acceleration
    )

    # Compensate for tilt
    tilt_factor = (
        np.cos(roll)
        * np.cos(pitch)
    )

    tilt_factor = max(
        tilt_factor,
        0.7,
    )

    total_thrust /= tilt_factor

    base = (
        total_thrust / 4.0
    )

    # --------------------------------------------------------
    # Attitude control
    # --------------------------------------------------------

    roll_error = (
        target_roll - roll
    )

    pitch_error = (
        target_pitch - pitch
    )

    yaw_error = wrap_angle(
        -yaw
    )

    roll_command = np.clip(
        KP_ROLL * roll_error
        - KD_ROLL * p,
        -0.5,
        0.5,
    )

    pitch_command = np.clip(
        KP_PITCH * pitch_error
        - KD_PITCH * q,
        -0.5,
        0.5,
    )

    yaw_command = np.clip(
        KP_YAW * yaw_error
        - KD_YAW * r,
        -0.2,
        0.2,
    )

    # --------------------------------------------------------
    # Motor mixing
    #
    # Motor locations:
    #
    # 1 = +X
    # 2 = -Y
    # 3 = -X
    # 4 = +Y
    # --------------------------------------------------------

    u1 = (
        base
        - pitch_command
        + yaw_command
    )

    u2 = (
        base
        - roll_command
        - yaw_command
    )

    u3 = (
        base
        + pitch_command
        + yaw_command
    )

    u4 = (
        base
        + roll_command
        - yaw_command
    )

    return np.clip(
        np.array(
            [
                u1,
                u2,
                u3,
                u4,
            ]
        ),
        MOTOR_MIN,
        MOTOR_MAX,
    )


# ============================================================
# RUN
# ============================================================

def run_experiment(mode):

    phase = get_circle_phase(mode)

    if mode == "T":
        output_path = TRAINING_PATH
    else:
        output_path = VALIDATION_PATH

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)

    # --------------------------------------------------------
    # Start on the ground at the phase-specific
    # starting point of the circle.
    # --------------------------------------------------------

    data.qpos[0] = (
        CIRCLE_RADIUS
        * np.cos(phase)
    )

    data.qpos[1] = (
        CIRCLE_RADIUS
        * np.sin(phase)
    )

    data.qpos[2] = 0.15

    # Identity quaternion
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

    print()
    print("=" * 65)
    print("MUJOCO TAKEOFF -> CIRCLE -> LAND")
    print("=" * 65)
    print()

    print(
        f"Mode            : {mode}"
    )

    print(
        f"Phase           : {phase:.6f} rad"
    )

    print(
        f"Takeoff         : {TAKEOFF_TIME:.1f} s"
    )

    print(
        f"Circular flight : {CIRCLE_TIME:.1f} s"
    )

    print(
        f"Landing         : {LAND_TIME:.1f} s"
    )

    print(
        f"Circle radius   : {CIRCLE_RADIUS:.1f} m"
    )

    print(
        f"Circle period   : {CIRCLE_PERIOD:.1f} s"
    )

    print(
        f"Flight altitude : {FLIGHT_ALTITUDE:.1f} m"
    )

    print()

    print(
        "The camera is FIXED so the circular motion is visible."
    )

    print()

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        # ----------------------------------------------------
        # Fixed camera
        # ----------------------------------------------------

        with viewer.lock():

            viewer.cam.lookat[:] = np.array(
                [
                    0.0,
                    0.0,
                    0.7,
                ]
            )

            viewer.cam.distance = 4.5
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -35

        viewer.sync()

        while (
            viewer.is_running()
            and data.time < TOTAL_TIME
        ):

            step_start = time.perf_counter()

            # ------------------------------------------------
            # Controller
            # ------------------------------------------------

            u = controller(
                data,
                mode,
            )

            data.ctrl[:] = u

            # ------------------------------------------------
            # Physics
            # ------------------------------------------------

            mujoco.mj_step(
                model,
                data,
            )

            # ------------------------------------------------
            # Reference
            # ------------------------------------------------

            (
                x_ref,
                y_ref,
                z_ref,
                vx_ref,
                vy_ref,
                vz_ref,
            ) = flight_reference(
                data.time,
                mode,
            )

            # ------------------------------------------------
            # Record
            # ------------------------------------------------

            roll, pitch, yaw = quat_to_euler(
                data.qpos[3:7]
            )

            records.append(
                [
                    data.time,

                    # State
                    data.qpos[0],
                    data.qpos[1],
                    data.qpos[2],

                    data.qvel[0],
                    data.qvel[1],
                    data.qvel[2],

                    roll,
                    pitch,
                    yaw,

                    data.qvel[3],
                    data.qvel[4],
                    data.qvel[5],

                    # Reference
                    x_ref,
                    y_ref,
                    z_ref,

                    vx_ref,
                    vy_ref,
                    vz_ref,

                    # Inputs
                    *data.ctrl.copy(),
                ]
            )

            viewer.sync()

            # ------------------------------------------------
            # Real-time simulation
            # ------------------------------------------------

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
    # SAVE
    # ========================================================

    columns = [

        "time",

        "x",
        "y",
        "z",

        "vx",
        "vy",
        "vz",

        "roll",
        "pitch",
        "yaw",

        "p",
        "q",
        "r",

        "x_ref",
        "y_ref",
        "z_ref",

        "vx_ref",
        "vy_ref",
        "vz_ref",

        "u1",
        "u2",
        "u3",
        "u4",
    ]

    df = pd.DataFrame(
        records,
        columns=columns,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 65)
    print("EXPERIMENT COMPLETE")
    print("=" * 65)

    print(
        f"Samples : {len(df)}"
    )

    print(
        f"Saved   : {output_path}"
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("T = Training trajectory")
    print("V = Validation trajectory")
    print("Q = Quit")
    print()

    while True:

        command = input(
            "Enter command [T/V/Q]: "
        ).strip().upper()

        if command == "T":

            run_experiment("T")

        elif command == "V":

            run_experiment("V")

        elif command == "Q":

            break

        else:

            print(
                "Please enter T, V, or Q."
            )


if __name__ == "__main__":
    main()