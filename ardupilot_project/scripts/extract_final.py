import sys
import csv
import math
import numpy as np

from pymavlink import mavutil


if len(sys.argv) != 3:
    print("Usage:")
    print("python extract_final.py INPUT.BIN OUTPUT.csv")
    sys.exit(1)


INPUT = sys.argv[1]
OUTPUT = sys.argv[2]


# =========================================================
# Quaternion from ArduPilot ATT values
#
# IMPORTANT:
# ArduPilot ATT Roll/Pitch are radians.
# ArduPilot ATT Yaw is degrees.
# =========================================================

def quaternion_from_att(roll, pitch, yaw_deg):

    yaw = math.radians(yaw_deg)

    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)

    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)

    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)

    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy

    q = np.array([q0, q1, q2, q3], dtype=float)

    return q / np.linalg.norm(q)


# =========================================================
# Read streams independently
# =========================================================

mav = mavutil.mavlink_connection(INPUT)

ATT = []
RATE = []
RCOU = []


while True:

    msg = mav.recv_match(
        type=["ATT", "RATE", "RCOU"],
        blocking=False
    )

    if msg is None:
        break

    typ = msg.get_type()

    t = getattr(msg, "TimeUS", None)

    if t is None:
        continue

    t = float(t)

    if typ == "ATT":

        ATT.append([
            t,
            float(msg.Roll),
            float(msg.Pitch),
            float(msg.Yaw)
        ])

    elif typ == "RATE":

        RATE.append([
            t,
            float(msg.R),
            float(msg.P),
            float(msg.Y)
        ])

    elif typ == "RCOU":

        RCOU.append([
            t,
            float(msg.C1),
            float(msg.C2),
            float(msg.C3),
            float(msg.C4)
        ])


ATT = np.asarray(ATT)
RATE = np.asarray(RATE)
RCOU = np.asarray(RCOU)


print("ATT samples :", len(ATT))
print("RATE samples:", len(RATE))
print("RCOU samples:", len(RCOU))


# =========================================================
# Sort
# =========================================================

ATT = ATT[np.argsort(ATT[:, 0])]
RATE = RATE[np.argsort(RATE[:, 0])]
RCOU = RCOU[np.argsort(RCOU[:, 0])]


# =========================================================
# ATT timestamps = common sampling grid
# =========================================================

t = ATT[:, 0]


# =========================================================
# Interpolate RATE onto ATT timestamps
# =========================================================

wx = np.interp(
    t,
    RATE[:, 0],
    RATE[:, 1]
)

wy = np.interp(
    t,
    RATE[:, 0],
    RATE[:, 2]
)

wz = np.interp(
    t,
    RATE[:, 0],
    RATE[:, 3]
)


# =========================================================
# Interpolate motor outputs onto ATT timestamps
# =========================================================

m1 = np.interp(
    t,
    RCOU[:, 0],
    RCOU[:, 1]
)

m2 = np.interp(
    t,
    RCOU[:, 0],
    RCOU[:, 2]
)

m3 = np.interp(
    t,
    RCOU[:, 0],
    RCOU[:, 3]
)

m4 = np.interp(
    t,
    RCOU[:, 0],
    RCOU[:, 4]
)


# =========================================================
# Quaternions
# =========================================================

quaternions = []

for i in range(len(ATT)):

    q = quaternion_from_att(
        ATT[i, 1],
        ATT[i, 2],
        ATT[i, 3]
    )

    quaternions.append(q)


Q = np.asarray(quaternions)


# =========================================================
# Quaternion sign continuity
#
# q and -q represent the same orientation.
# For a continuous trajectory, choose the sign closest
# to the previous quaternion.
# =========================================================

for k in range(1, len(Q)):

    if np.dot(Q[k - 1], Q[k]) < 0:
        Q[k] *= -1


# =========================================================
# Final synchronized dataset
# =========================================================

with open(OUTPUT, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "time_us",
        "q0", "q1", "q2", "q3",
        "wx", "wy", "wz",
        "motor1", "motor2", "motor3", "motor4"
    ])

    for i in range(len(t)):

        writer.writerow([
            int(t[i]),
            Q[i, 0],
            Q[i, 1],
            Q[i, 2],
            Q[i, 3],
            wx[i],
            wy[i],
            wz[i],
            m1[i],
            m2[i],
            m3[i],
            m4[i]
        ])


print()
print("Saved:", len(t), "synchronized samples")
print(
    "Duration:",
    (t[-1] - t[0]) / 1e6,
    "seconds"
)
print(
    "Mean dt:",
    np.mean(np.diff(t)) / 1e6,
    "seconds"
)
print("Output:", OUTPUT)
