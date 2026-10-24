"""
gazebo_env.py
-------------
ROS 2 wrapper around the Gazebo quadrotor.

The original project uses the six translational DMDc states
[x, y, z, vx, vy, vz]. Gazebo publishes the model odometry reliably (via
the OdometryPublisher plugin in model.sdf), so this wrapper reads pose out
of the bridged odometry topic and numerically differentiates position and
attitude itself to obtain velocity and angular-rate estimates for the PID
loop (rather than trusting the plugin's own twist field, whose frame
convention we don't want to depend on).
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators


def quat_to_euler(x, y, z, w):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class GazeboDroneEnv(Node):
    def __init__(self, dt=0.02, model_name="x500_dmdc", max_omega=1100.0):
        super().__init__("gazebo_drone_env")
        self.dt = dt
        self.model_name = model_name
        self.max_omega = max_omega

        # [x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r]
        self._state = np.zeros(12, dtype=float)
        self._have_pose = False
        self._last_position = None
        self._last_attitude = None
        self._last_time = None

        # Pose is bridged from Gazebo's /model/x500_dmdc/odometry. The model
        # only has an OdometryPublisher plugin (no PosePublisher), so we
        # pull position/orientation out of the Odometry message rather than
        # subscribing to a /pose topic that nothing actually publishes.
        # Sensor-data QoS is deliberately used so the subscriber also accepts
        # best-effort bridge publishers.
        self.create_subscription(
            Odometry,
            f"/{model_name}/odometry",
            self._pose_callback,
            qos_profile_sensor_data,
        )

        self.motor_pub = self.create_publisher(
            Actuators,
            f"/{model_name}/command/motor_speed",
            10,
        )

    def _pose_callback(self, msg: Odometry):
        now = time.monotonic()
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        position = np.array([p.x, p.y, p.z], dtype=float)
        attitude = np.array(quat_to_euler(q.x, q.y, q.z, q.w), dtype=float)

        velocity = np.zeros(3)
        angular_velocity = np.zeros(3)

        if self._last_position is not None and self._last_time is not None:
            dtime = now - self._last_time
            if dtime > 1e-4:
                velocity = (position - self._last_position) / dtime
                da = np.array([
                    wrap_angle(attitude[0] - self._last_attitude[0]),
                    wrap_angle(attitude[1] - self._last_attitude[1]),
                    wrap_angle(attitude[2] - self._last_attitude[2]),
                ])
                angular_velocity = da / dtime

                # Light clipping prevents one delayed WSL GUI frame from
                # producing an absurd derivative that destabilizes the PID.
                velocity = np.clip(velocity, -8.0, 8.0)
                angular_velocity = np.clip(angular_velocity, -15.0, 15.0)

        self._state = np.concatenate([
            position, velocity, attitude, angular_velocity
        ])
        self._last_position = position
        self._last_attitude = attitude
        self._last_time = now
        self._have_pose = True

    def wait_for_odometry(self, timeout=15.0):
        """Backward-compatible name: wait until bridged Gazebo pose arrives."""
        start = time.time()
        while rclpy.ok() and not self._have_pose:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout:
                raise TimeoutError(
                    "No Gazebo odometry received. Make sure the simulation "
                    "and ros_gz_bridge are running. Expected ROS topic: "
                    f"/{self.model_name}/odometry"
                )

    def get_state(self):
        rclpy.spin_once(self, timeout_sec=0.0)
        return self._state.copy()

    def get_dmdc_state(self):
        rclpy.spin_once(self, timeout_sec=0.0)
        return self._state[:6].copy()

    def get_attitude(self):
        return self._state[6:9].copy(), self._state[9:12].copy()

    def send_action(self, omega):
        omega = np.clip(np.asarray(omega, dtype=float), 0.0, self.max_omega)
        msg = Actuators()
        msg.velocity = omega.tolist()
        self.motor_pub.publish(msg)
        # Ensure the ROS publisher has a chance to hand the message to the bridge.
        rclpy.spin_once(self, timeout_sec=0.0)

    def step(self, omega):
        self.send_action(omega)
        time.sleep(self.dt)
        rclpy.spin_once(self, timeout_sec=0.0)
        state = self.get_state()

        terminated = bool(
            abs(state[0]) > 8.0
            or abs(state[1]) > 8.0
            or abs(state[6]) > np.deg2rad(70)
            or abs(state[7]) > np.deg2rad(70)
        )
        return state, 0.0, terminated, False, {}

    def close(self):
        try:
            self.send_action(np.zeros(4))
        except Exception:
            pass
        self.destroy_node()
