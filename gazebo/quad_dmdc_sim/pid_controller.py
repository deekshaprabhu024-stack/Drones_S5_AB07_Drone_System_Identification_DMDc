"""
Stable cascaded PID controller for the Gazebo X-configuration quadrotor.

The Gazebo motor plugin expects rotor angular velocity in rad/s.
This controller:
  position error -> desired acceleration
  -> desired roll/pitch
  -> physical attitude torques (N m)
  -> rotor forces (N)
  -> rotor angular velocities (rad/s)
"""

import numpy as np


class PIDController:

    def __init__(self, hover_omega, dt, mass=0.9, gravity=9.81,
                 motor_constant=8.54858e-06, arm_length=0.174,
                 max_omega=1100.0):

        self.hover_omega = hover_omega
        self.dt = dt
        self.mass = mass
        self.gravity = gravity
        self.k_f = motor_constant
        self.arm_length = arm_length
        self.max_omega = max_omega

        # Position controller.
        # Horizontal gains are deliberately conservative.
        self.kp_pos = np.array([1.2, 1.2, 4.0])
        self.kd_pos = np.array([1.8, 1.8, 3.0])

        # Never ask the vehicle for a large attitude change.
        self.max_tilt = np.deg2rad(8.0)

        # Physical attitude gains.
        # With Ixx=Iyy ~= 0.0053 kg m^2, these give a
        # moderately damped response instead of huge torques.
        self.kp_att = np.array([0.12, 0.12, 0.06])
        self.kd_att = np.array([0.035, 0.035, 0.020])

        # Gazebo motor model uses momentConstant/thrustConstant
        # as the yaw torque / thrust ratio.
        self.moment_constant = 0.016

        self.yaw_target = 0.0

    def reset(self):
        pass

    @staticmethod
    def _wrap_angle(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _position_loop(self, position, velocity, target):

        error = target - position
        accel_cmd = self.kp_pos * error - self.kd_pos * velocity

        # Limit commanded acceleration so the controller cannot
        # demand an abrupt takeoff or aggressive lateral tilt.
        accel_cmd[0:2] = np.clip(accel_cmd[0:2], -2.0, 2.0)
        accel_cmd[2] = np.clip(accel_cmd[2], -3.0, 3.0)

        pitch_des = np.clip(
            accel_cmd[0] / self.gravity, -1.0, 1.0
        ) * self.max_tilt

        roll_des = np.clip(
            -accel_cmd[1] / self.gravity, -1.0, 1.0
        ) * self.max_tilt

        # Compensate slightly for the loss of vertical thrust when tilted.
        tilt_comp = max(
            np.cos(roll_des) * np.cos(pitch_des), 0.85
        )

        thrust = self.mass * (self.gravity + accel_cmd[2]) / tilt_comp

        # Keep total thrust well inside the motor capability.
        thrust = np.clip(thrust, 0.0, 25.0)

        return thrust, roll_des, pitch_des

    def _attitude_loop(self, attitude, angular_velocity,
                       roll_des, pitch_des):

        roll, pitch, yaw = attitude
        p, q, r = angular_velocity

        att_error = np.array([
            roll_des - roll,
            pitch_des - pitch,
            self._wrap_angle(self.yaw_target - yaw),
        ])

        rate = np.array([p, q, r])

        # These are actual desired body torques in N m.
        return self.kp_att * att_error - self.kd_att * rate

    def compute(self, position, velocity, target, attitude,
                angular_velocity, **kwargs):

        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        target = np.asarray(target, dtype=float)
        attitude = np.asarray(attitude, dtype=float)
        angular_velocity = np.asarray(angular_velocity, dtype=float)

        thrust, roll_des, pitch_des = self._position_loop(
            position, velocity, target
        )

        roll_t, pitch_t, yaw_t = self._attitude_loop(
            attitude, angular_velocity, roll_des, pitch_des
        )

        # ---------------------------------------------------------
        # Physical X-configuration mixer.
        #
        # Rotor positions:
        #   0 = (+L,-L) CCW
        #   1 = (-L,+L) CCW
        #   2 = (+L,+L) CW
        #   3 = (-L,-L) CW
        #
        # A differential force 'a' on the four rotors produces
        # roll/pitch torque = 4*L*a.
        # ---------------------------------------------------------
        roll_force = roll_t / (4.0 * self.arm_length)
        pitch_force = pitch_t / (4.0 * self.arm_length)

        # For yaw, each rotor's reaction torque is approximately
        # moment_constant * rotor thrust.
        yaw_force = yaw_t / (4.0 * self.moment_constant)

        # Keep yaw correction conservative.
        yaw_force = np.clip(yaw_force, -0.08, 0.08)

        f_base = thrust / 4.0

        f0 = f_base - roll_force - pitch_force - yaw_force
        f1 = f_base + roll_force + pitch_force - yaw_force
        f2 = f_base + roll_force - pitch_force + yaw_force
        f3 = f_base - roll_force + pitch_force + yaw_force

        forces = np.clip(
            np.array([f0, f1, f2, f3], dtype=float),
            0.0,
            None
        )

        omega = np.sqrt(forces / self.k_f)
        omega = np.clip(omega, 0.0, self.max_omega)

        return omega
