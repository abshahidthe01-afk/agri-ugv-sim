"""
A 2D extended Kalman filter for the robot's pose: position, heading and gyro bias.

State x = [px, py, yaw, gyro_bias, gnss_east, gnss_north] in the world frame (metres,
radians, rad/s, metres). The last two are the GNSS error shared by both antennas: it drifts
slowly, so the filter models it instead of trusting every fix as independent; otherwise it
would report a position five times more certain than it is.
Predict: wheel odometry gives the body velocity (how far), the gyroscope the turn rate
(how much it turned); both carry noise, so the uncertainty P grows. Correct: a GNSS
antenna position or the dual-antenna heading pulls the state towards the measurement,
weighted by how certain each side is (the Kalman gain).
"""

import math

import numpy as np


def wrap(angle):
    """Return the angle in [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class PoseEkf:
    """Pose filter: predict with wheels and gyro, correct with GNSS positions and heading."""

    def __init__(self, x, y, yaw, sigma_position, sigma_yaw, sigma_bias=0.001,
                 gnss_sigma=0.010, gnss_tau=60.0):
        """Start at a measured pose; gnss_sigma and gnss_tau describe the shared GNSS drift."""
        self.x = np.array([x, y, yaw, 0.0, 0.0, 0.0])
        self.P = np.diag([sigma_position ** 2, sigma_position ** 2, sigma_yaw ** 2,
                          sigma_bias ** 2, gnss_sigma ** 2, gnss_sigma ** 2])
        self.gnss_sigma, self.gnss_tau = gnss_sigma, gnss_tau

    def predict(self, dt, vx, vy, gyro, sigma_speed, sigma_gyro, sigma_bias_walk=1e-5):
        """
        Move the state by the body velocity (vx, vy) and measured turn rate for dt seconds.

        sigma_speed and sigma_gyro are the standard deviations of one velocity and one gyro
        sample; sigma_bias_walk [rad/s per sqrt(s)] lets the bias drift slowly.
        """
        if dt <= 0:
            return
        px, py, yaw, bias, east, north = self.x
        rate = gyro - bias
        mid = yaw + rate * dt / 2
        c, s = math.cos(mid), math.sin(mid)
        dx_dyaw = (-s * vx - c * vy) * dt          # how the step moves when the heading moves
        dy_dyaw = (c * vx - s * vy) * dt
        fade = math.exp(-dt / self.gnss_tau)               # the shared GNSS error drifts
        self.x = np.array([px + (c * vx - s * vy) * dt, py + (s * vx + c * vy) * dt,
                           wrap(yaw + rate * dt), bias, fade * east, fade * north])
        F = np.eye(6)
        F[0, 2], F[1, 2], F[2, 3] = dx_dyaw, dy_dyaw, -dt
        F[0, 3], F[1, 3] = -dx_dyaw * dt / 2, -dy_dyaw * dt / 2
        F[4, 4] = F[5, 5] = fade
        G = np.zeros((6, 3))                         # effect of (vx, vy, gyro) noise
        G[:3] = [[c * dt, -s * dt, dx_dyaw * dt / 2], [s * dt, c * dt, dy_dyaw * dt / 2],
                 [0.0, 0.0, dt]]
        Q = G @ np.diag([sigma_speed ** 2, sigma_speed ** 2, sigma_gyro ** 2]) @ G.T
        Q[3, 3] += sigma_bias_walk ** 2 * dt
        Q[4, 4] += self.gnss_sigma ** 2 * (1 - fade ** 2)
        Q[5, 5] += self.gnss_sigma ** 2 * (1 - fade ** 2)
        self.P = F @ self.P @ F.T + Q

    def update_position(self, measured, lever, covariance):
        """
        Correct with one antenna's measured world position (x, y).

        lever is the antenna's horizontal offset from the robot's base in the body frame
        (forward, left), already corrected for the robot's tilt; covariance (2 x 2) is the
        antenna's own noise only: the shared drift is part of the state.
        """
        px, py, yaw, _, east, north = self.x
        c, s = math.cos(yaw), math.sin(yaw)
        predicted = np.array([px + c * lever[0] - s * lever[1] + east,
                              py + s * lever[0] + c * lever[1] + north])
        H = np.array([[1.0, 0.0, -s * lever[0] - c * lever[1], 0.0, 1.0, 0.0],
                      [0.0, 1.0, c * lever[0] - s * lever[1], 0.0, 0.0, 1.0]])
        self._correct(np.asarray(measured) - predicted, H, np.asarray(covariance))

    def update_yaw(self, measured, variance):
        """Correct with a measured heading [rad] and its variance."""
        H = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        self._correct(np.array([wrap(measured - self.x[2])]), H, np.array([[variance]]))

    def _correct(self, innovation, H, R):
        """Apply the Kalman correction for an innovation (measured minus predicted)."""
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.x[2] = wrap(self.x[2])
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T      # Joseph form: stays symmetric


def tilted_lever(lever, pitch, roll):
    """
    Return the horizontal (forward, left) offset of a point mounted at lever (x, y, z).

    A tilted robot moves high points sideways: nose-down pitch moves them forward, a
    positive roll (left side up) moves them to the right.
    """
    x, y, z = lever
    return (x * math.cos(pitch) + z * math.sin(pitch),
            y * math.cos(roll) - z * math.sin(roll))


def wheel_speed_sigma(base_sigma, speed, turn_rate, per_speed, per_turn):
    """
    Return how uncertain one wheel-odometry velocity sample is [m/s].

    Wheels miss slides: tyres creep at speed, and when the robot turns (on the spot, or a
    body twist while the wheels re-steer at standstill) the base slides sideways. The turn
    rate comes from the gyro, which sees turns the wheels miss. The factors are per sample
    (100 Hz white noise), so they look large: 1.0 m/s per rad/s lets the position drift by
    about 3 cm per second while turning at 0.3 rad/s.
    """
    return math.sqrt(base_sigma ** 2 + (per_speed * speed) ** 2 + (per_turn * turn_rate) ** 2)
