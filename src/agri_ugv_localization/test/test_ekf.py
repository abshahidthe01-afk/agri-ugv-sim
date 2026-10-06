"""Tests for the pose EKF: single steps, and whole simulated drives with realistic errors."""

import math

from agri_ugv_localization.ekf import PoseEkf, tilted_lever, wheel_speed_sigma, wrap
from agri_ugv_localization.gnss_errors import GaussMarkov
import numpy as np
import pytest

LEVERS = {'front': (0.6, 0.0, 1.9), 'rear': (-0.6, 0.0, 1.9)}   # antennas on the robot


def test_predict_moves_along_the_heading_and_grows_uncertainty():
    ekf = PoseEkf(0.0, 0.0, math.pi / 2, 0.01, 0.01)
    before = ekf.P[0, 0]
    for _ in range(100):
        ekf.predict(0.01, 1.0, 0.0, 0.0, 0.02, 0.0005)
    assert ekf.x[:3] == pytest.approx([0.0, 1.0, math.pi / 2], abs=1e-9)
    assert ekf.P[0, 0] > before


def test_the_gyro_bias_is_subtracted():
    ekf = PoseEkf(0.0, 0.0, 0.0, 0.01, 0.01)
    ekf.x[3] = 0.1
    ekf.predict(1.0, 0.0, 0.0, 0.6, 0.02, 0.0005)
    assert ekf.x[2] == pytest.approx(0.5)


def test_a_position_update_pulls_the_state_and_shrinks_uncertainty():
    ekf = PoseEkf(0.0, 0.0, 0.0, 1.0, 0.01, gnss_sigma=0.01)
    ekf.update_position((1.0, 0.0), (0.0, 0.0), np.diag([1e-6, 1e-6]))
    assert ekf.x[0] + ekf.x[4] == pytest.approx(1.0, abs=1e-3)     # robot + shared GNSS error
    assert ekf.x[0] == pytest.approx(1.0, abs=0.01)
    assert ekf.P[0, 0] < 0.01 ** 2 * 1.1


def test_the_antenna_lever_turns_with_the_robot():
    ekf = PoseEkf(0.0, 0.0, math.pi / 2, 0.1, 0.01)
    ekf.update_position((0.0, 0.6), (0.6, 0.0), np.diag([1e-4, 1e-4]))   # facing north
    assert ekf.x[:3] == pytest.approx([0.0, 0.0, math.pi / 2], abs=1e-9)


def test_a_heading_update_wraps_around_pi():
    ekf = PoseEkf(0.0, 0.0, math.radians(179), 0.1, 0.1)
    ekf.update_yaw(math.radians(-179), 1e-8)
    assert abs(wrap(ekf.x[2] - math.radians(-179))) < 1e-3


def test_the_tilted_lever_matches_the_antenna_shift_measured_in_gazebo():
    forward, left = tilted_lever(LEVERS['front'], math.radians(0.71), math.radians(-0.21))
    assert (forward, left) == pytest.approx((0.6235, 0.0070), abs=1e-4)


def simulate(seed, gyro_bias=0.0003):
    """Drive straight, turn, twist at standstill, straight, sideways; return errors and filter."""
    rng = np.random.default_rng(seed)
    dt, pitch, roll = 0.01, math.radians(0.71), math.radians(-0.21)
    plan = [(0.5, 0.0, 0.0, 10.0, True), (0.0, 0.0, 0.3, math.pi / 2 / 0.3, True),
            (0.0, 0.0, math.radians(2.0), 1.0, False),     # body twist the wheels do not see
            (0.5, 0.0, 0.0, 10.0, True), (0.0, 0.3, 0.0, 6.0, True)]
    truth, shared = np.zeros(3), GaussMarkov([0.01, 0.01], 60.0, rng)
    ekf = PoseEkf(0.01, -0.01, math.radians(0.3), 0.02, math.radians(0.5))
    errors, step = [], 0
    for vx, vy, wz, seconds, wheels_see in plan:
        for _ in range(int(round(seconds / dt))):
            c, s = math.cos(truth[2]), math.sin(truth[2])
            truth = truth + [(c * vx - s * vy) * dt, (s * vx + c * vy) * dt, wz * dt]
            seen = (vx, vy) if wheels_see else (0.0, 0.0)
            ekf.predict(dt, seen[0] + rng.normal(0, 0.02), seen[1] + rng.normal(0, 0.02),
                        wz + gyro_bias + rng.normal(0, 0.0005), 0.02, 0.0005)
            step += 1
            if step % 10 == 0:                     # GNSS at 10 Hz
                shared.step(0.1)
                fixes = {}
                for name, lever in LEVERS.items():
                    fx, fy = tilted_lever(lever, pitch, roll)
                    c, s = math.cos(truth[2]), math.sin(truth[2])
                    fixes[name] = (truth[:2] + [c * fx - s * fy, s * fx + c * fy]
                                   + shared.value + rng.normal(0, 0.003, 2))
                    ekf.update_position(fixes[name], (fx, fy), np.diag([9e-6, 9e-6]))
                d = fixes['front'] - fixes['rear']
                ekf.update_yaw(math.atan2(d[1], d[0]), math.radians(0.2026) ** 2)
            errors.append((np.hypot(*(ekf.x[:2] - truth[:2])), wrap(ekf.x[2] - truth[2])))
    return np.array(errors), ekf


def test_a_simulated_drive_is_accurate_and_learns_the_gyro_bias():
    errors, ekf = simulate(seed=1)
    assert errors[200:, 0].max() < 0.03                     # within 3 cm after 2 s
    assert abs(math.degrees(errors[-1, 1])) < 0.1
    assert ekf.x[3] == pytest.approx(0.0003, abs=0.0001)


def test_the_reported_uncertainty_is_honest():
    ratios = []
    for seed in range(10, 20):
        errors, ekf = simulate(seed)
        ratios.append(errors[-1, 0] ** 2 / ekf.P[0, 0])
    assert 0.5 < np.mean(ratios) < 5.0           # an honest 2D filter averages about 2


def test_wheels_count_for_less_at_speed_and_while_turning():
    assert wheel_speed_sigma(0.02, 0.0, 0.0, 0.5, 1.0) == pytest.approx(0.02)
    assert wheel_speed_sigma(0.02, 0.5, 0.0, 0.5, 1.0) == pytest.approx(math.hypot(0.02, 0.25))
    assert wheel_speed_sigma(0.02, 0.0, 0.3, 0.5, 1.0) == pytest.approx(math.hypot(0.02, 0.3))
