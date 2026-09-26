"""Automated tests for the four-wheel-steering kinematics."""

import math
import random

from agri_ugv_control.kinematics import inverse_kinematics, WheelModule
import pytest

# Geometry of the robot (docs/robot_spec.md). The tests only need realistic numbers.
WHEEL_RADIUS = 0.205
MODULES = [
    WheelModule('front_left', 0.675, 0.75),
    WheelModule('front_right', 0.675, -0.75),
    WheelModule('rear_left', -0.675, 0.75),
    WheelModule('rear_right', -0.675, -0.75),
]


def test_straight_forward():
    """Driving straight: all wheels point ahead and spin forward at v / r."""
    for cmd in inverse_kinematics(1.0, 0.0, 0.0, MODULES, WHEEL_RADIUS):
        assert cmd.steer_angle == pytest.approx(0.0)
        assert cmd.wheel_speed == pytest.approx(1.0 / WHEEL_RADIUS)


def test_straight_backward():
    """Reversing: wheels stay pointing ahead and spin backwards (no 180 deg flip)."""
    for cmd in inverse_kinematics(-1.0, 0.0, 0.0, MODULES, WHEEL_RADIUS):
        assert cmd.steer_angle == pytest.approx(0.0)
        assert cmd.wheel_speed == pytest.approx(-1.0 / WHEEL_RADIUS)


def test_spin_in_place_wheels_are_tangent():
    """Spinning on the spot: every wheel is at a right angle to its line from the centre."""
    wz = 0.5
    for module, cmd in zip(MODULES, inverse_kinematics(0.0, 0.0, wz, MODULES, WHEEL_RADIUS)):
        direction = (math.cos(cmd.steer_angle), math.sin(cmd.steer_angle))
        # Perpendicular vectors have a dot product of zero
        dot = direction[0] * module.x + direction[1] * module.y
        assert dot == pytest.approx(0.0, abs=1e-9)
        # Ground speed = turn rate x distance from the centre
        distance = math.hypot(module.x, module.y)
        assert abs(cmd.wheel_speed) * WHEEL_RADIUS == pytest.approx(wz * distance)


def test_crab_sideways():
    """Moving purely sideways: all wheels point at 90 deg and spin forward."""
    for cmd in inverse_kinematics(0.0, 0.5, 0.0, MODULES, WHEEL_RADIUS):
        assert cmd.steer_angle == pytest.approx(math.pi / 2)
        assert cmd.wheel_speed == pytest.approx(0.5 / WHEEL_RADIUS)


def test_stopped_keeps_current_angle():
    """Zero command: wheels do not move, and their angle is left unchanged (None)."""
    for cmd in inverse_kinematics(0.0, 0.0, 0.0, MODULES, WHEEL_RADIUS):
        assert cmd.steer_angle is None
        assert cmd.wheel_speed == 0.0


def test_random_motions_reproduce_required_wheel_velocity():
    """For any motion, each wheel's commanded ground velocity is exactly what is needed."""
    rng = random.Random(42)   # fixed seed: the same random cases on every run
    for _ in range(1000):
        vx, vy, wz = rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-2, 2)
        commands = inverse_kinematics(vx, vy, wz, MODULES, WHEEL_RADIUS)
        for module, cmd in zip(MODULES, commands):
            # What the wheel must do (the rigid-body formula) ...
            needed_x = vx - wz * module.y
            needed_y = vy + wz * module.x
            # ... versus what the command actually makes it do
            ground_speed = cmd.wheel_speed * WHEEL_RADIUS
            actual_x = ground_speed * math.cos(cmd.steer_angle)
            actual_y = ground_speed * math.sin(cmd.steer_angle)
            assert actual_x == pytest.approx(needed_x, abs=1e-9)
            assert actual_y == pytest.approx(needed_y, abs=1e-9)
            # And the steering joint never exceeds its +-90 deg limit
            assert -math.pi / 2 <= cmd.steer_angle <= math.pi / 2
