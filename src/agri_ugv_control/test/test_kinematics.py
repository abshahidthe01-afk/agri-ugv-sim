"""Automated tests for the four-wheel-steering kinematics."""

import math
import random

from agri_ugv_control.kinematics import (forward_kinematics, integrate_pose,
                                         inverse_kinematics, WheelModule)
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


def measured(vx, vy, wz):
    """Return the steering angles and wheel speeds the inverse kinematics would command."""
    commands = inverse_kinematics(vx, vy, wz, MODULES, WHEEL_RADIUS)
    return ([c.steer_angle or 0.0 for c in commands], [c.wheel_speed for c in commands])


@pytest.mark.parametrize('motion', [(0.5, 0.0, 0.0), (-0.3, 0.0, 0.0), (0.0, 0.4, 0.0),
                                    (0.0, 0.0, 0.3), (0.5, 0.2, -0.25), (-0.4, -0.3, 0.6)])
def test_forward_kinematics_undoes_the_inverse(motion):
    angles, speeds = measured(*motion)
    vx, vy, wz, residual = forward_kinematics(angles, speeds, MODULES, WHEEL_RADIUS)
    assert (vx, vy, wz) == pytest.approx(motion, abs=1e-12)
    assert residual == pytest.approx(0.0, abs=1e-12)


def test_random_motions_round_trip():
    rng = random.Random(7)
    for _ in range(200):
        motion = (rng.uniform(-1.5, 1.5), rng.uniform(-1, 1), rng.uniform(-1, 1))
        assert forward_kinematics(*measured(*motion), MODULES, WHEEL_RADIUS)[:3] == \
            pytest.approx(motion, abs=1e-9)


def test_a_slipping_wheel_shows_in_the_residual_and_is_averaged_out():
    angles, speeds = measured(0.5, 0.0, 0.0)
    speeds[0] *= 1.2                                   # one wheel spins 20 % too fast
    vx, vy, wz, residual = forward_kinematics(angles, speeds, MODULES, WHEEL_RADIUS)
    assert vx == pytest.approx(0.525)                  # 0.5 + 0.1 / 4: shared by 4 wheels
    assert residual > 0.02


def test_forward_kinematics_rejects_mismatched_input():
    with pytest.raises(ValueError):
        forward_kinematics([0.0], [1.0, 1.0], MODULES, WHEEL_RADIUS)


def test_integrating_a_constant_turn_closes_the_circle():
    x = y = yaw = 0.0
    dt = 2 * math.pi / 0.5 / 1257                  # 1257 steps of about 10 ms: one full turn
    for _ in range(1257):
        x, y, yaw = integrate_pose(x, y, yaw, 1.0, 0.0, 0.5, dt)
    assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert yaw == pytest.approx(2 * math.pi, abs=1e-9)


def test_integrating_sideways_motion_follows_the_heading():
    assert integrate_pose(1.0, 2.0, math.pi / 2, 0.0, 0.5, 0.0, 2.0) == pytest.approx(
        (0.0, 2.0, math.pi / 2))


def test_a_wheel_just_past_its_limit_stays_there_with_a_margin():
    """Crabbing left with a slight backward part: the wheels stay at +90 deg."""
    past = math.atan2(0.01, 0.5)
    for cmd in inverse_kinematics(-0.01, 0.5, 0.0, MODULES, WHEEL_RADIUS, margin=0.15):
        assert cmd.steer_angle == pytest.approx(math.pi / 2)
        assert cmd.wheel_speed == pytest.approx(0.5 / WHEEL_RADIUS)    # the part along it
    for cmd in inverse_kinematics(-0.01, 0.5, 0.0, MODULES, WHEEL_RADIUS):   # no margin
        assert cmd.steer_angle == pytest.approx(-math.pi / 2 + past)       # swung round
        assert cmd.wheel_speed < 0


def test_a_wheel_further_past_its_limit_than_the_margin_still_swings_round():
    angle = math.atan2(0.5, -0.2)                   # 22 deg past the limit
    for cmd in inverse_kinematics(-0.2, 0.5, 0.0, MODULES, WHEEL_RADIUS, margin=0.15):
        assert cmd.steer_angle == pytest.approx(angle - math.pi)
        assert cmd.wheel_speed == pytest.approx(-math.hypot(0.2, 0.5) / WHEEL_RADIUS)
    with pytest.raises(ValueError):
        inverse_kinematics(0.5, 0.0, 0.0, MODULES, WHEEL_RADIUS, margin=-0.1)
