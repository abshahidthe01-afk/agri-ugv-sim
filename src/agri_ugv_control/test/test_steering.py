"""Tests for steering first and driving only when the wheels point the right way."""

import math

from agri_ugv_control.kinematics import inverse_kinematics, WheelModule
from agri_ugv_control.steering import steer_first, steering_misalignment
from agri_ugv_control.velocity_limiter import VelocityLimits
import pytest

RADIUS = 0.205
MODULES = [WheelModule('front_left', 0.675, 0.75), WheelModule('front_right', 0.675, -0.75),
           WheelModule('rear_left', -0.675, 0.75), WheelModule('rear_right', -0.675, -0.75)]
LIMITS = VelocityLimits(1.5, 1.0, 1.0, 1.0, 1.0, 1.0)
STRAIGHT = [0.0, 0.0, 0.0, 0.0]


def spin_angles():
    """Return the steering angles for turning on the spot."""
    return [c.steer_angle for c in inverse_kinematics(0.0, 0.0, 0.3, MODULES, RADIUS)]


def test_misalignment_is_the_largest_difference_and_ignores_kept_angles():
    assert steering_misalignment([0.1, None, -0.2, 0.0], [0.0, 1.0, 0.0, 0.05]) == \
        pytest.approx(0.2)
    with pytest.raises(ValueError):
        steering_misalignment([0.0], [0.0, 0.0])


def test_aligned_wheels_drive_with_the_usual_ramp():
    velocity, commands = steer_first((0.5, 0.0, 0.0), (0.0, 0.0, 0.0), STRAIGHT, MODULES,
                                     RADIUS, LIMITS, 0.02, 0.05)
    assert velocity == pytest.approx((0.02, 0.0, 0.0))          # 1 m/s^2 for 20 ms
    assert all(c.wheel_speed == pytest.approx(0.02 / RADIUS) for c in commands)


def test_standing_robot_turns_its_wheels_before_rolling():
    velocity, commands = steer_first((0.0, 0.0, 0.3), (0.0, 0.0, 0.0), STRAIGHT, MODULES,
                                     RADIUS, LIMITS, 0.02, 0.05)
    assert velocity == (0.0, 0.0, 0.0)
    assert [c.steer_angle for c in commands] == pytest.approx(spin_angles())
    assert all(c.wheel_speed == 0.0 for c in commands)
    assert max(abs(a) for a in spin_angles()) > math.radians(40)    # a big change


def test_once_aligned_the_spin_ramps_up():
    velocity, commands = steer_first((0.0, 0.0, 0.3), (0.0, 0.0, 0.0), spin_angles(), MODULES,
                                     RADIUS, LIMITS, 0.02, 0.05)
    assert velocity == pytest.approx((0.0, 0.0, 0.02))
    assert all(abs(c.wheel_speed) > 0 for c in commands)


def test_moving_robot_brakes_gently_before_it_steers():
    velocity, commands = steer_first((0.0, 0.3, 0.0), (0.5, 0.0, 0.0), STRAIGHT, MODULES,
                                     RADIUS, LIMITS, 0.02, 0.05)
    assert velocity == pytest.approx((0.48, 0.0, 0.0))           # braking at 1 m/s^2
    assert [c.steer_angle for c in commands] == pytest.approx(STRAIGHT)   # still straight
