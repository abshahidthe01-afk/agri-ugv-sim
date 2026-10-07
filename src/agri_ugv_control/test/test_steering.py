"""Tests for steering first and driving only when the wheels point the right way."""

import math

from agri_ugv_control.kinematics import inverse_kinematics, WheelModule
from agri_ugv_control.steering import creep, StallWatch, steer_first, steering_misalignment
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


def test_a_mixed_command_speeds_up_with_the_wheels_at_its_angles():
    """Ramping together keeps the command's wheel angles; ramping apart points them at 45 deg."""
    target = (0.5, 0.03, 0.0)
    aim = [c.steer_angle for c in inverse_kinematics(*target, MODULES, RADIUS)]
    _, commands = steer_first(target, (0.0, 0.0, 0.0), aim, MODULES, RADIUS, LIMITS, 0.02, 0.05)
    assert [c.steer_angle for c in commands] == pytest.approx(aim)
    _, commands = steer_first(target, (0.0, 0.0, 0.0), aim, MODULES, RADIUS, LIMITS, 0.02, 0.05,
                              together=False)
    assert [c.steer_angle for c in commands] == pytest.approx([math.pi / 4] * 4)


def test_a_small_correction_while_crabbing_does_not_stop_the_robot_with_a_margin():
    crab = [math.pi / 2] * 4
    velocity, commands = steer_first((-0.01, 0.5, 0.0), (0.0, 0.5, 0.0), crab, MODULES, RADIUS,
                                     LIMITS, 0.02, 0.05, margin=0.15)
    assert velocity == pytest.approx((-0.01, 0.5, 0.0))
    assert [c.steer_angle for c in commands] == pytest.approx(crab)
    velocity, _ = steer_first((-0.01, 0.5, 0.0), (0.0, 0.5, 0.0), crab, MODULES, RADIUS, LIMITS,
                              0.02, 0.05)
    assert velocity == pytest.approx((0.0, 0.48, 0.0))     # brakes to swing the wheels round


def test_a_steering_joint_that_stops_short_is_noticed_and_a_creep_timed():
    watch = StallWatch(stall_time=1.5, creep_time=2.0, progress=0.02)
    assert not watch.update(0.25, 0.8, 0.05, standing=True)   # notes where it stands
    steps = [watch.update(0.25, 0.8, 0.05, standing=True) for _ in range(5)]
    assert not any(steps)                                  # 1.25 s without progress
    assert watch.update(0.25, 0.8, 0.05, standing=True)    # 1.5 s: creep
    assert watch.count == 1
    creeping = [watch.update(0.25, 0.8, 0.05, standing=False) for _ in range(7)]
    assert all(creeping) and not watch.update(0.25, 0.8, 0.05, standing=False)  # 2 s


def test_steering_that_makes_progress_or_is_aligned_never_creeps():
    watch = StallWatch()
    angle = 1.5
    for _ in range(200):                                   # a slow but steady turn
        angle -= 0.004
        assert not watch.update(0.02, angle, 0.05, standing=True)
    assert not watch.update(0.02, 0.01, 0.05, standing=True)
    with pytest.raises(ValueError):
        StallWatch(stall_time=0.0)


def test_creeping_rolls_each_wheel_along_where_it_points():
    stuck = [0.0, -math.radians(54), 0.0, 0.0]             # front right still turned
    velocity, commands = creep((0.5, 0.0, 0.0), (0.0, 0.0, 0.0), stuck, MODULES, RADIUS,
                               LIMITS, 0.1, 0.05)
    assert velocity == pytest.approx((0.05, 0.0, 0.0))     # 1 m/s^2 for 0.1 s, at most 5 cm/s
    assert [c.steer_angle for c in commands] == pytest.approx([0.0] * 4)   # still steering
    speeds = [c.wheel_speed * RADIUS for c in commands]
    assert speeds[0] == pytest.approx(0.05) and speeds[1] == pytest.approx(0.05 * math.cos(
        math.radians(54)))                                 # the turned wheel rolls less
