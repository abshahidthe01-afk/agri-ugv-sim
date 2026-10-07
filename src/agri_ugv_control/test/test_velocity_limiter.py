"""Tests for the speed and acceleration limiter."""

from agri_ugv_control.velocity_limiter import limit_velocity, VelocityLimits
import pytest

LIMITS = VelocityLimits(max_vx=1.5, max_vy=1.0, max_wz=1.0,
                        max_ax=1.0, max_ay=1.0, max_az=1.0)
DT = 0.02   # [s] one step at 50 Hz


def run(target, start, seconds):
    """Apply the limiter repeatedly for the given time and return the final velocity."""
    velocity = start
    for _ in range(round(seconds / DT)):
        velocity = limit_velocity(target, velocity, LIMITS, DT)
    return velocity


def test_speed_is_capped():
    """A command far above the limit ends at the maximum speed, never above it."""
    vx, vy, wz = run(target=(5.4, 3.0, -4.0), start=(0.0, 0.0, 0.0), seconds=10.0)
    assert vx == pytest.approx(1.5)
    assert vy == pytest.approx(1.0)
    assert wz == pytest.approx(-1.0)


def test_acceleration_is_gradual():
    """From standstill at 1 m/s^2, the robot reaches 0.5 m/s after 0.5 s, not at once."""
    vx, _, _ = run(target=(1.5, 0.0, 0.0), start=(0.0, 0.0, 0.0), seconds=0.5)
    assert vx == pytest.approx(0.5)


def test_braking_is_gradual():
    """A sudden stop command from 1.5 m/s takes 1.5 s at 1 m/s^2."""
    vx, _, _ = run(target=(0.0, 0.0, 0.0), start=(1.5, 0.0, 0.0), seconds=1.0)
    assert vx == pytest.approx(0.5)
    vx, _, _ = run(target=(0.0, 0.0, 0.0), start=(1.5, 0.0, 0.0), seconds=1.5)
    assert vx == pytest.approx(0.0, abs=1e-9)


def test_no_overshoot():
    """The velocity settles exactly on the target and does not go past it."""
    velocity = (0.0, 0.0, 0.0)
    for _ in range(500):
        velocity = limit_velocity((0.7, -0.3, 0.2), velocity, LIMITS, DT)
        assert velocity[0] <= 0.7 + 1e-12
        assert velocity[1] >= -0.3 - 1e-12
        assert velocity[2] <= 0.2 + 1e-12
    assert velocity == pytest.approx((0.7, -0.3, 0.2))


def test_components_ramp_together_and_keep_the_direction():
    """A forward command with a small sideways part keeps its direction while speeding up."""
    target = (0.5, 0.02, 0.01)
    velocity = limit_velocity(target, (0.0, 0.0, 0.0), LIMITS, DT)
    assert velocity == pytest.approx((0.02, 0.0008, 0.0004))      # 4 % of the way
    assert run(target, (0.0, 0.0, 0.0), 0.5) == pytest.approx(target)


def test_components_on_their_own_reach_a_small_part_at_once():
    """Ramping each component separately reaches the small parts first: 45 deg sideways."""
    velocity = limit_velocity((0.5, 0.02, 0.01), (0.0, 0.0, 0.0), LIMITS, DT, together=False)
    assert velocity == pytest.approx((0.02, 0.02, 0.01))
