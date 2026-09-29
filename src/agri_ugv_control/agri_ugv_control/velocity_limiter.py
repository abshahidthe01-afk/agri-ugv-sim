"""
Speed and acceleration limits for velocity commands: pure math, no ROS.

Protects the robot from commands it cannot follow safely: speeds are capped,
and changes in speed are spread out over time instead of happening at once.
"""

from dataclasses import dataclass
from typing import Tuple

Velocity = Tuple[float, float, float]   # (vx [m/s], vy [m/s], wz [rad/s])


@dataclass(frozen=True)
class VelocityLimits:
    """Maximum speeds and accelerations, per motion component."""

    max_vx: float   # forward / backward [m/s]
    max_vy: float   # sideways (crab) [m/s]
    max_wz: float   # turning [rad/s]
    max_ax: float   # forward / backward acceleration [m/s^2]
    max_ay: float   # sideways acceleration [m/s^2]
    max_az: float   # turning acceleration [rad/s^2]


def _clamp(value: float, limit: float) -> float:
    """Keep a value within -limit ... +limit."""
    return max(-limit, min(limit, value))


def limit_velocity(target: Velocity, current: Velocity,
                   limits: VelocityLimits, dt: float) -> Velocity:
    """
    Return the next velocity: one time step dt from current towards target.

    The target is first capped at the maximum speeds. Then each component
    moves towards it by at most (maximum acceleration x dt).
    """
    max_speeds = (limits.max_vx, limits.max_vy, limits.max_wz)
    max_accels = (limits.max_ax, limits.max_ay, limits.max_az)
    result = []
    for goal, now, max_speed, max_accel in zip(target, current, max_speeds, max_accels):
        goal = _clamp(goal, max_speed)
        step = _clamp(goal - now, max_accel * dt)
        result.append(now + step)
    return (result[0], result[1], result[2])
