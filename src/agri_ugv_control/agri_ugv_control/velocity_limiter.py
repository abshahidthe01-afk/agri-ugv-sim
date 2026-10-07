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
                   limits: VelocityLimits, dt: float, together: bool = True) -> Velocity:
    """
    Return the next velocity: one time step dt from current towards target.

    The target is first capped at the maximum speeds. Then each component
    moves towards it by at most (maximum acceleration x dt).

    With together (the default), all components move by the same fraction of
    their remaining change, the one with the least room setting the pace: the
    velocity moves on a straight line towards the target. Starting or stopping,
    a command keeps its direction, and so do the wheels: their steering angles
    depend only on the direction of (vx, vy, wz), not on its size. Otherwise
    each component ramps on its own (a small sideways part is reached long
    before the forward part, pointing the wheels sideways while speeding up).
    """
    max_speeds = (limits.max_vx, limits.max_vy, limits.max_wz)
    max_accels = (limits.max_ax, limits.max_ay, limits.max_az)
    change = [_clamp(goal, max_speed) - now
              for goal, now, max_speed in zip(target, current, max_speeds)]
    if together:
        fraction = min([1.0] + [max_accel * dt / abs(c)
                                for c, max_accel in zip(change, max_accels) if c != 0.0])
        steps = [fraction * c for c in change]
    else:
        steps = [_clamp(c, max_accel * dt) for c, max_accel in zip(change, max_accels)]
    result = [now + step for now, step in zip(current, steps)]
    return (result[0], result[1], result[2])
