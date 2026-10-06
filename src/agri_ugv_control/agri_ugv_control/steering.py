"""Steer first, then drive: never roll the wheels while they point the wrong way."""

from typing import List, Optional, Tuple

from agri_ugv_control.kinematics import inverse_kinematics, WheelCommand, WheelModule
from agri_ugv_control.velocity_limiter import limit_velocity, VelocityLimits

STOPPED = 1e-3          # [m/s or rad/s] below this the robot counts as standing still


def steering_misalignment(targets: List[Optional[float]], measured: List[float]) -> float:
    """Return the largest difference [rad] between target and measured steering angles."""
    if len(targets) != len(measured):
        raise ValueError(f'{len(targets)} targets but {len(measured)} measured angles')
    return max((abs(t - m) for t, m in zip(targets, measured) if t is not None), default=0.0)


def steer_first(target: Tuple[float, float, float], velocity: Tuple[float, float, float],
                measured: List[float], modules: List[WheelModule], wheel_radius: float,
                limits: VelocityLimits, dt: float,
                tolerance: float) -> Tuple[Tuple[float, float, float], List[WheelCommand]]:
    """
    Return the next (velocity, wheel commands) for a commanded target velocity.

    If the wheels point within tolerance [rad] of what the target motion needs, the robot
    drives as usual (speed and acceleration limited). If not, it first brakes (limited)
    and, once standing, turns the wheels to the new angles with the wheels not rolling.
    Rolling wheels that point the wrong way would skid and corrupt the wheel odometry.
    """
    aim = inverse_kinematics(*target, modules, wheel_radius)
    if steering_misalignment([c.steer_angle for c in aim], measured) <= tolerance:
        velocity = limit_velocity(target, velocity, limits, dt)
        return velocity, inverse_kinematics(*velocity, modules, wheel_radius)
    if max(abs(v) for v in velocity) > STOPPED:
        velocity = limit_velocity((0.0, 0.0, 0.0), velocity, limits, dt)   # brake first
        return velocity, inverse_kinematics(*velocity, modules, wheel_radius)
    return (0.0, 0.0, 0.0), [WheelCommand(steer_angle=c.steer_angle, wheel_speed=0.0)
                             for c in aim]
