"""Steer first, then drive: never roll the wheels while they point the wrong way."""

import math
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
                limits: VelocityLimits, dt: float, tolerance: float, together: bool = True,
                margin: float = 0.0) -> Tuple[Tuple[float, float, float], List[WheelCommand]]:
    """
    Return the next (velocity, wheel commands) for a commanded target velocity.

    If the wheels point within tolerance [rad] of what the target motion needs, the robot
    drives as usual (speed and acceleration limited). If not, it first brakes (limited)
    and, once standing, turns the wheels to the new angles with the wheels not rolling.
    Rolling wheels that point the wrong way would skid and corrupt the wheel odometry.
    together and margin are passed on to limit_velocity and inverse_kinematics.
    """
    aim = inverse_kinematics(*target, modules, wheel_radius, margin)
    if steering_misalignment([c.steer_angle for c in aim], measured) <= tolerance:
        velocity = limit_velocity(target, velocity, limits, dt, together)
        return velocity, inverse_kinematics(*velocity, modules, wheel_radius, margin)
    if max(abs(v) for v in velocity) > STOPPED:
        velocity = limit_velocity((0.0, 0.0, 0.0), velocity, limits, dt, together)   # brake
        return velocity, inverse_kinematics(*velocity, modules, wheel_radius, margin)
    return (0.0, 0.0, 0.0), [WheelCommand(steer_angle=c.steer_angle, wheel_speed=0.0)
                             for c in aim]


class StallWatch:
    """
    Notice steering that has stopped making progress while the robot stands.

    A tyre turning on the spot can get stuck, e.g. in a groove of the ground: its steering
    joint stops short of the target and steer_first would wait for ever. update() reports
    when to creep instead: after stall_time [s] standing without the misalignment
    shrinking by 'progress' [rad], for creep_time [s] or until the wheels are aligned.
    """

    def __init__(self, stall_time: float = 1.5, creep_time: float = 2.0,
                 progress: float = 0.02):
        """Set the times [s] and the least progress [rad] that counts as steering."""
        if stall_time <= 0 or creep_time <= 0 or progress <= 0:
            raise ValueError('stall_time, creep_time and progress must be positive')
        self.stall_time, self.creep_time, self.progress = stall_time, creep_time, progress
        self.best, self.waited, self.creep_left, self.count = None, 0.0, 0.0, 0

    def update(self, dt: float, misalignment: float, tolerance: float,
               standing: bool) -> bool:
        """Return True while the robot should creep to free a stuck wheel."""
        if misalignment <= tolerance:
            self.best, self.waited, self.creep_left = None, 0.0, 0.0
            return False
        if self.creep_left > 0:
            self.creep_left -= dt
            return True
        if not standing:
            self.best, self.waited = None, 0.0
            return False
        if self.best is None or misalignment < self.best - self.progress:
            self.best, self.waited = misalignment, 0.0
            return False
        self.waited += dt
        if self.waited < self.stall_time:
            return False
        self.best, self.waited, self.creep_left = None, 0.0, self.creep_time - dt
        self.count += 1
        return True


def creep(target: Tuple[float, float, float], velocity: Tuple[float, float, float],
          measured: List[float], modules: List[WheelModule], wheel_radius: float,
          limits: VelocityLimits, dt: float, speed: float,
          margin: float = 0.0) -> Tuple[Tuple[float, float, float], List[WheelCommand]]:
    """
    Return (velocity, wheel commands) that creep in the target's direction at 'speed' [m/s].

    The wheels keep steering towards the target's angles, and each rolls at the part of its
    needed ground speed along the direction it points now (measured): a wheel that is
    still turned rolls less instead of being dragged sideways.
    """
    size = math.hypot(target[0], target[1])
    if size < 1e-9:
        aim_velocity = (0.0, 0.0, 0.0)
    else:
        scale = speed / size
        aim_velocity = (target[0] * scale, target[1] * scale, target[2] * scale)
    velocity = limit_velocity(aim_velocity, velocity, limits, dt)
    aim = inverse_kinematics(*target, modules, wheel_radius, margin)
    commands = []
    for m, a, angle in zip(modules, aim, measured):
        ground_x = velocity[0] - velocity[2] * m.y
        ground_y = velocity[1] + velocity[2] * m.x
        rolling = ground_x * math.cos(angle) + ground_y * math.sin(angle)
        commands.append(WheelCommand(steer_angle=a.steer_angle,
                                     wheel_speed=rolling / wheel_radius))
    return velocity, commands
