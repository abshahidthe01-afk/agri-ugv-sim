"""
Four-wheel-steering kinematics: pure math, no ROS.

Conventions (ROS REP 103): x forward, y left, z up.
Angles in radians, counter-clockwise positive.
"""

from dataclasses import dataclass
import math
from typing import List, Optional


@dataclass(frozen=True)
class WheelModule:
    """Position of one steerable, driven wheel relative to the robot centre."""

    name: str   # e.g. front_left
    x: float    # steering axis, forward of the centre [m]
    y: float    # steering axis, left of the centre [m]


@dataclass(frozen=True)
class WheelCommand:
    """What one wheel module should do."""

    steer_angle: Optional[float]  # [rad] in [-pi/2, pi/2]; None = keep current angle
    wheel_speed: float            # [rad/s]; positive = rolling forward


def inverse_kinematics(vx: float, vy: float, wz: float,
                       modules: List[WheelModule],
                       wheel_radius: float) -> List[WheelCommand]:
    """
    Convert a robot velocity into a steering angle and spin speed per wheel.

    vx, vy: robot velocity forward and to the left [m/s]
    wz:     robot turn rate, counter-clockwise [rad/s]
    """
    commands = []
    for m in modules:
        # Velocity of this wheel = robot motion + extra motion from the rotation
        wheel_vx = vx - wz * m.y
        wheel_vy = vy + wz * m.x
        speed = math.hypot(wheel_vx, wheel_vy)   # [m/s]

        if speed < 1e-6:
            # This wheel is not moving: its direction is meaningless, keep it as it is
            commands.append(WheelCommand(steer_angle=None, wheel_speed=0.0))
            continue

        angle = math.atan2(wheel_vy, wheel_vx)

        # Steering joints turn at most +-90 deg. Beyond that, point the wheel
        # the opposite way and spin it backwards: same motion on the ground.
        if angle > math.pi / 2:
            angle -= math.pi
            speed = -speed
        elif angle < -math.pi / 2:
            angle += math.pi
            speed = -speed

        commands.append(WheelCommand(steer_angle=angle, wheel_speed=speed / wheel_radius))
    return commands
