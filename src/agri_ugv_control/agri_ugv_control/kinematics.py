"""
Four-wheel-steering kinematics: pure math, no ROS.

Conventions (ROS REP 103): x forward, y left, z up.
Angles in radians, counter-clockwise positive.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple


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
                       wheel_radius: float, margin: float = 0.0) -> List[WheelCommand]:
    """
    Convert a robot velocity into a steering angle and spin speed per wheel.

    vx, vy: robot velocity forward and to the left [m/s]
    wz:     robot turn rate, counter-clockwise [rad/s]
    margin: [rad] a wheel that should point up to this far past its +-90 deg
            steering limit stays at the limit (see below); 0 = exact motion
    """
    if margin < 0:
        raise ValueError(f'margin must not be negative, got {margin}')
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
        # Just past the limit (within margin), keep the wheel at the limit and
        # roll it at the part of the speed along it; the small rest is left out.
        # Crabbing at 90 deg, a tiny correction would otherwise swing the wheel
        # round by 180 deg, every time its sign changes.
        past = abs(angle) - math.pi / 2
        if 0 < past <= margin:
            angle = math.copysign(math.pi / 2, angle)
            speed *= math.cos(past)
        elif past > 0:
            angle -= math.copysign(math.pi, angle)
            speed = -speed

        commands.append(WheelCommand(steer_angle=angle, wheel_speed=speed / wheel_radius))
    return commands


def forward_kinematics(steer_angles: List[float], wheel_speeds: List[float],
                       modules: List[WheelModule],
                       wheel_radius: float) -> Tuple[float, float, float, float]:
    """
    Recover the robot velocity from measured steering angles and wheel speeds (odometry).

    Each wheel's ground velocity must equal (vx - wz * y, vy + wz * x): with four wheels
    that is 8 equations for 3 unknowns, solved by least squares. Returns
    (vx, vy, wz, residual), the residual being the RMS mismatch [m/s] between the wheels:
    zero when they agree, larger when wheels slip or scrub.
    """
    if not len(steer_angles) == len(wheel_speeds) == len(modules) >= 2:
        raise ValueError('need the same number (at least 2) of angles, speeds and modules')
    measured = [(wheel_radius * w * math.cos(a), wheel_radius * w * math.sin(a))
                for a, w in zip(steer_angles, wheel_speeds)]
    n = len(modules)
    sx, sy = sum(m.x for m in modules), sum(m.y for m in modules)
    srr = sum(m.x * m.x + m.y * m.y for m in modules)
    # Normal equations (A^T A) p = A^T b for the rows [1, 0, -y] and [0, 1, x]
    a = [[n, 0.0, -sy], [0.0, n, sx], [-sy, sx, srr]]
    b = [sum(vx for vx, _ in measured), sum(vy for _, vy in measured),
         sum(m.x * vy - m.y * vx for m, (vx, vy) in zip(modules, measured))]
    vx, vy, wz = _solve3(a, b)
    squares = sum((vx - wz * m.y - mx) ** 2 + (vy + wz * m.x - my) ** 2
                  for m, (mx, my) in zip(modules, measured))
    return vx, vy, wz, math.sqrt(squares / (2 * n))


def _solve3(a: List[List[float]], b: List[float]) -> Tuple[float, float, float]:
    """Solve a 3 x 3 linear system with Cramer's rule."""
    def det(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    d = det(a)
    if abs(d) < 1e-12:
        raise ValueError('wheel layout does not determine the robot motion')
    return tuple(det([[b[r] if c == k else a[r][c] for c in range(3)] for r in range(3)]) / d
                 for k in range(3))


def integrate_pose(x: float, y: float, yaw: float, vx: float, vy: float, wz: float,
                   dt: float) -> Tuple[float, float, float]:
    """
    Move a 2D pose by a body velocity held for dt seconds (dead reckoning).

    Uses the heading at the middle of the step, which is exact for straight lines and
    within a fraction of a millimetre per step on curves at 100 Hz.
    """
    middle = yaw + wz * dt / 2
    c, s = math.cos(middle), math.sin(middle)
    return x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt, yaw + wz * dt
