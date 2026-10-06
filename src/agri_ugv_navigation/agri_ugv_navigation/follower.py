"""Follow a straight segment with a four-wheel-steered robot: drive, crab, hold heading."""

import math

import numpy as np


def wrap(angle):
    """Return the angle in [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def follow(pose, start, end, heading, speed, k_cross=1.0, k_heading=1.0,
           max_correction=0.2, max_turn=0.3, braking=0.5, tolerance=0.03):
    """
    Return (vx, vy, wz, done): body velocities that drive the robot along start -> end.

    The robot moves along the segment at 'speed' (slowing down to stop at the end with
    'braking' m/s^2), moves sideways towards the line in proportion to its distance from
    it (k_cross per second, capped), and turns towards 'heading' (k_heading per second).
    It never needs to face the direction it moves in: four-wheel steering can crab.
    done is True within 'tolerance' of the end or once past it.
    """
    x, y, yaw = pose
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    direction = end - start
    length = float(np.hypot(*direction))
    tangent = direction / length
    normal = np.array([-tangent[1], tangent[0]])
    offset = np.array([x, y]) - start
    progress, cross = float(offset @ tangent), float(offset @ normal)
    remaining = length - progress
    if remaining <= tolerance:
        return 0.0, 0.0, 0.0, True
    along = min(speed, math.sqrt(2 * braking * remaining))
    sideways = -max(-max_correction, min(max_correction, k_cross * cross))
    world = along * tangent + sideways * normal
    c, s = math.cos(yaw), math.sin(yaw)
    turn = max(-max_turn, min(max_turn, k_heading * wrap(heading - yaw)))
    return c * world[0] + s * world[1], -s * world[0] + c * world[1], turn, False
