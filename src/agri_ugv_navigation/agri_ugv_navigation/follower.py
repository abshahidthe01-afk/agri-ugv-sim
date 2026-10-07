"""Follow a straight segment with a four-wheel-steered robot: drive, crab, hold heading."""

import math

import numpy as np


def wrap(angle):
    """Return the angle in [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def follow(pose, start, end, heading, speed, k_cross=1.0, k_heading=1.0,
           max_correction=0.2, max_turn=0.3, braking=0.5, tolerance=0.03,
           max_angle=0.1, lever=0.75):
    """
    Return (vx, vy, wz, done): body velocities that drive the robot along start -> end.

    The robot moves along the segment at 'speed' (slowing down to stop at the end with
    'braking' m/s^2), moves sideways towards the line in proportion to its distance from
    it (k_cross per second, capped), and turns towards 'heading' (k_heading per second).
    It never needs to face the direction it moves in: four-wheel steering can crab.
    done is True within 'tolerance' of the end or once past it.

    Corrections are gentle: the sideways part and the wheels' part of the turn (turn rate x
    'lever', the wheels' distance from the centre) each stay within 'max_angle' [rad] of
    the motion along the segment. Crabbing, the wheels point at their 90 deg steering
    limit, and a bigger correction would make some of them swing round (stop, re-steer).
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
    gentle = along * math.tan(max_angle)
    limit = min(max_correction, gentle)
    sideways = -max(-limit, min(limit, k_cross * cross))
    world = along * tangent + sideways * normal
    c, s = math.cos(yaw), math.sin(yaw)
    limit = min(max_turn, gentle / lever)
    turn = max(-limit, min(limit, k_heading * wrap(heading - yaw)))
    return c * world[0] + s * world[1], -s * world[0] + c * world[1], turn, False
