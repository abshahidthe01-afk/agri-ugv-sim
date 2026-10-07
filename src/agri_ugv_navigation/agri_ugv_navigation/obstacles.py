"""
Obstacles in the robot's way, seen by the 3D LiDAR: anything taller than the crops.

The robot drives over the crop (the plants pass under its body), so plants are no obstacle;
something that stands higher than the tallest crop in the strip the robot is about to
sweep is (a person, a post, another vehicle). Low objects are not detected.
"""

import math
from xml.etree import ElementTree

from agri_ugv_navigation.planner import ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH
import numpy as np


def rpy_matrix(roll, pitch, yaw):
    """Return the rotation matrix of roll, pitch and yaw (URDF convention)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def mount(urdf_text, link='lidar_link', base='base_footprint'):
    """Return (rotation, translation) of a link in the base frame, through the joints' origins."""
    joints = {}
    for joint in ElementTree.fromstring(urdf_text).findall('joint'):
        origin = joint.find('origin')
        xyz, rpy = ([0.0] * 3 if origin is None else
                    [float(v) for v in origin.get(key, '0 0 0').split()] for key in ('xyz', 'rpy'))
        child, parent = joint.find('child').get('link'), joint.find('parent').get('link')
        joints[child] = (parent, rpy_matrix(*rpy), np.array(xyz))
    rotation, translation = np.eye(3), np.zeros(3)
    while link != base:
        if link not in joints:
            raise ValueError(f'no chain of joints from {base} to {link}')
        parent, r, t = joints[link]
        rotation, translation = r @ rotation, r @ translation + t
        link = parent
    return rotation, translation


def cloud_points(fields, point_step, count, data):
    """Return the finite x, y, z points (N x 3) of PointCloud2 data, little-endian float32."""
    offsets = {f.name: f.offset for f in fields}
    dtype = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4'] * 3,
                      'offsets': [offsets['x'], offsets['y'], offsets['z']],
                      'itemsize': point_step})
    raw = np.frombuffer(bytes(data), dtype=dtype, count=count)
    xyz = np.column_stack([raw['x'], raw['y'], raw['z']]).astype(float)
    return xyz[np.all(np.isfinite(xyz), axis=1)]


def obstacle_gap(points, direction, reach=3.0, height=1.0, hidden=2.8, near_height=1.2,
                 margin=0.2, cell=0.4, min_points=3, own=0.05):
    """
    Return the free distance [m] from the robot's edge to an obstacle in its way, or None.

    points are LiDAR points in the robot frame (x forward, y left, z up from the ground
    under the robot); direction is the way the robot moves (x, y). The robot's footprint,
    widened by 'margin' on both sides, sweeps a strip in that direction; only obstacles
    within 'reach' of its edge count. A point is part of an obstacle if it stands 'height'
    above the lowest point in its neighbourhood (cells of 'cell' metres and the eight around
    them: the ground, or the crop's top in a dense crop). Closer than 'hidden' to the robot's
    centre the LiDAR cannot see the ground: there a point must be 'near_height' above the
    ground under the robot. At least 'min_points' points make an obstacle. Points within
    'own' of the robot's outline belong to the robot itself (the rolled-up curtains reach
    1 cm beyond the tyres) or are under it, and are left out.
    """
    d = np.asarray(direction, dtype=float)
    if np.hypot(*d) < 1e-9:
        return None
    d = d / np.hypot(*d)
    n = np.array([-d[1], d[0]])
    front = abs(d[0]) * ROBOT_HALF_LENGTH + abs(d[1]) * ROBOT_HALF_WIDTH
    side = abs(n[0]) * ROBOT_HALF_LENGTH + abs(n[1]) * ROBOT_HALF_WIDTH + margin
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[(np.abs(points[:, 0]) > ROBOT_HALF_LENGTH + own) |
                    (np.abs(points[:, 1]) > ROBOT_HALF_WIDTH + own)]
    s, lateral, z = points[:, :2] @ d, points[:, :2] @ n, points[:, 2]
    around = (s > front - cell) & (s <= front + reach + cell) & (np.abs(lateral) <= side + cell)
    s, lateral, z = s[around], lateral[around], z[around]
    if len(z) == 0:
        return None
    i = np.floor((s - (front - cell)) / cell).astype(int) + 1        # +1: a border of cells
    j = np.floor((lateral + side + cell) / cell).astype(int) + 1
    lowest = np.full((i.max() + 2, j.max() + 2), np.inf)
    np.minimum.at(lowest, (i, j), z)
    padded = np.pad(lowest, 1, constant_values=np.inf)
    ground = np.min([padded[1 + a:padded.shape[0] - 1 + a, 1 + b:padded.shape[1] - 1 + b]
                     for a in (-1, 0, 1) for b in (-1, 0, 1)], axis=0)[i, j]
    near = np.hypot(s, lateral) < hidden                     # (the LiDAR is on the centre)
    tall = np.where(near, z >= near_height, z - ground >= height)
    inside = (s > front) & (s <= front + reach) & (np.abs(lateral) <= side)
    found = np.sort(s[tall & inside])
    if len(found) < min_points:
        return None
    return float(found[min_points - 1] - front)
