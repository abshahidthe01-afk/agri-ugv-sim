"""
Crop rows seen by the 3D LiDAR: where the robot really is across the rows.

A plot's rows are parallel lines a known distance apart, as sown. Measured across the
rows, the plant points pile up on them: their positions, taken modulo the row spacing,
cluster around one value, the rows' offset, and they cluster best when measured exactly
across the rows' direction. (Circular statistics: the mean of the phases 2 pi u / spacing.)
The LiDAR sees each plant's near side, a little off the row towards it; that is undone.
Compared with where the field map and the estimated pose put the rows, this tells how far
the robot really is to the side of its estimated position, and how far it is turned.
"""

import math

from agri_ugv_field.plants import CROPS, plot_frame, row_offsets
import numpy as np


def wrap(value, period):
    """Return value shifted by whole periods into [-period / 2, period / 2)."""
    return (value + period / 2) % period - period / 2


def local_heights(xy, z, cell=0.4):
    """Return each point's height above the lowest point in its cell and the 8 around it."""
    if len(z) == 0:
        return np.zeros(0)
    ij = np.floor((xy - xy.min(axis=0)) / cell).astype(int) + 1
    lowest = np.full(tuple(ij.max(axis=0) + 2), np.inf)
    np.minimum.at(lowest, (ij[:, 0], ij[:, 1]), z)
    padded = np.pad(lowest, 1, constant_values=np.inf)
    rows, cols = lowest.shape
    ground = np.min([padded[1 + a:1 + a + rows, 1 + b:1 + b + cols]
                     for a in (-1, 0, 1) for b in (-1, 0, 1)], axis=0)
    return z - ground[ij[:, 0], ij[:, 1]]


def row_pattern(xy, spacing, span=math.radians(4.0), coarse=math.radians(0.25),
                fine=math.radians(0.02), max_shift=0.4):
    """
    Return (angle, offset, strength, shift) of rows 'spacing' apart among points xy (N x 2).

    angle [rad] is the rows' direction from the x axis (within +-span), offset the rows'
    position across them, measured along the normal (-sin, cos) and taken modulo the
    spacing, and strength how sharply the points cluster on lines (0: not at all, 1: all
    exactly on lines). The LiDAR, at the origin, sees the side of each plant that faces
    it, so the points lie off their row towards it, across the rows by about 'shift' times
    the sine of the angle between the ray and the rows. The shift (0 to max_shift) is found
    by regression: how the points' distance from the rows' mean grows with that sine; it
    is then undone. The direction is searched in 'coarse' steps, then in 'fine' ones.
    """
    xy = np.asarray(xy, dtype=float)
    reach = np.hypot(xy[:, 0], xy[:, 1])

    def across(angle, shift):
        u = xy @ np.array([-math.sin(angle), math.cos(angle)])
        return u + shift * u / reach

    def mean(angle, shift):
        return np.exp(2j * math.pi * across(angle, shift) / spacing).mean()

    def fitted_shift(angle):
        u = xy @ np.array([-math.sin(angle), math.cos(angle)])
        sine = u / reach
        if np.var(sine) < 1e-6:
            return 0.0
        shift = 0.0
        for _ in range(4):
            phase = 2 * math.pi * (u + shift * sine) / spacing
            off = np.angle(np.exp(1j * (phase - np.angle(np.exp(1j * phase).mean()))))
            shift -= np.cov(off * spacing / (2 * math.pi), sine)[0, 1] / np.var(sine)
            shift = min(max(shift, 0.0), max_shift)
        return float(shift)

    angles = np.arange(-span, span + coarse / 2, coarse)
    angle = angles[int(np.argmax([abs(mean(a, 0.0)) for a in angles]))]
    shift = fitted_shift(angle)
    angles = np.arange(angle - coarse, angle + coarse + fine / 2, fine)
    strength = np.array([abs(mean(a, shift)) for a in angles])
    k = int(np.argmax(strength))
    angle = angles[k]
    if 0 < k < len(angles) - 1:                    # the peak between the samples
        left, centre, right = strength[k - 1:k + 2]
        bend = left - 2 * centre + right
        if bend < 0:
            angle += fine * 0.5 * (left - right) / bend
    shift = fitted_shift(angle)
    result = mean(angle, shift)
    return float(angle), float(np.angle(result) / (2 * math.pi) * spacing % spacing), \
        float(abs(result)), shift


def plot_under(pose, plots, reach=2.0):
    """Return the plot under the robot, or whose end is within 'reach' along its rows."""
    best = None
    for plot in plots:
        centre, along, across = plot_frame(plot)
        d = np.array(pose[:2]) - centre
        a, b = abs(d @ along), abs(d @ across)
        if a <= plot['length'] / 2 + reach and b <= plot['width'] / 2:
            if best is None or a < best[0]:
                best = (a, plot)
    return None if best is None else best[1]


def plant_points(points, pose, plot, low=0.06, high=1.0, side=0.3, max_points=3000):
    """
    Return the LiDAR points (robot frame) on the plants of a plot, at most max_points.

    Plant points stand 'low' to 'high' above the lowest point around them, within the
    plot's length and up to 'side' beyond its sides, placed with the estimated pose
    (x, y, yaw); evenly picked if there are more than max_points.
    """
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    world = points[:, :2] @ np.array([[c, s], [-s, c]]) + [x, y]
    centre, along, across = plot_frame(plot)
    a, b = (world - centre) @ along, (world - centre) @ across
    inside = (np.abs(a) <= plot['length'] / 2) & (np.abs(b) <= plot['width'] / 2 + side)
    height = local_heights(world[inside], points[inside, 2])
    plants = points[inside][(height > low) & (height < high)]
    return plants[::-(-len(plants) // max_points) or 1]


def row_measurement(points, pose, plot, low=0.06, high=1.0, side=0.3, min_points=200,
                    min_strength=0.25, max_points=3000):
    """
    Measure where the robot is across a plot's crop rows, from LiDAR points; or None.

    points are LiDAR points in the robot frame (x forward, y left, z up from the ground
    under the robot), pose the estimated pose (x, y, yaw) in the world, plot a field layout
    row. Plant points: see plant_points ('side' takes in whole outer rows even if the
    pose is off; the next plots are further away). Returns {'left': how far the robot
    really is to its left of the estimated position, across the rows [m]; 'turn': how far
    it is really turned anticlockwise from the estimated heading [rad]; 'strength', 'shift'
    (see row_pattern) and 'count'}, or None for a dense crop (no rows to see), too few
    plant points or rows too faint. The rows repeat, so 'left' is only known within half a
    row spacing. At most 'max_points' plant points are used, for speed.
    """
    crop = CROPS[plot['crop']]
    if crop['shape'] == 'cereal':
        return None
    plants = plant_points(points, pose, plot, low, high, side, max_points)
    if len(plants) < min_points:
        return None
    x, y, yaw = pose
    centre, along, across = plot_frame(plot)
    spacing = crop['rows']
    angle, offset, strength, shift = row_pattern(plants[:, :2], spacing)
    if strength < min_strength:
        return None
    facing = 1.0 if math.cos(yaw - math.atan2(along[1], along[0])) >= 0 else -1.0
    rows_dir = facing * along                         # the rows' direction the robot faces
    left_dir = facing * across                        # across the rows, to the robot's left
    first = row_offsets(plot['width'], spacing)[0] * facing
    expected = (first - (np.array([x, y]) - centre) @ left_dir) % spacing
    expected_angle = wrap(math.atan2(rows_dir[1], rows_dir[0]) - yaw, 2 * math.pi)
    return {'left': float(-wrap(offset - expected, spacing)),
            'turn': float(-(angle - expected_angle)), 'strength': strength,
            'shift': shift, 'count': int(len(plants))}
