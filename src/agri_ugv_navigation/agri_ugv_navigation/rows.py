"""
Crop rows seen by the 3D LiDAR: where the robot really is across the rows.

A plot's rows are parallel lines a known distance apart, as sown. Measured across the
rows, the plant points pile up on them: their positions, taken modulo the row spacing,
cluster around one value, the rows' offset, and they cluster best when measured exactly
across the rows' direction. (Circular statistics: the mean of the phases 2 pi u / spacing.)
The LiDAR sees each plant's near side, a little off the row towards it; that is undone.
Compared with where the field map and the estimated pose put the rows, this tells how far
the robot really is to the side of its estimated position, and how far it is turned.
The rows repeat, so that alone is known only within half a row spacing; but a plot has a
fixed number of rows with bare soil beside them, and only one whole number of spacings
puts its comb of rows onto the plants (which_row).
"""

import math

from agri_ugv_field.plants import CROPS, plot_frame, row_offsets
import numpy as np


def wrap(value, period):
    """Return value shifted by whole periods into [-period / 2, period / 2)."""
    return (value + period / 2) % period - period / 2


def around(grid, pick, fill):
    """Return pick (np.min or np.max) of each cell of a 2D grid and the 8 cells around it."""
    padded = np.pad(grid, 1, constant_values=fill)
    rows, cols = grid.shape
    return pick([padded[1 + a:1 + a + rows, 1 + b:1 + b + cols]
                 for a in (-1, 0, 1) for b in (-1, 0, 1)], axis=0)


def local_heights(xy, z, cell=0.4):
    """
    Return each point's height above the ground around it.

    The ground of a cell: the lowest point of each cell and the 8 around it, then the
    highest of those lowest values of the cell and the 8 around it (an 'opening'). On a
    slope this gives the slope itself (the lowest point alone lies a cell further down the
    slope: 2 to 6 cm on the field's slopes, which looked like plants), and a dip narrower
    than about a metre (a wheel track) does not lower the ground beside it. Plants up to
    a cell wide between bare patches do not raise it.
    """
    if len(z) == 0:
        return np.zeros(0)
    ij = np.floor((xy - xy.min(axis=0)) / cell).astype(int) + 1
    lowest = np.full(tuple(ij.max(axis=0) + 2), np.inf)
    np.minimum.at(lowest, (ij[:, 0], ij[:, 1]), z)
    eroded = around(lowest, np.min, np.inf)
    opened = around(np.where(np.isfinite(eroded), eroded, -np.inf), np.max, -np.inf)
    ground = np.where(np.isfinite(opened), opened, eroded)
    return z - ground[ij[:, 0], ij[:, 1]]


def peak(values, step):
    """Return where a parabola through three samples 'step' apart peaks, from the middle one."""
    left, centre, right = values
    bend = left - 2 * centre + right
    return 0.5 * step * (left - right) / bend if bend < 0 else 0.0


def row_pattern(xy, spacing, span=math.radians(4.0), coarse=math.radians(0.25),
                fine=math.radians(0.05), max_shift=0.4, sample=500):
    """
    Return (angle, offset, strength, shift) of rows 'spacing' apart among points xy (N x 2).

    angle [rad] is the rows' direction from the x axis (within +-span), offset the rows'
    position across them, measured along the normal (-sin, cos) and taken modulo the
    spacing, and strength how sharply the points cluster on lines (0: not at all, 1: all
    exactly on lines). The LiDAR, at the origin, sees the side of each plant that faces
    it, so the points lie off their row towards it, across the rows by about 'shift' times
    the sine of the angle between the ray and the rows. The shift (-max_shift to max_shift)
    is found by regression: how the points' distance from the rows' mean grows with that
    sine; it is then undone. It can come out negative (ray casts of the simulated sugar
    beet: about +7 cm with leaves seen from one side, about -3 cm with leaves seen from
    both). The direction is searched in 'coarse' steps on 'sample' of the points, then
    refined with parabolas through the strengths 'fine' apart, on all of them.
    """
    xy = np.asarray(xy, dtype=float)
    reach = np.hypot(xy[:, 0], xy[:, 1])
    few = slice(None, None, max(1, len(xy) // sample))

    def across(angle, shift, part=slice(None)):
        u = xy[part] @ np.array([-math.sin(angle), math.cos(angle)])
        return u + shift * u / reach[part]

    def mean(angle, shift, part=slice(None)):
        return np.exp(2j * math.pi * across(angle, shift, part) / spacing).mean()

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
            shift = min(max(shift, -max_shift), max_shift)
        return float(shift)

    angles = np.arange(-span, span + coarse / 2, coarse)
    strength = [abs(mean(a, 0.0, few)) for a in angles]
    k = int(np.argmax(strength))
    angle = angles[k] + (peak(strength[k - 1:k + 2], coarse) if 0 < k < len(angles) - 1 else 0)
    shift = fitted_shift(angle)
    for _ in range(2):                             # the peak between samples 'fine' apart
        angle += peak([abs(mean(angle + d, shift)) for d in (-fine, 0.0, fine)], fine)
    shift = fitted_shift(angle)
    result = mean(angle, shift)
    return float(angle), float(np.angle(result) / (2 * math.pi) * spacing % spacing), \
        float(abs(result)), shift


def which_row(across, left, rows, spacing, max_rows=3):
    """
    Return (k, margin, beyond): the whole row spacings k to add to 'left' so the rows fit.

    across are the plant points' positions across the plot towards the robot's left [m],
    placed with the estimated pose; left is how far the robot really is to the left of its
    estimate, known within half a spacing; rows are the rows' positions on the same axis.
    With the right k, the points moved by left + k * spacing lie on the rows and none lie
    beyond the outer rows on the bare soil. Each k from -max_rows to max_rows scores the
    points within a quarter spacing of a row minus those over half a spacing beyond the
    outer rows; margin is how much the best beats the next best, and beyond the share of
    points beyond the outer rows with the best k (plants where the soil should be bare:
    the fit is not to be trusted), both as shares of the points.
    """
    across, rows = np.asarray(across, dtype=float), np.sort(np.asarray(rows, dtype=float))
    if len(across) == 0:
        return 0, 0.0, 0.0
    shifts = np.arange(-max_rows, max_rows + 1)
    scores, outside = [], []
    for k in shifts:
        moved = across + left + k * spacing
        near = np.min(np.abs(moved[:, None] - rows[None, :]), axis=1) < spacing / 4
        beyond = (moved < rows[0] - spacing / 2) | (moved > rows[-1] + spacing / 2)
        scores.append(int(near.sum()) - int(beyond.sum()))
        outside.append(int(beyond.sum()))
    order = np.argsort(scores)[::-1]
    best = order[0]
    return (int(shifts[best]), (scores[best] - scores[order[1]]) / len(across),
            outside[best] / len(across))


def pose_at(history, t, late=0.05):
    """
    Return the pose (x, y, yaw) at time t from a history of (t, x, y, yaw), or None.

    A time up to 'late' seconds after the newest pose gets the newest pose (a scan can
    arrive before the pose of its own time).
    """
    if len(history) < 2 or not history[0][0] <= t <= history[-1][0] + late:
        return None
    h = np.array(history, dtype=float)
    yaw = np.unwrap(h[:, 3])
    return (float(np.interp(t, h[:, 0], h[:, 1])), float(np.interp(t, h[:, 0], h[:, 2])),
            float(wrap(np.interp(t, h[:, 0], yaw), 2 * math.pi)))


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
                    min_strength=0.25, max_points=3000, edge_side=1.0, min_margin=0.02,
                    max_beyond=0.10):
    """
    Measure where the robot is across a plot's crop rows, from LiDAR points; or None.

    points are LiDAR points in the robot frame (x forward, y left, z up from the ground
    under the robot), pose the estimated pose (x, y, yaw) in the world, plot a field layout
    row. Plant points: see plant_points; the rows are measured on those up to 'side'
    beyond the plot's sides (whole outer rows even if the pose is off), which row is which
    on those up to 'edge_side' beyond them (the bare soil beside the plot; the next plot is
    further away). Returns {'left': how far the robot really is to its left of the
    estimated position, across the rows [m], known within half a row spacing; 'left_plot':
    the same with the right row found (which_row), or None if the best fit does not beat
    the next by min_margin or leaves more than max_beyond of the points on what should be
    bare soil; 'turn': how far it is really turned anticlockwise from the
    estimated heading [rad]; 'rows_moved' (k of which_row), 'margin', 'beyond',
    'strength', 'shift'
    (see row_pattern), 'count', 'spacing' and 'left_dir' (the world direction of 'left')},
    or None for a dense crop (no rows to see), too few plant points or rows too faint. At
    most 'max_points' plant points are used, for speed.
    """
    crop = CROPS[plot['crop']]
    if crop['shape'] == 'cereal':
        return None
    plants = plant_points(points, pose, plot, low, high, max(side, edge_side), max_points)
    x, y, yaw = pose
    centre, along, across = plot_frame(plot)
    facing = 1.0 if math.cos(yaw - math.atan2(along[1], along[0])) >= 0 else -1.0
    rows_dir = facing * along                         # the rows' direction the robot faces
    left_dir = facing * across                        # across the rows, to the robot's left
    c, s = math.cos(yaw), math.sin(yaw)
    world = plants[:, :2] @ np.array([[c, s], [-s, c]]) + [x, y]
    placed = (world - centre) @ left_dir              # across the plot, to the robot's left
    near = np.abs(placed) <= plot['width'] / 2 + side
    if np.sum(near) < min_points:
        return None
    spacing = crop['rows']
    angle, offset, strength, shift = row_pattern(plants[near, :2], spacing)
    if strength < min_strength:
        return None
    rows = row_offsets(plot['width'], spacing) * facing
    expected = (rows[0] - (np.array([x, y]) - centre) @ left_dir) % spacing
    expected_angle = wrap(math.atan2(rows_dir[1], rows_dir[0]) - yaw, 2 * math.pi)
    left = float(-wrap(offset - expected, spacing))
    moved, margin, beyond = which_row(placed, left, rows, spacing)
    sure = margin >= min_margin and beyond <= max_beyond
    return {'left': left, 'turn': float(-(angle - expected_angle)), 'strength': strength,
            'left_plot': left + moved * spacing if sure else None, 'rows_moved': moved,
            'margin': float(margin), 'beyond': float(beyond), 'shift': shift,
            'count': int(np.sum(near)), 'spacing': spacing,
            'left_dir': (float(left_dir[0]), float(left_dir[1]))}
