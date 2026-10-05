"""Plot layout of a field trial: rectangles in the world frame of the terrain, as a CSV table."""

import csv
import io
import math
import re

import numpy as np

COLUMNS = ['type', 'plot_id', 'crop', 'genotype', 'variant', 'centre_x', 'centre_y',
           'width', 'length', 'heading_deg', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']


def outline_geometry(ring, close_tol=0.001, shape_tol=0.5):
    """
    Describe a closed outline of 4 corners that is roughly a rectangle (a plot or the trial).

    Returns a dict with centre (x, y), width and length (means of the short and of the long
    opposite sides), heading (mean direction of the long sides, radians in (-pi/2, pi/2]),
    corners (4 x 2, counter-clockwise, starting where a long side begins), and the measured
    imperfection: gap (distance between first and last point) and deviation (largest
    difference between opposite sides or between the diagonals; 0 for a perfect rectangle).
    Hand-drawn outlines are accepted; above shape_tol the shape is rejected as wrong.
    Rows of a plot run along its long side.
    """
    points = np.asarray(ring, dtype=float)
    if points.shape != (5, 2):
        raise ValueError(f'expected 5 points (4 corners, first repeated), got {points.shape}')
    gap = float(np.hypot(*(points[4] - points[0])))
    if gap > close_tol:
        raise ValueError(f'the outline is not closed: gap of {gap:.4f} m')
    corners = points[:4]
    x, y = corners[:, 0], corners[:, 1]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:   # negative area: clockwise
        corners = corners[::-1]
    sides = np.roll(corners, -1, axis=0) - corners
    lengths = np.hypot(sides[:, 0], sides[:, 1])
    diagonals = np.hypot(*(corners[2] - corners[0])), np.hypot(*(corners[3] - corners[1]))
    deviation = float(max(abs(lengths[0] - lengths[2]), abs(lengths[1] - lengths[3]),
                          abs(diagonals[0] - diagonals[1])))
    if deviation > shape_tol:
        raise ValueError(f'not roughly a rectangle: sides or diagonals differ by '
                         f'{deviation:.3f} m')
    long = 0 if lengths[0] + lengths[2] >= lengths[1] + lengths[3] else 1
    direction = sides[long] - sides[long + 2]    # opposite sides point in opposite directions
    heading = math.atan2(direction[1], direction[0])
    if heading <= -math.pi / 2:
        heading += math.pi
    elif heading > math.pi / 2:
        heading -= math.pi
    u = np.array([math.cos(heading), math.sin(heading)])   # along the long side
    v = np.array([-u[1], u[0]])                            # 90 degrees to the left
    centre = corners.mean(axis=0)
    start = int(np.argmin((corners - centre) @ (u + v)))   # the corner at (-length, -width)/2
    return {
        'centre': centre,
        'width': float((lengths[1 - long] + lengths[3 - long]) / 2),
        'length': float((lengths[long] + lengths[long + 2]) / 2),
        'heading': heading,
        'corners': np.roll(corners, -start, axis=0),
        'gap': gap,
        'deviation': deviation,
    }


def attribute(record, name):
    """Return a record's attribute by name, ignoring upper and lower case."""
    for key, value in record.items():
        if key.lower() == name.lower():
            return value
    raise ValueError(f'attribute {name!r} missing; found {sorted(record)}')


def field_layout(shapes, to_world):
    """
    Turn shapefile outlines into (boundary, plots) in the world frame.

    shapes is a list of (points, record): points an (N, 2) array in the file's coordinates,
    record a dict of attributes with plot_ID, crop, genotype and var. to_world converts an
    (N, 2) array to world coordinates. The single outline with plot_ID 0 is the trial
    boundary. Every entry is a dict with the CSV columns plus 'gap' and 'deviation'.
    """
    boundary, plots, seen = None, [], set()
    for points, record in shapes:
        plot_id = int(attribute(record, 'plot_ID'))
        geometry = outline_geometry(to_world(np.asarray(points, dtype=float)))
        entry = {
            'type': 'boundary' if plot_id == 0 else 'plot',
            'plot_id': plot_id,
            'crop': str(attribute(record, 'crop')).strip(),
            'genotype': str(attribute(record, 'genotype')).strip(),
            'variant': str(attribute(record, 'var')).strip(),
            'centre_x': geometry['centre'][0],
            'centre_y': geometry['centre'][1],
            'width': geometry['width'],
            'length': geometry['length'],
            'heading_deg': math.degrees(geometry['heading']),
            'gap': geometry['gap'],
            'deviation': geometry['deviation'],
        }
        for k, (x, y) in enumerate(geometry['corners'], start=1):
            entry[f'x{k}'], entry[f'y{k}'] = x, y
        if plot_id == 0:
            if boundary is not None:
                raise ValueError('more than one outline with plot_ID 0 (the trial boundary)')
            boundary = entry
        else:
            if plot_id in seen:
                raise ValueError(f'plot_ID {plot_id} appears twice')
            seen.add(plot_id)
            plots.append(entry)
    if boundary is None:
        raise ValueError('no outline with plot_ID 0 (the trial boundary)')
    return boundary, sorted(plots, key=lambda p: p['plot_id'])


def layout_csv(entries, comments=()):
    """Write entries as CSV text: '#' comment lines, a header, one row per entry (mm precision)."""
    out = io.StringIO()
    for comment in comments:
        out.write(f'# {comment}\n')
    writer = csv.writer(out, lineterminator='\n')
    writer.writerow(COLUMNS)
    for entry in entries:
        writer.writerow([f'{entry[c]:.3f}' if isinstance(entry[c], (float, np.floating))
                         else entry[c] for c in COLUMNS])
    return out.getvalue()


def read_layout_csv(text):
    """Read CSV text written by layout_csv; returns a list of dicts with numbers as floats."""
    lines = [line for line in text.splitlines() if not line.startswith('#')]
    rows = list(csv.DictReader(lines))
    if not rows or list(rows[0]) != COLUMNS:
        raise ValueError(f'expected the columns {COLUMNS}')
    for row in rows:
        row['plot_id'] = int(row['plot_id'])
        for column in COLUMNS[5:]:
            row[column] = float(row[column])
    return rows


def terrain_origin(model_config_text):
    """
    Read the world origin from a terrain model.config written by agri_ugv_terrain.

    Returns (crs, east, north), e.g. ('EPSG:32632', 357472.44, 5610188.04): the map point
    that is the world origin (0, 0) of the terrain.
    """
    match = re.search(r'origin at (EPSG:\d+) E (-?\d+(?:\.\d+)?) N (-?\d+(?:\.\d+)?)',
                      model_config_text)
    if match is None:
        raise ValueError('no "origin at EPSG:<code> E <east> N <north>" in the model.config')
    return match.group(1), float(match.group(2)), float(match.group(3))
