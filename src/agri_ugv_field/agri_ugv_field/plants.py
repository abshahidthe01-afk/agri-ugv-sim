"""Crop plants in field plots: rows, plant positions and leaf meshes for textured leaves."""

import math

import numpy as np

# Assumed typical values for late June (crops 6-9 weeks old), not measured: the MuST-C files
# give no row spacings. Spacings and sizes in metres; 'plant' = distance between plants (or
# leaf clumps) in a row; 'width' is the typical leaf spread that the leaf angles in plant()
# produce. 'cereal' crops are drawn as one picture strip per row plus a canopy.
CROPS = {
    'Sugar Beet': {'key': 'sugar_beet', 'rows': 0.50, 'plant': 0.20, 'shape': 'rosette',
                   'height': 0.30, 'width': 0.40},
    'Sugar Corn': {'key': 'sweet_corn', 'rows': 0.75, 'plant': 0.15, 'shape': 'stalk',
                   'height': 0.70, 'width': 0.80},
    'Potato': {'key': 'potato', 'rows': 0.75, 'plant': 0.33, 'shape': 'bush',
               'height': 0.40, 'width': 0.65},
    'Soybean': {'key': 'soybean', 'rows': 0.45, 'plant': 0.12, 'shape': 'soy',
                'height': 0.30, 'width': 0.30},
    'Summerwheat': {'key': 'wheat', 'rows': 0.125, 'plant': None, 'shape': 'cereal',
                    'height': 0.70, 'width': 0.08},
}
for _name in ['Mixture (faba-wheat)', 'Mixture (faba)', 'Mixture (wheat)']:
    CROPS[_name] = {**CROPS['Summerwheat'], 'key': 'mixtures'}

# Texture drawn on each part of a crop's mesh (see textures.py); '' = the whole plant.
PARTS = {'rosette': {'': 'sugar_beet_leaf'}, 'stalk': {'': 'maize_leaf'},
         'bush': {'': 'potato_leaf'}, 'soy': {'': 'soybean_leaf'},
         'cereal': {'_rows': 'wheat_row', '_canopy': 'wheat_canopy'}}


def row_offsets(width, spacing):
    """Return the across-plot offsets of the rows: as many as fit, centred in the width."""
    if spacing <= 0 or width <= 0:
        raise ValueError(f'width and spacing must be positive, got {width}, {spacing}')
    n = int(round(width / spacing))
    return (np.arange(n) - (n - 1) / 2) * spacing


def plant_offsets(length, spacing, rng, jitter=0.02, missing=0.05):
    """
    Return along-row offsets of single plants, centred in the length.

    Each plant is shifted by up to +-jitter and a fraction 'missing' of them is left out,
    drawn from the random generator rng (same seed, same field).
    """
    n = int(length // spacing)
    along = (np.arange(n) - (n - 1) / 2) * spacing + rng.uniform(-jitter, jitter, n)
    return along[rng.random(n) >= missing]


def leaves(azimuth, length, width, rise, droop, base_z=0.0, segments=2, u=(0.0, 1.0)):
    """
    Return (vertices, uvs, faces) of curved leaf strips growing from the point (0, 0, base_z).

    Arguments are arrays with one value per leaf (or scalars): the midrib points at azimuth
    (radians from +x), starts at 'rise' radians above horizontal and bends down by 'droop'
    radians towards the tip. Each strip is 'width' wide, horizontal across the midrib. The
    texture runs from v = 0 at the base to v = 1 at the tip, and from u[0] to u[1] across.
    """
    azimuth = np.atleast_1d(np.asarray(azimuth, dtype=float))
    n = len(azimuth)
    length, width, rise, droop, base_z = (np.broadcast_to(np.asarray(x, dtype=float), (n,))
                                          for x in (length, width, rise, droop, base_z))
    angle = rise[:, None] - droop[:, None] * (np.arange(segments) + 0.5) / segments
    step = (length / segments)[:, None]
    start = np.zeros((n, 1))
    out = np.hstack([start, np.cumsum(step * np.cos(angle), axis=1)])
    up = np.hstack([start, np.cumsum(step * np.sin(angle), axis=1)]) + base_z[:, None]
    c, s = np.cos(azimuth)[:, None], np.sin(azimuth)[:, None]
    middle = np.stack([out * c, out * s, up], axis=-1)              # (n, stations, 3)
    side = np.stack([-s, c, np.zeros_like(c)], axis=-1) * (width / 2)[:, None, None]
    vertices = np.stack([middle - side, middle + side], axis=2).reshape(-1, 3)
    v = np.arange(segments + 1) / segments
    uvs = np.tile(np.stack([np.stack([np.full_like(v, u[0]), v], axis=-1),
                            np.stack([np.full_like(v, u[1]), v], axis=-1)], axis=1).reshape(-1, 2),
                  (n, 1))
    k = 2 * np.arange(segments)
    one = np.concatenate([np.column_stack([k, k + 1, k + 3]), np.column_stack([k, k + 3, k + 2])])
    faces = (one[None, :, :] + 2 * (segments + 1) * np.arange(n)[:, None, None]).reshape(-1, 3)
    return vertices, uvs, faces


def fit(vertices, height):
    """Scale a plant standing at the origin evenly to the given height, in place."""
    vertices *= height / vertices[:, 2].max()
    return vertices


def plant(shape, height, rng):
    """
    Return (vertices, uvs, faces) of one plant standing at the origin.

    'rosette' (sugar beet): 8-10 leaves rising and arching outwards; 'stalk' (maize): a stalk
    and alternating arching leaves; 'bush' (potato): 6 leaves; 'soy': a clump of 3 leaves.
    """
    turn = rng.uniform(0, 2 * math.pi)
    if shape == 'rosette':
        n = int(rng.integers(8, 11))
        parts = [leaves(turn + np.arange(n) * 2 * math.pi / n + rng.uniform(-0.2, 0.2, n),
                        rng.uniform(0.8, 1.0, n), 0.5, rng.uniform(1.25, 1.45, n),
                        rng.uniform(0.5, 0.8, n))]
    elif shape == 'stalk':
        n = int(rng.integers(6, 8))
        parts = [leaves([turn, turn + math.pi / 2], 0.85, 0.035, math.pi / 2, 0.0,
                        segments=1, u=(0.45, 0.55)),                 # stalk: two crossed strips
                 leaves(turn + np.arange(n) * math.pi + rng.uniform(-0.3, 0.3, n),
                        rng.uniform(0.6, 0.9, n), rng.uniform(0.6, 0.9, n) / 8,
                        rng.uniform(1.15, 1.35, n), rng.uniform(0.9, 1.3, n),
                        base_z=0.1 + 0.6 * np.arange(n) / n, segments=3)]
    elif shape == 'bush':
        parts = [leaves(turn + np.arange(6) * math.pi / 3 + rng.uniform(-0.3, 0.3, 6),
                        rng.uniform(0.8, 1.0, 6), 0.75, rng.uniform(1.1, 1.35, 6),
                        rng.uniform(0.3, 0.6, 6))]
    elif shape == 'soy':
        parts = [leaves(turn + np.arange(3) * 2 * math.pi / 3, rng.uniform(0.8, 1.0, 3), 0.6,
                        rng.uniform(1.25, 1.45, 3), rng.uniform(0.1, 0.3, 3), segments=1)]
    else:
        raise ValueError(f'unknown plant shape {shape!r}')
    vertices, uvs, faces = merge(parts)
    return fit(vertices, height), uvs, faces


def row_strip(along, height, ground, u_offset=0.0):
    """
    Return (vertices, uvs, faces) of an upright picture strip standing along a crop row.

    along is a list of (x, y) points on the row; the strip's foot follows the ground and its
    texture repeats every metre (u = distance along the row + u_offset).
    """
    xy = np.asarray(along, dtype=float)
    z = ground(xy[:, 0], xy[:, 1])
    u = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]) + u_offset
    vertices = np.column_stack([np.repeat(xy, 2, axis=0),
                                np.column_stack([z, z + height]).ravel()])
    uvs = np.column_stack([np.repeat(u, 2), np.tile([0.0, 1.0], len(xy))])
    k = 2 * np.arange(len(xy) - 1)
    faces = np.concatenate([np.column_stack([k, k + 2, k + 3]),
                            np.column_stack([k, k + 3, k + 1])])
    return vertices, uvs, faces


def vertex_normals(vertices, faces):
    """Return unit normals per vertex: the area-weighted mean of the faces around it."""
    v = np.asarray(vertices, dtype=float)
    cross = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    normals = np.zeros_like(v)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], cross)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(length > 0, length, 1.0)


def plot_frame(plot):
    """Return (centre, along unit vector, across unit vector) of a plot's fitted rectangle."""
    h = math.radians(plot['heading_deg'])
    u = np.array([math.cos(h), math.sin(h)])
    return np.array([plot['centre_x'], plot['centre_y']]), u, np.array([-u[1], u[0]])


def plot_mesh(plot, crop, ground, rng, margin=0.05):
    """
    Return ({part: (vertices, uvs, faces)}, count) of all plants in one plot, world metres.

    ground(x, y) gives the terrain height. Rows run along the plot's long side. count is the
    number of plants (or leaf clumps), or of row strips for cereals. Parts are named as in
    PARTS: '' for single plants, '_rows' and '_canopy' for cereals.
    """
    centre, u, v = plot_frame(plot)
    rows = row_offsets(plot['width'], crop['rows'])
    if crop['shape'] != 'cereal':
        parts = []
        for b in rows:
            for a in plant_offsets(plot['length'], crop['plant'], rng):
                scale = rng.uniform(0.8, 1.2)
                verts, uvs, faces = plant(crop['shape'], crop['height'] * scale, rng)
                base = centre + a * u + b * v
                verts[:, :2] += base
                verts[:, 2] += ground(base[0], base[1])
                parts.append((verts, uvs, faces))
        return {'': merge(parts)}, len(parts)
    half = plot['length'] / 2 - margin
    stations = np.linspace(-half, half, int(math.ceil(2 * half)) + 1)    # at most 1 m apart
    strips = [row_strip([centre + a * u + b * v for a in stations],
                        crop['height'] * rng.uniform(0.9, 1.05), ground, rng.uniform(0, 1))
              for b in rows]
    across = np.linspace(rows[0], rows[-1], int(math.ceil(rows[-1] - rows[0])) + 1)
    grid = np.array([centre + a * u + b * v for a in stations for b in across])
    top = ground(grid[:, 0], grid[:, 1]) + 0.9 * crop['height']
    uvs = np.array([(a - stations[0], b - across[0]) for a in stations for b in across])
    m = len(across)
    k = np.array([i * m + j for i in range(len(stations) - 1) for j in range(m - 1)])
    faces = np.concatenate([np.column_stack([k, k + m, k + m + 1]),
                            np.column_stack([k, k + m + 1, k + 1])])
    canopy = (np.column_stack([grid, top]), uvs, faces)
    return {'_rows': merge(strips), '_canopy': canopy}, len(strips)


def merge(parts):
    """Join (vertices, uvs, faces) parts into one mesh, renumbering the faces."""
    if not parts:
        return np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 3), dtype=int)
    offsets = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
    return (np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts]),
            np.vstack([p[2] + o for p, o in zip(parts, offsets)]))


def obj_text(vertices, uvs, faces, normals):
    """Write a mesh with texture coordinates and normals as Wavefront OBJ (mm precision)."""
    lines = ['# Generated by agri_ugv_field']
    lines += [f'v {x:.3f} {y:.3f} {z:.3f}' for x, y, z in vertices]
    lines += [f'vt {a:.3f} {b:.3f}' for a, b in uvs]
    lines += [f'vn {x:.2f} {y:.2f} {z:.2f}' for x, y, z in normals]
    lines += [f'f {i}/{i}/{i} {j}/{j}/{j} {k}/{k}/{k}' for i, j, k in faces + 1]
    return '\n'.join(lines) + '\n'
