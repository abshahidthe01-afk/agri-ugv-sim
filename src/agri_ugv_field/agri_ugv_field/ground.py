"""Ground height under any point, read from a terrain mesh made by agri_ugv_terrain."""

import numpy as np


def read_obj_grid(obj_text):
    """
    Read the regular height grid stored in a terrain OBJ file.

    The vertices of an agri_ugv_terrain mesh are grid points, row by row from the south,
    x changing fastest. Returns (x axis, y axis, heights[rows, cols]).
    """
    vertices = np.array([line.split()[1:4] for line in obj_text.splitlines()
                         if line.startswith('v ')], dtype=float)
    if len(vertices) == 0:
        raise ValueError('no vertices found in the OBJ text')
    xs, ys = np.unique(vertices[:, 0]), np.unique(vertices[:, 1])
    if len(xs) * len(ys) != len(vertices) or len(xs) < 2 or len(ys) < 2:
        raise ValueError(f'{len(vertices)} vertices do not form a grid of {len(ys)} x {len(xs)}')
    grid = vertices.reshape(len(ys), len(xs), 3)
    if not (np.allclose(grid[:, :, 0], xs) and np.allclose(grid[:, :, 1], ys[:, None])):
        raise ValueError('vertices are not ordered row by row from the south, x fastest')
    return xs, ys, grid[:, :, 2]


def ground_height(grid, x, y):
    """
    Return the ground height at points (x, y), interpolated bilinearly in the grid.

    Within a 0.45 m cell this differs from the mesh's two flat triangles by millimetres.
    Points outside the grid get the height of the nearest edge.
    """
    xs, ys, heights = grid
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    fx = np.clip(np.interp(x, xs, np.arange(len(xs))), 0, len(xs) - 1)
    fy = np.clip(np.interp(y, ys, np.arange(len(ys))), 0, len(ys) - 1)
    j, i = np.minimum(fx.astype(int), len(xs) - 2), np.minimum(fy.astype(int), len(ys) - 2)
    tx, ty = fx - j, fy - i
    return ((1 - ty) * ((1 - tx) * heights[i, j] + tx * heights[i, j + 1])
            + ty * ((1 - tx) * heights[i + 1, j] + tx * heights[i + 1, j + 1]))
