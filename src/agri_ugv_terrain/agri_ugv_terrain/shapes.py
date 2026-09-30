"""Height grids for test terrains."""

import numpy as np


def grid_shape(length, width, spacing):
    """
    Return (rows, cols) of a grid covering length (along x) by width (along y) metres.

    Both sides must be whole multiples of the spacing, so the grid ends exactly at the edges.
    """
    if not (np.isfinite(spacing) and spacing > 0.0):
        raise ValueError(f'spacing must be a positive number, got {spacing}')
    counts = []
    for label, side in (('width', width), ('length', length)):
        cells = side / spacing
        if not np.isfinite(cells) or cells < 1 or abs(cells - round(cells)) > 1e-9:
            raise ValueError(
                f'{label} ({side} m) must be a positive whole multiple of spacing ({spacing} m)')
        counts.append(int(round(cells)) + 1)
    return tuple(counts)


def flat(length, width, spacing):
    """Return an all-zero height grid: flat ground at height 0."""
    return np.zeros(grid_shape(length, width, spacing))
