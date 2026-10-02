"""Height grids for test terrains."""

from agri_ugv_terrain.heightfield import grid_axes
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


def waves(length, width, spacing, amplitude, wavelength):
    """
    Return waves along x: height = amplitude * cos(2 pi x / wavelength), the same for every y.

    x is measured from the centre of the grid, so there is a crest at x = 0 and the
    heights go from -amplitude (troughs) to +amplitude (crests).
    """
    if not (np.isfinite(amplitude) and amplitude >= 0.0):
        raise ValueError(f'amplitude must be zero or positive, got {amplitude}')
    if not (np.isfinite(wavelength) and wavelength >= 4.0 * spacing):
        raise ValueError(
            f'wavelength ({wavelength} m) needs at least 4 grid points, so >= {4 * spacing} m')
    rows, cols = grid_shape(length, width, spacing)
    x, _ = grid_axes(rows, cols, spacing)
    return np.tile(amplitude * np.cos(2.0 * np.pi * x / wavelength), (rows, 1))


def ramp(length, width, spacing, grade):
    """
    Return a straight slope along x: height = grade * x, so the origin is at height 0.

    grade is rise over run (0.10 = 10 %, about 5.7 deg); negative values slope down along +x.
    """
    if not (np.isfinite(grade) and abs(grade) <= 1.0):
        raise ValueError(f'grade must be between -1 and 1 (45 deg), got {grade}')
    rows, cols = grid_shape(length, width, spacing)
    x, _ = grid_axes(rows, cols, spacing)
    return np.tile(grade * x, (rows, 1))
