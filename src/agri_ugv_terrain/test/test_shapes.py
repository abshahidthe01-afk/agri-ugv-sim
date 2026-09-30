"""Tests for the test-terrain height grids."""

from agri_ugv_terrain.shapes import flat, grid_shape
import numpy as np
import pytest


def test_grid_shape_counts_points_not_cells():
    assert grid_shape(20.0, 10.0, 0.5) == (21, 41)  # rows along y, columns along x


@pytest.mark.parametrize('length, width, spacing', [
    (20.0, 10.3, 0.5),   # width is not a whole multiple of the spacing
    (0.2, 10.0, 0.5),    # shorter than one cell
    (20.0, 10.0, 0.0),
    (20.0, 10.0, -0.5),
    (float('nan'), 10.0, 0.5),
])
def test_bad_sizes_are_rejected(length, width, spacing):
    with pytest.raises(ValueError):
        grid_shape(length, width, spacing)


def test_flat_is_all_zero():
    heights = flat(2.0, 1.0, 0.5)
    assert heights.shape == (3, 5)
    assert np.all(heights == 0.0)
