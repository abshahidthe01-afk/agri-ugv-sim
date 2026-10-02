"""Tests for the test-terrain height grids."""

from agri_ugv_terrain.heightfield import mesh_from_heights
from agri_ugv_terrain.shapes import flat, grid_shape, ramp, waves
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


def test_waves_crests_and_troughs_land_on_grid_points():
    heights = waves(27.0, 6.0, 0.15, 0.05, 2.7)
    assert heights.shape == (41, 181)
    assert heights.max() == pytest.approx(0.05)
    assert heights.min() == pytest.approx(-0.05)
    assert heights[0, 90] == pytest.approx(0.05)   # centre column, x = 0: crest
    assert heights[0, 99] == pytest.approx(-0.05)  # 9 columns = 1.35 m further: trough


def test_waves_are_the_same_across_y():
    heights = waves(27.0, 6.0, 0.15, 0.05, 2.7)
    assert np.all(heights == heights[0])


def test_waves_match_the_mesh_coordinates():
    heights = waves(5.4, 1.5, 0.15, 0.05, 2.7)
    vertices, _ = mesh_from_heights(heights, 0.15)
    expected = 0.05 * np.cos(2.0 * np.pi * vertices[:, 0] / 2.7)
    assert vertices[:, 2] == pytest.approx(expected)


@pytest.mark.parametrize('amplitude, wavelength', [
    (-0.05, 2.7),         # negative amplitude
    (float('nan'), 2.7),
    (0.05, 0.0),
    (0.05, 0.5),          # fewer than 4 points per wave at 0.15 m spacing
])
def test_bad_waves_are_rejected(amplitude, wavelength):
    with pytest.raises(ValueError):
        waves(27.0, 6.0, 0.15, amplitude, wavelength)


def test_ramp_rises_with_the_grade_and_is_zero_at_the_origin():
    heights = ramp(30.0, 6.0, 0.5, 0.10)
    assert heights.shape == (13, 61)
    assert heights[0, 30] == pytest.approx(0.0)              # centre column: x = 0
    assert heights[0, -1] - heights[0, 0] == pytest.approx(3.0)  # 30 m at 10 %
    assert np.all(heights == heights[0])


@pytest.mark.parametrize('grade', [1.5, -1.01, float('nan')])
def test_bad_grades_are_rejected(grade):
    with pytest.raises(ValueError):
        ramp(30.0, 6.0, 0.5, grade)
