"""Tests for turning a DEM raster into a terrain height grid."""

from agri_ugv_terrain.dem import block_percentile, fill_gaps, heights_from_dem, read_dem
import numpy as np
import pytest


def test_block_percentile_shape_drops_leftover_pixels():
    result = block_percentile(np.zeros((7, 11)), np.ones((7, 11), bool), 3, 10)
    assert result.shape == (2, 3)


def test_low_percentile_ignores_spikes_and_plants():
    heights = np.full((10, 10), 1.0)       # soil at 1.0 m
    heights[::2, ::2] = 1.3                # a quarter of the pixels are plants, 30 cm higher
    heights[3, 3] = 9.0                    # one spike
    result = block_percentile(heights, np.ones((10, 10), bool), 5, 10)
    assert result == pytest.approx(np.full((2, 2), 1.0))


def test_cells_with_too_few_valid_pixels_become_nan():
    valid = np.ones((4, 4), bool)
    valid[:2, :2] = False                  # top-left cell fully invalid
    valid[2, 2:] = False                   # bottom-right cell: 2 of 4 pixels valid
    result = block_percentile(np.ones((4, 4)), valid, 2, 10, min_valid=0.75)
    assert np.isnan(result[0, 0]) and np.isnan(result[1, 1])
    assert result[0, 1] == 1.0 and result[1, 0] == 1.0


@pytest.mark.parametrize('block, percentile', [(0, 10), (2.5, 10), (2, -1), (2, 101), (6, 10)])
def test_bad_block_arguments_are_rejected(block, percentile):
    with pytest.raises(ValueError):
        block_percentile(np.zeros((10, 10)), np.ones((10, 10), bool), block, percentile)


def test_fill_gaps_fills_a_corner_from_its_neighbours():
    grid = np.array([[np.nan, np.nan, 2.0],
                     [np.nan, 2.0, 2.0],
                     [2.0, 2.0, 2.0]])
    filled = fill_gaps(grid)
    assert not np.isnan(filled).any()
    assert filled == pytest.approx(np.full((3, 3), 2.0))


def test_fill_gaps_keeps_known_cells_and_stays_within_their_range():
    grid = np.array([[0.0, np.nan, np.nan, 1.0]])
    grid = np.vstack([grid, [[0.0, 0.0, 1.0, 1.0]]])
    filled = fill_gaps(grid)
    assert filled[0, 0] == 0.0 and filled[0, 3] == 1.0
    assert 0.0 <= filled[0, 1] <= 1.0 and 0.0 <= filled[0, 2] <= 1.0


def test_fill_gaps_needs_at_least_one_known_cell():
    with pytest.raises(ValueError):
        fill_gaps(np.full((3, 3), np.nan))


def test_heights_from_dem_puts_north_at_positive_y_and_the_origin_at_zero():
    heights = np.tile(np.linspace(225.0, 221.0, 9)[:, None], (1, 9))  # row 0 (north) highest
    grid, reference = heights_from_dem(heights, np.ones((9, 9), bool), 3, 50)
    assert grid.shape == (3, 3)
    assert grid[-1, 0] > grid[0, 0]        # last row = north = highest
    assert grid[1, 1] == pytest.approx(0.0)
    assert reference == pytest.approx(223.0)


def test_read_dem_round_trip(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    data = np.array([[1.0, 2.0, -32767.0], [4.0, 5.0, 6.0]], dtype='float32')
    path = tmp_path / 'dem.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=2, width=3, count=1, dtype='float32',
                       crs='EPSG:32632', transform=from_origin(1000.0, 2000.0, 0.03, 0.03),
                       nodata=-32767.0) as dst:
        dst.write(data, 1)
    heights, valid, pixel, left, top, crs = read_dem(path)
    assert valid.tolist() == [[True, True, False], [True, True, True]]
    assert heights[1, 2] == 6.0
    assert pixel == pytest.approx(0.03)
    assert (left, top) == pytest.approx((1000.0, 2000.0))
    assert crs == 'EPSG:32632'
