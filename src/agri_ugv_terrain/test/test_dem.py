"""Tests for turning a DEM raster into a terrain height grid."""

from agri_ugv_terrain.dem import (block_percentile, fill_gaps, fill_photo_gaps, heights_from_dem,
                                  pad_grid, read_dem, read_ortho, texture_from_ortho)
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


def test_texture_from_ortho_crops_the_terrain_area_and_averages():
    rgb = np.zeros((7, 13, 3), dtype=np.uint8)
    rgb[:2, :2] = [100, 200, 50]           # top-left 2 x 2 pixels
    rgb[:, 12] = 255                       # the leftover column is not part of the terrain
    small = texture_from_ortho(rgb, 3, 2, 4, 2)   # terrain: 2 x 4 cells of 3 x 3 pixels
    assert small.shape == (3, 6, 3)
    assert small[0, 0].tolist() == [100, 200, 50]
    assert small.max() < 255


@pytest.mark.parametrize('shape, factor', [((7, 13), 2), ((5, 12, 3), 2), ((6, 12, 3), 4)])
def test_bad_photos_or_factors_are_rejected(shape, factor):
    with pytest.raises(ValueError):
        texture_from_ortho(np.zeros(shape, dtype=np.uint8), 3, 2, 4, factor)


def test_read_ortho_round_trip(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    data = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(3, 2, 3)   # bands, rows, cols
    path = tmp_path / 'ortho.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=2, width=3, count=3, dtype='uint8',
                       crs='EPSG:32632', transform=from_origin(1000.0, 2000.0, 0.03, 0.03)) as dst:
        dst.write(data)
    rgb, pixel, left, top = read_ortho(path)
    assert rgb.shape == (2, 3, 3)
    assert rgb[1, 2].tolist() == [5, 11, 17]   # pixel (1, 2) across the three bands
    assert pixel == pytest.approx(0.03)
    assert (left, top) == pytest.approx((1000.0, 2000.0))


def test_pad_grid_continues_the_edges_and_keeps_the_inside():
    grid = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    padded = pad_grid(grid, 2)
    assert padded.shape == (6, 7)
    assert np.array_equal(padded[2:4, 2:5], grid)
    assert padded[0, 3] == 2.0 and padded[5, 3] == 5.0       # straight out from the edges
    assert padded[3, 0] == 4.0 and padded[2, 6] == 3.0
    assert padded.min() >= 1.0 and padded.max() <= 6.0
    assert np.array_equal(pad_grid(grid, 0), grid)
    with pytest.raises(ValueError):
        pad_grid(grid, 1.5)


def test_fill_photo_gaps_gives_plain_soil_and_keeps_the_photo():
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    valid = np.zeros((60, 80), bool)
    valid[:, :40] = True
    rgb[:, :40] = [120, 90, 60]                  # soil
    rgb[:30, 35:40] = [60, 160, 60]             # a green strip along the edge
    out = fill_photo_gaps(rgb, valid, block=5, fade=2.0, grain=0.0)
    assert np.array_equal(out[valid], rgb[valid])
    assert out[45, 41].tolist() == [120, 90, 60]          # beside soil: soil
    assert out[10, 41, 1] > 120                           # beside the strip: greenish ...
    assert np.abs(out[10, 79].astype(int) - [120, 90, 60]).max() <= 3   # ... fading to soil
    again = fill_photo_gaps(rgb, valid, block=5)          # grain has a fixed seed
    assert np.array_equal(again, fill_photo_gaps(rgb, valid, block=5))
    with pytest.raises(ValueError):
        fill_photo_gaps(rgb, np.zeros((60, 80), bool))


def test_texture_from_ortho_can_pad_and_fill():
    rgb = np.full((6, 12, 3), 100, dtype=np.uint8)
    rgb[:, :3] = 0                               # no data in the western cell column
    plain = texture_from_ortho(rgb, 3, 2, 4, 3)
    assert np.array_equal(texture_from_ortho(rgb, 3, 2, 4, 3, pad=0, fill=False), plain)
    padded = texture_from_ortho(rgb, 3, 2, 4, 3, pad=1)
    assert padded.shape == (4, 6, 3) and padded[0].max() == 0     # black border
    assert np.array_equal(padded[1:3, 1:5], plain)
    filled = texture_from_ortho(rgb, 3, 2, 4, 3, pad=1, fill=True, margin=0)
    assert filled.shape == (4, 6, 3) and filled.min() > 80        # no black left
    assert np.array_equal(filled[1:3, 2:5], plain[:, 1:])          # the photo is kept
    with pytest.raises(ValueError):
        texture_from_ortho(rgb, 3, 2, 4, 4, pad=1)               # 12 x 18 px: 4 does not fit
