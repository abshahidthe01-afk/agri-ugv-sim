"""Turn a DEM (a raster of ground heights, e.g. from a drone survey) into a terrain height grid."""

import warnings

import numpy as np


def block_percentile(heights, valid, block, percentile, min_valid=0.5):
    """
    Reduce a fine raster to one height per cell of block x block pixels.

    Each cell gets the given percentile of its valid pixels: a low percentile follows the
    soil between plants and ignores spikes. Cells with fewer than `min_valid` (a fraction)
    valid pixels become NaN. Leftover pixels at the right and bottom edges are dropped.
    """
    heights = np.asarray(heights, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if heights.ndim != 2 or valid.shape != heights.shape:
        raise ValueError('heights must be 2D and valid must have the same shape')
    if int(block) != block or block < 1:
        raise ValueError(f'block must be a whole number of pixels >= 1, got {block}')
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f'percentile must be between 0 and 100, got {percentile}')
    block = int(block)
    rows, cols = heights.shape[0] // block, heights.shape[1] // block
    if rows < 2 or cols < 2:
        raise ValueError(f'raster {heights.shape} is too small for blocks of {block} pixels')

    trimmed = np.where(valid, heights, np.nan)[:rows * block, :cols * block]
    cells = trimmed.reshape(rows, block, cols, block).swapaxes(1, 2).reshape(rows, cols, -1)
    count = np.sum(~np.isnan(cells), axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # cells without any valid pixel
        result = np.nanpercentile(cells, percentile, axis=2)
    result[count < min_valid * block * block] = np.nan
    return result


def fill_gaps(grid):
    """
    Fill NaN cells from their neighbours, growing inwards from the known cells.

    Each pass gives every empty cell next to known cells the mean of those neighbours
    (up, down, left, right), until no gaps are left. Used for the corners outside the field.
    """
    grid = np.array(grid, dtype=float)
    if grid.ndim != 2 or np.all(np.isnan(grid)):
        raise ValueError('need a 2D grid with at least one known cell')
    while np.isnan(grid).any():
        padded = np.pad(grid, 1, constant_values=np.nan)
        neighbours = np.stack([padded[:-2, 1:-1], padded[2:, 1:-1],
                               padded[1:-1, :-2], padded[1:-1, 2:]])
        known = ~np.isnan(neighbours)
        count = known.sum(axis=0)
        total = np.where(known, neighbours, 0.0).sum(axis=0)
        fill = np.isnan(grid) & (count > 0)
        grid[fill] = total[fill] / count[fill]
    return grid


def heights_from_dem(heights, valid, block, percentile, min_valid=0.5):
    """
    Return (grid, reference): a terrain height grid ready for mesh_from_heights.

    Steps: a low percentile per cell, fill the gaps, flip the rows (raster row 0 is the
    north edge, but our grid rows run from south to north along +y), and subtract the
    height at the centre, so the robot's spawn point (the origin) is at about z = 0.
    `reference` is the subtracted height, to convert back to the original heights.
    """
    grid = np.flipud(fill_gaps(block_percentile(heights, valid, block, percentile, min_valid)))
    rows, cols = grid.shape
    r, c = (rows - 1) / 2.0, (cols - 1) / 2.0
    centre = grid[int(np.floor(r)):int(np.ceil(r)) + 1, int(np.floor(c)):int(np.ceil(c)) + 1]
    reference = float(centre.mean())
    return grid - reference, reference


def texture_from_ortho(rgb, block, rows, cols, factor):
    """
    Cut the area covered by a rows x cols terrain grid out of an aerial photo and shrink it.

    rgb is the photo as (height, width, 3), on the same pixel grid as the DEM; the terrain
    used the first rows * block by cols * block pixels. factor > 1 averages factor x factor
    pixels into one, to keep the texture file small.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f'expected an RGB image of shape (height, width, 3), got {rgb.shape}')
    height, width = rows * block, cols * block
    if rgb.shape[0] < height or rgb.shape[1] < width:
        raise ValueError(f'photo {rgb.shape[:2]} is smaller than the terrain {height, width}')
    if int(factor) != factor or factor < 1 or height % factor or width % factor:
        raise ValueError(f'factor must be a whole number that divides {height} and {width}')
    factor = int(factor)
    area = rgb[:height, :width].astype(float)
    small = area.reshape(height // factor, factor, width // factor, factor, 3).mean(axis=(1, 3))
    return np.round(small).astype(np.uint8)


def read_ortho(path):
    """Read an RGB GeoTIFF; returns (rgb as (height, width, 3) uint8, pixel_size, left, top)."""
    import rasterio  # only needed for real photo files, so imported here

    with rasterio.open(path) as src:
        if src.count != 3 or src.dtypes[0] != 'uint8':
            raise ValueError(f'{path}: expected 3 bands of uint8, got {src.count} {src.dtypes[0]}')
        transform = src.transform
        rgb = np.moveaxis(src.read(), 0, -1)
        return rgb, float(transform.a), float(transform.c), float(transform.f)


def read_dem(path):
    """
    Read a single-band, north-up GeoTIFF with square pixels.

    Returns (heights, valid, pixel_size, left, top, crs): left and top are the map
    coordinates of the top-left corner, crs the coordinate system, e.g. 'EPSG:32632'.
    """
    import rasterio  # only needed for real DEM files, so imported here

    with rasterio.open(path) as src:
        transform = src.transform
        if src.count != 1:
            raise ValueError(f'{path}: expected 1 band, found {src.count}')
        if transform.b != 0.0 or transform.d != 0.0 or transform.e >= 0.0:
            raise ValueError(f'{path}: raster must be north-up (no rotation)')
        if abs(transform.a + transform.e) > 1e-9:
            raise ValueError(f'{path}: pixels must be square, got {src.res}')
        data = src.read(1, masked=True)
        valid = ~np.ma.getmaskarray(data)
        return (np.asarray(data.data, dtype=float), valid, float(transform.a),
                float(transform.c), float(transform.f), str(src.crs))
