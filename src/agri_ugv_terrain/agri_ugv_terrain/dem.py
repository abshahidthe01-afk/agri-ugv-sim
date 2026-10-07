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


def pad_grid(grid, cells):
    """
    Extend a height grid by `cells` on every side, continuing the edge heights outwards.

    The new cells are filled like gaps (fill_gaps), so the ground goes on level from the
    edge instead of ending. The original cells keep their values.
    """
    if int(cells) != cells or cells < 0:
        raise ValueError(f'cells must be a whole number >= 0, got {cells}')
    grid = np.asarray(grid, dtype=float)
    if cells == 0:
        return grid.copy()
    return fill_gaps(np.pad(grid, int(cells), constant_values=np.nan))


def fill_photo_gaps(rgb, valid, block=10, fade=3.0, grain=4.0, seed=1):
    """
    Fill the pixels of a photo that have no data with plain soil colour.

    The colour starts as the photo's colour at the nearest edge of the data (block means
    of block x block pixels, spread outwards like fill_gaps) and fades into the photo's
    typical (median) colour over about `fade` blocks; a little random grain (standard
    deviation `grain`, fixed seed) keeps it from looking painted. No detail is invented.
    """
    rgb = np.asarray(rgb)
    valid = np.asarray(valid, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[:2] != valid.shape or rgb.shape[2] != 3:
        raise ValueError(f'need an RGB photo and a mask of the same size, got {rgb.shape} '
                         f'and {valid.shape}')
    if not valid.any():
        raise ValueError('the photo has no valid pixel')
    from PIL import Image  # only needed for textured terrains

    h, w = valid.shape
    rows, cols = -(-h // block), -(-w // block)
    image = np.zeros((rows * block, cols * block, 3))
    image[:h, :w] = rgb
    known = np.zeros((rows * block, cols * block), bool)
    known[:h, :w] = valid
    inside = np.zeros_like(known)
    inside[:h, :w] = True
    sums = (image * known[..., None]).reshape(rows, block, cols, block, 3).sum(axis=(1, 3))
    counts = known.reshape(rows, block, cols, block).sum(axis=(1, 3))
    full = counts > inside.reshape(rows, block, cols, block).sum(axis=(1, 3)) / 2
    if not full.any():
        full = counts > 0
    coarse = np.where(full[..., None], sums / np.maximum(counts, 1)[..., None], np.nan)
    distance = np.where(full, 0.0, np.inf)       # in blocks from the data, growing outwards
    step = 0
    while np.isinf(distance).any() and step < rows + cols:
        step += 1
        padded = np.pad(distance, 1, constant_values=np.inf)
        nearest = np.minimum.reduce([padded[:-2, 1:-1], padded[2:, 1:-1],
                                     padded[1:-1, :-2], padded[1:-1, 2:]])
        distance[np.isinf(distance) & (nearest == step - 1)] = step
    edge = np.stack([fill_gaps(coarse[..., k]) for k in range(3)], axis=-1)
    soil = np.median(rgb[valid].astype(float), axis=0)
    colour = soil + (edge - soil) * np.exp(-distance / fade)[..., None]
    small = Image.fromarray(np.clip(np.round(colour), 0, 255).astype(np.uint8))
    smooth = np.asarray(small.resize((cols * block, rows * block), Image.BILINEAR),
                        dtype=float)[:h, :w]
    smooth += np.random.default_rng(seed).normal(0.0, grain, (h, w))[..., None]
    out = rgb.copy()
    out[~valid] = np.clip(np.round(smooth[~valid]), 0, 255).astype(rgb.dtype)
    return out


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


def texture_from_ortho(rgb, block, rows, cols, factor, pad=0, fill=False, margin=2):
    """
    Cut the area covered by a rows x cols terrain grid out of an aerial photo and shrink it.

    rgb is the photo as (height, width, 3), on the same pixel grid as the DEM; the terrain
    used the first rows * block by cols * block pixels. factor > 1 averages factor x factor
    pixels into one, to keep the texture file small. pad adds that many cells on every side
    (as pad_grid does for the heights), black at first. With fill, the texture pixels
    without photo data (black in the photo, or added by pad), and `margin` pixels around
    them where the photo's edge is often dark, get plain soil colour (fill_photo_gaps).
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f'expected an RGB image of shape (height, width, 3), got {rgb.shape}')
    if int(pad) != pad or pad < 0:
        raise ValueError(f'pad must be a whole number of cells >= 0, got {pad}')
    height, width = rows * block, cols * block
    if rgb.shape[0] < height or rgb.shape[1] < width:
        raise ValueError(f'photo {rgb.shape[:2]} is smaller than the terrain {height, width}')
    border = int(pad) * block
    full_h, full_w = height + 2 * border, width + 2 * border
    if int(factor) != factor or factor < 1 or full_h % factor or full_w % factor:
        raise ValueError(f'factor must be a whole number that divides {full_h} and {full_w}')
    factor = int(factor)
    area = np.pad(rgb[:height, :width], ((border, border), (border, border), (0, 0)))
    shape = (full_h // factor, factor, full_w // factor, factor)
    small = np.round(area.reshape(*shape, 3).mean(axis=(1, 3), dtype=float)).astype(np.uint8)
    if not fill:
        return small
    valid = (area.max(axis=2) > 0).reshape(shape).all(axis=(1, 3))
    for _ in range(margin):
        edge = np.pad(valid, 1, constant_values=False)
        valid &= edge[:-2, 1:-1] & edge[2:, 1:-1] & edge[1:-1, :-2] & edge[1:-1, 2:]
    return fill_photo_gaps(small, valid)


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
