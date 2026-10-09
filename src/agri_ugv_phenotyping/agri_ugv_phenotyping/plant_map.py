"""
Plant map from the laser line scanners: per plot, how high the plants are, seen from above.

Each scanner profile is a line of points across the robot (agri_ugv_description: the
scanners on the side panels look down and across, each to the far side). In the robot
frame a point's height z is its height above the plane the wheels stand on, the ground
between the wheels, so it is the plant's height there without knowing the robot's tilt or
the terrain's height. Placed in the field with the robot's pose (x, y, heading) at the
profile's time, the points fill one raster per plot, in the plot's own frame (along the
rows, across them), with 'cell' metres per cell: the highest point of each cell, how many
points fell into it, and how many of them are plant points (higher than 'low') and where
those lie on average across the rows (the rows run along the cells' edges: the cells
alone would move them in steps of a cell). A wrong pose puts plants where they are not:
two passes that disagree blur the rows, and rows placed off the field map show how far
the pose was off across them.

Traits per plot: the share of its area seen, the plant cover (cells higher than 'low'), the
canopy height (95th percentile of those cells' heights) and, for row crops, where the rows
lie in the map compared with the field map and how sharp they are.
"""

import json
import math

from agri_ugv_field.plants import CROPS, plot_frame, row_offsets
import numpy as np

PANEL_Y = 0.72          # [m] points at least this far to the side ...
PANEL_Z = 0.43          # [m] ... and this high, within the box, are the robot's side panels
BOX_X = 0.78            # [m] half the enclosure's length, with a margin


def profile_points(ranges, angle_min, angle_increment, range_min, range_max, mounting):
    """
    Return the points (robot frame, N x 3) of one scanner profile, without the robot itself.

    ranges, angle_min, angle_increment, range_min and range_max are those of the LaserScan;
    mounting is (rotation, translation) of the scanner's beam frame in the robot frame. Rays
    without a return (inf, or outside the range limits) give no point; the outermost rays
    end on the far side panel (the enclosure's inner faces are 0.73 m from the centre).
    """
    ranges = np.asarray(ranges, dtype=float)
    angles = angle_min + angle_increment * np.arange(len(ranges))
    ok = np.isfinite(ranges) & (ranges >= range_min) & (ranges < range_max)
    r, a = ranges[ok], angles[ok]
    rotation, translation = mounting
    points = np.column_stack([r * np.cos(a), r * np.sin(a), np.zeros(len(r))]) @ \
        np.asarray(rotation).T + translation
    panel = (np.abs(points[:, 1]) >= PANEL_Y) & (points[:, 2] >= PANEL_Z) & \
        (np.abs(points[:, 0]) <= BOX_X)
    return points[~panel]


def place(points, pose):
    """Return the world positions (N x 2) of robot-frame points seen from pose (x, y, yaw)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(points, dtype=float)[:, :2] @ np.array([[c, s], [-s, c]]) + [x, y]


class PlotMap:
    """The plants of one plot: per cell the highest point, the points and the plant points."""

    def __init__(self, plot, cell=0.05, low=0.06):
        """Start an empty raster over the plot: cells 'cell' metres along and across the rows."""
        if cell <= 0:
            raise ValueError(f'cell must be positive, got {cell}')
        self.plot, self.cell, self.low = plot, cell, low
        self.centre, self.along, self.across = plot_frame(plot)
        shape = (int(math.ceil(plot['length'] / cell - 1e-9)),
                 int(math.ceil(plot['width'] / cell - 1e-9)))
        self.top = np.full(shape, np.nan, dtype=np.float32)
        self.count = np.zeros(shape, dtype=np.uint32)
        self.plant_count = np.zeros(shape, dtype=np.uint32)     # points higher than 'low'
        self.across_sum = np.zeros(shape)       # sum of their positions across the plot [m]

    def add(self, xy, height):
        """Add points at world positions xy (N x 2) with heights; return how many fell inside."""
        d = np.asarray(xy, dtype=float) - self.centre
        across = d @ self.across
        i = np.floor((d @ self.along + self.plot['length'] / 2) / self.cell).astype(int)
        j = np.floor((across + self.plot['width'] / 2) / self.cell).astype(int)
        inside = (i >= 0) & (i < self.top.shape[0]) & (j >= 0) & (j < self.top.shape[1])
        if inside.any():
            height = np.asarray(height, dtype=float)[inside]
            i, j, across = i[inside], j[inside], across[inside]
            np.fmax.at(self.top, (i, j), height.astype(np.float32))
            np.add.at(self.count, (i, j), 1)
            plant = height > self.low
            np.add.at(self.plant_count, (i[plant], j[plant]), 1)
            np.add.at(self.across_sum, (i[plant], j[plant]), across[plant])
        return int(inside.sum())

    def plants(self):
        """Return the cells with plant points (boolean raster)."""
        return self.plant_count > 0

    def traits(self, min_cells=50):
        """
        Return the plot's traits as a dict.

        seen: share of the plot's cells with points; cover: share of the seen cells with
        plant points (higher than 'low'); height and height_median: 95th and 50th percentile
        of those cells' heights [m] (None without plant cells); points: all points placed in
        the plot. For row crops with at least min_cells plant cells: rows_offset, how far the
        rows lie in the map from where the field map puts them, across the plot ('across' of
        plot_frame) [m], within half a row spacing; rows_sharpness, how sharply the plant
        points line up on rows (0 to 1, circular statistics as in agri_ugv_navigation.rows;
        the points of a cell at their mean position).
        """
        seen = self.count > 0
        plants = self.plants()
        heights = self.top[plants].astype(float)
        result = {'plot': self.plot['plot_id'], 'crop': self.plot['crop'],
                  'seen': float(seen.mean()),
                  'cover': float(plants.sum() / seen.sum()) if seen.any() else 0.0,
                  'height': float(np.percentile(heights, 95)) if len(heights) else None,
                  'height_median': float(np.median(heights)) if len(heights) else None,
                  'points': int(self.count.sum()), 'rows_offset': None,
                  'rows_sharpness': None}
        crop = CROPS[self.plot['crop']]
        if crop['shape'] != 'cereal' and plants.sum() >= min_cells:
            spacing = crop['rows']
            weight = self.plant_count[plants]
            across = self.across_sum[plants] / weight
            first = row_offsets(self.plot['width'], spacing)[0]
            mean = np.average(np.exp(2j * math.pi * (across - first) / spacing), weights=weight)
            result['rows_offset'] = float(np.angle(mean) / (2 * math.pi) * spacing)
            result['rows_sharpness'] = float(abs(mean))
        return result

    def plant_cells(self):
        """Return the world positions (N x 2) and heights (N) of the cells with plants."""
        i, j = np.nonzero(self.plants())
        a = (i + 0.5) * self.cell - self.plot['length'] / 2
        b = (j + 0.5) * self.cell - self.plot['width'] / 2
        xy = self.centre + np.outer(a, self.along) + np.outer(b, self.across)
        return xy, self.top[i, j].astype(float)


class FieldMap:
    """Plant maps of all plots, filled with scanner profiles placed by the robot's pose."""

    def __init__(self, plots, cell=0.05, low=0.06, reach=1.5):
        """Prepare a map per plot; profiles reach at most 'reach' metres from the robot."""
        self.maps = {p['plot_id']: PlotMap(p, cell, low) for p in plots}
        self.cell, self.low = cell, low
        self.ids = np.array(list(self.maps))
        self.centres = np.array([[p['centre_x'], p['centre_y']] for p in plots]).reshape(-1, 2)
        self.radius = np.array([math.hypot(p['length'], p['width']) / 2 for p in plots]) + reach
        self.changed = {}                  # plot_id: time of the latest point placed in it

    def add(self, points, pose, t=0.0):
        """Add one profile's points (robot frame) seen from pose (x, y, yaw) at time t."""
        if len(points) == 0 or len(self.ids) == 0:
            return 0
        near = np.hypot(*(self.centres - pose[:2]).T) <= self.radius
        if not near.any():
            return 0
        xy, placed = place(points, pose), 0
        for plot_id in self.ids[near]:
            n = self.maps[plot_id].add(xy, points[:, 2])
            if n:
                self.changed[int(plot_id)] = t
                placed += n
        return placed

    def traits(self):
        """Return the traits of every plot with points (see PlotMap.traits), by plot_id."""
        return {plot_id: m.traits() for plot_id, m in self.maps.items() if m.count.any()}

    def save(self, path):
        """
        Save the plots with points: path.npz (rasters) and path.json (traits).

        The npz holds 'cell', 'low', 'plots' (the plots' layout rows as JSON text) and per
        plot 'top_<plot_id>' (highest point per cell [m], NaN where nothing was seen),
        'count_<plot_id>', 'plant_count_<plot_id>' and 'across_sum_<plot_id>'. Returns the
        number of plots saved.
        """
        maps = {plot_id: m for plot_id, m in self.maps.items() if m.count.any()}
        arrays = {}
        for plot_id, m in maps.items():
            for name in ('top', 'count', 'plant_count', 'across_sum'):
                arrays[f'{name}_{plot_id}'] = getattr(m, name)
        np.savez_compressed(f'{path}.npz', cell=self.cell, low=self.low,
                            plots=json.dumps([m.plot for m in maps.values()]), **arrays)
        with open(f'{path}.json', 'w') as out:
            json.dump([m.traits() for m in maps.values()], out, indent=1)
        return len(maps)
