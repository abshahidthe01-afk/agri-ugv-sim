"""Crop damage: how much of the wheels' travel runs over plants."""

import math

from agri_ugv_field.plants import CROPS, plot_frame, row_offsets
import numpy as np

STEM_RADIUS = 0.03      # [m] around a row line the tyre runs over the stems: plants crushed


class CropMap:
    """
    The crop rows of a field: tells whether a tyre at a ground point hits plants.

    Rows are lines along each plot's long side, placed exactly like the generated plants;
    a row's leaves spread half the crop's width to each side. Dense crops (cereals) fill
    the whole plot. Gaps between plants within a row are ignored (leaves overlap).
    """

    def __init__(self, plots):
        """Build the map from layout rows (type 'plot'), as read by read_layout_csv."""
        self.plots = []
        for plot in plots:
            crop = CROPS[plot['crop']]
            centre, along, across = plot_frame(plot)
            dense = crop['shape'] == 'cereal'
            rows = np.array([]) if dense else row_offsets(plot['width'], crop['rows'])
            reach = math.hypot(plot['length'], plot['width']) / 2 + 0.5
            self.plots.append((plot['plot_id'], centre, along, across, plot['length'] / 2,
                               plot['width'] / 2, rows, crop['width'] / 2, dense, reach))

    def contact(self, x, y, tyre_half):
        """
        Return (plot_id or None, crushes plants, touches leaves) for a tyre centred at (x, y).

        tyre_half is half the tyre width; the tyre runs along the rows or crosses them.
        """
        point = np.array([x, y])
        for plot_id, centre, along, across, half_length, half_width, rows, canopy, dense, \
                reach in self.plots:
            d = point - centre
            if abs(d[0]) > reach or abs(d[1]) > reach:
                continue
            a, b = d @ along, d @ across
            if abs(a) > half_length or abs(b) > half_width + tyre_half:
                continue
            if dense:
                return plot_id, True, True
            nearest = float(np.min(np.abs(rows - b)))
            return plot_id, nearest < tyre_half + STEM_RADIUS, nearest < tyre_half + canopy
        return None, False, False


def damage(path, wheels, crop_map, tyre_width):
    """
    Add up the wheels' travel over the crop along a driven path.

    path is a list of (x, y, yaw) poses of the robot base; wheels maps a name to the wheel's
    (x, y) position on the robot. Returns {wheel: {'in_plots', 'leaves', 'crushed'}} in metres
    and the crushed metres per plot.
    """
    totals = {name: {'in_plots': 0.0, 'leaves': 0.0, 'crushed': 0.0} for name in wheels}
    per_plot = {}
    for (x0, y0, yaw0), (x1, y1, yaw1) in zip(path[:-1], path[1:]):
        for name, (wx, wy) in wheels.items():
            start = (x0 + math.cos(yaw0) * wx - math.sin(yaw0) * wy,
                     y0 + math.sin(yaw0) * wx + math.cos(yaw0) * wy)
            end = (x1 + math.cos(yaw1) * wx - math.sin(yaw1) * wy,
                   y1 + math.sin(yaw1) * wx + math.cos(yaw1) * wy)
            step = math.hypot(end[0] - start[0], end[1] - start[1])
            plot_id, crushed, leaves = crop_map.contact((start[0] + end[0]) / 2,
                                                        (start[1] + end[1]) / 2, tyre_width / 2)
            if plot_id is None:
                continue
            totals[name]['in_plots'] += step
            totals[name]['leaves'] += step * leaves
            totals[name]['crushed'] += step * crushed
            per_plot[plot_id] = per_plot.get(plot_id, 0.0) + step * crushed
    return totals, per_plot
