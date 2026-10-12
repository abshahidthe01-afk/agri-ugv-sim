"""
Tramlines: the cereal rows left unsown where the robot's wheels run.

Farmers leave tramlines in drilled cereals so that the machines' wheels do not crush the
crop. Here they are sown for this robot: its passes through a plot (pass_offsets) and its
track give the lines its wheels run on, and a row closer than WHEEL_TO_ROW to one of them
is not sown. The mission planner (agri_ugv_navigation.planner) drives the same passes.
"""

import math

import numpy as np

TRACK = 1.5             # [m] the robot's track: wheel centre to wheel centre, across
SLACK = 0.05            # [m] width a plot's passes may leave uncovered
# [m] least distance from a wheel's centre to a crop row: half a tyre (0.0825), the stems
# (0.03) and 3 cm for steering errors
WHEEL_TO_ROW = 0.14


def even_offsets(width, n, track=TRACK):
    """Return n pass centre lines spread evenly, the outer wheels on the plot's edges."""
    if n == 1:
        return [0.0]
    edge = (width - track) / 2
    return sorted(-edge + k * 2 * edge / (n - 1) for k in range(n))


def pass_offsets(width, track=TRACK, slack=SLACK):
    """
    Return evenly spaced pass centre lines across a plot (the simple, row-blind way).

    As many passes as needed to cover the width with the track, allowing 'slack' metres
    left over (plots measure 5.98-6.02 m: a 2 cm sliver is not worth a fifth pass), and an
    even number of them, so that the robot leaves every plot into the lane it came from.
    """
    n = max(1, math.ceil((width - slack) / track))
    return even_offsets(width, n + n % 2, track)


def wheel_lines(width, track=TRACK, slack=SLACK):
    """Return the lines across a plot that the wheels of its passes (pass_offsets) run on."""
    return sorted(o + side * track / 2 for o in pass_offsets(width, track, slack)
                  for side in (-1, 1))


def tramline_rows(rows, width, track=TRACK, room=WHEEL_TO_ROW):
    """
    Return a mask of the rows that lie on tramlines: closer than 'room' to a wheel line.

    rows are the rows' offsets across a plot 'width' metres wide (plants.row_offsets).
    """
    rows = np.asarray(rows, dtype=float)
    lines = np.array(wheel_lines(width, track))
    return np.min(np.abs(rows[:, None] - lines[None, :]), axis=1) < room
