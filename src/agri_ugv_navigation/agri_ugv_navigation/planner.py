"""
Mission planner: passes through every plot and the lane moves between them.

The robot keeps facing along the crop rows the whole time (four-wheel steering): passes
are driven forwards or backwards in turn, and the next pass is reached by moving sideways
in the lane at the plot's end. Plot rows are served from their wider neighbouring lane
and taken in a serpentine; changing plot rows goes around the end of the plot block.
"""

import math

import numpy as np

ROBOT_HALF_WIDTH = 0.84       # [m] wheel centre to the outer tyre edge, plus a little
ROW_GAP = 3.0                 # [m] plot centres further apart across the field: next plot row


def field_frame(boundary):
    """Return (centre, along, across): the trial boundary's centre and its axes."""
    h = math.radians(boundary['heading_deg'])
    along = np.array([math.cos(h), math.sin(h)])
    return (np.array([boundary['centre_x'], boundary['centre_y']]), along,
            np.array([-along[1], along[0]]))


def plot_axes(plot):
    """Return (centre, rows direction, across the rows) of a plot (world frame)."""
    h = math.radians(plot['heading_deg'])
    u = np.array([math.cos(h), math.sin(h)])
    return np.array([plot['centre_x'], plot['centre_y']]), u, np.array([-u[1], u[0]])


def plot_rows(plots, frame):
    """Group the plots into plot rows across the field: north first, each west to east."""
    centre, along, across = frame
    ordered = sorted(plots, key=lambda p: -(plot_axes(p)[0] - centre) @ across)
    rows, current = [], [ordered[0]]
    for plot in ordered[1:]:
        gap = ((plot_axes(current[-1])[0] - plot_axes(plot)[0]) @ across)
        if gap > ROW_GAP:
            rows.append(current)
            current = []
        current.append(plot)
    rows.append(current)
    return [sorted(r, key=lambda p: (plot_axes(p)[0] - centre) @ along) for r in rows]


def lanes(rows, frame, headland=1.0):
    """
    Return the across-field position of the lane centre lines, north to south.

    Between two plot rows the lane centre is halfway; outside the block it is 'headland'
    metres beyond the plot ends, enough for the robot's body to leave the plot.
    """
    centre, _, across = frame
    north = [max((plot_axes(p)[0] - centre) @ across + p['length'] / 2 for p in r) for r in rows]
    south = [min((plot_axes(p)[0] - centre) @ across - p['length'] / 2 for p in r) for r in rows]
    inner = [(south[k] + north[k + 1]) / 2 for k in range(len(rows) - 1)]
    widths = [math.inf] + [south[k] - north[k + 1] for k in range(len(rows) - 1)] + [math.inf]
    return [north[0] + headland] + inner + [south[-1] - headland], widths


def pass_offsets(width, track=1.5, slack=0.05):
    """
    Return evenly spaced pass centre lines across a plot (the simple, row-blind way).

    As many passes as needed to cover the width with the track, allowing 'slack' metres
    left over (plots measure 5.98-6.02 m: a 2 cm sliver is not worth a fifth pass).
    """
    n = max(1, math.ceil((width - slack) / track))
    return [(k - (n - 1) / 2) * width / n for k in range(n)]


def point_on_lane(plot, offset, lane, frame):
    """Return where a plot's pass line (offset across the rows) meets a lane centre line."""
    centre, _, across = frame
    c, u, v = plot_axes(plot)
    base = c + offset * v
    t = (lane - (base - centre) @ across) / (u @ across)
    return base + t * u


def plan_mission(plots, boundary, headland=1.0, track=1.5, field=None):
    """
    Return the mission as a list of straight segments to drive in order.

    Each segment is a dict: kind ('pass', 'shift' between passes, 'transfer' between plots
    or plot rows), start and end (world x, y), and plot_id for passes. The heading stays
    the plots' mean row direction throughout. field lists all plots of the trial when
    'plots' is only a part of it: lanes and the way around the block's end then come from
    the whole field, so a mission over a few plots drives where the full mission would
    (and not into the neighbouring plots).
    """
    field = plots if field is None else field
    missing = {p['plot_id'] for p in plots} - {p['plot_id'] for p in field}
    if missing:
        raise ValueError(f'plots {sorted(missing)} are not part of the field')
    chosen = {p['plot_id'] for p in plots}
    frame = field_frame(boundary)
    centre, along, across = frame
    rows = plot_rows(field, frame)
    lane_lines, lane_widths = lanes(rows, frame, headland)
    block_end = max(abs((plot_axes(p)[0] - centre) @ along) + p['width'] / 2 for p in field)
    beyond = block_end + ROBOT_HALF_WIDTH + 0.8           # around the block's end
    served = 0
    points, segments = [], []

    def go(kind, end, plot_id=None):
        start = points[-1] if points else end
        if np.hypot(*(end - start)) > 1e-6:
            segments.append({'kind': kind, 'start': tuple(start), 'end': tuple(end),
                             'plot_id': plot_id})
        points.append(np.asarray(end, dtype=float))

    for r, row in enumerate(rows):
        row = [p for p in row if p['plot_id'] in chosen]
        if not row:
            continue
        wide = 0 if lane_widths[r] >= lane_widths[r + 1] else 1
        home, far = lane_lines[r + wide], lane_lines[r + 1 - wide]
        eastwards = served % 2 == 0
        served += 1
        order = row if eastwards else row[::-1]
        if points and abs((points[-1] - centre) @ across - home) > 1e-6:   # change lane
            side = math.copysign(beyond, (points[-1] - centre) @ along)
            corner = centre + side * along + ((points[-1] - centre) @ across) * across
            go('transfer', corner)
            go('transfer', centre + side * along + home * across)
        for plot in order:
            offsets = sorted(pass_offsets(plot['width'], track),
                             key=lambda o: (plot_axes(plot)[2] * o) @ along)
            if not eastwards:
                offsets = offsets[::-1]
            for k, offset in enumerate(offsets):
                near, other = (home, far) if k % 2 == 0 else (far, home)
                go('transfer' if k == 0 else 'shift', point_on_lane(plot, offset, near, frame))
                go('pass', point_on_lane(plot, offset, other, frame), plot['plot_id'])
    heading = math.atan2(*np.mean([plot_axes(p)[1] for p in plots], axis=0)[::-1])
    return segments, heading
