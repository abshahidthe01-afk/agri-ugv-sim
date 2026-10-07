"""
Mission planner: passes through every plot and the lane moves between them.

The robot keeps facing along the crop rows the whole time (four-wheel steering): passes
are driven forwards or backwards in turn, and the next pass is reached by moving sideways
in the lane at the plot's end. Plot rows are served from their wider neighbouring lane
and taken in a serpentine; changing plot rows goes around the end of the plot block.
In the lanes the tyres keep a clearance from the plot ends where there is room for it;
where a lane is narrower, the robot moves in its middle, measured at each plot's end.
"""

import math

import numpy as np

ROBOT_HALF_WIDTH = 0.84       # [m] wheel centre to the outer tyre edge, plus a little
ROBOT_HALF_LENGTH = 0.76      # [m] centre to the tyre edge along the rows (tyres steered across)
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


def corners(plot):
    """Return the four corners of a plot (world x, y), in order around it."""
    c, u, v = plot_axes(plot)
    a, b = plot['length'] / 2, plot['width'] / 2
    return [c + s * a * u + t * b * v for s, t in ((1, 1), (-1, 1), (-1, -1), (1, -1))]


def reach(row, side, frame):
    """Return how far a plot row reaches north (side +1) or south (-1, as a distance)."""
    centre, _, across = frame
    return max(side * ((c - centre) @ across) for p in row for c in corners(p))


def lane_widths(rows, frame):
    """Return the free width of the lanes north of each plot row and south of the last."""
    inner = [-reach(rows[k], -1, frame) - reach(rows[k + 1], 1, frame)
             for k in range(len(rows) - 1)]
    return [math.inf] + inner + [math.inf]


def stop_between(near, far, clearance, where):
    """
    Return where to stop between near and far, the closest and furthest places that fit.

    'clearance' beyond near if the room allows that on both sides, otherwise halfway.
    """
    if far < near:
        raise ValueError(f'the robot does not fit into the lane {where}: '
                         f'{near - far:.2f} m too narrow')
    return near + clearance if far - near >= 2 * clearance else (near + far) / 2


def lane_line(rows, r, side, frame, clearance):
    """
    Return the across-field position of the line along which the robot moves past plot row r.

    The line runs in the lane north (side +1) or south (-1) of the row, so that the tyres
    keep 'clearance' from every plot end of the row and of the row across the lane.
    """
    near = reach(rows[r], side, frame) + ROBOT_HALF_LENGTH
    k = r - side                                    # the plot row across the lane
    far = -reach(rows[k], -side, frame) - ROBOT_HALF_LENGTH if 0 <= k < len(rows) \
        else math.inf
    return side * stop_between(near, far, clearance, f'beside plot row {r + 1}')


def clip_across(points, low, high):
    """Return the part of a convex polygon [(a, b), ...] with low <= b <= high."""
    for limit, sign in ((low, 1), (high, -1)):
        kept = []
        for p, q in zip(points, points[1:] + points[:1]):
            if sign * (p[1] - limit) >= 0:
                kept.append(p)
            if (sign * (p[1] - limit) >= 0) != (sign * (q[1] - limit) >= 0):
                t = (limit - p[1]) / (q[1] - p[1])
                kept.append((p[0] + t * (q[0] - p[0]), limit))
        points = kept
    return points


def lane_stop(plot, offsets, side, field, clearance):
    """
    Return where the robot moves sideways at one end of a plot, between passes at offsets.

    The result is the distance from the plot centre along its rows, towards the end the
    rows point to (side +1) or the other end (-1). The robot's tyres keep 'clearance' from
    the plot's end and from the plots across the lane where there is room for both;
    otherwise the robot moves in the middle of the free space. Only the plots that the
    robot's width passes over count (field: all plots).
    """
    c, u, v = plot_axes(plot)
    low, high = min(offsets) - ROBOT_HALF_WIDTH, max(offsets) + ROBOT_HALF_WIDTH
    end = plot['length'] / 2
    near, far = end, math.inf
    for other in field:
        part = clip_across([(side * ((q - c) @ u), (q - c) @ v) for q in corners(other)],
                           low, high)
        if not part:
            continue
        a = [p[0] for p in part]
        if min(a) > end:                            # across the lane
            far = min(far, min(a))
        else:                                       # the plot itself, or one beside it
            near = max(near, max(a))
    return side * stop_between(near + ROBOT_HALF_LENGTH, far - ROBOT_HALF_LENGTH, clearance,
                               f'at the end of plot {plot["plot_id"]}')


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


def plan_mission(plots, boundary, clearance=0.3, track=1.5, field=None):
    """
    Return the mission as a list of straight segments to drive in order.

    Each segment is a dict: kind ('pass', 'shift' between passes, 'transfer' between plots
    or plot rows), start and end (world x, y), and plot_id for passes. The heading stays
    the plots' mean row direction throughout. In the lanes the tyres keep 'clearance' (m)
    from the plot ends where the lane is wide enough. field lists all plots of the trial
    when 'plots' is only a part of it: lanes and the way around the block's end then come
    from the whole field, so a mission over a few plots drives where the full mission would
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
    widths = lane_widths(rows, frame)
    block_end = max(abs((plot_axes(p)[0] - centre) @ along) + p['width'] / 2 for p in field)
    beyond = block_end + ROBOT_HALF_WIDTH + 0.8           # around the block's end
    served, lane = 0, None
    points, segments = [], []

    def go(kind, end, plot_id=None):
        start = points[-1] if points else end
        if np.hypot(*(end - start)) > 1e-6:
            segments.append({'kind': kind, 'start': tuple(start), 'end': tuple(end),
                             'plot_id': plot_id})
        points.append(np.asarray(end, dtype=float))

    for r, row in enumerate(rows):
        chosen_row = [p for p in row if p['plot_id'] in chosen]
        if not chosen_row:
            continue
        home = 1 if widths[r] >= widths[r + 1] else -1     # the wider lane: north or south
        line = lane_line(rows, r, home, frame, clearance)
        here = r if home > 0 else r + 1                     # lanes numbered north to south
        eastwards = served % 2 == 0
        served += 1
        order = chosen_row if eastwards else chosen_row[::-1]
        if points and here != lane:                         # another lane: around the end
            side = math.copysign(beyond, (points[-1] - centre) @ along)
            corner = centre + side * along + ((points[-1] - centre) @ across) * across
            go('transfer', corner)
            go('transfer', centre + side * along + line * across)
        lane = here
        for plot in order:
            c, u, v = plot_axes(plot)
            offsets = sorted(pass_offsets(plot['width'], track), key=lambda o: (v * o) @ along)
            if not eastwards:
                offsets = offsets[::-1]
            towards_home = 1 if home * (u @ across) > 0 else -1   # the plot's end at home
            stop = None                         # where the last pass ended (along the rows)
            for k, offset in enumerate(offsets):
                if stop is None:
                    go('transfer', point_on_lane(plot, offset, line, frame))
                else:
                    go('shift', c + stop * u + offset * v)
                ending = -towards_home if k % 2 == 0 else towards_home
                if k == len(offsets) - 1 and ending == towards_home:
                    go('pass', point_on_lane(plot, offset, line, frame), plot['plot_id'])
                else:
                    stop = lane_stop(plot, offsets[k:k + 2], ending, field, clearance)
                    go('pass', c + stop * u + offset * v, plot['plot_id'])
    heading = math.atan2(*np.mean([plot_axes(p)[1] for p in plots], axis=0)[::-1])
    return segments, heading
