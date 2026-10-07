"""
A coverage mission: the planned segments, driven one after another.

Pure logic, no ROS: which plots to cover, where the robot starts, and the velocity to
command for the robot's current pose (mission_node.py runs it, mission.launch.py uses it
to place the robot at the start).
"""

import json
import math

from agri_ugv_field.ground import ground_height
from agri_ugv_field.layout import read_layout_csv
from agri_ugv_navigation.follower import follow
from agri_ugv_navigation.planner import plan_mission
import numpy as np


def select_plots(plots, spec):
    """
    Return the plots named in spec, plot_IDs separated by commas ('198' or '178,177,198').

    An empty spec selects every plot. Unknown, repeated or malformed IDs raise a ValueError.
    """
    text = str(spec).strip()
    if not text:
        return list(plots)
    by_id = {p['plot_id']: p for p in plots}
    chosen = []
    for token in text.split(','):
        try:
            plot_id = int(token)
        except ValueError:
            raise ValueError(f'plots must be plot_IDs separated by commas, got {token!r} '
                             f'in {text!r}') from None
        if plot_id not in by_id:
            raise ValueError(f'no plot {plot_id} in the layout (plot_IDs '
                             f'{min(by_id)} to {max(by_id)})')
        if plot_id in chosen:
            raise ValueError(f'plot {plot_id} is listed twice in {text!r}')
        chosen.append(plot_id)
    return [by_id[plot_id] for plot_id in chosen]


def load_plan(layout_text, spec=''):
    """
    Plan the mission over the plots named in spec; return (segments, heading, plot_ids).

    Lanes come from the whole field, so a few plots are driven as the full mission would.
    """
    rows = read_layout_csv(layout_text)
    boundary = [r for r in rows if r['type'] == 'boundary']
    if len(boundary) != 1:
        raise ValueError(f'the layout needs exactly one boundary row, found {len(boundary)}')
    field = [r for r in rows if r['type'] == 'plot']
    plots = select_plots(field, spec)
    segments, heading = plan_mission(plots, boundary[0], field=field)
    return segments, heading, [p['plot_id'] for p in plots]


def start_pose(segments, heading):
    """Return the pose (x, y, yaw) where the mission begins: the first segment's start."""
    x, y = segments[0]['start']
    return float(x), float(y), float(heading)


def spawn_height(grid, x, y, radius=1.2, clearance=0.1):
    """
    Return the height at which to place the robot centred at (x, y) to drop it onto the ground.

    The highest ground within 'radius' (the wheels reach 1.0 m from the centre) plus
    'clearance': on a slope, the uphill wheels would otherwise start inside the ground.
    grid is a terrain height grid as read by read_obj_grid.
    """
    steps = np.linspace(-radius, radius, 17)
    dx, dy = np.meshgrid(steps, steps)
    inside = np.hypot(dx, dy) <= radius + 1e-9
    heights = ground_height(grid, x + dx[inside], y + dy[inside])
    return float(np.max(heights)) + clearance


def summary(segments):
    """Return {kind: (number of segments, total length in metres)}."""
    totals = {}
    for s in segments:
        count, length = totals.get(s['kind'], (0, 0.0))
        totals[s['kind']] = (count + 1, length + math.dist(s['start'], s['end']))
    return totals


def plan_to_json(segments, heading, plot_ids):
    """Return the plan as JSON text (published on /mission/plan)."""
    return json.dumps({
        'heading': float(heading), 'plots': [int(p) for p in plot_ids],
        'segments': [{'kind': s['kind'], 'plot_id': s['plot_id'],
                      'start': [float(v) for v in s['start']],
                      'end': [float(v) for v in s['end']]} for s in segments]})


def plan_from_json(text):
    """Read a plan written by plan_to_json; return (segments, heading, plot_ids)."""
    plan = json.loads(text)
    segments = [{'kind': s['kind'], 'plot_id': s['plot_id'], 'start': tuple(s['start']),
                 'end': tuple(s['end'])} for s in plan['segments']]
    return segments, plan['heading'], plan['plots']


class Mission:
    """
    Drive the planned segments one after another with the segment follower.

    step(pose) gives the body velocity to command for the robot's pose. When the follower
    reports a segment done, the next one starts at once: the robot stops at the corner
    anyway, because the driver turns the wheels to the new direction before rolling.
    """

    def __init__(self, segments, heading, speed=0.5):
        """Prepare to drive 'segments' facing 'heading' [rad] at 'speed' [m/s]."""
        if not segments:
            raise ValueError('the mission has no segments')
        if speed <= 0:
            raise ValueError(f'speed must be positive, got {speed}')
        self.segments, self.heading, self.speed = segments, heading, speed
        self.index = 0

    @property
    def finished(self):
        """Return True once every segment is done."""
        return self.index >= len(self.segments)

    def step(self, pose):
        """
        Return ((vx, vy, wz), changed) for the robot's pose (x, y, yaw).

        changed is True when a segment ended during this step. After the last one the
        command is zero.
        """
        changed = False
        while not self.finished:
            segment = self.segments[self.index]
            vx, vy, wz, done = follow(pose, segment['start'], segment['end'], self.heading,
                                      self.speed)
            if not done:
                return (vx, vy, wz), changed
            self.index += 1
            changed = True
        return (0.0, 0.0, 0.0), changed
