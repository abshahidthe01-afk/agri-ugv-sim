"""Tests for the coverage mission: plot selection, plan, start, and driving it through."""

import math
from pathlib import Path

from agri_ugv_navigation.mission import (load_plan, Mission, plan_from_json, plan_to_json,
                                         select_plots, spawn_height, start_pose, summary)
from agri_ugv_navigation.planner import plan_mission
import numpy as np
import pytest

LAYOUT = Path(__file__).resolve().parents[2] / 'agri_ugv_field' / 'data' / \
    'must_c_field_plots.csv'


def plot(plot_id, east, north):
    """Return a 6 x 8 m plot whose rows run north, centred at (east, north)."""
    return {'plot_id': plot_id, 'crop': 'Sugar Beet', 'centre_x': east, 'centre_y': north,
            'width': 6.0, 'length': 8.0, 'heading_deg': 90.0}


BOUNDARY = {'centre_x': 7.5, 'centre_y': 0.0, 'width': 30.0, 'length': 40.0,
            'heading_deg': 0.0}
PLOTS = [plot(10 + k, 7.5 * k, 6.0) for k in range(3)] + \
        [plot(20 + k, 7.5 * k, -6.0) for k in range(3)]


@pytest.fixture(scope='module')
def layout():
    """Return the committed MuST-C layout text, or skip if it has not been made yet."""
    if not LAYOUT.exists():
        pytest.skip(f'{LAYOUT.name} not generated yet')
    return LAYOUT.read_text()


def test_plots_are_chosen_by_id_in_the_given_order():
    assert [p['plot_id'] for p in select_plots(PLOTS, ' 21, 10')] == [21, 10]
    assert select_plots(PLOTS, '') == PLOTS
    assert [p['plot_id'] for p in select_plots(PLOTS, 11)] == [11]      # a single number


@pytest.mark.parametrize('spec, message', [('10;11', 'separated by commas'),
                                           ('99', 'no plot 99'), ('10,10', 'twice')])
def test_bad_plot_lists_are_refused(spec, message):
    with pytest.raises(ValueError, match=message):
        select_plots(PLOTS, spec)


def test_the_whole_field_is_planned_as_before(layout):
    segments, heading, plot_ids = load_plan(layout)
    assert len(plot_ids) == 80 and math.degrees(heading) == pytest.approx(84.55, abs=0.01)
    counts = {kind: n for kind, (n, _) in summary(segments).items()}
    assert counts == {'pass': 336, 'shift': 256, 'transfer': 83}     # 6 passes in soybean
    assert sum(length for _, length in summary(segments).values()) == \
        pytest.approx(4025.4, abs=0.1)


def test_one_plot_is_passed_exactly_where_the_whole_mission_passes_it(layout):
    def lines(segments):
        return {tuple(sorted((tuple(np.round(s['start'], 6)), tuple(np.round(s['end'], 6)))))
                for s in segments if s['kind'] == 'pass' and s['plot_id'] == 198}
    block, heading, _ = load_plan(layout, '198')
    assert [s['kind'] for s in block] == ['pass', 'shift'] * 3 + ['pass']
    assert len(lines(block)) == 4 and lines(block) == lines(load_plan(layout)[0])
    x, y, yaw = start_pose(block, heading)
    assert (x, y) == block[0]['start'] and yaw == heading


def test_the_robot_is_placed_above_the_highest_ground_under_it():
    xs, ys = np.arange(-5.0, 5.01, 0.5), np.arange(-5.0, 5.01, 0.5)
    grid = (xs, ys, 0.1 * xs[None, :] + 0.0 * ys[:, None])      # 10 % slope up to the east
    assert spawn_height(grid, 0.0, 0.0) == pytest.approx(0.1 * 1.2 + 0.1)


def test_the_plan_survives_the_trip_through_json():
    segments, heading = plan_mission(PLOTS, BOUNDARY)
    again, heading_again, plots = plan_from_json(plan_to_json(segments, heading, [10, 11]))
    assert heading_again == heading and plots == [10, 11]
    assert [s['kind'] for s in again] == [s['kind'] for s in segments]
    assert all(a['start'] == pytest.approx(s['start']) and a['end'] == pytest.approx(s['end'])
               and a['plot_id'] == s['plot_id'] for a, s in zip(again, segments))


def test_a_perfect_robot_drives_the_whole_mission_and_stops_at_the_end():
    segments, heading = plan_mission(PLOTS, BOUNDARY)
    mission = Mission(segments, heading)
    x, y = segments[0]['start']
    yaw, dt, changes = heading, 0.05, 0
    for _ in range(20000):
        (vx, vy, wz), changed = mission.step((x, y, yaw))
        changes += changed
        if mission.finished:
            break
        x += (math.cos(yaw) * vx - math.sin(yaw) * vy) * dt
        y += (math.sin(yaw) * vx + math.cos(yaw) * vy) * dt
        yaw += wz * dt
    assert mission.finished and changes == len(segments)
    assert (x, y) == pytest.approx(segments[-1]['end'], abs=0.03)
    assert mission.step((x, y, yaw)) == ((0.0, 0.0, 0.0), False)


def test_a_mission_needs_segments_and_a_speed():
    with pytest.raises(ValueError):
        Mission([], 0.0)
    with pytest.raises(ValueError):
        Mission(plan_mission(PLOTS, BOUNDARY)[0], 0.0, speed=0.0)


def test_follower_options_reach_the_follower_and_can_change():
    segments, heading = plan_mission(PLOTS, BOUNDARY)
    mission = Mission(segments, heading, max_angle=0.1)
    x, y = segments[0]['start']
    gentle, _ = mission.step((x + 0.1, y, heading))             # 10 cm off the first pass
    mission.options['max_angle'] = 1.5
    strong, _ = mission.step((x + 0.1, y, heading))
    assert abs(gentle[0]) < abs(strong[0]) or abs(gentle[1]) < abs(strong[1])
