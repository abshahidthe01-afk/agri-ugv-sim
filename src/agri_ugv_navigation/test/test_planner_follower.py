"""Tests for the mission planner and the segment follower."""

import math

from agri_ugv_navigation.follower import follow
from agri_ugv_navigation.planner import field_frame, lanes, pass_offsets, plan_mission, plot_rows
import pytest


def plot(plot_id, east, north, length=8.0, width=6.0):
    """Return a plot whose rows run north, centred at (east, north)."""
    return {'plot_id': plot_id, 'crop': 'Sugar Beet', 'centre_x': east, 'centre_y': north,
            'width': width, 'length': length, 'heading_deg': 90.0}


BOUNDARY = {'centre_x': 7.5, 'centre_y': 0.0, 'width': 30.0, 'length': 40.0,
            'heading_deg': 0.0}
# two plot rows (north at y = 6, south at y = -6, 4 m lane between) of three plots each
PLOTS = [plot(10 + k, 7.5 * k, 6.0) for k in range(3)] + \
        [plot(20 + k, 7.5 * k, -6.0) for k in range(3)]


@pytest.mark.parametrize('width, expected', [(6.0, [-2.25, -0.75, 0.75, 2.25]),
                                             (6.02, [-2.2575, -0.7525, 0.7525, 2.2575]),
                                             (3.0, [-0.75, 0.75])])
def test_passes_cover_the_plot_without_a_pass_for_a_sliver(width, expected):
    assert pass_offsets(width) == pytest.approx(expected)


def test_plots_are_grouped_into_rows_and_lanes_found():
    frame = field_frame(BOUNDARY)
    rows = plot_rows(PLOTS, frame)
    assert [[p['plot_id'] for p in r] for r in rows] == [[10, 11, 12], [20, 21, 22]]
    lines, widths = lanes(rows, frame)
    assert lines == pytest.approx([11.0, 0.0, -11.0])
    assert widths[1] == pytest.approx(4.0)


def test_the_mission_passes_every_plot_and_keeps_the_segments_connected():
    segments, heading = plan_mission(PLOTS, BOUNDARY)
    assert math.degrees(heading) == pytest.approx(90.0)
    passes = [s for s in segments if s['kind'] == 'pass']
    assert len(passes) == 24 and sorted({s['plot_id'] for s in passes}) == [10, 11, 12, 20, 21,
                                                                            22]
    for a, b in zip(segments, segments[1:]):
        assert a['end'] == pytest.approx(b['start'])
    for s in passes:                     # each pass crosses its whole plot, lane to lane
        assert abs(s['end'][1] - s['start'][1]) == pytest.approx(11.0)


def test_moves_between_passes_and_plots_stay_in_the_lanes():
    segments, _ = plan_mission(PLOTS, BOUNDARY)
    for s in segments:
        if s['kind'] != 'pass':
            ys = (s['start'][1], s['end'][1])
            assert all(abs(abs(y) - 6.0) >= 4.0 - 1e-9 or abs(y) > 10.9 for y in ys)


def drive(pose, start, end, heading, steps=3000, dt=0.02):
    """Simulate a perfect four-wheel-steered robot following a segment."""
    x, y, yaw = pose
    for _ in range(steps):
        vx, vy, wz, done = follow((x, y, yaw), start, end, heading, 0.5)
        if done:
            break
        x += (math.cos(yaw) * vx - math.sin(yaw) * vy) * dt
        y += (math.sin(yaw) * vx + math.cos(yaw) * vy) * dt
        yaw += wz * dt
    return x, y, yaw, done


def test_the_follower_pulls_back_onto_the_line_and_stops_at_the_end():
    x, y, yaw, done = drive((0.0, 0.10, math.radians(3)), (0.0, 0.0), (10.0, 0.0), 0.0)
    assert done and x == pytest.approx(10.0, abs=0.03) and y == pytest.approx(0.0, abs=0.005)
    assert yaw == pytest.approx(0.0, abs=0.002)


def test_the_follower_drives_backwards_and_crabs_without_turning():
    x, y, yaw, done = drive((5.0, 0.0, math.pi / 2), (5.0, 0.0), (5.0, -6.0), math.pi / 2)
    assert done and y == pytest.approx(-6.0, abs=0.03) and yaw == pytest.approx(math.pi / 2)
    x, y, yaw, done = drive((0.0, 0.0, math.pi / 2), (0.0, 0.0), (1.5, 0.0), math.pi / 2)
    assert done and x == pytest.approx(1.5, abs=0.03) and yaw == pytest.approx(math.pi / 2)


def test_a_block_of_plots_uses_the_lanes_of_the_whole_field():
    """Plot 11 alone: its passes end in the field's lanes (y 11 and 0), not 1 m past it."""
    def pass_ends(segments):
        return sorted({round(y, 9) for s in segments if s['kind'] == 'pass'
                       for y in (s['start'][1], s['end'][1])})
    assert pass_ends(plan_mission(PLOTS[1:2], BOUNDARY, field=PLOTS)[0]) == [0.0, 11.0]
    assert pass_ends(plan_mission(PLOTS[1:2], BOUNDARY)[0]) == [1.0, 11.0]


def test_a_block_over_two_plot_rows_goes_around_the_end_of_the_field():
    segments, _ = plan_mission([PLOTS[1], PLOTS[4]], BOUNDARY, field=PLOTS)
    assert sorted({s['plot_id'] for s in segments if s['kind'] == 'pass'}) == [11, 21]
    beyond = 7.5 + 10.5 + 0.84 + 0.8                 # past the field's last plot, not plot 11
    corners = [s['end'] for s in segments if s['kind'] == 'transfer']
    assert corners[0] == pytest.approx((beyond, 11.0))
    assert corners[1] == pytest.approx((beyond, -11.0))
    with pytest.raises(ValueError, match='not part of the field'):
        plan_mission([plot(99, 0.0, 0.0)], BOUNDARY, field=PLOTS)


def test_corrections_stay_gentle_while_crabbing():
    """Crabbing after a twist: the sideways and turn parts stay within 0.1 rad of the motion."""
    pose = (0.0, 0.05, math.pi / 2 + math.radians(3.5))         # 5 cm off, turned 3.5 deg
    vx, vy, wz, _ = follow(pose, (0.0, 0.0), (1.5, 0.0), math.pi / 2, 0.5)
    along = 0.5 * math.tan(0.1)
    world_y = math.sin(pose[2]) * vx + math.cos(pose[2]) * vy    # back towards the line
    assert -along - 1e-9 <= world_y < 0 and abs(wz) * 0.75 <= along + 1e-9
    vx, vy, wz, _ = follow(pose, (0.0, 0.0), (1.5, 0.0), math.pi / 2, 0.5, max_angle=1.5)
    assert abs(wz) == pytest.approx(math.radians(3.5))           # without: k_heading x error
