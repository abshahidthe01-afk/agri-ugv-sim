"""Tests for the mission planner and the segment follower."""

import math

from agri_ugv_field.plants import row_offsets
from agri_ugv_navigation.follower import follow
from agri_ugv_navigation.planner import (covers, field_frame, lane_widths, pass_offsets,
                                         plan_mission, plot_offsets, plot_rows,
                                         row_aware_offsets, wheel_room)
import numpy as np
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
# the same with a lane only 1.8 m wide: the robot (1.52 m over its tyres) just fits
NARROW = [plot(10 + k, 7.5 * k, 4.9) for k in range(3)] + \
         [plot(20 + k, 7.5 * k, -4.9) for k in range(3)]
REACH = 0.76 + 0.3            # robot centre to tyre edge, plus the clearance kept in lanes


def clearance(segments, plots, kinds=('shift', 'transfer')):
    """Return the least gap between the robot's tyres and any plot along the moves."""
    gaps = []
    for s in segments:
        if s['kind'] in kinds:
            for t in np.linspace(0.0, 1.0, 41):
                x, y = np.array(s['start']) + t * (np.array(s['end']) - np.array(s['start']))
                for p in plots:                  # robot facing north: 0.84 to the side
                    dx = max(0.0, abs(x - p['centre_x']) - 0.84 - p['width'] / 2)
                    dy = max(0.0, abs(y - p['centre_y']) - 0.76 - p['length'] / 2)
                    gaps.append(math.hypot(dx, dy))
    return min(gaps)


@pytest.mark.parametrize('width, expected', [(6.0, [-2.25, -0.75, 0.75, 2.25]),
                                             (6.02, [-2.26, -0.75333, 0.75333, 2.26]),
                                             (3.0, [-0.75, 0.75]),
                                             (4.5, [-1.5, -0.5, 0.5, 1.5])])
def test_passes_cover_the_plot_in_an_even_number_without_one_for_a_sliver(width, expected):
    assert pass_offsets(width) == pytest.approx(expected, abs=1e-5)
    assert covers(pass_offsets(width), width)


def test_wheels_run_between_the_rows_of_row_crops():
    """Soybean rows 0.45 m apart: no 4 passes of a 1.5 m track keep off the rows, 6 do."""
    soy = plot_offsets({'crop': 'Soybean', 'width': 6.0})
    assert soy == pytest.approx([-2.325, -1.35, -0.45, 0.45, 1.35, 2.325])
    rows = row_offsets(6.0, 0.45)
    assert min(wheel_room(o, rows) for o in soy) == pytest.approx(0.15)
    assert covers(soy, 6.0)
    for crop in ('Sugar Beet', 'Potato'):          # rows 0.5 and 0.75 m apart: already clear
        assert plot_offsets({'crop': crop, 'width': 6.0}) == pytest.approx(pass_offsets(6.0))
    assert plot_offsets({'crop': 'Summerwheat', 'width': 6.02}) == pass_offsets(6.02)


def test_rows_too_close_for_the_wheels_still_give_covering_passes():
    rows = np.arange(-2.95, 3.0, 0.1)               # 0.1 m apart: no room for a tyre
    offsets = row_aware_offsets(6.0, rows)
    assert covers(offsets, 6.0) and len(offsets) in (4, 6, 8)


def test_plots_are_grouped_into_rows_and_lanes_found():
    frame = field_frame(BOUNDARY)
    rows = plot_rows(PLOTS, frame)
    assert [[p['plot_id'] for p in r] for r in rows] == [[10, 11, 12], [20, 21, 22]]
    assert lane_widths(rows, frame) == pytest.approx([math.inf, 4.0, math.inf])


def test_the_mission_passes_every_plot_and_keeps_the_segments_connected():
    segments, heading = plan_mission(PLOTS, BOUNDARY)
    assert math.degrees(heading) == pytest.approx(90.0)
    passes = [s for s in segments if s['kind'] == 'pass']
    assert len(passes) == 24 and sorted({s['plot_id'] for s in passes}) == [10, 11, 12, 20, 21,
                                                                            22]
    for a, b in zip(segments, segments[1:]):
        assert a['end'] == pytest.approx(b['start'])
    for s in passes:                     # each pass crosses its whole plot, lane to lane
        assert abs(s['end'][1] - s['start'][1]) == pytest.approx(8.0 + 2 * REACH)


def test_moves_between_passes_and_plots_keep_the_clearance():
    segments, _ = plan_mission(PLOTS, BOUNDARY)
    assert clearance(segments, PLOTS) == pytest.approx(0.3)
    segments, _ = plan_mission(NARROW, BOUNDARY)          # 0.28 m to spare: half each side
    assert clearance(segments, NARROW, kinds=('shift',)) == pytest.approx(0.14)
    assert clearance(segments, NARROW) == pytest.approx(0.14)


def test_a_plot_end_that_sticks_out_moves_only_the_stops_at_that_plot():
    """Plot 11 reaches 10 cm further into the narrow lane: there the robot keeps 5 cm away."""
    longer = [dict(p, length=8.1, centre_y=4.85) if p['plot_id'] == 11 else p for p in NARROW]
    segments, _ = plan_mission(longer, BOUNDARY)
    stops = {}
    for a, b in zip(segments, segments[1:]):
        if a['kind'] == 'pass' and b['kind'] == 'shift' and abs(a['end'][1]) < 0.9:
            stops.setdefault(a['plot_id'], set()).add(round(a['end'][1], 9))
    assert stops == {10: {0.0}, 11: {-0.05}, 12: {0.0}, 20: {0.0}, 21: {-0.05}, 22: {0.0}}
    assert clearance(segments, longer) == pytest.approx(0.09)


def test_a_lane_too_narrow_for_the_robot_is_refused():
    tight = [dict(p, centre_y=math.copysign(4.7, p['centre_y'])) for p in NARROW]   # 1.4 m
    with pytest.raises(ValueError, match='does not fit into the lane at the end of plot 1'):
        plan_mission(tight, BOUNDARY)


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
    """Plot 11 alone: in the narrow lane it keeps to the middle, as the whole mission does."""
    def pass_ends(segments):
        return sorted({round(y, 9) for s in segments if s['kind'] == 'pass'
                       for y in (s['start'][1], s['end'][1])})
    assert pass_ends(plan_mission(NARROW[1:2], BOUNDARY, field=NARROW)[0]) == \
        pytest.approx([0.0, 8.9 + REACH])
    assert pass_ends(plan_mission(NARROW[1:2], BOUNDARY)[0]) == \
        pytest.approx([0.9 - REACH, 8.9 + REACH])


def test_a_block_over_two_plot_rows_goes_around_the_end_of_the_field():
    segments, _ = plan_mission([PLOTS[1], PLOTS[4]], BOUNDARY, field=PLOTS)
    assert sorted({s['plot_id'] for s in segments if s['kind'] == 'pass'}) == [11, 21]
    beyond = 7.5 + 10.5 + 0.84 + 0.8                 # past the field's last plot, not plot 11
    corners = [s['end'] for s in segments if s['kind'] == 'transfer']
    assert corners[0] == pytest.approx((beyond, 10.0 + REACH))
    assert corners[1] == pytest.approx((beyond, -10.0 - REACH))
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
