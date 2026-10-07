"""Tests for finding the crop rows in LiDAR points and the robot's place across them."""

import math

from agri_ugv_field.plants import row_offsets
from agri_ugv_navigation.rows import local_heights, plant_points, plot_under, \
    row_measurement, row_pattern, wrap
import numpy as np
import pytest

SOY = {'plot_id': 1, 'crop': 'Soybean', 'centre_x': 0.0, 'centre_y': 0.0, 'width': 6.0,
       'length': 8.0, 'heading_deg': 90.0}          # rows run north; 'across' points west


def ground():
    """Return world points of the flat ground around the plot."""
    x, y = np.meshgrid(np.arange(-5.0, 5.0, 0.1), np.arange(-9.0, 9.0, 0.1))
    return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])


def field(plot, seed=0):
    """Return world points of a plot's plants on its rows, and of the ground around it."""
    rng = np.random.default_rng(seed)
    points = []
    for b in row_offsets(plot['width'], 0.45):
        for a in np.arange(-3.9, 3.9, 0.12):         # a plant every 12 cm, 30 leaf points
            side = np.clip(rng.normal(0.0, 0.06, 30), -0.15, 0.15)
            points.append(np.column_stack([-(b + side), a + rng.uniform(-0.03, 0.03, 30),
                                           rng.uniform(0.1, 0.33, 30)]))
    return np.vstack(points + [ground()])


def near_sides(plot, sensor, radius=0.15, top=0.3, seed=0):
    """Return world points on the side of each plant (a ball on its row) facing the sensor."""
    rng = np.random.default_rng(seed)
    points = []
    for b in row_offsets(plot['width'], 0.45):
        for a in np.arange(-3.9, 3.9, 0.12):
            normal = rng.normal(size=(60, 3))
            normal /= np.linalg.norm(normal, axis=1)[:, None]
            surface = np.array([-b, a, top - radius]) + radius * normal
            facing = np.einsum('ij,ij->i', normal, np.array(sensor) - surface) > 0
            points.append(surface[facing & (normal[:, 2] > -0.2)])
    return np.vstack(points + [ground()])


def seen_from(world, pose):
    """Return the points in the robot frame, without those hidden below the LiDAR."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = world[:, :2] - [x, y]
    local = np.column_stack([d @ [c, s], d @ [-s, c], world[:, 2]])
    return local[np.hypot(local[:, 0], local[:, 1]) > 2.7]


def test_rows_are_found_with_their_direction_and_offset():
    xy = np.array([[a, b + k * 0.5] for a in np.arange(3.0, 9.0, 0.1) for k in range(8)
                   for b in (0.2,)])
    turn = math.radians(1.5)
    rotated = xy @ np.array([[math.cos(turn), math.sin(turn)], [-math.sin(turn), math.cos(turn)]])
    angle, offset, strength, shift = row_pattern(rotated, 0.5)
    assert math.degrees(angle) == pytest.approx(1.5, abs=0.02)
    assert offset == pytest.approx(0.2, abs=0.005) and strength > 0.99
    assert shift == pytest.approx(0.0, abs=0.005)


@pytest.mark.parametrize('facing', [0.0, math.pi])
def test_the_robot_is_found_across_the_rows(facing):
    """The estimate puts the robot 8 cm too far left and 0.5 deg turned too far."""
    world = field(SOY)
    true = (0.3, -2.0, math.pi / 2 + facing + math.radians(0.3))
    left = np.array([-math.sin(true[2]), math.cos(true[2])])
    estimate = (true[0] + 0.08 * left[0], true[1] + 0.08 * left[1], true[2] + math.radians(0.5))
    m = row_measurement(seen_from(world, true), estimate, SOY)
    assert m['left'] == pytest.approx(-0.08, abs=0.01)
    assert math.degrees(m['turn']) == pytest.approx(-0.5, abs=0.1)
    m = row_measurement(seen_from(world, true), true, SOY)          # a right estimate
    assert m['left'] == pytest.approx(0.0, abs=0.01) and m['strength'] > 0.4


def test_the_near_side_of_the_plants_is_undone():
    """From an outer pass the LiDAR sees the plants' sides towards it: rows seem nearer."""
    true = (-2.325, 0.5, math.pi / 2 + math.radians(0.3))
    points = seen_from(near_sides(SOY, (true[0], true[1], 2.7)), true)
    m = row_measurement(points, true, SOY)
    assert m['left'] == pytest.approx(0.0, abs=0.025) and 0.02 < m['shift'] < 0.08
    plants = plant_points(points, true, SOY)[:, :2]
    seen, undone = row_pattern(plants, 0.45, max_shift=0.0)[1], row_pattern(plants, 0.45)[1]
    assert abs(wrap(seen - undone, 0.45)) > 0.025                   # 3 cm without it


def test_no_rows_in_cereals_or_in_scattered_points():
    wheat = dict(SOY, crop='Summerwheat')
    pose = (0.3, -2.0, math.pi / 2)
    assert row_measurement(seen_from(field(SOY), pose), pose, wheat) is None
    rng = np.random.default_rng(1)
    scattered = np.column_stack([rng.uniform(-3, 3, 5000), rng.uniform(-4, 4, 5000),
                                 rng.uniform(0.1, 0.3, 5000)])
    assert row_measurement(seen_from(np.vstack([scattered, ground()]), pose), pose, SOY) is None


def test_heights_above_the_local_ground_and_wrapping():
    xy = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
    assert local_heights(xy, np.array([0.2, 0.5, 1.0])).tolist() == pytest.approx([0.0, 0.3, 0.0])
    assert wrap(0.4, 0.45) == pytest.approx(-0.05) and wrap(-0.3, 0.45) == pytest.approx(0.15)


def test_the_plot_under_the_robot_or_just_ahead():
    assert plot_under((0.3, -2.0), [SOY])['plot_id'] == 1
    assert plot_under((0.3, -5.5), [SOY])['plot_id'] == 1           # in the lane, 1.5 m out
    assert plot_under((0.3, -6.5), [SOY]) is None
    assert plot_under((3.5, 0.0), [SOY]) is None                    # beside the plot
