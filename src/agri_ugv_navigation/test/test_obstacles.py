"""Tests for finding obstacles in the robot's way in LiDAR points."""

import math
from types import SimpleNamespace

from agri_ugv_navigation.obstacles import cloud_points, mount, obstacle_gap
import numpy as np
import pytest

LIDAR_HEIGHT = 2.70


def visible(points):
    """Keep the points the LiDAR on the mast sees: at most 45 degrees below it."""
    points = np.asarray(points, dtype=float)
    return points[LIDAR_HEIGHT - points[:, 2] <= np.hypot(points[:, 0], points[:, 1])]


def ground(size=9.0, step=0.1):
    """Return flat ground points around the robot."""
    xs = np.arange(-size, size, step)
    x, y = np.meshgrid(xs, xs)
    return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])


def person(x, y, tall=1.7, radius=0.25, bottom=0.0):
    """Return points on the robot-facing half of a standing cylinder."""
    angles = np.linspace(-math.pi / 2, math.pi / 2, 15) + math.atan2(-y, -x)
    heights = np.arange(bottom, tall + 1e-9, 0.05)
    a, h = np.meshgrid(angles, heights)
    return np.column_stack([x + radius * np.cos(a.ravel()), y + radius * np.sin(a.ravel()),
                            h.ravel()])


def scene(*parts):
    return visible(np.vstack([ground()] + list(parts)))


def crop_rows(top=0.84, spacing=0.75):
    """Return points on tall crop rows (sweet corn, the tallest plants) all around."""
    along = np.arange(-9, 9, 0.05)
    rows = [np.column_stack([along, np.full_like(along, y), np.full_like(along, z)])
            for y in np.arange(-9, 9, spacing) for z in (0.3, 0.6, top)]
    return np.vstack(rows)


def test_a_person_ahead_is_found_and_the_crop_is_not():
    assert obstacle_gap(scene(crop_rows()), (1.0, 0.0)) is None
    gap = obstacle_gap(scene(crop_rows(), person(3.5, 0.0)), (1.0, 0.0))
    assert gap == pytest.approx(3.5 - 0.25 - 0.76, abs=0.02)


def test_only_the_way_the_robot_moves_counts():
    behind = scene(person(-3.5, 0.0))
    assert obstacle_gap(behind, (1.0, 0.0)) is None
    assert obstacle_gap(behind, (-1.0, 0.0)) == pytest.approx(2.49, abs=0.02)
    beside = scene(person(2.0, 1.5))                    # 1.25 m out: the strip ends at 1.04
    assert obstacle_gap(beside, (1.0, 0.0)) is None
    sideways = obstacle_gap(scene(person(0.0, 3.5)), (0.0, 1.0))          # crabbing
    assert sideways == pytest.approx(3.25 - 0.84, abs=0.02)
    assert obstacle_gap(scene(person(5.0, 0.0)), (1.0, 0.0)) is None     # beyond reach
    assert obstacle_gap(scene(person(3.5, 0.0)), (0.0, 0.0)) is None     # not moving


def test_the_robot_itself_is_no_obstacle():
    """The LiDAR sees the ends of the rolled-up curtains, 1 cm beyond the tyres."""
    rng = np.random.default_rng(1)
    ends = np.array([[x, y, 1.78] for x in (0.77, -0.77) for y in (0.73, -0.73)])
    curtains = np.repeat(ends, 20, axis=0) + rng.normal(0.0, 0.01, (80, 3))
    for direction in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.7, 0.7)]:
        assert obstacle_gap(np.vstack([scene(), curtains]), direction) is None


def test_close_to_the_robot_where_the_ground_is_hidden():
    close = scene(person(1.8, 0.0))                     # only its upper part is in view
    assert np.min(close[np.hypot(close[:, 0], close[:, 1]) < 2.1][:, 2]) > 0.85
    assert obstacle_gap(close, (1.0, 0.0)) == pytest.approx(1.8 - 0.25 - 0.76, abs=0.02)


def test_in_a_dense_crop_the_canopy_is_the_ground():
    x, y = np.meshgrid(np.arange(2.0, 8.0, 0.05), np.arange(-3.0, 3.0, 0.05))
    canopy = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.7)])
    assert obstacle_gap(visible(canopy), (1.0, 0.0)) is None
    tall = visible(np.vstack([canopy, person(4.0, 0.0, tall=1.8, bottom=0.7)]))
    assert obstacle_gap(tall, (1.0, 0.0)) == pytest.approx(4.0 - 0.25 - 0.76, abs=0.02)


def test_the_lidar_mount_follows_the_joints():
    urdf = ('<robot name="r"><link name="base_footprint"/><link name="base_link"/>'
            '<link name="lidar_link"/>'
            '<joint name="a" type="fixed"><parent link="base_footprint"/>'
            '<child link="base_link"/><origin xyz="0 0 0.205" rpy="0 0 0"/></joint>'
            '<joint name="b" type="fixed"><parent link="base_link"/>'
            '<child link="lidar_link"/><origin xyz="0.1 0 2.495" rpy="0 0 1.5707963"/>'
            '</joint></robot>')
    rotation, translation = mount(urdf)
    assert translation == pytest.approx([0.1, 0.0, 2.70])
    assert rotation @ [1.0, 0.0, 0.0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
    with pytest.raises(ValueError, match='no chain'):
        mount(urdf, link='camera_link')


def test_cloud_points_drop_rays_without_a_return():
    layout = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring'],
                       'formats': ['<f4', '<f4', '<f4', '<f4', '<u2'],
                       'offsets': [0, 4, 8, 12, 16], 'itemsize': 18})
    data = np.zeros(3, layout)
    data['x'], data['y'], data['z'] = [1.0, np.inf, 4.0], [2.0, np.inf, 5.0], [3.0, 0.0, -6.0]
    fields = [SimpleNamespace(name=n, offset=o)
              for n, o in [('x', 0), ('y', 4), ('z', 8), ('intensity', 12), ('ring', 16)]]
    points = cloud_points(fields, 18, 3, data.tobytes())
    assert points.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, -6.0]]
