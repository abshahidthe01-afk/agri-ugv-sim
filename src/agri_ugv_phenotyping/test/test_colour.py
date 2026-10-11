"""Tests for colouring the scanners' points from the camera pictures (colour.py)."""

from agri_ugv_phenotyping.cloud import display_rows, PlotCloud, read_ply
from agri_ugv_phenotyping.colour import Colourist, excess_green, GREEN, nearest_in_front, \
    project, robot_points
import numpy as np
import pytest

# a camera 1.5 m above the robot's centre looking straight down; the top of the picture
# towards the robot's front (+x), its left towards the robot's left (+y)
DOWN = (np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        np.array([0.0, 0.0, 1.5]))
K = [200.0, 0.0, 100.0, 0.0, 200.0, 75.0, 0.0, 0.0, 1.0]       # a 200 x 150 picture
LEAF, SOIL = (40, 120, 30), (110, 85, 60)


def test_projection_of_points_ahead_left_and_up():
    u, v, depth = project(np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.0, 0.0],
                                    [0.0, 0.0, 0.5]]), DOWN, K)
    assert (u[0], v[0], depth[0]) == pytest.approx((100.0, 75.0, 1.5))   # straight below
    assert u[1] < 100.0                          # to the robot's left: left in the picture
    assert v[2] < 75.0                           # ahead of the robot: up in the picture
    assert depth[3] == pytest.approx(1.0)        # half a metre up: nearer
    assert robot_points([[1.0, 2.0]], [0.3], (1.0, 1.0, np.pi / 2))[0] == \
        pytest.approx([1.0, 0.0, 0.3])           # 1 m north of a robot facing north


def test_a_point_behind_another_is_hidden_also_through_a_small_gap():
    u = np.array([50.0, 50.0, 62.0, 150.0])
    v = np.array([50.0, 50.0, 50.0, 50.0])
    depth = np.array([1.0, 1.2, 1.2, 1.2])
    seen = nearest_in_front(u, v, depth, (150, 200), block=4, window=7, tolerance=0.05)
    assert list(seen) == [True, False, False, True]      # 12 px away is still behind it
    assert list(nearest_in_front(u, v, depth, (150, 200), window=1)) == \
        [True, False, True, True]                         # without the closing it is not


def picture():
    """Return what DOWN sees: soil, and a leaf 30 cm above the centre (its edge 12 cm out)."""
    rows, cols = np.mgrid[0:150, 0:200]
    ahead, left = (75 - rows) / 200.0, (100 - cols) / 200.0     # per metre of depth
    on_leaf = (np.abs(ahead * 1.2) <= 0.12) & (np.abs(left * 1.2) <= 0.12)
    image = np.zeros((150, 200, 3), dtype=np.uint8)
    image[:] = SOIL
    image[on_leaf] = LEAF
    return image


def scene():
    """Return the profiles' points: soil every 5 cm along x, 1 cm across; the leaf."""
    x, y = np.meshgrid(np.arange(-0.5, 0.51, 0.05), np.arange(-0.5, 0.51, 0.01))
    soil = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    x, y = np.meshgrid(np.arange(-0.1, 0.101, 0.05), np.arange(-0.1, 0.101, 0.01))
    leaf = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.3)])
    return soil, leaf


def test_the_leaf_is_green_and_the_soil_under_it_gets_no_colour():
    soil, leaf = scene()
    points = np.vstack([soil, leaf])
    colourist = Colourist({16: (DOWN, K)}, delay=1.0)
    colourist.add_picture(0.0, 16, picture())
    colourist.add_profile(0.0, points[:, :2], points[:, 2])
    assert colourist.ready(lambda t: (0.0, 0.0, 0.0)) == []        # waits for later profiles
    colourist.add_profile(1.5, np.zeros((0, 2)), np.zeros(0))
    [(xy, height, colours)] = colourist.ready(lambda t: (0.0, 0.0, 0.0))
    green = excess_green(colours) > GREEN
    on_leaf = height > 0.2
    assert on_leaf.sum() == len(leaf) and green[on_leaf].all()
    under = ~on_leaf & np.all(np.abs(xy) <= 0.1, axis=1)
    assert not under.any()                                        # hidden by the leaf
    far = ~on_leaf & np.any(np.abs(xy) >= 0.25, axis=1)
    assert far.sum() > 0.8 * np.sum(np.any(np.abs(soil[:, :2]) >= 0.25, axis=1))
    assert not green[far].any()
    assert len(colourist.pictures) == 0                           # used up


def test_old_points_are_forgotten_and_unknown_cameras_ignored():
    colourist = Colourist({16: (DOWN, K)}, keep=3.0)
    colourist.add_profile(0.0, np.zeros((5, 2)), np.zeros(5))
    colourist.add_profile(5.0, np.ones((5, 2)), np.zeros(5))
    assert [p[0] for p in colourist.profiles] == [5.0]
    colourist.add_picture(4.0, 17, picture())
    assert len(colourist.pictures) == 0
    for t in range(10):                       # pictures without profiles do not pile up
        colourist.add_picture(10.0 + t, 16, picture())
    assert [p[0] for p in colourist.pictures] == [15.0, 16.0, 17.0, 18.0, 19.0]


def test_excess_green_tells_leaves_from_soil():
    value = excess_green(np.array([LEAF, SOIL, (0, 0, 0)]))
    assert value[0] > GREEN > value[1] and value[2] == 0.0


def test_a_cloud_keeps_the_mean_colour_and_views_per_voxel(tmp_path):
    plot = {'type': 'plot', 'plot_id': 198, 'crop': 'Sugar Beet', 'genotype': '',
            'variant': '', 'centre_x': 0.0, 'centre_y': 0.0, 'width': 6.0, 'length': 8.0,
            'heading_deg': 0.0}
    cloud = PlotCloud(plot, voxel=0.01)
    xy = np.array([[0.0025, 0.0025], [1.0025, 1.0025]])
    cloud.add(xy, [0.3025, 0.0025])
    cloud.add_colours(xy[:1], [0.3025], [(40, 120, 30)])
    cloud.add_colours(xy[:1], [0.3025], [(60, 100, 50)])
    points, _ = cloud.points()
    colours, views = cloud.colours()
    leaf = np.argmax(points[:, 2])
    assert colours[leaf] == pytest.approx([50, 110, 40]) and views[leaf] == 2
    assert views[1 - leaf] == 0
    cloud.save(tmp_path / 'c.ply')
    _, fields, comments = read_ply(tmp_path / 'c.ply')
    assert sorted(fields['views']) == [0, 2]
    grey = fields['views'] == 0
    assert fields['red'][grey][0] == fields['green'][grey][0] == 128
    assert any(c.startswith('colour:') for c in comments)


def test_a_cloud_to_show_lies_in_the_world_brighter_and_grey_where_unseen():
    plot = {'type': 'plot', 'plot_id': 198, 'crop': 'Sugar Beet', 'genotype': '',
            'variant': '', 'centre_x': 10.0, 'centre_y': 5.0, 'width': 6.0, 'length': 8.0,
            'heading_deg': 90.0}                                   # rows run north
    cloud = PlotCloud(plot, voxel=0.01)
    xy = np.array([[10.0025, 5.0025], [11.0025, 6.0025], [10.5025, 4.0025]])
    cloud.add(xy, [0.3025, 0.0025, 0.1025])
    cloud.add([[30.0, 5.0]], [0.0])                                # outside the plot
    assert cloud.version == 1
    cloud.add_colours(xy[:2], [0.3025, 0.0025], [(40, 120, 30), (200, 100, 50)])
    assert cloud.version == 2
    points, colours = cloud.world(gain=2.0)
    order = np.argsort(points[:, 2])[::-1]                         # leaf, middle, soil
    assert points[order] == pytest.approx(np.column_stack([xy[[0, 2, 1]],
                                                           [0.3025, 0.1025, 0.0025]]))
    assert colours[order].tolist() == [[80, 240, 60], [128, 128, 128], [255, 200, 100]]
    rows = display_rows(points[order], colours[order])
    assert rows.itemsize == 16 and rows['rgb'][0] == (80 << 16) | (240 << 8) | 60
    assert rows['z'][0] == pytest.approx(0.3025)
    assert len(display_rows(np.zeros((0, 3)), np.zeros((0, 3)))) == 0
