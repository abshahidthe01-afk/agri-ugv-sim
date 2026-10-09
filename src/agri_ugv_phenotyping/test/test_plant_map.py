import json
import math

from agri_ugv_field.plants import row_offsets
from agri_ugv_phenotyping.plant_map import FieldMap, place, PlotMap, profile_points
import numpy as np
import pytest

# rows run north (heading 90 deg): along the rows = +y, across = -x
BEET = {'type': 'plot', 'plot_id': 198, 'crop': 'Sugar Beet', 'genotype': '', 'variant': '',
        'centre_x': 0.0, 'centre_y': 0.0, 'width': 6.0, 'length': 8.0, 'heading_deg': 90.0}
WHEAT = dict(BEET, plot_id=197, crop='Summerwheat', centre_x=10.0)
FAN, CENTRAL = math.radians(35), math.radians(50)


def beet_plants(step=0.01, width=0.2, height=0.3):
    """Return world points (N x 3) over BEET: 'height' within width/2 of a row, else ground."""
    a, b = np.meshgrid(np.arange(-3.995, 4.0, step), np.arange(-2.995, 3.0, step))
    a, b = a.ravel(), b.ravel()
    rows = row_offsets(BEET['width'], 0.50)
    near = np.min(np.abs(b[:, None] - rows[None, :]), axis=1) < width / 2
    return np.column_stack([-b, a, np.where(near, height, 0.0)])     # x = -across, y = along


def robot_frame(world, pose):
    """Return world points (x, y, z) in the robot frame of pose (x, y, yaw)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = world[:, :2] - [x, y]
    return np.column_stack([d @ [c, s], d @ [-s, c], world[:, 2]])


def tilted_scanner():
    """Return the mounting of a left scanner: 1.2 m high, central ray 50 deg towards -y."""
    s, c = math.sin(CENTRAL), math.cos(CENTRAL)
    rotation = np.column_stack([[0.0, -s, -c], [0.0, c, -s], [1.0, 0.0, 0.0]])
    return rotation, np.array([0.0, 0.67, 1.2])


def test_a_profile_over_flat_ground_lies_on_it_without_the_far_side_panel():
    angles = np.linspace(-FAN / 2, FAN / 2, 300)
    from_vertical = CENTRAL - angles
    ground = 1.2 / np.cos(from_vertical)
    panel = 1.40 / np.sin(from_vertical)                 # to y = -0.73, the panel's inner face
    hits_panel = 1.2 - panel * np.cos(from_vertical) >= 0.45
    ranges = np.where(hits_panel, panel, ground)         # over 2 m: no return
    ranges[150] = np.inf                                 # no return
    points = profile_points(ranges, -FAN / 2, FAN / 299, 0.39, 2.0, tilted_scanner())
    assert hits_panel.sum() == 49                        # as in Gazebo: the outermost rays
    assert len(points) == np.sum(~hits_panel & (ground < 2.0)) - 1
    assert np.allclose(points[:, 2], 0.0, atol=1e-9)
    assert np.allclose(points[:, 0], 0.0, atol=1e-9)
    assert points[:, 1].min() == pytest.approx(-0.93, abs=0.01)     # 2 m away
    assert points[:, 1].max() == pytest.approx(0.67 - 1.2 * math.tan(CENTRAL - FAN / 2),
                                               abs=1e-6)


def test_points_high_inside_the_box_are_kept_unless_on_a_side_panel():
    along_y = (np.eye(3), np.array([0.0, 0.0, 0.6]))
    points = profile_points([0.5, 0.74], math.pi / 2, 0.0, 0.1, 2.0, along_y)
    assert points[:, 1] == pytest.approx([0.5])           # a plant, not the panel
    low = (np.eye(3), np.array([0.0, 0.0, 0.3]))          # under the panels' bottom edge
    assert profile_points([0.8], math.pi / 2, 0.0, 0.1, 2.0, low)[:, 1] == pytest.approx([0.8])


def test_place_turns_and_moves_robot_frame_points():
    xy = place(np.array([[1.0, 0.0, 0.3], [0.0, 2.0, 0.0]]), (5.0, 1.0, math.pi / 2))
    assert xy == pytest.approx(np.array([[5.0, 2.0], [3.0, 1.0]]))


def test_a_plot_mapped_from_the_true_pose_has_its_rows_where_the_field_map_puts_them():
    m = PlotMap(BEET, cell=0.05)
    world = beet_plants()
    assert m.add(world[:, :2], world[:, 2]) == len(world)
    t = m.traits()
    assert m.top.shape == (160, 120)
    assert t['seen'] == pytest.approx(1.0)
    assert t['cover'] == pytest.approx(0.4, abs=0.01)    # 0.2 m of every 0.5 m
    assert t['height'] == pytest.approx(0.3)
    assert t['rows_offset'] == pytest.approx(0.0, abs=0.003)
    assert t['rows_sharpness'] == pytest.approx(0.77, abs=0.02)
    assert t['points'] == len(world)


def test_a_pose_off_across_the_rows_moves_the_rows_in_the_map():
    true, off = (0.3, 0.0, math.pi / 2), (0.3 - 0.04, 0.0, math.pi / 2)   # 4 cm to 'across'
    world = beet_plants()
    field = FieldMap([BEET, WHEAT])
    field.add(robot_frame(world, true), off, t=12.5)
    assert field.traits()[198]['rows_offset'] == pytest.approx(0.04, abs=0.01)
    assert list(field.traits()) == [198]
    assert field.changed == {198: 12.5}


def test_two_passes_that_disagree_blur_the_rows():
    world = beet_plants()
    pose = (0.0, 0.0, math.pi / 2)
    one, two = FieldMap([BEET]), FieldMap([BEET])
    one.add(robot_frame(world, pose), pose)
    two.add(robot_frame(world, pose), pose)
    two.add(robot_frame(world, pose), (0.125, 0.0, math.pi / 2))   # a quarter row spacing
    sharp, blurred = one.traits()[198], two.traits()[198]
    assert blurred['rows_sharpness'] < 0.75 * sharp['rows_sharpness']
    assert blurred['cover'] > 1.4 * sharp['cover']
    assert blurred['height'] == pytest.approx(sharp['height'])


def test_cereals_have_no_rows_to_measure_and_ground_has_no_plants():
    m = PlotMap(WHEAT)
    a, b = np.meshgrid(np.arange(-3.9, 4.0, 0.02), np.arange(-2.9, 3.0, 0.02))
    xy = np.column_stack([10.0 - b.ravel(), a.ravel()])
    m.add(xy, np.full(len(xy), 0.7))
    t = m.traits()
    assert t['rows_offset'] is None and t['cover'] == pytest.approx(1.0)
    bare = PlotMap(BEET)
    bare.add(np.zeros((5, 2)), np.full(5, 0.02))
    t = bare.traits()
    assert t['cover'] == 0.0 and t['height'] is None and t['rows_offset'] is None


def test_points_outside_the_plots_and_profiles_far_away_are_left_out():
    field = FieldMap([BEET, WHEAT])
    points = np.array([[0.0, 4.0, 0.3], [0.0, 3.0, 0.3], [0.0, -2.0, 0.3]])
    # robot at x = 5 facing north: robot y (left) is world -x; points at x = 1, 2 and 7
    assert field.add(points, (5.0, 0.0, math.pi / 2)) == 2
    assert field.add(points, (50.0, 0.0, 0.0)) == 0
    assert field.add(np.zeros((0, 3)), (5.0, 0.0, 0.0)) == 0


def test_the_plant_cells_are_where_the_plants_are():
    m = PlotMap(BEET)
    xy = np.array([[-0.26, 1.0], [0.74, 1.0], [0.24, 1.0], [0.24, 1.01]])
    m.add(xy, np.array([0.3, 0.02, 0.02, 0.2]))        # a plant, ground, ground under a leaf
    cells, height = m.plant_cells()
    order = np.argsort(cells[:, 0])
    assert cells[order] == pytest.approx(np.array([[-0.275, 1.025], [0.225, 1.025]]))
    assert height[order] == pytest.approx([0.3, 0.2])
    assert m.plant_count.sum() == 2 and m.count.sum() == 4


def test_the_map_is_saved_with_its_traits(tmp_path):
    field = FieldMap([BEET, WHEAT], cell=0.1)
    world = beet_plants(step=0.05)
    field.add(robot_frame(world, (0.0, 0.0, 0.0)), (0.0, 0.0, 0.0))
    assert field.save(tmp_path / 'map') == 1
    saved = np.load(tmp_path / 'map.npz')
    assert float(saved['cell']) == 0.1
    assert [p['plot_id'] for p in json.loads(str(saved['plots']))] == [198]
    assert saved['top_198'].shape == (80, 60) and saved['count_198'].sum() == len(world)
    traits = json.loads((tmp_path / 'map.json').read_text())
    assert traits[0]['plot'] == 198 and traits[0]['height'] == pytest.approx(0.3)


def test_a_cell_must_have_a_size():
    with pytest.raises(ValueError):
        PlotMap(BEET, cell=0.0)
