import json

from agri_ugv_field.layout import field_layout, layout_csv
from agri_ugv_phenotyping.cloud import write_ply
from agri_ugv_phenotyping.plant_map import FieldMap
from agri_ugv_phenotyping.truth import canopy_top, leaf_points, main, map_traits, traits, \
    true_canopy, true_leaves
import numpy as np
import pytest

BEET = {'type': 'plot', 'plot_id': 198, 'crop': 'Sugar Beet', 'genotype': '', 'variant': '',
        'centre_x': 0.0, 'centre_y': 0.0, 'width': 6.0, 'length': 8.0, 'heading_deg': 90.0}


def flat(x, y):
    return np.zeros(np.shape(x))


def square(x0, y0, size, z, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
    """Return (vertices, uvs, faces) of a horizontal square leaf."""
    vertices = np.array([[x0, y0, z], [x0 + size, y0, z], [x0 + size, y0 + size, z],
                         [x0, y0 + size, z]])
    return vertices, np.array(uv, dtype=float), np.array([[0, 1, 2], [0, 2, 3]])


def test_leaf_points_cover_the_triangles_and_skip_transparent_texels():
    points = leaf_points(*square(0.0, 0.0, 0.1, 0.3), step=0.01)
    assert points[:, 2] == pytest.approx(0.3)
    assert points[:, :2].min(axis=0) == pytest.approx([0, 0])
    assert points[:, :2].max(axis=0) == pytest.approx([0.1, 0.1])
    left_opaque = np.array([[True, False], [True, False]])        # u < 0.5 opaque
    half = leaf_points(*square(0.0, 0.0, 0.1, 0.3), alpha=left_opaque, step=0.01)
    assert half[:, 0].max() < 0.05 + 1e-9
    assert 0.4 < len(half) / len(points) < 0.6


def test_the_canopy_of_a_plot_has_its_cover_and_height():
    # a 1 m x 6 m leaf at 0.3 m across the plot's middle: 1/8 of the plot (rows run north)
    vertices, uvs, faces = square(-3.0, -0.5, 6.0, 0.3)
    vertices[:, 1] = np.where(vertices[:, 1] > 0, 0.5, -0.5)
    top = canopy_top(BEET, leaf_points(vertices, uvs, faces, step=0.02), flat)
    t = traits(top)
    assert top.shape == (160, 120)
    assert t['cover'] == pytest.approx(1 / 8, abs=0.01)
    assert t['height'] == pytest.approx(0.3)
    assert traits(np.full((4, 4), -np.inf))['height'] is None


def test_a_saved_map_is_compared_on_the_cells_it_has_seen(tmp_path):
    field = FieldMap([BEET])
    xy = np.array([[0.0, 0.01], [0.0, 0.12], [1.0, 1.01]])
    field.add(np.column_stack([xy, [0.3, 0.02, 0.2]]), (0.0, 0.0, 0.0))
    field.save(tmp_path / 'map')
    seen, measured = map_traits(np.load(tmp_path / 'map.npz'), 198)
    assert seen.sum() == 3 and measured['cover'] == pytest.approx(2 / 3)
    assert measured['height'] == pytest.approx(0.295, abs=0.01)
    top = np.full(seen.shape, 0.4)
    assert traits(top, cells=seen)['cover'] == 1.0


def rectangle(cx, cy, length, width, heading_deg):
    h = np.radians(heading_deg)
    u, v = np.array([np.cos(h), np.sin(h)]), np.array([-np.sin(h), np.cos(h)])
    corners = [np.array([cx, cy]) + a * length / 2 * u + b * width / 2 * v
               for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    return np.array(corners + corners[:1])


def test_the_true_canopy_grows_the_same_plants_again_and_main_prints_it(tmp_path, capsys):
    pytest.importorskip('PIL')
    from agri_ugv_terrain.make_terrain import write_model

    edge = {'plot_ID': 0, 'crop': '', 'genotype': '', 'var': ''}
    shapes = [(rectangle(0.0, 0.0, 30.0, 12.0, -5.5), edge)]
    shapes += [(rectangle(-3.5 + 7.0 * k, 0.0, 7.7, 6.0, 84.5),
                {'plot_ID': 100 + k, 'crop': crop, 'genotype': '', 'var': ''})
               for k, crop in enumerate(['Sugar Beet', 'Soybean'])]
    boundary, plots = field_layout(shapes, lambda points: points)
    (tmp_path / 'plots.csv').write_text(layout_csv([boundary] + plots))
    write_model(tmp_path, 'ground', np.zeros((41, 81)), 0.5, 'flat test ground')
    one = true_canopy(plots, flat, {101})
    two = true_canopy(plots, flat, {100, 101})
    assert list(one) == [101] and np.array_equal(one[101], two[101])
    beet = traits(two[100])
    assert 0.3 < beet['cover'] < 0.95
    assert 0.25 < beet['height'] <= 0.30 * 1.2 + 1e-6
    main(['--layout', str(tmp_path / 'plots.csv'), '--terrain-model', str(tmp_path / 'ground'),
          '--plots', '100,101'])
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith('plot 100 (Sugar Beet): cover')
    assert out[1].startswith('plot 101 (Soybean): cover')
    field = FieldMap([p for p in plots if p['plot_id'] == 100])
    field.add(np.array([[0.0, 0.0, 0.3]]), (plots[0]['centre_x'], plots[0]['centre_y'], 0.0))
    field.save(tmp_path / 'map')
    main(['--layout', str(tmp_path / 'plots.csv'), '--terrain-model', str(tmp_path / 'ground'),
          '--map', str(tmp_path / 'map.npz')])
    out = capsys.readouterr().out
    assert 'map minus truth: cover' in out and json.loads(str(np.load(
        tmp_path / 'map.npz')['plots']))[0]['plot_id'] == 100
    # the true leaves in the plot's frame, and a cloud that is exactly them
    leaves = true_leaves(plots, flat, {100})[100]
    assert 0.25 < leaves[:, 2].max() <= 0.30 * 1.2 + 1e-6      # heights above the ground
    assert np.abs(leaves[:, 1]).max() < 3.0 + 0.3               # across the 6 m wide plot
    part = leaves[np.abs(leaves[:, 0]) < 1.0]                    # 2 m of the plot's length
    write_ply(tmp_path / 'cloud.ply', part, comments=['plot 100 (Sugar Beet), voxel 0.01 m'])
    main(['--layout', str(tmp_path / 'plots.csv'), '--terrain-model', str(tmp_path / 'ground'),
          '--cloud', str(tmp_path / 'cloud.ply')])
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith('plot 100 (Sugar Beet), 3D cloud:')
    assert out[1].startswith('  accuracy: to the true leaves median 0.0 cm')
    assert '100 % has a point within 2 cm' in out[2]
