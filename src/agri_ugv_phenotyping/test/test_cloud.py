"""Tests for the plots' 3D point clouds (cloud.py)."""

from agri_ugv_phenotyping.cloud import compare, nearest_distances, PlotCloud, read_ply, \
    to_plot_frame, top_surface, voxel_means, write_ply
from agri_ugv_phenotyping.plant_map import FieldMap
import numpy as np
import pytest

# rows run north: along the plot = world y, across it = world -x
BEET = {'type': 'plot', 'plot_id': 198, 'crop': 'Sugar Beet', 'genotype': '', 'variant': '',
        'centre_x': 10.0, 'centre_y': 20.0, 'width': 6.0, 'length': 8.0, 'heading_deg': 90.0}


def test_points_are_kept_in_the_plot_frame_and_merged_per_voxel():
    cloud = PlotCloud(BEET, voxel=0.01)
    xy = np.array([[9.9975, 21.0025], [9.9965, 21.0045], [9.0, 20.0], [30.0, 20.0]])
    assert cloud.add(xy, [0.302, 0.306, 0.1, 0.2]) == 3          # the last is outside
    points, counts = cloud.points()
    order = np.argsort(points[:, 0])
    assert points[order] == pytest.approx(np.array([[0.0, 1.0, 0.1],
                                                    [1.0035, 0.003, 0.304]]), abs=1e-6)
    assert list(counts[order]) == [1, 2]
    assert to_plot_frame(BEET, [[10.0, 21.0]], [0.3])[0] == pytest.approx([1.0, 0.0, 0.3])


def test_merging_in_batches_gives_the_same_cloud():
    rng = np.random.default_rng(1)
    xy = np.column_stack([rng.uniform(8, 12, 5000), rng.uniform(17, 23, 5000)])
    height = rng.uniform(0, 0.4, 5000)
    one, many = PlotCloud(BEET, 0.05, batch=10 ** 6), PlotCloud(BEET, 0.05, batch=7)
    for k in range(0, 5000, 100):
        one.add(xy[k:k + 100], height[k:k + 100])
        many.add(xy[k:k + 100], height[k:k + 100])
    (a, ca), (b, cb) = one.points(), many.points()
    assert ca.sum() == cb.sum() == 5000
    assert a[np.lexsort(a.T)] == pytest.approx(b[np.lexsort(b.T)], abs=1e-5)


def test_a_ply_file_keeps_points_counts_colours_and_comments(tmp_path):
    points = np.array([[0.0, 1.0, 0.2], [-1.5, 2.25, 0.0]])
    write_ply(tmp_path / 'a.ply', points, [3, 1], [[10, 200, 30], [90, 80, 70]],
              comments=['plot 198 (Sugar Beet)'])
    read, fields, comments = read_ply(tmp_path / 'a.ply')
    assert read == pytest.approx(points)
    assert list(fields['count']) == [3, 1] and list(fields['green']) == [200, 80]
    assert comments == ['plot 198 (Sugar Beet)']
    assert (tmp_path / 'a.ply').read_bytes().startswith(b'ply\nformat binary_little_endian')


def test_nearest_distances_are_those_of_a_full_search_within_the_radius():
    rng = np.random.default_rng(2)
    points, queries = rng.uniform(0, 1, (3000, 3)), rng.uniform(-0.1, 1.1, (500, 3))
    full = np.sqrt(((queries[:, None] - points[None]) ** 2).sum(axis=2)).min(axis=1)
    found = nearest_distances(queries, points, 0.06)
    near = full <= 0.06
    assert found[near] == pytest.approx(full[near], abs=1e-12)
    assert np.all(np.isinf(found[~near]))
    assert np.all(np.isinf(nearest_distances(queries, np.zeros((0, 3)), 0.06)))


def test_voxel_means_and_the_top_surface():
    points = np.array([[0.001, 0.001, 0.101], [0.003, 0.002, 0.103], [0.5, 0.5, 0.2]])
    assert sorted(map(tuple, voxel_means(points, 0.01).round(4))) == [
        (0.002, 0.0015, 0.102), (0.5, 0.5, 0.2)]
    top = top_surface(points, cell=0.01)
    assert sorted(top[:, 2]) == [0.103, 0.2]               # the highest point per column


def leaf_sheet(height, step=0.004, across=None):
    """Return a flat 'leaf' 1 m x 1 m at a height, sampled every 'step' (and 'across') m."""
    a, b = np.meshgrid(np.arange(0.0, 1.0, step), np.arange(0.0, 1.0, across or step))
    return np.column_stack([a.ravel(), b.ravel(), np.full(a.size, height)])


def test_a_cloud_on_the_leaves_is_accurate_and_complete_and_a_raised_one_is_not():
    leaves = leaf_sheet(0.3)
    on = leaf_sheet(0.3, step=0.01) + [0.002, 0.002, 0.0]
    good = compare(on, leaves)
    assert good['accuracy_median'] < 0.003 and good['within_1cm'] == 1.0
    assert good['complete_2cm'] == 1.0
    raised = compare(on + [0, 0, 0.03], leaves)               # 3 cm too high
    assert raised['accuracy_median'] == pytest.approx(0.03, abs=0.001)
    assert raised['within_1cm'] == 0.0
    assert raised['complete_2cm'] == 0.0 and raised['complete_5cm'] == 1.0
    sparse = compare(leaf_sheet(0.3, 0.01, across=0.1), leaves)     # profiles 10 cm apart
    assert sparse['complete_2cm'] < 0.5 and sparse['complete_5cm'] > 0.99


def test_a_field_map_with_voxels_saves_a_cloud_per_plot(tmp_path):
    field = FieldMap([BEET], voxel=0.01)
    field.add(np.array([[1.0, 0.0, 0.3], [1.0, 0.5, 0.0]]), (10.0, 20.0, 0.0))
    assert field.save(tmp_path / 'map') == 1
    points, fields, comments = read_ply(tmp_path / 'map_cloud_198.ply')
    assert len(points) == 2 and comments[0].startswith('plot 198 (Sugar Beet)')
    plain = FieldMap([BEET])
    plain.add(np.array([[1.0, 0.0, 0.3]]), (10.0, 20.0, 0.0))
    plain.save(tmp_path / 'plain')
    assert not (tmp_path / 'plain_cloud_198.ply').exists()
