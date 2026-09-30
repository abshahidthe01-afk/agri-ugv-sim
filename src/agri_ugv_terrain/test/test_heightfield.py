"""Tests for the height grid to mesh conversion."""

from xml.etree import ElementTree

from agri_ugv_terrain.heightfield import (mesh_from_heights, model_config, model_sdf,
                                          obj_text, vertex_normals)
import numpy as np
import pytest


def bumpy_grid(rows=5, cols=7):
    """Return a small grid with a gentle wave, so the heights are not all equal."""
    j = np.arange(cols)
    i = np.arange(rows)[:, None]
    return 0.05 * np.sin(j) + 0.03 * np.cos(i)


def face_normals(vertices, faces):
    """Return the (unnormalised) normal of every triangle."""
    p0, p1, p2 = (vertices[faces[:, k]] for k in range(3))
    return np.cross(p1 - p0, p2 - p0)


def test_counts():
    vertices, faces = mesh_from_heights(np.zeros((3, 4)), 0.5)
    assert vertices.shape == (12, 3)
    assert faces.shape == (2 * 2 * 3, 3)  # two triangles per cell, 2 x 3 cells


def test_grid_is_centred_on_origin():
    vertices, _ = mesh_from_heights(np.zeros((3, 5)), 0.5)
    assert vertices[:, 0].min() == pytest.approx(-1.0)
    assert vertices[:, 0].max() == pytest.approx(1.0)
    assert vertices[:, 1].min() == pytest.approx(-0.5)
    assert vertices[:, 1].max() == pytest.approx(0.5)


def test_rows_run_along_y_and_columns_along_x():
    heights = np.zeros((3, 4))
    heights[2, 1] = 0.7  # last row, second column
    vertices, _ = mesh_from_heights(heights, 1.0)
    top = vertices[np.argmax(vertices[:, 2])]
    assert top == pytest.approx([-0.5, 1.0, 0.7])


def test_all_triangles_point_up():
    vertices, faces = mesh_from_heights(bumpy_grid(), 0.5)
    assert np.all(face_normals(vertices, faces)[:, 2] > 0.0)


def test_flat_triangles_cover_the_grid_exactly_once():
    vertices, faces = mesh_from_heights(np.zeros((4, 6)), 0.5)
    total_area = 0.5 * np.linalg.norm(face_normals(vertices, faces), axis=1).sum()
    assert total_area == pytest.approx((3 * 0.5) * (5 * 0.5))


@pytest.mark.parametrize('heights, spacing', [
    (np.zeros(5), 0.5),                  # not a 2D grid
    (np.zeros((1, 5)), 0.5),             # only one row
    (np.array([[0.0, np.nan], [0.0, 0.0]]), 0.5),
    (np.zeros((2, 2)), 0.0),             # zero spacing
    (np.zeros((2, 2)), -1.0),
    (np.zeros((2, 2)), float('nan')),
])
def test_invalid_input_is_rejected(heights, spacing):
    with pytest.raises(ValueError):
        mesh_from_heights(heights, spacing)


def test_flat_ground_normals_point_straight_up():
    normals = vertex_normals(np.zeros((3, 4)), 0.5)
    assert normals.shape == (12, 3)
    assert normals == pytest.approx(np.tile([0.0, 0.0, 1.0], (12, 1)))


def test_slope_normals_tilt_against_the_slope():
    x = np.arange(5) * 0.5
    heights = np.tile(0.1 * x, (3, 1))  # rises 0.1 m per metre along +x
    normals = vertex_normals(heights, 0.5)
    expected = np.array([-0.1, 0.0, 1.0]) / np.sqrt(1.01)
    assert normals == pytest.approx(np.tile(expected, (15, 1)))


def test_normals_have_unit_length():
    normals = vertex_normals(bumpy_grid(), 0.5)
    assert np.linalg.norm(normals, axis=1) == pytest.approx(np.ones(len(normals)))


def test_obj_text():
    heights = bumpy_grid()
    vertices, faces = mesh_from_heights(heights, 0.5)
    normals = vertex_normals(heights, 0.5)
    lines = obj_text(vertices, normals, faces).splitlines()
    v_lines = [line for line in lines if line.startswith('v ')]
    vn_lines = [line for line in lines if line.startswith('vn ')]
    f_lines = [line for line in lines if line.startswith('f ')]
    assert len(v_lines) == len(vertices)
    assert len(vn_lines) == len(vertices)  # Gazebo crashes without a normal per vertex
    assert len(f_lines) == len(faces)
    corners = [corner.split('//') for line in f_lines for corner in line.split()[1:]]
    assert all(v == n for v, n in corners)  # corner i uses vertex i and normal i
    indices = np.array([int(v) for v, _ in corners])
    assert indices.min() == 1 and indices.max() == len(vertices)  # OBJ counts from 1
    first = [float(n) for n in v_lines[0].split()[1:]]
    assert first == pytest.approx(vertices[0], abs=1e-4)


def test_obj_text_needs_one_normal_per_vertex():
    vertices, faces = mesh_from_heights(np.zeros((2, 2)), 0.5)
    with pytest.raises(ValueError):
        obj_text(vertices, vertices[:3], faces)


def test_model_sdf_uses_the_mesh_for_collision_and_visual():
    model = ElementTree.fromstring(model_sdf('flat_mesh', 'flat_mesh.obj')).find('model')
    assert model.get('name') == 'flat_mesh'
    assert model.findtext('static') == 'true'
    uris = [uri.text for uri in model.iter('uri')]
    assert uris == ['meshes/flat_mesh.obj', 'meshes/flat_mesh.obj']
    assert model.find('link/collision/geometry/mesh') is not None


def test_model_config():
    config = ElementTree.fromstring(model_config('flat_mesh', 'A flat test terrain.'))
    assert config.findtext('name') == 'flat_mesh'
    assert config.findtext('sdf') == 'model.sdf'
