"""Tests for crop rows, plant positions and leaf meshes."""

import math

from agri_ugv_field.plants import (CROPS, grow, leaf_outline, leaves, merge, obj_text, PARTS,
                                   plant, plant_offsets, plot_mesh, row_offsets, row_strip,
                                   two_sided, vertex_normals)
from agri_ugv_field.textures import TEXTURES
import numpy as np
import pytest

PLOT = {'centre_x': 10.0, 'centre_y': -5.0, 'width': 6.0, 'length': 7.7, 'heading_deg': 84.5}


def flat(x, y):
    """Return a flat ground at height 0.5 m."""
    return np.full(np.shape(x), 0.5)


def plot_coordinates(vertices):
    """Return vertex positions along and across PLOT, relative to its centre."""
    h = math.radians(PLOT['heading_deg'])
    d = vertices[:, :2] - [PLOT['centre_x'], PLOT['centre_y']]
    return d @ [math.cos(h), math.sin(h)], d @ [-math.sin(h), math.cos(h)]


@pytest.mark.parametrize('width', [5.979, 6.0, 6.02])
@pytest.mark.parametrize('spacing, rows', [(0.5, 12), (0.75, 8), (0.45, 13), (0.125, 48)])
def test_rows_fill_the_plot_width_and_are_centred(width, spacing, rows):
    offsets = row_offsets(width, spacing)
    assert len(offsets) == rows
    assert offsets.mean() == pytest.approx(0.0, abs=1e-12)
    assert np.diff(offsets) == pytest.approx(np.full(rows - 1, spacing))


def test_plants_are_jittered_thinned_and_repeatable():
    along = plant_offsets(7.7, 0.2, np.random.default_rng(1))
    assert np.abs(along).max() < 7.7 / 2
    assert 30 <= len(along) <= 38                # 38 places, about 5 % missing
    assert along == pytest.approx(plant_offsets(7.7, 0.2, np.random.default_rng(1)))


def test_leaf_strips_have_the_requested_length_direction_and_texture_range():
    vertices, uvs, faces = leaves([0.0, math.pi / 2], 1.0, 0.2, 0.0, 0.0, segments=2,
                                  u=(0.25, 0.75))
    assert vertices.shape == (12, 3) and faces.shape == (8, 3)
    tips = vertices.reshape(2, 3, 2, 3)[:, -1].mean(axis=1)     # middle of each leaf tip
    assert tips == pytest.approx(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), abs=1e-12)
    assert sorted(set(uvs[:, 0])) == [0.25, 0.75]
    assert sorted(set(uvs[:, 1])) == [0.0, 0.5, 1.0]
    upright = leaves(0.0, 0.8, 0.1, math.pi / 2, 0.0, base_z=0.3)[0]
    assert upright[:, 2].max() == pytest.approx(1.1)


@pytest.mark.parametrize('name', ['Sugar Beet', 'Sugar Corn', 'Potato', 'Soybean'])
def test_plants_have_the_table_height_and_about_the_table_width(name):
    crop, rng = CROPS[name], np.random.default_rng(2)
    widths = []
    for _ in range(40):
        vertices, uvs, faces = plant(crop['shape'], crop['height'], rng)
        assert vertices[:, 2].max() == pytest.approx(crop['height'])
        assert vertices[:, 2].min() >= -1e-9          # no leaf goes below the ground
        assert len(uvs) == len(vertices) and faces.max() < len(vertices)
        widths.append(2 * np.hypot(vertices[:, 0], vertices[:, 1]).max())
    assert np.mean(widths) == pytest.approx(crop['width'], rel=0.35)


def test_an_unknown_plant_shape_is_refused():
    with pytest.raises(ValueError, match='shape'):
        plant('tree', 1.0, np.random.default_rng(1))


def test_a_row_strip_stands_on_the_ground_and_repeats_its_picture_every_metre():
    vertices, uvs, faces = row_strip([(0.0, 0.0), (0.6, 0.8), (1.2, 1.6)], 0.7, flat, 0.25)
    assert vertices[0::2, 2] == pytest.approx([0.5] * 3)
    assert vertices[1::2, 2] == pytest.approx([1.2] * 3)
    assert uvs[0::2, 0] == pytest.approx([0.25, 1.25, 2.25])
    assert len(faces) == 4


def test_single_plants_stand_in_rows_inside_the_plot():
    meshes, count = plot_mesh(PLOT, CROPS['Sugar Beet'], flat, np.random.default_rng(1))
    vertices, uvs, faces = meshes['']
    assert np.array_equal(faces[len(faces) // 2:], faces[:len(faces) // 2, ::-1] +
                          len(vertices) // 2)                          # every face both ways
    along, across = plot_coordinates(vertices)
    assert 12 * 30 <= count <= 12 * 38
    assert np.abs(along).max() < 7.7 / 2 + 0.3 and np.abs(across).max() < 3.0 + 0.1
    assert vertices[:, 2].min() == pytest.approx(0.5)


def test_cereals_get_one_strip_per_row_and_a_canopy_inside_the_plot():
    meshes, count = plot_mesh(PLOT, CROPS['Summerwheat'], flat, np.random.default_rng(1))
    assert sorted(meshes) == ['_canopy', '_rows'] and count == 48
    for vertices, _, _ in meshes.values():
        along, across = plot_coordinates(vertices)
        assert np.abs(along).max() == pytest.approx(7.7 / 2 - 0.05)
        assert np.abs(across).max() == pytest.approx(2.9375)
    assert meshes['_canopy'][0][:, 2] == pytest.approx(np.full(len(meshes['_canopy'][0]),
                                                               0.5 + 0.9 * 0.7))


def test_every_crop_shape_has_textures_that_exist():
    assert {crop['shape'] for crop in CROPS.values()} <= set(PARTS)
    assert {t for parts in PARTS.values() for t in parts.values()} <= set(TEXTURES)


def test_merge_renumbers_faces_and_obj_text_lists_every_vertex_once():
    part = (np.zeros((3, 3)), np.zeros((3, 2)), np.array([[0, 1, 2]]))
    vertices, uvs, faces = merge([part, part])
    assert faces.tolist() == [[0, 1, 2], [3, 4, 5]]
    lines = obj_text(vertices, uvs, faces, vertex_normals(vertices, faces)).splitlines()
    assert [sum(line.startswith(k) for line in lines) for k in ('v ', 'vt ', 'vn ', 'f ')] == [
        6, 6, 6, 2]
    assert lines[-1] == 'f 4/4/4 5/5/5 6/6/6'


def test_vertex_normals_are_unit_vectors_pointing_up_on_flat_ground():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    normals = vertex_normals(vertices, np.array([[0, 1, 2], [0, 2, 3]]))
    assert normals == pytest.approx(np.tile([0.0, 0.0, 1.0], (4, 1)))


def test_a_leaf_outline_gives_the_drawn_width_of_each_band_from_the_base():
    opaque = np.zeros((8, 10), dtype=bool)
    opaque[0:4, 4:6] = True                     # upper half (towards the tip): narrow
    opaque[4:8, 1:9] = True                     # lower half (the base): wide
    opaque[5, 9] = True
    assert leaf_outline(opaque, 2) == pytest.approx(np.array([[0.1, 1.0], [0.4, 0.6]]))
    assert leaf_outline(np.zeros((4, 4), dtype=bool), 2) == pytest.approx(np.full((2, 2), 0.5))


def test_an_outlined_leaf_is_one_quad_per_band_as_wide_as_the_drawn_leaf():
    outline = np.array([[0.4, 0.6], [0.0, 1.0], [0.25, 0.75]])
    vertices, uvs, faces = leaves(0.0, 1.0, 0.2, 0.0, 0.0, outline=outline)  # flat, along +x
    assert vertices.shape == (12, 3) and faces.shape == (6, 3)
    quads = vertices.reshape(3, 4, 3)
    assert quads[:, :, 1].min(axis=1) == pytest.approx([-0.02, -0.1, -0.05])
    assert quads[:, :, 1].max(axis=1) == pytest.approx([0.02, 0.1, 0.05])
    assert quads[:, :, 0].min(axis=1) == pytest.approx([0.0, 1 / 3, 2 / 3])
    assert uvs.reshape(3, 4, 2)[1] == pytest.approx(
        np.array([[0.0, 1 / 3], [1.0, 1 / 3], [0.0, 2 / 3], [1.0, 2 / 3]]))
    plain_v, _, plain_f = leaves(0.0, 1.0, 0.2, 0.0, 0.0)
    for v, f in ((vertices, faces), (plain_v, plain_f)):    # the same winding as plain strips
        edges = v[f[:, 1:]] - v[f[:, :1]]
        assert (np.cross(edges[:, 0], edges[:, 1])[:, 2] < 0).all()


def test_two_sided_meshes_face_both_ways_on_their_own_vertices():
    mesh = (np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.zeros((3, 2)),
            np.array([[0, 1, 2]]))
    vertices, uvs, faces = two_sided(mesh)
    assert len(vertices) == 6 and faces.tolist() == [[0, 1, 2], [5, 4, 3]]
    normals = vertex_normals(vertices, faces)
    assert normals[:, 2] == pytest.approx([1.0] * 3 + [-1.0] * 3)


@pytest.mark.parametrize('name', ['Sugar Beet', 'Sugar Corn', 'Potato', 'Soybean'])
def test_outlined_plants_keep_their_height_and_use_the_same_random_numbers(name):
    crop = CROPS[name]
    outline = np.array([[0.45, 0.55], [0.1, 0.9], [0.2, 0.8], [0.4, 0.6]])
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    shaped = plant(crop['shape'], crop['height'], a, outline)
    plant(crop['shape'], crop['height'], b)
    assert shaped[0][:, 2].max() == pytest.approx(crop['height'])
    assert a.random() == b.random()             # the next plant is the same either way


def test_grow_gives_each_crop_its_leaf_outline():
    plots = [dict(PLOT, plot_id=1, crop='Sugar Beet'), dict(PLOT, plot_id=2, crop='Summerwheat')]
    narrow = {'sugar_beet_leaf': np.array([[0.45, 0.55]] * 4)}
    (_, beet, _), (_, wheat, _) = grow(plots, flat, 1, narrow)
    (_, plain, _), (_, plain_wheat, _) = grow(plots, flat, 1)
    assert np.ptp(beet[''][2]) and len(beet[''][2]) > len(plain[''][2])     # 4 bands, not 2
    assert np.array_equal(wheat['_rows'][0], plain_wheat['_rows'][0])
