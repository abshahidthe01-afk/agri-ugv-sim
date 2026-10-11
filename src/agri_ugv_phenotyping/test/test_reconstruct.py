"""Tests for the 3D reconstruction from depth images (reconstruct.py)."""

from agri_ugv_phenotyping.colour import excess_green, GREEN
from agri_ugv_phenotyping.reconstruct import area, colour, fuse, leaf_scores, mesh, \
    rotation_rpy, write_mesh_ply
import numpy as np
import pytest

K = [150.0, 0.0, 80.0, 0.0, 150.0, 60.0, 0.0, 0.0, 1.0]       # a 160 x 120 picture
LEAF, SOIL = (40, 120, 30), (110, 85, 60)
LOW, HIGH, VOXEL = (-0.3, -0.3, -0.05), (0.3, 0.3, 0.3), 0.01


def looking_at(eye, target=(0.0, 0.0, 0.1)):
    """Return the mounting (rotation, translation) of a camera at eye looking at target."""
    ahead = np.subtract(target, eye) / np.linalg.norm(np.subtract(target, eye))
    left = np.cross([0.0, 0.0, 1.0], ahead)
    left /= np.linalg.norm(left)
    return np.column_stack([ahead, left, np.cross(ahead, left)]), np.asarray(eye, dtype=float)


def scene(mounting):
    """Return the picture and depth image of soil at z 0 under a 20 cm square leaf at 0.2 m."""
    rotation, eye = mounting
    rows, cols = np.mgrid[0:120, 0:160]
    view = np.stack([np.ones(rows.shape), -(cols - K[2]) / K[0], -(rows - K[5]) / K[4]], -1)
    ray = view @ rotation.T                        # robot frame, per metre along the view
    depth = np.full(rows.shape, np.inf)
    picture = np.zeros(rows.shape + (3,), dtype=np.uint8)
    for height, colours in ((0.0, SOIL), (0.2, LEAF)):
        with np.errstate(divide='ignore'):
            along = (height - eye[2]) / ray[..., 2]
        hit = eye + along[..., None] * ray
        on = (along > 0) & (along < depth)
        if height > 0:
            on &= np.all(np.abs(hit[..., :2]) <= 0.1, axis=-1)
        depth[on] = along[on]
        picture[on] = colours
    return picture, depth


CAMERAS = [(looking_at(eye), K) for eye in
           [(0.1, 0.0, 1.2), (-0.4, 0.3, 1.0), (0.4, 0.3, 1.0), (0.0, -0.5, 1.0)]]


def reconstruction():
    pictures, depths = zip(*[scene(mounting) for mounting, _ in CAMERAS])
    distance, weight = fuse(depths, CAMERAS, LOW, HIGH, VOXEL, trunc=3)
    vertices, faces = mesh(distance, weight, LOW, VOXEL)
    colours, views, kept = colour(vertices, faces, pictures, depths, CAMERAS, tolerance=0.015)
    return vertices, faces, colours, views, kept


def test_the_leaf_and_the_soil_come_out_in_their_place_and_colour():
    vertices, faces, colours, views, kept = reconstruction()
    used = np.unique(kept)
    v, c = vertices[used], colours[used]
    on_leaf = np.all(np.abs(v[:, :2]) <= 0.09, axis=1) & (v[:, 2] > 0.1)
    assert on_leaf.sum() > 100
    assert np.abs(v[on_leaf, 2] - 0.2).max() < 0.012                # the leaf's top at 0.2 m
    assert (excess_green(c[on_leaf]) > GREEN).mean() > 0.95
    soil = np.any(np.abs(v[:, :2]) >= 0.18, axis=1)
    assert np.abs(v[soil, 2]).max() < 0.012 and (excess_green(c[soil]) < GREEN).mean() > 0.95
    assert len(kept) < len(faces)          # left-overs no picture sees (under the soil)
    # every 2 cm of the leaf's top has a vertex within 1.5 cm
    a, b = np.meshgrid(np.arange(-0.09, 0.091, 0.02), np.arange(-0.09, 0.091, 0.02))
    grid = np.column_stack([a.ravel(), b.ravel(), np.full(a.size, 0.2)])
    gaps = np.linalg.norm(grid[:, None] - v[None, on_leaf], axis=-1).min(axis=1)
    assert gaps.max() < 0.015


def test_scores_area_and_the_mesh_file(tmp_path):
    vertices, faces, colours, views, kept = reconstruction()
    a, b = np.meshgrid(np.arange(-0.1, 0.1, 0.004), np.arange(-0.1, 0.1, 0.004))
    leaf = np.column_stack([a.ravel(), b.ravel(), np.full(a.size, 0.2)])
    s = leaf_scores(vertices, kept, colours, leaf, reach=0.3)
    assert s['accuracy_median'] < 0.006 and s['complete_1cm'] > 0.9 and s['top_cells'] > 300
    assert 0.3 < area(vertices, kept) < 0.6                 # 0.36 m2 of soil and leaf in view
    count = write_mesh_ply(tmp_path / 'm.ply', vertices, kept, colours, ['test'])
    text = (tmp_path / 'm.ply').read_bytes()
    header = text[:text.index(b'end_header')].decode()
    assert f'element vertex {count}' in header and f'element face {len(kept)}' in header
    assert len(text) - len(header) - len('end_header\n') == count * 15 + len(kept) * 13


def test_an_empty_volume_has_no_surface_and_rotations_compose():
    distance, weight = np.ones((5, 5, 5), np.float32), np.zeros((5, 5, 5), np.float32)
    vertices, faces = mesh(distance, weight, LOW, VOXEL)
    assert len(vertices) == 0 and len(faces) == 0
    assert rotation_rpy(0.0, 0.0, np.pi / 2) @ [1.0, 0.0, 0.0] == pytest.approx([0, 1, 0])
    assert rotation_rpy(0.1, 0.0, 0.0)[2, 1] == pytest.approx(np.sin(0.1))
