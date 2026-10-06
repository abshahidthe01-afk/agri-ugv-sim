"""Tests for reading the terrain grid from its OBJ mesh and the ground height lookup."""

from agri_ugv_field.ground import ground_height, read_obj_grid
from agri_ugv_terrain.heightfield import mesh_from_heights, obj_text, vertex_normals
import numpy as np
import pytest


def terrain_obj(heights, spacing):
    """Return OBJ text exactly as agri_ugv_terrain writes it."""
    vertices, faces = mesh_from_heights(heights, spacing)
    return obj_text(vertices, vertex_normals(heights, spacing), faces)


def test_grid_is_read_back_from_the_obj():
    heights = np.arange(12.0).reshape(3, 4) / 10
    xs, ys, grid = read_obj_grid(terrain_obj(heights, 0.5))
    assert xs == pytest.approx([-0.75, -0.25, 0.25, 0.75])
    assert ys == pytest.approx([-0.5, 0.0, 0.5])
    assert grid == pytest.approx(heights)


def test_a_sloping_plane_is_interpolated_exactly_and_edges_are_held():
    xx, yy = np.meshgrid(np.arange(4) * 0.5 - 0.75, np.arange(3) * 0.5 - 0.5)
    grid = read_obj_grid(terrain_obj(0.1 * xx + 0.2 * yy + 1.0, 0.5))
    x, y = np.array([0.1, -0.6, 0.7]), np.array([0.3, -0.4, 0.45])
    assert ground_height(grid, x, y) == pytest.approx(0.1 * x + 0.2 * y + 1.0)
    assert ground_height(grid, 5.0, 0.0) == pytest.approx(0.1 * 0.75 + 1.0)   # beyond +x


def test_text_without_a_vertex_grid_is_rejected():
    with pytest.raises(ValueError, match='no vertices'):
        read_obj_grid('# empty\n')
    with pytest.raises(ValueError, match='grid'):
        read_obj_grid('v 0 0 0\nv 1 0 0\nv 0 1 0\n')
    shuffled = '\n'.join(reversed(terrain_obj(np.zeros((2, 3)), 1.0).splitlines()))
    with pytest.raises(ValueError, match='ordered'):
        read_obj_grid(shuffled)
