"""Tests for the leaf and crop textures drawn by code."""

from agri_ugv_field.textures import draw_texture, TEXTURES
import numpy as np
import pytest


@pytest.mark.parametrize('name', sorted(TEXTURES))
def test_textures_have_transparent_background_and_opaque_leaves(name):
    pytest.importorskip('PIL')
    image = draw_texture(name)
    assert image.mode == 'RGBA'
    alpha = np.asarray(image)[:, :, 3]
    assert 0.1 < (alpha >= 128).mean() < 0.9           # Gazebo keeps pixels from 50 % up
    green = np.asarray(image)[alpha >= 128][:, 1].mean()
    assert green > np.asarray(image)[alpha >= 128][:, 2].mean()   # leaves are green, not blue


def test_the_same_seed_draws_the_same_picture():
    pytest.importorskip('PIL')
    assert draw_texture('wheat_row', 3).tobytes() == draw_texture('wheat_row', 3).tobytes()
    assert draw_texture('wheat_row', 3).tobytes() != draw_texture('wheat_row', 4).tobytes()
