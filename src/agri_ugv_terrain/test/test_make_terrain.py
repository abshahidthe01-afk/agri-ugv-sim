"""Tests for writing terrain model folders."""

from agri_ugv_terrain.make_terrain import main, write_model
import numpy as np
import pytest


def test_write_model_creates_the_three_files(tmp_path):
    model_dir = write_model(tmp_path, 'test_ground', np.zeros((3, 4)), 0.5, 'A test.')
    assert model_dir == tmp_path / 'test_ground'
    assert (model_dir / 'model.config').is_file()
    assert (model_dir / 'model.sdf').is_file()
    obj = (model_dir / 'meshes' / 'test_ground.obj').read_text()
    assert obj.count('\nv ') == 12
    assert obj.count('\nf ') == 12


@pytest.mark.parametrize('name', ['', '1st', 'has space', 'a/b', 'x"y'])
def test_unsafe_names_are_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        write_model(tmp_path, name, np.zeros((2, 2)), 0.5, 'A test.')


def test_command_line(tmp_path, capsys):
    main(['--shape', 'flat', '--name', 'flat_mesh', '--length', '2', '--width', '1',
          '--spacing', '0.5', '--output-dir', str(tmp_path)])
    assert (tmp_path / 'flat_mesh' / 'meshes' / 'flat_mesh.obj').is_file()
    assert '3 x 5 points, 16 triangles' in capsys.readouterr().out
