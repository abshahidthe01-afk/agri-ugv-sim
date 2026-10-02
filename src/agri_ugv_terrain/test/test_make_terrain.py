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


def test_command_line_waves(tmp_path, capsys):
    main(['--shape', 'waves', '--name', 'waves', '--length', '27', '--width', '6',
          '--spacing', '0.15', '--output-dir', str(tmp_path)])
    assert '41 x 181 points, 14400 triangles' in capsys.readouterr().out
    config = (tmp_path / 'waves' / 'model.config').read_text()
    assert 'waves along x +-0.05 m every 2.7 m' in config


def test_command_line_ramp(tmp_path, capsys):
    main(['--shape', 'ramp', '--name', 'ramp', '--length', '30', '--width', '6',
          '--grade', '0.1', '--output-dir', str(tmp_path)])
    assert '13 x 61 points, 1440 triangles, heights -1.500 to +1.500 m' in capsys.readouterr().out


def test_command_line_dem(tmp_path, capsys):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    data = np.full((20, 30), 222.0, dtype='float32')
    data[:10] += 0.5  # northern half 0.5 m higher
    path = tmp_path / 'dem.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=20, width=30, count=1, dtype='float32',
                       crs='EPSG:32632', transform=from_origin(1000.0, 2000.0, 0.03, 0.03),
                       nodata=-32767.0) as dst:
        dst.write(data, 1)
    main(['--shape', 'dem', '--name', 'field', '--dem-file', str(path), '--spacing', '0.15',
          '--output-dir', str(tmp_path)])
    out = capsys.readouterr().out
    assert '4 x 6 points, 30 triangles' in out
    assert 'E 1000.45 N 1999.70' in out


def test_dem_spacing_must_fit_the_pixels(tmp_path):
    rasterio = pytest.importorskip('rasterio')
    from rasterio.transform import from_origin
    path = tmp_path / 'dem.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=20, width=30, count=1, dtype='float32',
                       crs='EPSG:32632', transform=from_origin(1000.0, 2000.0, 0.03, 0.03)) as dst:
        dst.write(np.zeros((20, 30), dtype='float32'), 1)
    with pytest.raises(ValueError):
        main(['--shape', 'dem', '--name', 'field', '--dem-file', str(path), '--spacing', '0.1',
              '--output-dir', str(tmp_path)])
