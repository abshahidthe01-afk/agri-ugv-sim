"""Tests for the command-line tool that writes the crop plants model."""

from xml.etree import ElementTree

from agri_ugv_field.layout import field_layout, layout_csv
from agri_ugv_field.make_plants import main
from agri_ugv_terrain.make_terrain import write_model
import numpy as np
import pytest


def rectangle(cx, cy, length, width, heading_deg):
    """Return a closed ring of 5 points with the long side at heading_deg."""
    h = np.radians(heading_deg)
    u, v = np.array([np.cos(h), np.sin(h)]), np.array([-np.sin(h), np.cos(h)])
    corners = [np.array([cx, cy]) + a * length / 2 * u + b * width / 2 * v
               for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    return np.array(corners + corners[:1])


def record(plot_id, crop):
    """Return shapefile attributes for an outline."""
    return {'plot_ID': plot_id, 'crop': crop, 'genotype': '', 'var': ''}


def field(folder, crops):
    """Write a layout CSV with one plot per crop and a flat terrain model; return args."""
    shapes = [(rectangle(0.0, 0.0, 30.0, 12.0, -5.5), record(0, ''))]
    shapes += [(rectangle(-7.0 + 7.0 * k, 0.0, 7.7, 6.0, 84.5), record(100 + k, crop))
               for k, crop in enumerate(crops)]
    boundary, plots = field_layout(shapes, lambda points: points)
    (folder / 'plots.csv').write_text(layout_csv([boundary] + plots))
    write_model(folder, 'ground', np.zeros((41, 81)), 0.5, 'flat test ground')
    return ['--layout', str(folder / 'plots.csv'), '--terrain-model', str(folder / 'ground'),
            '--name', 'plants', '--output-dir', str(folder)]


def test_plants_model_has_textured_double_sided_meshes(tmp_path):
    pytest.importorskip('PIL')
    main(field(tmp_path, ['Sugar Beet', 'Summerwheat']))
    model_dir = tmp_path / 'plants'
    model = ElementTree.parse(model_dir / 'model.sdf').getroot().find('model')
    assert model.find('static').text == 'true'
    assert model.find('link/collision') is None
    visuals = {v.get('name'): v for v in model.findall('link/visual')}
    assert sorted(visuals) == ['sugar_beet', 'wheat_canopy', 'wheat_rows']
    for name, visual in visuals.items():
        assert visual.find('material/double_sided').text == 'true'
        texture = visual.find('material/pbr/metal/albedo_map').text
        assert (model_dir / texture).is_file()
        obj = (model_dir / 'meshes' / f'{name}.obj').read_text().splitlines()
        counts = [sum(line.startswith(k) for line in obj) for k in ('v ', 'vt ', 'vn ')]
        assert counts[0] == counts[1] == counts[2] > 0
    assert 'assumed' in (model_dir / 'model.config').read_text()


def test_an_unknown_crop_is_refused(tmp_path):
    with pytest.raises(ValueError, match='Rice'):
        main(field(tmp_path, ['Rice']))
