"""Tests for the command-line tool that turns a shapefile into the plot layout CSV."""

import math

from agri_ugv_field.layout import read_layout_csv
import numpy as np
import pytest

ORIGIN = (357472.44, 5610188.04)          # E, N of the must_c_field terrain centre
CONFIG = ('<?xml version="1.0"?>\n<model>\n  <name>field</name>\n  <description>terrain; '
          'origin at EPSG:32632 E 357472.44 N 5610188.04, 221.845 m above sea level'
          '</description>\n</model>\n')


def utm_rectangle(east, north, length, width, heading_deg):
    """Return a closed ring of 5 map points (m) with the long side at heading_deg."""
    h = math.radians(heading_deg)
    u, v = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])
    corners = [np.array([east, north]) + a * length / 2 * u + b * width / 2 * v
               for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    return [list(p) for p in corners + [corners[0]]]


def write_field(folder, polygons=False, prj=True):
    """Write a small MuST-C-like shapefile (ETRS89 / UTM 32N) and a terrain model folder."""
    shapefile = pytest.importorskip('shapefile')
    pyproj = pytest.importorskip('pyproj')
    writer = shapefile.Writer(str(folder / 'field'),
                              shapeType=shapefile.POLYGON if polygons else shapefile.POLYLINE)
    for name, kind in [('local_idx', 'N'), ('plot_ID', 'N'), ('crop', 'C'), ('Genotype', 'C'),
                       ('var', 'C'), ('RM_plot_id', 'C')]:
        writer.field(name, kind)
    rings = [(0, utm_rectangle(ORIGIN[0], ORIGIN[1] - 2.0, 169.5, 44.6, -5.5), ''),
             (198, utm_rectangle(ORIGIN[0] - 60.0, ORIGIN[1] + 5.0, 7.7, 6.0, 84.5),
              'Sugar Beet'),
             (162, utm_rectangle(ORIGIN[0] + 80.0, ORIGIN[1] + 15.0, 8.1, 6.0, 84.5),
              'Mixture (wheat)')]
    for k, (plot_id, ring, crop) in enumerate(rings, start=1):
        if polygons:
            writer.poly([ring])
        else:
            writer.line([ring])
        writer.record(k, plot_id, crop, 'G', '', '')
    writer.close()
    if prj:
        (folder / 'field.prj').write_text(pyproj.CRS.from_epsg(25832).to_wkt('WKT1_ESRI'))
    (folder / 'terrain').mkdir()
    (folder / 'terrain' / 'model.config').write_text(CONFIG)


@pytest.mark.parametrize('polygons', [False, True])
def test_shapefile_to_world_frame_csv(tmp_path, polygons):
    write_field(tmp_path, polygons)
    from agri_ugv_field.make_field_layout import main
    output = tmp_path / 'out' / 'plots.csv'
    main(['--shapefile', str(tmp_path / 'field.shp'), '--terrain-model',
          str(tmp_path / 'terrain'), '--output', str(output)])
    text = output.read_text()
    assert 'ETRS89 / UTM zone 32N' in text and 'EPSG:32632' in text and 'CC BY 4.0' in text
    rows = read_layout_csv(text)
    expected = [('boundary', 0), ('plot', 162), ('plot', 198)]
    assert [(r['type'], r['plot_id']) for r in rows] == expected
    sugar_beet = rows[2]
    assert (sugar_beet['centre_x'], sugar_beet['centre_y']) == pytest.approx((-60.0, 5.0),
                                                                             abs=0.002)
    assert (sugar_beet['width'], sugar_beet['length']) == pytest.approx((6.0, 7.7), abs=0.002)
    assert sugar_beet['heading_deg'] == pytest.approx(84.5, abs=0.01)
    assert rows[0]['heading_deg'] == pytest.approx(-5.5, abs=0.01)


def test_a_shapefile_without_prj_is_refused(tmp_path):
    write_field(tmp_path, prj=False)
    from agri_ugv_field.make_field_layout import main
    with pytest.raises(ValueError, match='prj'):
        main(['--shapefile', str(tmp_path / 'field.shp'), '--terrain-model',
              str(tmp_path / 'terrain'), '--output', str(tmp_path / 'plots.csv')])
