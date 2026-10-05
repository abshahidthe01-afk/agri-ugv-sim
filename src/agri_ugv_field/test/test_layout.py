"""Tests for plot rectangles, the field layout and its CSV file."""

import math

from agri_ugv_field.layout import (field_layout, layout_csv, outline_geometry,
                                   read_layout_csv, terrain_origin)
import numpy as np
import pytest


def rectangle(cx, cy, length, width, heading_deg, clockwise=False, start=0):
    """Return a closed ring of 5 points: a rectangle whose long side points at heading_deg."""
    h = math.radians(heading_deg)
    u = np.array([math.cos(h), math.sin(h)])
    v = np.array([-u[1], u[0]])
    corners = [np.array([cx, cy]) + a * length / 2 * u + b * width / 2 * v
               for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    corners = corners[start:] + corners[:start]
    if clockwise:
        corners = corners[::-1]
    return np.array(corners + [corners[0]])


def outline(plot_id, ring, crop='', genotype='', var=''):
    """Return (points, record) like a shapefile outline with MuST-C attribute names."""
    return ring, {'local_idx': 1, 'plot_ID': plot_id, 'crop': crop, 'Genotype': genotype,
                  'var': var, 'RM_plot_id': ''}


def test_rectangle_size_heading_and_centre():
    g = outline_geometry(rectangle(3.0, -2.0, 8.1, 6.0, 84.5))
    assert g['length'] == pytest.approx(8.1)
    assert g['width'] == pytest.approx(6.0)
    assert math.degrees(g['heading']) == pytest.approx(84.5)
    assert g['centre'] == pytest.approx(np.array([3.0, -2.0]))
    assert g['gap'] == 0.0
    assert g['deviation'] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize('clockwise', [False, True])
@pytest.mark.parametrize('start', [0, 1, 2, 3])
def test_corners_run_counter_clockwise_from_the_start_of_the_long_side(clockwise, start):
    expected = rectangle(0.0, 0.0, 8.0, 6.0, -5.5)[:4]
    g = outline_geometry(rectangle(0.0, 0.0, 8.0, 6.0, -5.5, clockwise, start))
    assert g['corners'] == pytest.approx(expected)


@pytest.mark.parametrize('heading', [84.5, 264.5, -95.5])
def test_heading_is_a_line_direction_between_minus_and_plus_90_degrees(heading):
    g = outline_geometry(rectangle(0.0, 0.0, 8.0, 6.0, heading))
    assert math.degrees(g['heading']) == pytest.approx(84.5)


def test_open_outlines_and_wrong_point_counts_are_rejected():
    ring = rectangle(0.0, 0.0, 8.0, 6.0, 0.0)
    ring[4] += 0.01                              # the outline ends 1 cm from its start
    with pytest.raises(ValueError, match='not closed'):
        outline_geometry(ring)
    with pytest.raises(ValueError, match='5 points'):
        outline_geometry(rectangle(0.0, 0.0, 8.0, 6.0, 0.0)[:4])


def test_a_strongly_sheared_outline_is_rejected():
    ring = rectangle(0.0, 0.0, 8.0, 6.0, 0.0)
    ring[[2, 3]] += [1.0, 0.0]                   # far side shifted 1 m sideways
    with pytest.raises(ValueError, match='not roughly a rectangle'):
        outline_geometry(ring)


def test_the_hand_drawn_must_c_boundary_is_accepted_and_its_imperfection_measured():
    corners = [[357558.56, 5610200.04], [357389.87, 5610216.3], [357385.38, 5610171.98],
               [357554.32, 5610155.85]]             # md_FieldSHP.shp, plot_ID 0 (rounded)
    g = outline_geometry(np.array(corners + corners[:1]) - [357472.44, 5610188.04])
    assert g['deviation'] == pytest.approx(0.236, abs=0.002)   # 169.708 vs 169.472 m
    assert (g['width'], g['length']) == pytest.approx((44.47, 169.59), abs=0.01)
    assert math.degrees(g['heading']) == pytest.approx(-5.48, abs=0.01)


def test_field_layout_separates_the_boundary_and_sorts_the_plots():
    shapes = [outline(241, rectangle(10.0, 0.0, 8.1, 6.0, 84.5), 'Sugar Beet', 'BTS 440',
                      'Herbicide: 0%'),
              outline(0, rectangle(0.0, 0.0, 169.5, 44.6, -5.5)),
              outline(162, rectangle(-10.0, 0.0, 7.7, 6.0, 84.5), 'Mixture (wheat)')]
    boundary, plots = field_layout(shapes, lambda points: points - [1.0, 2.0])
    assert boundary['type'] == 'boundary'
    assert boundary['length'] == pytest.approx(169.5)
    assert [p['plot_id'] for p in plots] == [162, 241]
    assert (plots[1]['centre_x'], plots[1]['centre_y']) == pytest.approx((9.0, -2.0))
    assert plots[1]['genotype'] == 'BTS 440'
    assert plots[1]['variant'] == 'Herbicide: 0%'


@pytest.mark.parametrize('ids', [[1, 2], [0, 0, 1], [0, 1, 1]])
def test_missing_or_double_boundary_and_repeated_plot_ids_are_rejected(ids):
    shapes = [outline(i, rectangle(10.0 * k, 0.0, 8.0, 6.0, 90.0)) for k, i in enumerate(ids)]
    with pytest.raises(ValueError):
        field_layout(shapes, lambda points: points)


def test_a_missing_attribute_is_named():
    with pytest.raises(ValueError, match='crop'):
        field_layout([(rectangle(0.0, 0.0, 8.0, 6.0, 0.0), {'plot_ID': 0})], lambda p: p)


def test_csv_round_trip_keeps_millimetres_and_text_with_commas():
    shapes = [outline(0, rectangle(0.0, 0.0, 169.5, 44.6, -5.5)),
              outline(198, rectangle(-60.1234, 7.5678, 7.7, 6.0, 84.5), 'Sugar Beet', 'A, B')]
    boundary, plots = field_layout(shapes, lambda points: points)
    text = layout_csv([boundary] + plots, ['first note', 'second note'])
    assert text.startswith('# first note\n# second note\ntype,plot_id,crop,')
    rows = read_layout_csv(text)
    assert [r['type'] for r in rows] == ['boundary', 'plot']
    assert rows[1]['plot_id'] == 198
    assert rows[1]['genotype'] == 'A, B'
    for column in ['centre_x', 'centre_y', 'width', 'length', 'heading_deg', 'x1', 'y4']:
        assert rows[1][column] == pytest.approx(plots[0][column], abs=0.0005)


def test_reading_a_csv_with_other_columns_fails():
    with pytest.raises(ValueError, match='columns'):
        read_layout_csv('a,b\n1,2\n')


def test_terrain_origin_is_read_from_the_model_config():
    text = ('<description>terrain from 230517_DEM.tif, 0.45 m grid; origin at EPSG:32632 '
            'E 357472.44 N 5610188.04, 221.845 m above sea level</description>')
    assert terrain_origin(text) == ('EPSG:32632', 357472.44, 5610188.04)
    with pytest.raises(ValueError, match='origin'):
        terrain_origin('<description>flat test terrain</description>')
