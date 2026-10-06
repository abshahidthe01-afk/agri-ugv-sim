"""Tests for the crop damage metric: wheels between rows, on rows, across rows, in wheat."""

import math

from agri_ugv_field.crop_damage import CropMap, damage, STEM_RADIUS
import pytest

TYRE = 0.165
BEET = {'plot_id': 198, 'crop': 'Sugar Beet', 'centre_x': 0.0, 'centre_y': 0.0,
        'width': 6.0, 'length': 8.0, 'heading_deg': 90.0}           # rows run north
WHEAT = dict(BEET, plot_id=197, crop='Summerwheat', centre_x=10.0)
ONE_WHEEL = {'w': (0.0, 0.0)}


def straight(x, y0, y1, steps=400):
    """Return a path driving north from (x, y0) to (x, y1)."""
    return [(x, y0 + (y1 - y0) * k / steps, math.pi / 2) for k in range(steps + 1)]


def test_a_wheel_between_rows_touches_leaves_but_crushes_nothing():
    # rows at -2.75 ... 2.75 every 0.5 m (east = -across when heading north); gap at x = 0.0
    totals, _ = damage(straight(0.0, -6.0, 6.0), ONE_WHEEL, CropMap([BEET]), TYRE)
    assert totals['w']['in_plots'] == pytest.approx(8.0, abs=0.05)
    assert totals['w']['crushed'] == pytest.approx(0.0)
    assert totals['w']['leaves'] == pytest.approx(8.0, abs=0.05)   # 7 cm gap < 16.5 cm tyre


def test_a_wheel_on_a_row_crushes_its_whole_length():
    totals, per_plot = damage(straight(0.25, -6.0, 6.0), ONE_WHEEL, CropMap([BEET]), TYRE)
    assert totals['w']['crushed'] == pytest.approx(8.0, abs=0.05)
    assert per_plot == {198: pytest.approx(8.0, abs=0.05)}


def test_crossing_the_rows_crushes_one_tyre_plus_stem_zone_per_row():
    path = [(-4.0 + 8.0 * k / 4000, 0.0, 0.0) for k in range(4001)]          # drive east
    totals, _ = damage(path, ONE_WHEEL, CropMap([BEET]), TYRE)
    assert totals['w']['crushed'] == pytest.approx(12 * (TYRE + 2 * STEM_RADIUS), abs=0.02)


def test_any_wheel_in_wheat_crushes_and_outside_plots_nothing_happens():
    totals, _ = damage(straight(10.0, -6.0, 6.0), ONE_WHEEL, CropMap([BEET, WHEAT]), TYRE)
    assert totals['w']['crushed'] == pytest.approx(8.0, abs=0.05)
    totals, _ = damage(straight(5.0, -6.0, 6.0), ONE_WHEEL, CropMap([BEET, WHEAT]), TYRE)
    assert totals['w'] == {'in_plots': 0.0, 'leaves': 0.0, 'crushed': 0.0}


def test_the_robot_straddling_a_row_keeps_both_wheel_lines_in_gaps():
    wheels = {'left': (0.0, 0.75), 'right': (0.0, -0.75)}              # track 1.5 m
    centred_on_row = straight(0.25, -6.0, 6.0)
    totals, _ = damage(centred_on_row, wheels, CropMap([BEET]), TYRE)
    assert sum(t['crushed'] for t in totals.values()) == pytest.approx(0.0)
    centred_on_gap = straight(0.0, -6.0, 6.0)
    totals, _ = damage(centred_on_gap, wheels, CropMap([BEET]), TYRE)
    assert sum(t['crushed'] for t in totals.values()) == pytest.approx(16.0, abs=0.1)
