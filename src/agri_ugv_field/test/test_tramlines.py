"""Tests for the tramlines: the cereal rows left unsown where the robot's wheels run."""

from agri_ugv_field.plants import CROPS, row_groups, row_offsets, sown_rows
from agri_ugv_field.tramlines import pass_offsets, tramline_rows, wheel_lines, WHEEL_TO_ROW
import numpy as np
import pytest


def test_the_wheels_of_neighbouring_passes_share_their_lines():
    assert pass_offsets(6.0) == pytest.approx([-2.25, -0.75, 0.75, 2.25])
    assert wheel_lines(6.0) == pytest.approx([-3.0, -1.5, -1.5, 0.0, 0.0, 1.5, 1.5, 3.0])


@pytest.mark.parametrize('width', [5.979, 6.0, 6.02])
def test_wheat_loses_two_rows_per_inner_wheel_line_and_one_at_each_edge(width):
    rows = row_offsets(width, 0.125)
    on = tramline_rows(rows, width)
    assert len(rows) == 48 and on.sum() == 8
    assert rows[on] == pytest.approx([-2.9375, -1.5625, -1.4375, -0.0625, 0.0625, 1.4375,
                                      1.5625, 2.9375])
    sown = sown_rows(width, CROPS['Summerwheat'])
    lines = np.array(wheel_lines(width))
    assert np.min(np.abs(sown[:, None] - lines[None, :])) >= WHEEL_TO_ROW


def test_only_the_drilled_cereals_have_tramlines():
    for crop in CROPS.values():
        fewer = len(sown_rows(6.0, crop)) < len(row_offsets(6.0, crop['rows']))
        assert fewer == (crop['shape'] == 'cereal')


def test_the_sown_wheat_rows_form_four_strips_between_the_tramlines():
    groups = np.array(row_groups(sown_rows(6.0, CROPS['Summerwheat']), 0.125))
    assert groups == pytest.approx(np.array([[-2.8125, -1.6875], [-1.3125, -0.1875],
                                             [0.1875, 1.3125], [1.6875, 2.8125]]))
    assert row_groups(row_offsets(6.0, 0.5), 0.5) == [(-2.75, 2.75)]
