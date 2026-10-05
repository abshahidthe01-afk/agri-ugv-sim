"""Checks of the committed MuST-C plot layout against facts known from the dataset."""

from collections import Counter
from pathlib import Path

from agri_ugv_field.layout import read_layout_csv, terrain_origin
import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
LAYOUT = PACKAGE / 'data' / 'must_c_field_plots.csv'
TERRAIN = PACKAGE.parent / 'agri_ugv_gazebo' / 'models' / 'must_c_field' / 'model.config'


@pytest.fixture(scope='module')
def rows():
    """Return the rows of the committed layout CSV, or skip if it has not been made yet."""
    if not LAYOUT.exists():
        pytest.skip(f'{LAYOUT.name} not generated yet')
    return read_layout_csv(LAYOUT.read_text())


def test_80_plots_with_the_published_ids_and_crops(rows):
    plots = [r for r in rows if r['type'] == 'plot']
    assert [r['type'] for r in rows].count('boundary') == 1
    assert sorted(r['plot_id'] for r in plots) == list(range(162, 242))
    assert Counter(r['crop'] for r in plots) == {
        'Sugar Beet': 16, 'Summerwheat': 16, 'Sugar Corn': 16, 'Potato': 16, 'Soybean': 8,
        'Mixture (faba-wheat)': 4, 'Mixture (faba)': 2, 'Mixture (wheat)': 2}
    by_id = {r['plot_id']: r['crop'] for r in plots}
    assert by_id[198] == 'Sugar Beet'      # the dataset's sample plot


def test_plots_are_6_m_wide_and_7_7_or_8_1_m_long_and_parallel(rows):
    plots = [r for r in rows if r['type'] == 'plot']
    assert all(abs(r['width'] - 6.0) < 0.1 for r in plots)
    # 7.66 or 8.08 m; 3 hand-drawn outlines (196, 203, 238) mix both, giving 7.87 m
    assert all(7.6 < r['length'] < 8.2 for r in plots)
    headings = np.array([r['heading_deg'] for r in plots])
    assert np.ptp(headings) < 1.0
    boundary = next(r for r in rows if r['type'] == 'boundary')
    assert boundary['heading_deg'] == pytest.approx(headings.mean() - 90.0, abs=1.0)


def test_plots_lie_on_the_terrain_and_use_its_origin(rows):
    if not TERRAIN.exists():
        pytest.skip('terrain model not found')
    crs, east, north = terrain_origin(TERRAIN.read_text())
    origin = f'{crs} and shifted by the origin of the must_c_field terrain (E {east:.2f} '
    assert origin + f'N {north:.2f})' in LAYOUT.read_text()
    corners = np.array([[r[f'x{k}'], r[f'y{k}']] for r in rows if r['type'] == 'plot'
                        for k in range(1, 5)])
    assert np.all(np.abs(corners) < [87.0, 31.7])   # inside the mesh (142 x 388 points, 0.45 m)
