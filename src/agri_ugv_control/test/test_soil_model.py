"""Tests for the soil profiles in the robot model (soil.xacro)."""

from pathlib import Path
from xml.etree import ElementTree

import pytest
import xacro

MODEL_FILE = (Path(__file__).resolve().parents[2]
              / 'agri_ugv_description' / 'urdf' / 'agri_ugv.urdf.xacro')
WHEEL_LINKS = sorted(f'{p}_wheel_link' for p in
                     ('front_left', 'front_right', 'rear_left', 'rear_right'))


def model(**args):
    """Run xacro on the real model with the given arguments and return the XML root."""
    return ElementTree.fromstring(xacro.process_file(str(MODEL_FILE), mappings=args).toxml())


def wheel_slip_plugins(root):
    """Return all WheelSlip plugins in the model."""
    return [p for p in root.iter('plugin') if p.get('name', '').endswith('WheelSlip')]


def test_rigid_soil_adds_no_slip_model():
    root = model(controllers_file='x')
    assert wheel_slip_plugins(root) == []
    assert root.find('.//mu1') is None


@pytest.mark.parametrize('soil, mu, compliance', [
    ('firm', 0.65, 0.2), ('soft', 0.45, 0.55), ('wet', 0.3, 1.0)])
def test_soil_profiles_set_all_four_wheels(soil, mu, compliance):
    root = model(controllers_file='x', soil=soil)
    [plugin] = wheel_slip_plugins(root)
    wheels = {w.get('link_name'): w for w in plugin.findall('wheel')}
    assert sorted(wheels) == WHEEL_LINKS
    for wheel in wheels.values():
        assert float(wheel.findtext('slip_compliance_longitudinal')) == pytest.approx(compliance)
        assert float(wheel.findtext('slip_compliance_lateral')) == pytest.approx(compliance)
        assert float(wheel.findtext('wheel_normal_force')) == pytest.approx(70.0 * 9.81)
        assert float(wheel.findtext('wheel_radius')) == pytest.approx(0.205)
    frictions = {g.get('reference'): g for g in root.findall('gazebo')
                 if g.find('mu1') is not None}
    assert sorted(frictions) == WHEEL_LINKS
    for friction in frictions.values():
        assert float(friction.findtext('mu1')) == pytest.approx(mu)
        assert float(friction.findtext('mu2')) == pytest.approx(mu)
        assert friction.findtext('fdir1').split() == ['0', '0', '1']  # along the axle


def test_no_slip_model_without_gazebo():
    assert wheel_slip_plugins(model(soil='wet')) == []


def test_unknown_soil_is_rejected():
    with pytest.raises(xacro.XacroException, match='unknown soil muddy'):
        model(controllers_file='x', soil='muddy')
