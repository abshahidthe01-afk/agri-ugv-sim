"""Tests for reading the geometry from the robot model."""

from pathlib import Path

from agri_ugv_control.robot_geometry import geometry_from_urdf
import pytest
import xacro

# The real robot model, found relative to this file: src/agri_ugv_description/urdf/
MODEL_FILE = (Path(__file__).resolve().parents[2]
              / 'agri_ugv_description' / 'urdf' / 'agri_ugv.urdf.xacro')


def load_real_model():
    """Run xacro on the real model file and return the resulting URDF text."""
    return xacro.process_file(str(MODEL_FILE)).toxml()


def test_real_model_matches_the_spec():
    """The geometry read from the real model matches docs/robot_spec.md."""
    modules, wheel_radius = geometry_from_urdf(load_real_model())
    assert wheel_radius == pytest.approx(0.205)
    positions = {m.name: (m.x, m.y) for m in modules}
    assert positions['front_left'] == pytest.approx((0.675, 0.75))
    assert positions['front_right'] == pytest.approx((0.675, -0.75))
    assert positions['rear_left'] == pytest.approx((-0.675, 0.75))
    assert positions['rear_right'] == pytest.approx((-0.675, -0.75))


def test_missing_joint_is_reported():
    """A model without the expected joints gives a clear error, not a silent wrong answer."""
    with pytest.raises(ValueError, match='front_left_steer_joint'):
        geometry_from_urdf('<robot name="empty"><link name="base_link"/></robot>')
