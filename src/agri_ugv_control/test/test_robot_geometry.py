"""Tests for reading the geometry and the mass properties from the robot model."""

import math
from pathlib import Path

from agri_ugv_control.robot_geometry import center_of_mass, geometry_from_urdf
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


def test_mass_and_centre_of_mass_match_the_spec():
    """Total mass is 280 kg; the centre of mass is centred and 0.894 m above the ground."""
    total_mass, (x, y, z) = center_of_mass(load_real_model())
    assert total_mass == pytest.approx(280.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert z == pytest.approx(0.894, abs=0.001)


def test_robot_is_hard_to_tip_over():
    """The static tip-over angle is far above field slopes, sideways and forwards."""
    urdf = load_real_model()
    modules, _ = geometry_from_urdf(urdf)
    _, (_, _, com_height) = center_of_mass(urdf)
    half_track = max(abs(m.y) for m in modules)
    half_wheelbase = max(abs(m.x) for m in modules)
    sideways = math.degrees(math.atan2(half_track, com_height))
    forwards = math.degrees(math.atan2(half_wheelbase, com_height))
    assert sideways > 35.0
    assert forwards > 35.0


def test_centre_of_mass_follows_rotated_joints():
    urdf = """<robot name="r">
      <link name="root"><inertial><mass value="2.0"/></inertial></link>
      <link name="arm"><inertial><origin xyz="1 0 0"/><mass value="2.0"/></inertial></link>
      <joint name="j" type="fixed"><parent link="root"/><child link="arm"/>
        <origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/></joint>
    </robot>"""
    mass, com = center_of_mass(urdf)          # the arm's mass sits at (1, 1, 0)
    assert mass == pytest.approx(4.0)
    assert com == pytest.approx((0.5, 0.5, 0.0))
