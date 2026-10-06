"""Tests for the world origin, fixes in world coordinates and antenna offsets."""

import math

from agri_ugv_localization.frames import antenna_levers, spherical_coordinates, world_from_fix
from agri_ugv_localization.gnss_errors import shift
import pytest

ORIGIN = (50.626127482, 6.984883225, 221.845, 1.558)        # must_c_field
WORLD = """<world><spherical_coordinates><surface_model>EARTH_WGS84</surface_model>
  <latitude_deg>50.626127482</latitude_deg><longitude_deg>6.984883225</longitude_deg>
  <elevation>221.845</elevation><heading_deg>1.558</heading_deg></spherical_coordinates></world>"""


def test_the_world_origin_is_read_and_defaults_without_coordinates():
    assert spherical_coordinates(WORLD) == pytest.approx(ORIGIN)
    assert spherical_coordinates('<world><gravity>0 0 -9.8</gravity></world>') == (0, 0, 0, 0)


def test_a_fix_east_of_the_origin_lands_turned_by_the_heading():
    fix = shift(*ORIGIN[:3], 10.0, 0.0, 1.5)                  # 10 m true east, 1.5 m up
    x, y, z = world_from_fix(ORIGIN, fix)
    h = math.radians(1.558)
    assert (x, y, z) == pytest.approx((10 * math.cos(h), -10 * math.sin(h), 1.5), abs=1e-6)


def test_antenna_levers_add_up_the_joints_from_the_base():
    urdf = """<robot name="r">
      <link name="base_footprint"/><link name="base_link"/><link name="gnss_front_link"/>
      <joint name="a" type="fixed"><parent link="base_footprint"/><child link="base_link"/>
        <origin xyz="0 0 0.205"/></joint>
      <joint name="b" type="fixed"><parent link="base_link"/><child link="gnss_front_link"/>
        <origin xyz="0.6 0 1.695"/></joint></robot>"""
    assert antenna_levers(urdf, ('gnss_front_link',)) == {
        'gnss_front_link': pytest.approx((0.6, 0.0, 1.9))}
    with pytest.raises(ValueError):
        antenna_levers(urdf, ('gnss_rear_link',))
