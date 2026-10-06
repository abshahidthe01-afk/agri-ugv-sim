"""Tests for the phenotyping payload in the robot model (payload.xacro)."""

import math
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
import xacro

MODEL_FILE = (Path(__file__).resolve().parents[2]
              / 'agri_ugv_description' / 'urdf' / 'agri_ugv.urdf.xacro')
WHEEL_RADIUS = 0.205          # base_link is this high above the ground


@pytest.fixture(scope='module')
def root():
    """Return the processed robot model as XML."""
    return ElementTree.fromstring(
        xacro.process_file(str(MODEL_FILE), mappings={'controllers_file': 'x'}).toxml())


def poses(root, prefix):
    """Return {joint name: (xyz above the ground, viewing direction)} for joints with prefix."""
    found = {}
    for joint in root.findall('joint'):
        if joint.get('name').startswith(prefix):
            origin = joint.find('origin')
            xyz = np.array([float(v) for v in origin.get('xyz').split()]) + [0, 0, WHEEL_RADIUS]
            _, pitch, yaw = (float(v) for v in origin.get('rpy').split())
            view = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw),
                             -math.sin(pitch)])
            found[joint.get('name')] = (xyz, view)
    return found


def test_payload_adds_no_mass_and_no_collision(root):
    assert sum(float(m.get('value')) for m in root.iter('mass')) == pytest.approx(280.0)
    payload = [link for link in root.findall('link') if link.get('name').startswith(
        ('dome_camera_', 'line_scanner_', 'led_panels', 'curtains', 'computers'))]
    assert len(payload) == 25          # 20 cameras, 2 scanners, LEDs, curtains, computers
    assert all(link.find('inertial') is None and link.find('collision') is None
               for link in payload)


def test_20_dome_cameras_inside_the_enclosure_aim_at_the_plant(root):
    cameras = poses(root, 'dome_camera_')
    assert len(cameras) == 20
    for xyz, view in cameras.values():
        to_plant = np.array([0.0, 0.0, 0.30]) - xyz
        assert view @ to_plant / np.linalg.norm(to_plant) == pytest.approx(1.0, abs=1e-9)
        assert np.all(np.abs(xyz[:2]) <= 0.66) and 0.45 < xyz[2] < 1.80


@pytest.mark.parametrize('side, y', [('left', 0.67), ('right', -0.67)])
def test_line_scanners_look_down_and_across_at_50_degrees(root, side, y):
    [(xyz, view)] = poses(root, f'line_scanner_{side}').values()
    assert xyz == pytest.approx([0.0, y, 1.20])
    assert math.degrees(math.acos(-view[2])) == pytest.approx(50.0)
    assert view[1] * y < 0                       # towards the middle of the robot
