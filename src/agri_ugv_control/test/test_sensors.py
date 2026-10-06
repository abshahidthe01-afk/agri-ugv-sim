"""Tests for the sensors in the robot model (sensors.xacro)."""

from pathlib import Path
from xml.etree import ElementTree

import pytest
import xacro

MODEL_FILE = (Path(__file__).resolve().parents[2]
              / 'agri_ugv_description' / 'urdf' / 'agri_ugv.urdf.xacro')


def model(**args):
    """Run xacro on the real model with the given arguments and return the XML root."""
    return ElementTree.fromstring(xacro.process_file(str(MODEL_FILE), mappings=args).toxml())


def test_imu_sits_on_the_body_near_the_centre_of_mass():
    root = model(controllers_file='x')
    [joint] = [j for j in root.findall('joint') if j.get('name') == 'imu_joint']
    assert joint.get('type') == 'fixed'
    assert joint.find('parent').get('link') == 'base_link'
    x, y, z = (float(v) for v in joint.find('origin').get('xyz').split())
    assert (x, y, z + 0.205) == pytest.approx((0.0, 0.0, 0.90))   # 0.90 m above the ground


def test_imu_publishes_on_ros_friendly_topic_and_frame():
    root = model(controllers_file='x')
    [sensor] = [g.find('sensor') for g in root.findall('gazebo')
                if g.get('reference') == 'imu_link']
    assert sensor.get('type') == 'imu'
    assert sensor.findtext('topic') == '/imu'
    assert sensor.findtext('ignition_frame_id') == 'imu_link'
    assert float(sensor.findtext('update_rate')) == 100.0
    assert any(p.get('filename') == 'ignition-gazebo-imu-system' for p in root.iter('plugin'))
