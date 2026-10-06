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


def test_imu_sits_under_the_roof_at_the_back():
    root = model(controllers_file='x')
    [joint] = [j for j in root.findall('joint') if j.get('name') == 'imu_joint']
    assert joint.get('type') == 'fixed'
    assert joint.find('parent').get('link') == 'base_link'
    x, y, z = (float(v) for v in joint.find('origin').get('xyz').split())
    assert (x, y, z + 0.205) == pytest.approx((-0.60, 0.0, 1.79))   # 1.79 m above the ground


def test_imu_publishes_on_ros_friendly_topic_and_frame():
    root = model(controllers_file='x')
    [sensor] = [g.find('sensor') for g in root.findall('gazebo')
                if g.get('reference') == 'imu_link']
    assert sensor.get('type') == 'imu'
    assert sensor.findtext('topic') == '/imu'
    assert sensor.findtext('ignition_frame_id') == 'imu_link'
    assert float(sensor.findtext('update_rate')) == 100.0
    assert any(p.get('filename') == 'ignition-gazebo-imu-system' for p in root.iter('plugin'))


@pytest.mark.parametrize('name, x', [('front', 0.6), ('rear', -0.6)])
def test_gnss_antennas_sit_on_the_roof_1_2_m_apart(name, x):
    root = model(controllers_file='x')
    [joint] = [j for j in root.findall('joint') if j.get('name') == f'gnss_{name}_joint']
    assert joint.find('parent').get('link') == 'base_link'
    position = [float(v) for v in joint.find('origin').get('xyz').split()]
    assert (position[0], position[1], position[2] + 0.205) == pytest.approx((x, 0.0, 1.90))
    [sensor] = [g.find('sensor') for g in root.findall('gazebo')
                if g.get('reference') == f'gnss_{name}_link']
    assert sensor.get('type') == 'navsat'
    assert sensor.findtext('topic') == f'/gnss/{name}/fix_ideal'
    assert sensor.findtext('ignition_frame_id') == f'gnss_{name}_link'
    assert float(sensor.findtext('update_rate')) == 10.0


def test_the_navsat_system_is_loaded_once():
    root = model(controllers_file='x')
    names = [p.get('filename') for p in root.iter('plugin')]
    assert names.count('ignition-gazebo-navsat-system') == 1


@pytest.mark.parametrize('quantity, white, bias, drift', [
    ('angular_velocity', 0.0005, 0.0003, 2e-5), ('linear_acceleration', 0.005, 0.01, 2e-4)])
def test_imu_noise_and_biases_on_all_three_axes(quantity, white, bias, drift):
    root = model(controllers_file='x')
    [sensor] = [g.find('sensor') for g in root.findall('gazebo')
                if g.get('reference') == 'imu_link']
    for axis in 'xyz':
        noise = sensor.find(f'imu/{quantity}/{axis}/noise')
        assert noise.get('type') == 'gaussian'
        assert float(noise.findtext('stddev')) == pytest.approx(white)
        assert float(noise.findtext('bias_stddev')) == pytest.approx(bias)
        tau = float(noise.findtext('dynamic_bias_correlation_time'))
        density = float(noise.findtext('dynamic_bias_stddev'))
        assert density * (tau / 2) ** 0.5 == pytest.approx(drift)   # Gazebo's long-run spread
