"""Tests for the sensors in the robot model (sensors.xacro)."""

import math
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


def lidar(root):
    """Return the LiDAR's gpu_lidar sensor element."""
    [sensor] = [g.find('sensor') for g in root.findall('gazebo')
                if g.get('reference') == 'lidar_link']
    return sensor


def test_lidar_sits_on_a_mast_and_its_lowest_beams_clear_the_roof():
    root = model(controllers_file='x')
    [joint] = [j for j in root.findall('joint') if j.get('name') == 'lidar_joint']
    assert joint.get('type') == 'fixed' and joint.find('parent').get('link') == 'base_link'
    x, y, z = (float(v) for v in joint.find('origin').get('xyz').split())
    height = z + 0.205                                          # above the ground
    assert (x, y, height) == pytest.approx((0.0, 0.0, 2.70))
    lowest = -float(lidar(root).findtext('lidar/scan/vertical/min_angle'))
    # roof edge mid-side (1.85 m, 0.75 m out), computers' outer corner (1.97 m), antennas
    for top, out in [(1.85, 0.75), (1.97, math.hypot(0.2, 0.5)), (1.90, 0.53)]:
        assert math.atan2(height - top, out) > lowest             # below the lowest beam


def test_lidar_is_like_an_ouster_os0_64_and_publishes_its_points():
    sensor = lidar(model(controllers_file='x'))
    assert sensor.get('type') == 'gpu_lidar'
    assert sensor.findtext('topic') == '/lidar'                  # points on /lidar/points
    assert sensor.findtext('ignition_frame_id') == 'lidar_link'
    assert float(sensor.findtext('update_rate')) == 10.0
    scan = sensor.find('lidar/scan')
    assert int(scan.findtext('horizontal/samples')) == 1024
    step = (float(scan.findtext('horizontal/max_angle'))
            - float(scan.findtext('horizontal/min_angle'))) / 1023
    assert step == pytest.approx(2 * math.pi / 1024)             # all around, no double ray
    assert int(scan.findtext('vertical/samples')) == 64
    assert float(scan.findtext('vertical/max_angle')) == pytest.approx(math.pi / 4)
    assert (float(sensor.findtext('lidar/range/min')),
            float(sensor.findtext('lidar/range/max'))) == (0.3, 50.0)
    assert float(sensor.findtext('lidar/noise/stddev')) == pytest.approx(0.01)


def test_the_lidar_and_its_rendering_system_can_be_left_out():
    with_lidar, without = model(controllers_file='x'), model(controllers_file='x', lidar='false')
    plugins = [p.get('filename') for p in with_lidar.iter('plugin')]
    assert plugins.count('ignition-gazebo-sensors-system') == 1
    [system] = [p for p in with_lidar.iter('plugin')
                if p.get('filename') == 'ignition-gazebo-sensors-system']
    assert system.findtext('render_engine') == 'ogre2'
    assert not [link for link in without.findall('link') if link.get('name') == 'lidar_link']
    assert 'ignition-gazebo-sensors-system' not in {p.get('filename')
                                                    for p in without.iter('plugin')}
