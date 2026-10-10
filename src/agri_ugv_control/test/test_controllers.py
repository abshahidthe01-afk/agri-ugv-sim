"""Tests for the controller settings (config/controllers.yaml) against the robot model."""

from pathlib import Path
from xml.etree import ElementTree

from agri_ugv_control.robot_geometry import geometry_from_urdf
import pytest
import xacro
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
MODEL_FILE = PACKAGE.parent / 'agri_ugv_description' / 'urdf' / 'agri_ugv.urdf.xacro'
PLUGIN_FILE = PACKAGE.parent / 'agri_ugv_four_ws' / 'four_ws_controller.xml'


@pytest.fixture(scope='module')
def controllers():
    """Return the settings of all controllers."""
    with open(PACKAGE / 'config' / 'controllers.yaml') as text:
        return yaml.safe_load(text)


def test_the_cpp_controller_has_the_geometry_of_the_robot_model(controllers):
    modules, radius = geometry_from_urdf(
        xacro.process_file(str(MODEL_FILE), mappings={'controllers_file': 'x'}).toxml())
    settings = controllers['four_ws_controller']['ros__parameters']
    assert settings['steering_joints'] == [f'{m.name}_steer_joint' for m in modules]
    assert settings['wheel_joints'] == [f'{m.name}_wheel_joint' for m in modules]
    assert settings['module_x'] == pytest.approx([m.x for m in modules], abs=1e-9)
    assert settings['module_y'] == pytest.approx([m.y for m in modules], abs=1e-9)
    assert settings['wheel_radius'] == pytest.approx(radius, abs=1e-9)
    # ROS would refuse a whole number here: the parameters are declared as doubles
    numbers = settings['module_x'] + settings['module_y'] + [settings['wheel_radius']]
    assert all(isinstance(value, float) for value in numbers)


def test_the_cpp_controller_drives_the_joints_of_the_joint_group_controllers(controllers):
    settings = controllers['four_ws_controller']['ros__parameters']
    assert settings['steering_joints'] == \
        controllers['steering_controller']['ros__parameters']['joints']
    assert settings['wheel_joints'] == \
        controllers['wheel_controller']['ros__parameters']['joints']


def test_the_cpp_controller_type_is_the_exported_plugin(controllers):
    kind = controllers['controller_manager']['ros__parameters']['four_ws_controller']['type']
    names = [c.get('name') for c in ElementTree.parse(PLUGIN_FILE).getroot().iter('class')]
    assert names == [kind] == ['agri_ugv_four_ws/FourWsController']
