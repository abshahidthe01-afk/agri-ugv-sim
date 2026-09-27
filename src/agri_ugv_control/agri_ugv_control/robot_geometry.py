"""Read the wheel-module geometry from the robot description (URDF text).

The robot model is the single source of truth for the geometry: this module
extracts the numbers from it, so they are never typed into the code.
"""

from typing import List, Tuple
import xml.etree.ElementTree as ElementTree

from agri_ugv_control.kinematics import WheelModule

MODULE_NAMES = ('front_left', 'front_right', 'rear_left', 'rear_right')


def geometry_from_urdf(urdf_text: str) -> Tuple[List[WheelModule], float]:
    """Return the four wheel modules and the wheel radius described in a URDF.

    Wheel positions come from the <origin> of each <name>_steer_joint, which
    must be attached to base_link. The wheel radius comes from the first
    cylinder in each <name>_wheel_link; all four wheels must agree.
    """
    robot = ElementTree.fromstring(urdf_text)
    joints = {joint.get('name'): joint for joint in robot.findall('joint')}
    links = {link.get('name'): link for link in robot.findall('link')}

    modules = []
    radii = []
    for name in MODULE_NAMES:
        # Position of the steering axis, relative to the robot centre
        joint = joints.get(f'{name}_steer_joint')
        if joint is None:
            raise ValueError(f'robot model has no joint named {name}_steer_joint')
        parent = joint.find('parent').get('link')
        if parent != 'base_link':
            raise ValueError(f'{name}_steer_joint must be attached to base_link, not {parent}')
        x, y, _ = (float(value) for value in joint.find('origin').get('xyz').split())
        modules.append(WheelModule(name, x, y))

        # Wheel radius: the tyre is the first cylinder in the wheel link
        link = links.get(f'{name}_wheel_link')
        cylinder = link.find('visual/geometry/cylinder') if link is not None else None
        if cylinder is None:
            raise ValueError(f'robot model has no tyre cylinder in {name}_wheel_link')
        radii.append(float(cylinder.get('radius')))

    if max(radii) - min(radii) > 1e-9:
        raise ValueError(f'wheel radii differ between modules: {radii}')
    return modules, radii[0]
