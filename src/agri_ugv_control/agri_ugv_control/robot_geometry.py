"""
Read the wheel-module geometry from the robot description (URDF text).

The robot model is the single source of truth for the geometry: this module
extracts the numbers from it, so they are never typed into the code.
"""

from typing import List, Tuple
import xml.etree.ElementTree as ElementTree

from agri_ugv_control.kinematics import WheelModule

MODULE_NAMES = ('front_left', 'front_right', 'rear_left', 'rear_right')


def geometry_from_urdf(urdf_text: str) -> Tuple[List[WheelModule], float]:
    """
    Return the four wheel modules and the wheel radius described in a URDF.

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


def _xyz(element, attribute):
    """Read an 'x y z' attribute as three floats; a missing one means zeros."""
    if element is None or element.get(attribute) is None:
        return (0.0, 0.0, 0.0)
    x, y, z = (float(value) for value in element.get(attribute).split())
    return (x, y, z)


def center_of_mass(urdf_text: str) -> Tuple[float, Tuple[float, float, float]]:
    """
    Return the total mass [kg] and the centre of mass (x, y, z) [m] of a URDF.

    The centre of mass is relative to the root link, with all joints at zero.
    Only valid when no joint origin rotates its child link (true for this
    robot); otherwise a ValueError is raised instead of a wrong answer.
    """
    robot = ElementTree.fromstring(urdf_text)

    # For every child link: its parent link, and its offset from that parent
    parent_of = {}
    for joint in robot.findall('joint'):
        origin = joint.find('origin')
        if any(abs(angle) > 1e-9 for angle in _xyz(origin, 'rpy')):
            raise ValueError(f'joint {joint.get("name")} rotates its child link')
        parent_of[joint.find('child').get('link')] = (
            joint.find('parent').get('link'), _xyz(origin, 'xyz'))

    def position(link_name):
        """Add up the offsets from the root link down to this link."""
        if link_name not in parent_of:
            return (0.0, 0.0, 0.0)
        parent, offset = parent_of[link_name]
        return tuple(p + o for p, o in zip(position(parent), offset))

    total_mass = 0.0
    weighted = [0.0, 0.0, 0.0]
    for link in robot.findall('link'):
        inertial = link.find('inertial')
        if inertial is None:
            continue   # a link without mass, such as base_footprint
        mass = float(inertial.find('mass').get('value'))
        link_position = position(link.get('name'))
        com_in_link = _xyz(inertial.find('origin'), 'xyz')
        total_mass += mass
        for i in range(3):
            weighted[i] += mass * (link_position[i] + com_in_link[i])
    com = (weighted[0] / total_mass, weighted[1] / total_mass, weighted[2] / total_mass)
    return total_mass, com
