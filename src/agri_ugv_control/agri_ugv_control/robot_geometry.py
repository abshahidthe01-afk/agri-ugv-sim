"""
Read the wheel-module geometry from the robot description (URDF text).

The robot model is the single source of truth for the geometry: this module
extracts the numbers from it, so they are never typed into the code.
"""

import math
from typing import List, Tuple
import xml.etree.ElementTree as ElementTree

from agri_ugv_control.kinematics import WheelModule

MODULE_NAMES = ('front_left', 'front_right', 'rear_left', 'rear_right')
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


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


def _rotation(rpy):
    """Return the 3 x 3 rotation (nested tuples) of URDF roll, pitch, yaw angles."""
    cr, sr = math.cos(rpy[0]), math.sin(rpy[0])
    cp, sp = math.cos(rpy[1]), math.sin(rpy[1])
    cy, sy = math.cos(rpy[2]), math.sin(rpy[2])
    return ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr))


def _apply(rotation, vector):
    """Return rotation times vector."""
    return tuple(sum(r * v for r, v in zip(row, vector)) for row in rotation)


def _compose(first, second):
    """Return the product of two rotations."""
    return tuple(tuple(sum(first[i][k] * second[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def center_of_mass(urdf_text: str) -> Tuple[float, Tuple[float, float, float]]:
    """
    Return the total mass [kg] and the centre of mass (x, y, z) [m] of a URDF.

    The centre of mass is relative to the root link, with all joints at zero. Joint
    origins may rotate their child links (the payload's cameras and scanners do).
    """
    robot = ElementTree.fromstring(urdf_text)

    # For every child link: its parent link, and its pose relative to that parent
    parent_of = {}
    for joint in robot.findall('joint'):
        origin = joint.find('origin')
        parent_of[joint.find('child').get('link')] = (
            joint.find('parent').get('link'), _xyz(origin, 'xyz'),
            _rotation(_xyz(origin, 'rpy')))

    def pose(link_name):
        """Chain the joint poses from the root link down to this link: (rotation, position)."""
        if link_name not in parent_of:
            return IDENTITY, (0.0, 0.0, 0.0)
        parent, offset, rotation = parent_of[link_name]
        parent_rotation, parent_position = pose(parent)
        position = tuple(p + o for p, o in zip(parent_position, _apply(parent_rotation, offset)))
        return _compose(parent_rotation, rotation), position

    total_mass = 0.0
    weighted = [0.0, 0.0, 0.0]
    for link in robot.findall('link'):
        inertial = link.find('inertial')
        if inertial is None:
            continue   # a link without mass, such as base_footprint
        mass = float(inertial.find('mass').get('value'))
        rotation, position = pose(link.get('name'))
        com = _apply(rotation, _xyz(inertial.find('origin'), 'xyz'))
        total_mass += mass
        for i in range(3):
            weighted[i] += mass * (position[i] + com[i])
    com = (weighted[0] / total_mass, weighted[1] / total_mass, weighted[2] / total_mass)
    return total_mass, com
