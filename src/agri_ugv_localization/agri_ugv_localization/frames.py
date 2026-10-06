"""Where things are: the world's GNSS origin, fixes in world coordinates, antenna offsets."""

import math
import re
from xml.etree import ElementTree

from agri_ugv_localization.gnss_errors import offset


def spherical_coordinates(sdf_text):
    """
    Return (latitude, longitude, elevation, heading_deg) of a world's origin.

    Worlds without <spherical_coordinates> get Gazebo's default (0, 0, 0, 0).
    """
    def value(tag, default):
        found = re.search(rf'<{tag}>([^<]+)</{tag}>', sdf_text)
        return float(found.group(1)) if found else default
    if '<spherical_coordinates>' not in sdf_text:
        return 0.0, 0.0, 0.0, 0.0
    tags = ('latitude_deg', 'longitude_deg', 'elevation', 'heading_deg')
    return tuple(value(t, 0.0) for t in tags)


def world_from_fix(origin, fix):
    """
    Return the world (x, y, z) [m] of a fix (lat, lon, alt).

    origin is (latitude, longitude, elevation, heading_deg) of the world; the world's axes
    are turned by heading_deg from true east and north (Gazebo's convention). Accurate to
    about a millimetre within a few hundred metres of the origin.
    """
    east, north, up = offset(origin[:3], fix)
    h = math.radians(origin[3])
    return (east * math.cos(h) + north * math.sin(h), -east * math.sin(h) + north * math.cos(h),
            up)


def antenna_levers(urdf_text, names=('gnss_front_link', 'gnss_rear_link'),
                   base='base_footprint'):
    """
    Return {link: (x, y, z)}: where each antenna sits relative to the robot's base [m].

    Adds up the joint origins from the base to each link (these joints do not rotate).
    """
    robot = ElementTree.fromstring(urdf_text)
    parent_of = {}
    for joint in robot.findall('joint'):
        origin = joint.find('origin')
        xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()] if origin is not None \
            else [0.0, 0.0, 0.0]
        parent_of[joint.find('child').get('link')] = (joint.find('parent').get('link'), xyz)
    levers = {}
    for name in names:
        position, link = [0.0, 0.0, 0.0], name
        while link != base:
            if link not in parent_of:
                raise ValueError(f'{name} is not connected to {base}')
            link, xyz = parent_of[link]
            position = [p + v for p, v in zip(position, xyz)]
        levers[name] = tuple(position)
    return levers
