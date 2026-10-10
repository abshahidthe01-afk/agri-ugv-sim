"""
Start the field simulation with the robot at the start of a coverage mission, and the mission.

    ros2 launch agri_ugv_navigation mission.launch.py plots:=198

plots: plot_IDs separated by commas, e.g. 178,177,198,197 (default: the whole field).
lidar: true (default) or false, with or without the 3D LiDAR on the roof.
scanners: true (default) or false, the laser line scanners measuring or not;
scanner_rate: their profiles per second (default 10).
controller: python (default) or cpp, the four-wheel-steering driver (see sim.launch.py).
The robot is placed where the mission begins, facing along the crop rows; the mission
node then waits for /mission/start (std_srvs/Trigger). The rows node measures the robot's
place across the crop rows in the LiDAR scans (/rows/measurement).
"""

import math
import os

from agri_ugv_field.ground import read_obj_grid
from agri_ugv_navigation.mission import load_plan, spawn_height, start_pose, summary
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, LogInfo,
                            OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def mission(context):
    """Plan the mission; start the simulation with the robot at its start, and the node."""
    plots = LaunchConfiguration('plots').perform(context).strip()
    field = get_package_share_directory('agri_ugv_field')
    gazebo = get_package_share_directory('agri_ugv_gazebo')
    with open(os.path.join(field, 'data', 'must_c_field_plots.csv')) as text:
        segments, heading, plot_ids = load_plan(text.read(), plots)
    x, y, yaw = start_pose(segments, heading)
    with open(os.path.join(gazebo, 'models', 'must_c_field', 'meshes',
                           'must_c_field.obj')) as text:
        z = spawn_height(read_obj_grid(text.read()), x, y)
    length = sum(metres for _, metres in summary(segments).values())
    parameters = {'use_sim_time': True}
    if plots:
        parameters['plots'] = ParameterValue(plots, value_type=str)
    return [
        LogInfo(msg=f'Mission over {len(plot_ids)} plot{"s" * (len(plot_ids) > 1)}: '
                    f'{len(segments)} segments, '
                    f'{length:.1f} m; robot placed at x {x:.2f} y {y:.2f} z {z:.2f}, '
                    f'facing {math.degrees(yaw):.2f} deg'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo, 'launch', 'sim.launch.py')),
            launch_arguments={'world': 'must_c_field', 'soil': LaunchConfiguration('soil'),
                              'lidar': LaunchConfiguration('lidar'),
                              'scanners': LaunchConfiguration('scanners'),
                              'scanner_rate': LaunchConfiguration('scanner_rate'),
                              'controller': LaunchConfiguration('controller'),
                              'x': f'{x:.3f}', 'y': f'{y:.3f}', 'z': f'{z:.3f}',
                              'yaw': f'{yaw:.5f}'}.items()),
        Node(package='agri_ugv_navigation', executable='mission', output='screen',
             parameters=[parameters]),
        Node(package='agri_ugv_navigation', executable='rows', output='screen',
             parameters=[{'use_sim_time': True}]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('plots', default_value='',
                              description='plot_IDs separated by commas (empty: all plots)'),
        DeclareLaunchArgument('soil', default_value='rigid',
                              description='Soil profile: rigid, firm, soft or wet'),
        DeclareLaunchArgument('lidar', default_value='true',
                              description='true: with the 3D LiDAR on the roof, false: without'),
        DeclareLaunchArgument('scanners', default_value='true',
                              description='true: the laser line scanners measure, false: not'),
        DeclareLaunchArgument('scanner_rate', default_value='10',
                              description='Profiles per second of each line scanner'),
        DeclareLaunchArgument('controller', default_value='python',
                              description='Four-wheel-steering driver: python or cpp'),
        OpaqueFunction(function=mission),
    ])
