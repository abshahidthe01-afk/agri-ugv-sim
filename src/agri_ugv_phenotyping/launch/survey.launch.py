"""
Start a phenotyping survey: the coverage mission, and the plant map of the line scanners.

    ros2 launch agri_ugv_phenotyping survey.launch.py plots:=198 [cameras:=top] [rviz:=true]

The mission is agri_ugv_navigation's mission.launch.py, with the same arguments; start it
with /mission/start (std_srvs/Trigger). The plant_map node places every scanner profile
with the pose on pose_topic (default /localization/odometry, the robot's own estimate;
/ground_truth/odom for Gazebo's true pose) and saves the map in ~/plant_maps when the
mission is complete. rviz:=true opens RViz with rviz/plant_cloud.rviz: the latest plots' 3D
points as they are measured, in the cameras' colours (grey without cameras).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

MISSION_ARGUMENTS = [
    ('plots', '', 'plot_IDs separated by commas (empty: all plots)'),
    ('soil', 'rigid', 'Soil profile: rigid, firm, soft or wet'),
    ('lidar', 'true', 'true: with the 3D LiDAR on the roof, false: without'),
    ('scanners', 'true', 'true: the laser line scanners measure, false: not'),
    ('scanner_rate', '10', 'Profiles per second of each line scanner'),
    ('cameras', 'none', 'Dome cameras that take pictures: none, top or all'),
    ('camera_rate', '2', 'Pictures per second of each camera'),
    ('camera_depth', 'false', 'true: the dome cameras also measure depth'),
    ('controller', 'cpp', 'Four-wheel-steering driver: cpp or python'),
]


def generate_launch_description():
    mission = os.path.join(get_package_share_directory('agri_ugv_navigation'), 'launch',
                           'mission.launch.py')
    view = os.path.join(get_package_share_directory('agri_ugv_phenotyping'), 'rviz',
                        'plant_cloud.rviz')
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=default, description=text)
          for name, default, text in MISSION_ARGUMENTS],
        DeclareLaunchArgument('pose_topic', default_value='/localization/odometry',
                              description='Pose the plant map places the profiles with'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='true: show the plant cloud live in RViz'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(mission),
            launch_arguments={name: LaunchConfiguration(name)
                              for name, _, _ in MISSION_ARGUMENTS}.items()),
        Node(package='agri_ugv_phenotyping', executable='plant_map', output='screen',
             parameters=[{'use_sim_time': True,
                          'pose_topic': LaunchConfiguration('pose_topic')}]),
        Node(package='rviz2', executable='rviz2', arguments=['-d', view],
             parameters=[{'use_sim_time': True}],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
