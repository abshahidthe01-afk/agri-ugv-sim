"""Ideal (physics-free) robot in RViz, driven by /cmd_vel."""

from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    model = PathJoinSubstitution(
        [FindPackageShare('agri_ugv_description'), 'urdf', 'agri_ugv.urdf.xacro'])
    rviz_config = PathJoinSubstitution(
        [FindPackageShare('agri_ugv_control'), 'rviz', 'kinematic_sim.rviz'])
    robot_description = ParameterValue(Command(['xacro ', model]), value_type=str)

    return LaunchDescription([
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}]),
        Node(package='agri_ugv_control', executable='kinematic_sim', output='screen'),
        Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_config]),
    ])
