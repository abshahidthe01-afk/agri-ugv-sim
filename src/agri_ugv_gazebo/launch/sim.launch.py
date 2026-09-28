"""Start Gazebo Fortress with a world and spawn the robot into it."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    world = LaunchConfiguration('world')
    model = PathJoinSubstitution(
        [FindPackageShare('agri_ugv_description'), 'urdf', 'agri_ugv.urdf.xacro'])
    robot_description = ParameterValue(Command(['xacro ', model]), value_type=str)

    return LaunchDescription([
        # Which world to load; can be changed from the command line with world:=...
        DeclareLaunchArgument(
            'world',
            default_value=PathJoinSubstitution(
                [FindPackageShare('agri_ugv_gazebo'), 'worlds', 'flat.sdf']),
            description='Full path of the Gazebo world file'),

        # Gazebo itself (-r: start running immediately)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])),
            launch_arguments={'gz_args': ['-r ', world]}.items()),

        # The robot model on /robot_description, running on simulation time
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description, 'use_sim_time': True}]),

        # Put the robot into the world, 10 cm above the ground
        Node(package='ros_gz_sim', executable='create', output='screen',
             arguments=['-topic', 'robot_description', '-name', 'agri_ugv', '-z', '0.1']),

        # Gazebo's clock -> ROS, so ROS programs run on simulation time
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock']),
    ])
