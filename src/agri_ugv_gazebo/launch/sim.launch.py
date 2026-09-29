"""Start Gazebo Fortress with a world, spawn the robot, and start its controllers."""

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
    controllers = PathJoinSubstitution(
        [FindPackageShare('agri_ugv_control'), 'config', 'controllers.yaml'])
    robot_description = ParameterValue(
        Command(['xacro ', model, ' controllers_file:=', controllers]), value_type=str)

    def spawner(controller):
        """Ask the controller manager to load and start one controller."""
        return Node(package='controller_manager', executable='spawner',
                    arguments=[controller], output='screen')

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

        # The robot model (with ros2_control) on /robot_description, on simulation time
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description, 'use_sim_time': True}]),

        # Put the robot into the world, 10 cm above the ground
        Node(package='ros_gz_sim', executable='create', output='screen',
             arguments=['-topic', 'robot_description', '-name', 'agri_ugv', '-z', '0.1']),

        # Gazebo's clock -> ROS, so ROS programs run on simulation time
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock']),

        # Controllers: they wait until the robot (and its controller manager) exists
        spawner('joint_state_broadcaster'),
        spawner('steering_controller'),
        spawner('wheel_controller'),
        
        # Our driver: /cmd_vel -> steering angles and wheel speeds, on simulation time
        Node(package='agri_ugv_control', executable='four_ws_driver', output='screen',
             parameters=[{'use_sim_time': True}]),
    ])
