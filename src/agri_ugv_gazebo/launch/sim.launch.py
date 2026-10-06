"""Start Gazebo Fortress with a world, spawn the robot, and start its controllers."""

from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare('agri_ugv_gazebo')
    world_file = PathJoinSubstitution([share, 'worlds', [LaunchConfiguration('world'), '.sdf']])
    model = PathJoinSubstitution(
        [FindPackageShare('agri_ugv_description'), 'urdf', 'agri_ugv.urdf.xacro'])
    controllers = PathJoinSubstitution(
        [FindPackageShare('agri_ugv_control'), 'config', 'controllers.yaml'])
    robot_description = ParameterValue(
        Command(['xacro ', model, ' controllers_file:=', controllers,
                 ' soil:=', LaunchConfiguration('soil')]), value_type=str)

    def spawner(controller):
        """Ask the controller manager to load and start one controller."""
        return Node(package='controller_manager', executable='spawner',
                    arguments=[controller], output='screen')

    return LaunchDescription([
        # Which world to load: a file name from this package's worlds folder, without .sdf
        DeclareLaunchArgument(
            'world', default_value='flat',
            description='World name, e.g. flat or flat_mesh (loads worlds/<name>.sdf)'),
        DeclareLaunchArgument(
            'soil', default_value='rigid',
            description='Soil profile: rigid (no slip model), firm, soft or wet'),

        # Let Gazebo find our generated terrain models (model://<name>) in the models folder
        AppendEnvironmentVariable(
            'IGN_GAZEBO_RESOURCE_PATH', PathJoinSubstitution([share, 'models'])),

        # Gazebo itself (-r: start running immediately)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])),
            launch_arguments={'gz_args': ['-r ', world_file]}.items()),

        # The robot model (with ros2_control) on /robot_description, on simulation time
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description, 'use_sim_time': True}]),

        # Put the robot into the world, 10 cm above the ground
        Node(package='ros_gz_sim', executable='create', output='screen',
             arguments=['-topic', 'robot_description', '-name', 'agri_ugv', '-z', '0.1']),

        # Bridge Gazebo -> ROS ('[' means one direction only, Gazebo to ROS):
        # - the simulation clock, so ROS programs run on simulation time
        # - the true pose and velocity of the robot, from the model's odometry plugin
        # - the IMU, and the GNSS antennas' ideal fixes (errors are added below)
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
                        '/ground_truth/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
                        '/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU',
                        '/gnss/front/fix_ideal@sensor_msgs/msg/NavSatFix[ignition.msgs.NavSat',
                        '/gnss/rear/fix_ideal@sensor_msgs/msg/NavSatFix[ignition.msgs.NavSat']),

        # Controllers: they wait until the robot (and its controller manager) exists
        spawner('joint_state_broadcaster'),
        spawner('steering_controller'),
        spawner('wheel_controller'),

        # Our driver: /cmd_vel -> steering angles and wheel speeds, on simulation time
        Node(package='agri_ugv_control', executable='four_ws_driver', output='screen',
             parameters=[{'use_sim_time': True}]),

        # Wheel odometry: body velocity from the measured steering angles and wheel speeds
        Node(package='agri_ugv_control', executable='wheel_odometry', output='screen',
             parameters=[{'use_sim_time': True}]),

        # Simulated RTK receiver: ideal fixes + realistic errors, and the dual-antenna heading
        Node(package='agri_ugv_localization', executable='gnss_errors', output='screen',
             parameters=[{'use_sim_time': True}]),

        # Localization: EKF fusing wheel odometry, gyro and both GNSS antennas
        Node(package='agri_ugv_localization', executable='localization', output='screen',
             parameters=[{'use_sim_time': True, 'world_file': world_file}]),
    ])
