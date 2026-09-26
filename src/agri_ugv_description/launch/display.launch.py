from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("agri_ugv_description")

    # Where the model file and the RViz view setup are, once installed
    xacro_file = PathJoinSubstitution([pkg_share, "urdf", "agri_ugv.urdf.xacro"])
    rviz_config = PathJoinSubstitution([pkg_share, "rviz", "display.rviz"])

    # Run xacro at launch time and keep the result as plain text
    robot_description = ParameterValue(Command(["xacro ", xacro_file]), value_type=str)

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", rviz_config],   # -d = load this display config
        ),
    ])
