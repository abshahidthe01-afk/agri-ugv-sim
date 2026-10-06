"""
ROS node: wheel odometry from the measured steering angles and wheel speeds.

Subscribes:  /robot_description (std_msgs/String, latched)  geometry source
             /joint_states      (sensor_msgs/JointState)    steering angles, wheel speeds
Publishes:   /wheel/odom        (nav_msgs/Odometry)         pose in odom, body velocity

The pose starts at zero and is dead-reckoned (it drifts); a fusion filter should use the
velocity, whose covariance grows when the wheels disagree (slip, scrubbing).
"""

import math

from agri_ugv_control.kinematics import forward_kinematics, integrate_pose
from agri_ugv_control.robot_geometry import geometry_from_urdf
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

UNUSED = 1e6               # variance of the axes a ground robot's odometry does not measure


class WheelOdometryNode(Node):
    """Turn joint states into body velocity and a dead-reckoned pose."""

    def __init__(self):
        """Read the noise parameters and connect the topics."""
        super().__init__('wheel_odometry')
        self.speed_sigma = self.declare_parameter('speed_sigma', 0.02).value   # [m/s]
        self.turn_sigma = self.declare_parameter('turn_sigma', 0.02).value     # [rad/s]
        self.modules, self.radius = None, None
        self.pose, self.last_time = (0.0, 0.0, 0.0), None
        self.publisher = self.create_publisher(Odometry, '/wheel/odom', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self.on_description, latched)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)

    def on_description(self, msg):
        """Read the wheel positions and radius from the robot model."""
        self.modules, self.radius = geometry_from_urdf(msg.data)
        self.lever = math.sqrt(sum(m.x ** 2 + m.y ** 2 for m in self.modules)
                               / len(self.modules))     # wheels' distance from the centre

    def on_joints(self, msg):
        """Compute the body velocity, advance the pose and publish both."""
        if self.modules is None:
            return
        index = {name: i for i, name in enumerate(msg.name)}
        try:
            angles = [msg.position[index[f'{m.name}_steer_joint']] for m in self.modules]
            speeds = [msg.velocity[index[f'{m.name}_wheel_joint']] for m in self.modules]
        except (KeyError, IndexError):
            return                     # not a complete message of all wheel modules
        vx, vy, wz, residual = forward_kinematics(angles, speeds, self.modules, self.radius)
        time = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if self.last_time is not None and time > self.last_time:
            self.pose = integrate_pose(*self.pose, vx, vy, wz, time - self.last_time)
        self.last_time = time

        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id, odom.child_frame_id = 'odom', 'base_footprint'
        x, y, yaw = self.pose
        odom.pose.pose.position.x, odom.pose.pose.position.y = x, y
        odom.pose.pose.orientation.z = math.sin(yaw / 2)
        odom.pose.pose.orientation.w = math.cos(yaw / 2)
        odom.twist.twist.linear.x, odom.twist.twist.linear.y = vx, vy
        odom.twist.twist.angular.z = wz
        speed_variance = self.speed_sigma ** 2 + residual ** 2
        covariance = [0.0] * 36
        covariance[0] = covariance[7] = speed_variance
        covariance[14] = covariance[21] = covariance[28] = UNUSED
        covariance[35] = self.turn_sigma ** 2 + (residual / self.lever) ** 2
        odom.twist.covariance = covariance
        self.publisher.publish(odom)


def main(args=None):
    """Run the node until Ctrl+C."""
    rclpy.init(args=args)
    node = WheelOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
