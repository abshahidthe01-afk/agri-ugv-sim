"""
Kinematic simulation of the four-wheel-steering robot (no physics).

Turns velocity commands into wheel motion and robot motion, exactly as an
ideal robot would move: no slip, no delays. Used to check the kinematics
visually in RViz before any physics simulation is involved.

Subscribes:  /robot_description  (std_msgs/String)       geometry source
             /cmd_vel            (geometry_msgs/Twist)    motion commands
Publishes:   /joint_states       (sensor_msgs/JointState) steering and wheel angles
             TF odom -> base_footprint                    where the robot is
"""

import math

from agri_ugv_control.kinematics import inverse_kinematics
from agri_ugv_control.robot_geometry import geometry_from_urdf
from geometry_msgs.msg import TransformStamped, Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

UPDATE_RATE = 50.0       # [Hz]
COMMAND_TIMEOUT = 0.5    # [s] no new command for this long -> stop (watchdog)


class KinematicSim(Node):
    """Ideal four-wheel-steering robot: moves exactly as commanded."""

    def __init__(self):
        super().__init__('kinematic_sim')

        # Geometry arrives from the robot model; nothing moves until it does
        self.modules = None
        self.wheel_radius = None

        # Latest command and when it arrived
        self.command = Twist()
        self.command_time = None

        # State of the simulated robot
        self.x = self.y = self.yaw = 0.0
        self.steer_angles = {}
        self.wheel_angles = {}

        # robot_state_publisher publishes the model once, as a 'noticeboard' message
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self.on_description, latched)
        self.create_subscription(Twist, '/cmd_vel', self.on_command, 10)

        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.dt = 1.0 / UPDATE_RATE
        self.create_timer(self.dt, self.update)

    def on_description(self, msg):
        """Read the wheel geometry from the robot model."""
        self.modules, self.wheel_radius = geometry_from_urdf(msg.data)
        for m in self.modules:
            self.steer_angles[m.name] = 0.0
            self.wheel_angles[m.name] = 0.0
        positions = ', '.join(f'{m.name} ({m.x:+.3f}, {m.y:+.3f})' for m in self.modules)
        self.get_logger().info(
            f'Geometry from robot model: wheel radius {self.wheel_radius:.3f} m, {positions}')

    def on_command(self, msg):
        """Store the newest velocity command and when it arrived."""
        self.command = msg
        self.command_time = self.get_clock().now()

    def update(self):
        """Advance the ideal robot by one time step and publish its state."""
        if self.modules is None:
            return   # no geometry yet

        # Watchdog: a command that has not been renewed recently means stop
        vx = vy = wz = 0.0
        if self.command_time is not None:
            age = (self.get_clock().now() - self.command_time).nanoseconds * 1e-9
            if age < COMMAND_TIMEOUT:
                vx = self.command.linear.x
                vy = self.command.linear.y
                wz = self.command.angular.z

        # Wheels: steering angle and spin, from the tested kinematics
        commands = inverse_kinematics(vx, vy, wz, self.modules, self.wheel_radius)
        for m, cmd in zip(self.modules, commands):
            if cmd.steer_angle is not None:
                self.steer_angles[m.name] = cmd.steer_angle
            self.wheel_angles[m.name] += cmd.wheel_speed * self.dt

        # Robot: move in the direction it is facing (robot frame -> odom frame)
        cos_yaw, sin_yaw = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (vx * cos_yaw - vy * sin_yaw) * self.dt
        self.y += (vx * sin_yaw + vy * cos_yaw) * self.dt
        self.yaw += wz * self.dt

        self.publish_state()

    def publish_state(self):
        """Publish joint angles (for the wheels) and TF (for the whole robot)."""
        now = self.get_clock().now().to_msg()

        names, positions = [], []
        for m in self.modules:
            names += [f'{m.name}_steer_joint', f'{m.name}_suspension_joint',
                      f'{m.name}_wheel_joint']
            positions += [self.steer_angles[m.name], 0.0, self.wheel_angles[m.name]]
        joints = JointState()
        joints.header.stamp = now
        joints.name = names
        joints.position = positions
        self.joint_pub.publish(joints)

        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_footprint'
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        # A rotation about the vertical axis, written as a quaternion
        tf.transform.rotation.z = math.sin(self.yaw / 2)
        tf.transform.rotation.w = math.cos(self.yaw / 2)
        self.tf_broadcaster.sendTransform(tf)


def main():
    """Start the node and keep it running until Ctrl+C."""
    rclpy.init()
    node = KinematicSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
