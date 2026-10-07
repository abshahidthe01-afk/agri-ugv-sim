"""
Four-wheel-steering driver: turns /cmd_vel into commands for the joint controllers.

Uses the same tested inverse kinematics as the kinematic simulation, with the
wheel geometry read from the robot model. Every command first passes through
a speed and acceleration limiter, so the robot never receives a motion it
cannot follow safely. Works with any ros2_control hardware (Gazebo now, real
motors later), because it only talks to the controllers.

Subscribes:  /robot_description             (std_msgs/String)            geometry source
             /cmd_vel                       (geometry_msgs/Twist)         motion commands
             /joint_states                  (sensor_msgs/JointState)      measured steering
Publishes:   /steering_controller/commands  (std_msgs/Float64MultiArray)  4 angles [rad]
             /wheel_controller/commands     (std_msgs/Float64MultiArray)  4 speeds [rad/s]
Parameters:  max_linear_x, max_linear_y [m/s], max_angular_z [rad/s],
             max_accel_x, max_accel_y [m/s^2], max_accel_z [rad/s^2]
             wait_for_steering (true): steer first, roll only when the wheels point
             within steer_tolerance [rad] of the new angles (avoids skidding wheels)
             ramp_together (true): speed up and brake in all directions together, so
             the wheels keep their angles meanwhile (false: each direction on its own)
             limit_margin [rad] (0.15): a wheel asked to point this little past its
             +-90 deg steering limit stays at the limit instead of swinging round
"""

from agri_ugv_control.kinematics import inverse_kinematics
from agri_ugv_control.robot_geometry import geometry_from_urdf
from agri_ugv_control.steering import steer_first
from agri_ugv_control.velocity_limiter import limit_velocity, VelocityLimits
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

UPDATE_RATE = 50.0       # [Hz]
COMMAND_TIMEOUT = 0.5    # [s] no new command for this long -> stop (watchdog)


class FourWsDriver(Node):
    """Send steering angles and wheel speeds that realise the commanded robot motion."""

    def __init__(self):
        super().__init__('four_ws_driver')

        # Limits are ROS parameters: they can be changed without editing the code
        self.limits = VelocityLimits(
            max_vx=self.declare_parameter('max_linear_x', 1.5).value,
            max_vy=self.declare_parameter('max_linear_y', 1.0).value,
            max_wz=self.declare_parameter('max_angular_z', 1.0).value,
            max_ax=self.declare_parameter('max_accel_x', 1.0).value,
            max_ay=self.declare_parameter('max_accel_y', 1.0).value,
            max_az=self.declare_parameter('max_accel_z', 1.0).value,
        )
        self.get_logger().info(f'Velocity limits: {self.limits}')
        self.declare_parameter('wait_for_steering', True)        # read every cycle
        self.declare_parameter('ramp_together', True)            # read every cycle
        self.declare_parameter('limit_margin', 0.15)             # read every cycle
        self.tolerance = self.declare_parameter('steer_tolerance', 0.05).value   # [rad]
        self.measured = None     # measured steering angle per module, from /joint_states

        self.modules = None
        self.wheel_radius = None
        self.steer_angles = []   # last commanded angle per module, kept while stopped

        self.command = Twist()
        self.command_time = None
        self.velocity = (0.0, 0.0, 0.0)   # the limited velocity actually being driven

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self.on_description, latched)
        self.create_subscription(Twist, '/cmd_vel', self.on_command, 10)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)

        self.steer_pub = self.create_publisher(
            Float64MultiArray, '/steering_controller/commands', 10)
        self.wheel_pub = self.create_publisher(
            Float64MultiArray, '/wheel_controller/commands', 10)

        self.dt = 1.0 / UPDATE_RATE
        self.create_timer(self.dt, self.update)

    def on_description(self, msg):
        """Read the wheel geometry from the robot model."""
        self.modules, self.wheel_radius = geometry_from_urdf(msg.data)
        self.steer_angles = [0.0] * len(self.modules)
        self.get_logger().info(
            f'Geometry from robot model: {len(self.modules)} wheel modules, '
            f'wheel radius {self.wheel_radius:.3f} m')

    def on_joints(self, msg):
        """Remember the measured steering angles, in module order."""
        if self.modules is None:
            return
        index = {name: i for i, name in enumerate(msg.name)}
        try:
            self.measured = [msg.position[index[f'{m.name}_steer_joint']]
                             for m in self.modules]
        except (KeyError, IndexError):
            pass                 # not a complete message of all steering joints

    def on_command(self, msg):
        """Store the newest velocity command and when it arrived."""
        self.command = msg
        self.command_time = self.get_clock().now()

    def update(self):
        """Compute and send the wheel commands for the current velocity command."""
        if self.modules is None:
            return   # no geometry yet

        # Watchdog: a command that has not been renewed recently means stop
        target = (0.0, 0.0, 0.0)
        if self.command_time is not None:
            age = (self.get_clock().now() - self.command_time).nanoseconds * 1e-9
            if age < COMMAND_TIMEOUT:
                target = (self.command.linear.x, self.command.linear.y,
                          self.command.angular.z)

        # Limiter: cap the speed and ramp towards it (also makes watchdog stops gradual).
        # With wait_for_steering, wheels only roll once they point the right way.
        together = self.get_parameter('ramp_together').value
        margin = self.get_parameter('limit_margin').value
        if self.get_parameter('wait_for_steering').value and self.measured is not None:
            self.velocity, commands = steer_first(
                target, self.velocity, self.measured, self.modules, self.wheel_radius,
                self.limits, self.dt, self.tolerance, together, margin)
        else:
            self.velocity = limit_velocity(target, self.velocity, self.limits, self.dt,
                                           together)
            commands = inverse_kinematics(*self.velocity, self.modules, self.wheel_radius,
                                          margin)
        for i, cmd in enumerate(commands):
            if cmd.steer_angle is not None:
                self.steer_angles[i] = cmd.steer_angle

        self.steer_pub.publish(Float64MultiArray(data=list(self.steer_angles)))
        self.wheel_pub.publish(Float64MultiArray(data=[cmd.wheel_speed for cmd in commands]))


def main():
    """Start the node and keep it running until Ctrl+C."""
    rclpy.init()
    node = FourWsDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
