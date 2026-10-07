"""
ROS node: drive the coverage mission, segment by segment.

Plans the mission over the chosen plots of the field layout at startup, then waits. Once
started, it commands the follower's velocity for the localization's pose at 'rate' Hz
until the last segment is done. Without a recent pose it commands a stop. With the 3D
LiDAR on the robot it pauses while an obstacle (anything taller than the crops) stands
within 'stop_distance' of the robot in the way it moves, or while the LiDAR is silent,
and drives on once the way has been clear for 'clear_time'.

Subscribes:  /localization/odometry (nav_msgs/Odometry)    the robot's estimated pose
             /lidar/points (sensor_msgs/PointCloud2)        the 3D LiDAR, for obstacles
             /robot_description (std_msgs/String, latched)  where the LiDAR sits
Publishes:   /cmd_vel (geometry_msgs/Twist)                 velocity commands
             /mission/plan (std_msgs/String, latched)       the planned segments, as JSON
             /mission/status (std_msgs/String, latched)     state and progress, as JSON
Services:    /mission/start, /mission/stop (std_srvs/Trigger)
Parameters:  layout_file (default: the MuST-C plot layout of agri_ugv_field)
             plots: plot_IDs separated by commas, e.g. '198' (empty: the whole field)
             speed [m/s], rate [Hz], pose_timeout [s]
             correction_angle [rad] (0.1, read every cycle): corrections stay within this
             angle of the motion (see follow); large values switch the limit off
             obstacle_stop (true), stop_distance [m] (3.0), obstacle_height [m] (1.0),
             clear_time [s] (1.0), all read every scan; scan_timeout [s] (0.5)
States:      waiting, running, paused (for an obstacle), complete, stopped
"""

import json
import math
import os

from agri_ugv_navigation.mission import load_plan, Mission, plan_to_json, summary
from agri_ugv_navigation.obstacles import cloud_points, mount, obstacle_gap
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger


def stamp_seconds(stamp):
    """Return a message time stamp in seconds."""
    return stamp.sec + 1e-9 * stamp.nanosec


class MissionNode(Node):
    """Run a planned coverage mission on the robot's estimated pose."""

    def __init__(self):
        """Read the parameters, plan the mission and connect the topics and services."""
        super().__init__('mission')
        layout = os.path.join(get_package_share_directory('agri_ugv_field'), 'data',
                              'must_c_field_plots.csv')
        layout_file = self.declare_parameter('layout_file', layout).value
        plots = self.declare_parameter(
            'plots', '', ParameterDescriptor(dynamic_typing=True)).value
        speed = self.declare_parameter('speed', 0.5).value
        rate = self.declare_parameter('rate', 20.0).value
        self.pose_timeout = self.declare_parameter('pose_timeout', 0.5).value
        self.declare_parameter('correction_angle', 0.1)
        self.declare_parameter('obstacle_stop', True)
        self.declare_parameter('stop_distance', 3.0)
        self.declare_parameter('obstacle_height', 1.0)
        self.declare_parameter('clear_time', 1.0)
        self.scan_timeout = self.declare_parameter('scan_timeout', 0.5).value
        with open(layout_file) as text:
            segments, heading, plot_ids = load_plan(text.read(), '' if plots is None else plots)
        self.mission = Mission(segments, heading, speed)
        parts = ', '.join(f'{n} {kind}{"es" if kind == "pass" else "s"} {length:.1f} m'
                          for kind, (n, length) in summary(segments).items())
        self.get_logger().info(f'Plan over {len(plot_ids)} plot{"s" * (len(plot_ids) > 1)}: '
                               f'{len(segments)} segments ({parts}), '
                               f'heading {math.degrees(heading):.2f} deg')

        self.pose, self.pose_time, self.started_at = None, None, None
        self.lidar, self.gap, self.scan_time = None, None, None   # LiDAR mount, newest scan
        self.paused_at, self.clear_since = None, None
        self.state = 'waiting'

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.command_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/mission/status', latched)
        self.plan_pub = self.create_publisher(String, '/mission/plan', latched)
        self.plan_pub.publish(String(data=plan_to_json(segments, heading, plot_ids)))
        self.create_subscription(Odometry, '/localization/odometry', self.on_pose, 10)
        self.create_subscription(String, '/robot_description', self.on_description, latched)
        # reliable: a scan is over 1 MB, and best-effort delivery lost whole scans in Gazebo
        self.create_subscription(PointCloud2, '/lidar/points', self.on_scan,
                                 QoSProfile(depth=1))
        self.create_service(Trigger, '/mission/start', self.on_start)
        self.create_service(Trigger, '/mission/stop', self.on_stop)
        self.publish_status()
        self.create_timer(1.0 / rate, self.update)
        self.get_logger().info('Waiting for /mission/start')

    def on_pose(self, msg):
        """Keep the newest estimated pose (x, y, yaw) and its time."""
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        self.pose_time = stamp_seconds(msg.header.stamp)

    def on_description(self, msg):
        """Find where the LiDAR sits on the robot (or that there is none)."""
        try:
            self.lidar = mount(msg.data)
        except ValueError:
            self.lidar = None
            self.get_logger().info('No LiDAR on the robot: no stops for obstacles')

    def on_scan(self, msg):
        """Measure the free way ahead in the newest LiDAR scan, along the current segment."""
        if self.lidar is None or self.pose is None or self.mission.finished:
            return
        rotation, translation = self.lidar
        points = cloud_points(msg.fields, msg.point_step, msg.width * msg.height, msg.data)
        segment = self.mission.segments[self.mission.index]
        dx, dy = (end - start for start, end in zip(segment['start'], segment['end']))
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        self.gap = obstacle_gap(points @ rotation.T + translation,
                                (c * dx + s * dy, -s * dx + c * dy),
                                reach=self.get_parameter('stop_distance').value,
                                height=self.get_parameter('obstacle_height').value)
        self.scan_time = stamp_seconds(msg.header.stamp)

    def on_start(self, request, response):
        """Start driving (only once)."""
        response.success = self.state == 'waiting'
        if response.success:
            self.state = 'running'
            self.started_at = self.now()
            self.publish_status()
            self.log_segment()
            response.message = 'mission started'
        else:
            response.message = f'mission is {self.state}, not waiting'
        return response

    def on_stop(self, request, response):
        """Stop driving before the mission is complete."""
        response.success = self.state in ('running', 'paused')
        if response.success:
            self.state = 'stopped'
            self.command_pub.publish(Twist())
            self.publish_status()
            self.get_logger().info(f'Mission stopped after {self.elapsed():.1f} s, '
                                   f'{self.mission.index} of {len(self.mission.segments)} '
                                   f'segments done')
        response.message = f'mission is {self.state}'
        return response

    def now(self):
        """Return the node's time in seconds (simulation time with use_sim_time)."""
        return self.get_clock().now().nanoseconds * 1e-9

    def elapsed(self):
        """Return the seconds since the start."""
        return self.now() - self.started_at

    def blocked(self):
        """Tell whether to stand still for an obstacle; switch between running and paused."""
        if self.lidar is None or not self.get_parameter('obstacle_stop').value:
            if self.state == 'paused':
                self.resume('obstacle stop switched off')
            return False
        now = self.now()
        silent = self.scan_time is None or now - self.scan_time > self.scan_timeout
        if silent or self.gap is not None:
            self.clear_since = None
            if self.state == 'running':
                self.state, self.paused_at = 'paused', now
                if not silent:
                    why = f'obstacle {self.gap:.2f} m ahead'
                elif self.scan_time is None:
                    why = 'no LiDAR scan yet'
                else:
                    why = f'no LiDAR scan for {now - self.scan_time:.2f} s'
                self.get_logger().info(f'Paused: {why} (t = {self.elapsed():.1f} s)')
                self.publish_status()
            return True
        if self.state == 'paused':
            if self.clear_since is None:
                self.clear_since = now
            if now - self.clear_since < self.get_parameter('clear_time').value:
                return True
            self.resume('the way is clear')
        return False

    def resume(self, why):
        """Drive on after a pause."""
        self.state = 'running'
        self.get_logger().info(f'Driving on: {why} (paused for '
                               f'{self.now() - self.paused_at:.1f} s, t = {self.elapsed():.1f} s)')
        self.publish_status()

    def update(self):
        """Command the follower's velocity for the newest pose."""
        if self.state not in ('running', 'paused'):
            return
        if self.pose is None or self.now() - self.pose_time > self.pose_timeout:
            self.command_pub.publish(Twist())
            self.get_logger().warn('No recent pose from /localization/odometry: standing still',
                                   throttle_duration_sec=5.0)
            return
        if self.blocked():
            self.command_pub.publish(Twist())
            return
        self.mission.options['max_angle'] = self.get_parameter('correction_angle').value
        (vx, vy, wz), changed = self.mission.step(self.pose)
        command = Twist()
        command.linear.x, command.linear.y, command.angular.z = vx, vy, wz
        self.command_pub.publish(command)
        if self.mission.finished:
            self.state = 'complete'
            self.get_logger().info(f'Mission complete: {len(self.mission.segments)} segments '
                                   f'in {self.elapsed():.1f} s')
        elif changed:
            self.log_segment()
        if changed:
            self.publish_status()

    def log_segment(self):
        """Log the segment that has just begun."""
        segment = self.mission.segments[self.mission.index]
        where = f' over plot {segment["plot_id"]}' if segment['plot_id'] is not None else ''
        length = math.dist(segment['start'], segment['end'])
        self.get_logger().info(f'Segment {self.mission.index + 1}/{len(self.mission.segments)}: '
                               f'{segment["kind"]}{where}, {length:.2f} m '
                               f'(t = {self.elapsed():.1f} s)')

    def publish_status(self):
        """Publish the state and progress as JSON."""
        self.status_pub.publish(String(data=json.dumps({
            'state': self.state, 'done': self.mission.index,
            'count': len(self.mission.segments)})))


def main(args=None):
    """Run the node until Ctrl+C."""
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
