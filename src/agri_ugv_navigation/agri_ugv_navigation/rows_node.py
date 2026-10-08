"""
ROS node: measure where the robot really is across the crop rows, with the 3D LiDAR.

For a LiDAR scan near a plot of row crops it compares the rows with where the field map
and the estimated pose put them (agri_ugv_navigation.rows): how far the robot really is
to the left of its estimate across the rows, with the right row found from the plot's
edges, and how far it is turned. It only measures; the localization can use it.

Subscribes:  /lidar/points (sensor_msgs/PointCloud2)        the 3D LiDAR (best effort)
             /localization/odometry (nav_msgs/Odometry)    the estimate, kept for 4 s
             /robot_description (std_msgs/String, latched)  where the LiDAR sits
Publishes:   /rows/measurement (std_msgs/String, JSON)      one per measured scan: t (the
             scan's time), plot, crop, left [m], left_plot [m] (null: which row is not
             clear), turn [rad], strength, margin, beyond, count, spacing [m], left_dir
             (world direction of 'left'), pose (the estimate the rows were compared with,
             x, y, yaw), ms (time it took)
Parameters:  layout_file (default: the MuST-C plot layout of agri_ugv_field)
             rate [Hz] (5, read every scan): scans closer together are skipped
             max_points (1500, read every scan): plant points used per scan, for speed
"""

from collections import deque
import json
import math
import os
import time

from agri_ugv_field.layout import read_layout_csv
from agri_ugv_navigation.obstacles import cloud_points, mount
from agri_ugv_navigation.rows import plot_under, pose_at, row_measurement
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

REPORT_EVERY = 100          # measured scans between two log lines


def stamp_seconds(msg):
    """Return a message's time stamp in seconds."""
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


def yaw_of(q):
    """Return the heading of a quaternion."""
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


def measurement_json(t, plot, pose, m, ms):
    """Return a row measurement (see row_measurement) as the JSON text of /rows/measurement."""
    return json.dumps({
        't': t, 'plot': plot['plot_id'], 'crop': plot['crop'], 'left': m['left'],
        'left_plot': m['left_plot'], 'turn': m['turn'], 'strength': m['strength'],
        'margin': m['margin'], 'beyond': m['beyond'], 'count': m['count'],
        'spacing': m['spacing'], 'left_dir': list(m['left_dir']), 'pose': list(pose),
        'ms': ms})


class RowsNode(Node):
    """Measure the robot's place across the crop rows in LiDAR scans."""

    def __init__(self):
        """Read the plot layout and connect the topics."""
        super().__init__('rows')
        layout = os.path.join(get_package_share_directory('agri_ugv_field'), 'data',
                              'must_c_field_plots.csv')
        with open(self.declare_parameter('layout_file', layout).value) as text:
            self.plots = [p for p in read_layout_csv(text.read()) if p['type'] == 'plot']
        self.declare_parameter('rate', 5.0)
        self.declare_parameter('max_points', 1500)
        self.history = deque(maxlen=400)                 # (t, x, y, yaw) of the estimate
        self.lidar, self.last = None, None
        self.ms, self.found, self.tried = [], 0, 0
        self.publisher = self.create_publisher(String, '/rows/measurement', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self.on_description, latched)
        self.create_subscription(Odometry, '/localization/odometry', self.on_pose, 100)
        # best effort: a scan lost now and then does not matter here (half are skipped
        # anyway), and lost scans are not sent again, which spares the bridge that also
        # feeds the mission node's obstacle check
        self.create_subscription(PointCloud2, '/lidar/points', self.on_scan,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))

    def on_description(self, msg):
        """Find where the LiDAR sits on the robot (or that there is none)."""
        try:
            self.lidar = mount(msg.data)
        except ValueError:
            self.lidar = None
            self.get_logger().info('No LiDAR on the robot: no row measurements')

    def on_pose(self, msg):
        """Keep the estimated poses of the last seconds."""
        p = msg.pose.pose.position
        self.history.append((stamp_seconds(msg), p.x, p.y, yaw_of(msg.pose.pose.orientation)))

    def on_scan(self, msg):
        """Measure the rows in a scan near a plot (at most 'rate' times per second)."""
        if self.lidar is None:
            return
        t = stamp_seconds(msg)
        rate = self.get_parameter('rate').value
        if self.last is not None and t - self.last < 1.0 / rate - 1e-3:
            return
        pose = pose_at(self.history, t)
        plot = None if pose is None else plot_under(pose, self.plots)
        if plot is None:
            return
        self.last = t
        start = time.perf_counter()
        rotation, translation = self.lidar
        points = cloud_points(msg.fields, msg.point_step, msg.width * msg.height, msg.data)
        m = row_measurement(points @ rotation.T + translation, pose, plot,
                            max_points=self.get_parameter('max_points').value)
        ms = 1000 * (time.perf_counter() - start)
        self.tried += 1
        self.ms.append(ms)
        if m is not None:
            self.found += m['left_plot'] is not None
            self.publisher.publish(String(data=measurement_json(t, plot, pose, m, ms)))
        if len(self.ms) == REPORT_EVERY:
            self.get_logger().info(
                f'Rows: {self.tried} scans near plots, the right row found in {self.found}; '
                f'{sum(self.ms) / len(self.ms):.0f} ms per scan (mean of {REPORT_EVERY})')
            self.ms, self.found, self.tried = [], 0, 0


def main(args=None):
    """Run the node until Ctrl+C."""
    rclpy.init(args=args)
    node = RowsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
