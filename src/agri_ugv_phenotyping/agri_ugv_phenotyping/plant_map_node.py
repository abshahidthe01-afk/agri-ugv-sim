"""
ROS node: map the plants of every plot with the laser line scanners while the robot drives.

Each profile is placed with the robot's pose at its time (agri_ugv_phenotyping.plant_map).
When no new points have reached a plot for 'quiet' seconds (the robot has left it), the
plot's traits are logged; when the mission is complete, or on request, the map is saved.

Subscribes:  /scanners/left, /scanners/right (sensor_msgs/LaserScan)   the profiles
             /localization/odometry (nav_msgs/Odometry)   the pose, kept for 4 s (topic:
                                            parameter pose_topic, e.g. /ground_truth/odom)
             /robot_description (std_msgs/String, latched)   where the scanners sit
             /mission/status (std_msgs/String, latched)   saves the map when 'complete'
Publishes:   /plant_map/cloud (sensor_msgs/PointCloud2, frame 'map')   the cells higher
             than 'low' at their height, every publish_period seconds
Services:    /plant_map/save (std_srvs/Trigger)   save the map now
Parameters:  layout_file (default: the MuST-C plot layout of agri_ugv_field), cell [m]
             (0.05), low [m] (0.06), pose_topic, publish_period [s] (5.0), quiet [s] (30.0),
             voxel [m] (0.01; 0: none): every plot's 3D point cloud, merged per voxel
             (agri_ugv_phenotyping.cloud), output (default ~/plant_maps/plant_map): the map
             is saved as output.npz (rasters), output.json (traits per plot) and
             output_cloud_<plot_id>.ply (the clouds)
"""

import array
from collections import deque
import json
import math
import os

from agri_ugv_field.layout import read_layout_csv
from agri_ugv_navigation.obstacles import mount
from agri_ugv_navigation.rows import pose_at
from agri_ugv_phenotyping.plant_map import FieldMap, profile_points
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import String
from std_srvs.srv import Trigger

SIDES = ('left', 'right')


def stamp_seconds(msg):
    """Return a message's time stamp in seconds."""
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


def yaw_of(q):
    """Return the heading of a quaternion."""
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


def traits_text(t):
    """Return one plot's traits (PlotMap.traits) as a log line."""
    text = (f'plot {t["plot"]} ({t["crop"]}): {100 * t["seen"]:.0f} % seen, cover '
            f'{100 * t["cover"]:.0f} %')
    if t['height'] is not None:
        text += (f', canopy height {100 * t["height"]:.0f} cm (median '
                 f'{100 * t["height_median"]:.0f} cm)')
    if t['rows_offset'] is not None:
        text += (f', rows {100 * t["rows_offset"]:+.1f} cm from the field map (sharpness '
                 f'{t["rows_sharpness"]:.2f})')
    return text


def cloud(xyz, stamp):
    """Return points (N x 3) as a PointCloud2 in the frame 'map'."""
    msg = PointCloud2()
    msg.header.stamp, msg.header.frame_id = stamp, 'map'
    msg.height, msg.width = 1, len(xyz)
    msg.fields = [PointField(name=name, offset=4 * k, datatype=PointField.FLOAT32, count=1)
                  for k, name in enumerate('xyz')]
    msg.is_bigendian, msg.point_step, msg.is_dense = False, 12, True
    msg.row_step = 12 * len(xyz)
    msg.data = array.array('B', np.asarray(xyz, dtype='<f4').tobytes())
    return msg


class PlantMapNode(Node):
    """Map the plants of every plot with the line scanners."""

    def __init__(self):
        """Read the plot layout and parameters, and connect the topics."""
        super().__init__('plant_map')
        layout = os.path.join(get_package_share_directory('agri_ugv_field'), 'data',
                              'must_c_field_plots.csv')
        with open(self.declare_parameter('layout_file', layout).value) as text:
            plots = [p for p in read_layout_csv(text.read()) if p['type'] == 'plot']
        value = {name: self.declare_parameter(name, default).value for name, default in [
            ('cell', 0.05), ('low', 0.06), ('pose_topic', '/localization/odometry'),
            ('publish_period', 5.0), ('quiet', 30.0), ('voxel', 0.01),
            ('output', os.path.join(os.path.expanduser('~'), 'plant_maps', 'plant_map'))]}
        self.quiet = value['quiet']
        self.period, self.output = value['publish_period'], value['output']
        self.map = FieldMap(plots, value['cell'], value['low'], voxel=value['voxel'])
        self.history = deque(maxlen=400)               # (t, x, y, yaw) of the pose
        self.mounts, self.reported = {}, {}
        self.last = {'publish': None, 'check': None}
        self.profiles, self.unplaced = 0, 0
        self.publisher = self.create_publisher(PointCloud2, '/plant_map/cloud', 1)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self.on_description, latched)
        self.create_subscription(String, '/mission/status', self.on_status, latched)
        self.create_subscription(Odometry, value['pose_topic'], self.on_pose, 100)
        for side in SIDES:
            self.create_subscription(LaserScan, f'/scanners/{side}',
                                     lambda msg, s=side: self.on_profile(s, msg), 50)
        self.create_service(Trigger, '/plant_map/save', self.on_save)

    def on_description(self, msg):
        """Find where the scanners sit on the robot (or that there are none)."""
        self.mounts = {}
        for side in SIDES:
            try:
                self.mounts[side] = mount(msg.data, link=f'line_scanner_{side}_beam_link')
            except ValueError:
                pass
        if not self.mounts:
            self.get_logger().info('No line scanners on the robot: no plant map')

    def on_pose(self, msg):
        """Keep the poses of the last seconds."""
        p = msg.pose.pose.position
        self.history.append((stamp_seconds(msg), p.x, p.y, yaw_of(msg.pose.pose.orientation)))

    def on_profile(self, side, msg):
        """Place one profile's points in the map."""
        if side not in self.mounts:
            return
        t = stamp_seconds(msg)
        pose = pose_at(self.history, t)
        self.profiles += 1
        if pose is None:
            self.unplaced += 1
            return
        points = profile_points(msg.ranges, msg.angle_min, msg.angle_increment,
                                msg.range_min, msg.range_max, self.mounts[side])
        self.map.add(points, pose, t)
        self.check(t)
        if self.last['publish'] is None or t - self.last['publish'] >= self.period:
            self.last['publish'] = t
            self.publish(msg.header.stamp)

    def check(self, t):
        """Once a second: log the traits of the plots the robot has left since they changed."""
        if self.last['check'] is not None and t - self.last['check'] < 1.0:
            return
        self.last['check'] = t
        for plot_id, changed in self.map.changed.items():
            if t - changed >= self.quiet and self.reported.get(plot_id) != changed:
                self.report(plot_id)

    def report(self, plot_id):
        """Log one plot's traits."""
        self.reported[plot_id] = self.map.changed[plot_id]
        self.get_logger().info('Plant map: ' + traits_text(self.map.maps[plot_id].traits()))

    def publish(self, stamp):
        """Publish the plant cells of all plots at their heights."""
        parts = [m.plant_cells() for m in self.map.maps.values() if m.count.any()]
        xyz = np.vstack([np.column_stack([xy, h]) for xy, h in parts]) if parts else \
            np.zeros((0, 3))
        self.publisher.publish(cloud(xyz, stamp))

    def save(self):
        """Log the traits of the plots not yet reported; save the map. Return a message."""
        for plot_id, changed in self.map.changed.items():
            if self.reported.get(plot_id) != changed:
                self.report(plot_id)
        if not self.map.changed:
            return False, 'nothing mapped yet'
        os.makedirs(os.path.dirname(self.output) or '.', exist_ok=True)
        count = self.map.save(self.output)
        text = (f'{count} plot{"s" * (count != 1)} saved to {self.output}.npz and .json'
                + (', 3D clouds to _cloud_<plot>.ply' if self.map.clouds else '')
                + f' ({self.profiles} profiles, {self.unplaced} without a pose)')
        self.get_logger().info('Plant map: ' + text)
        return True, text

    def on_status(self, msg):
        """Save the map once the mission is complete."""
        if json.loads(msg.data).get('state') == 'complete':
            self.save()

    def on_save(self, request, response):
        """Save the map on request (ros2 service call /plant_map/save std_srvs/srv/Trigger)."""
        response.success, response.message = self.save()
        return response


def main(args=None):
    """Run the node until Ctrl+C."""
    rclpy.init(args=args)
    node = PlantMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
