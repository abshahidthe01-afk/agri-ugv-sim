"""
ROS node: map the plants of every plot with the laser line scanners while the robot drives.

Each profile is placed with the robot's pose at its time (agri_ugv_phenotyping.plant_map).
When no new points have reached a plot for 'quiet' seconds (the robot has left it), the
plot's traits are logged; when the mission is complete, or on request, the map is saved.
When the robot's dome cameras take pictures (sim.launch.py cameras:=top), the 3D points
get the colours the cameras see (agri_ugv_phenotyping.colour).

Subscribes:  /scanners/left, /scanners/right (sensor_msgs/LaserScan)   the profiles
             /localization/odometry (nav_msgs/Odometry)   the pose, kept for 4 s (topic:
                                            parameter pose_topic, e.g. /ground_truth/odom)
             /robot_description (std_msgs/String, latched)   where the scanners sit
             /mission/status (std_msgs/String, latched)   saves the map when 'complete'
             /cameras/dome_<i>/image, /camera_info (sensor_msgs/Image, CameraInfo)   the
                                            pictures of the cameras the robot model has
Publishes:   /plant_map/cloud (sensor_msgs/PointCloud2, frame 'map')   the cells higher
             than 'low' at their height, every publish_period seconds
             /plant_map/cloud_3d (sensor_msgs/PointCloud2, frame 'map', with 'rgb')   the 3D
                                            points of the latest cloud_plots plots, coloured
                                            by the cameras (grey: unseen), every cloud_period
                                            seconds (rviz/plant_cloud.rviz shows them)
Services:    /plant_map/save (std_srvs/Trigger)   save the map now
Parameters:  layout_file (default: the MuST-C plot layout of agri_ugv_field), cell [m]
             (0.05), low [m] (0.06), pose_topic, publish_period [s] (5.0), quiet [s] (30.0),
             voxel [m] (0.01; 0: none): every plot's 3D point cloud, merged per voxel
             (agri_ugv_phenotyping.cloud), colour_delay [s] (1.0): a picture colours the
             points once the profiles measured this long after it have arrived (colour.
             Colourist), cloud_plots (4): how many plots, the latest mapped, the 3D topic
             shows (0: none; one plot is about 3 MB), cloud_period [s] (2.0), display_gain
             (2.0): the colours shown are brighter by this factor (the plants lie in the
             roof's shade; the saved clouds keep the true colours),
             output (default ~/plant_maps/plant_map): the map
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
from agri_ugv_phenotyping.cloud import display_rows
from agri_ugv_phenotyping.colour import Colourist
from agri_ugv_phenotyping.plant_map import FieldMap, place, profile_points
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2, PointField
from std_msgs.msg import String
from std_srvs.srv import Trigger

SIDES = ('left', 'right')
CAMERAS = range(20)     # the dome cameras' numbers (agri_ugv_description payload.xacro)


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


def cloud(xyz, stamp, rgb=None):
    """
    Return points (N x 3) as a PointCloud2 in the frame 'map'.

    With rgb (rows of cloud.display_rows: x, y, z and the packed colour) instead of xyz,
    the cloud has a field 'rgb'.
    """
    if rgb is None:
        data, names = np.asarray(xyz, dtype='<f4'), ('x', 'y', 'z')
    else:
        data, names = rgb, ('x', 'y', 'z', 'rgb')
    msg = PointCloud2()
    msg.header.stamp, msg.header.frame_id = stamp, 'map'
    msg.height, msg.width = 1, len(data)
    msg.fields = [PointField(name=name, offset=4 * k, datatype=PointField.FLOAT32, count=1)
                  for k, name in enumerate(names)]
    msg.is_bigendian, msg.point_step, msg.is_dense = False, 4 * len(names), True
    msg.row_step = msg.point_step * len(data)
    msg.data = array.array('B', data.tobytes())
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
            ('publish_period', 5.0), ('quiet', 30.0), ('voxel', 0.01), ('colour_delay', 1.0),
            ('cloud_plots', 4), ('cloud_period', 2.0), ('display_gain', 2.0),
            ('output', os.path.join(os.path.expanduser('~'), 'plant_maps', 'plant_map'))]}
        self.quiet = value['quiet']
        self.period, self.output = value['publish_period'], value['output']
        self.map = FieldMap(plots, value['cell'], value['low'], voxel=value['voxel'])
        self.colour_delay = value['colour_delay']
        self.colourist, self.camera_mounts, self.pictures = None, {}, 0
        self.history = deque(maxlen=400)               # (t, x, y, yaw) of the pose
        self.mounts, self.reported = {}, {}
        self.last = {'publish': None, 'check': None, 'cloud': None}
        self.shown = {}                                # plot_id: (version, display rows)
        self.cloud_plots, self.cloud_period = value['cloud_plots'], value['cloud_period']
        self.display_gain = value['display_gain']
        self.profiles, self.unplaced = 0, 0
        self.publisher = self.create_publisher(PointCloud2, '/plant_map/cloud', 1)
        self.cloud_publisher = self.create_publisher(PointCloud2, '/plant_map/cloud_3d', 1) \
            if self.map.clouds and self.cloud_plots > 0 else None
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
        if self.camera_mounts or not self.map.clouds or not self.mounts:
            return                 # cameras found before, or no 3D clouds or scanners to colour
        for i in CAMERAS:
            try:
                self.camera_mounts[i] = mount(msg.data, link=f'dome_camera_{i}_optical_link')
            except ValueError:
                continue
            self.create_subscription(Image, f'/cameras/dome_{i}/image',
                                     lambda m, i=i: self.on_picture(i, m), 4)
            self.create_subscription(CameraInfo, f'/cameras/dome_{i}/camera_info',
                                     lambda m, i=i: self.on_calibration(i, m), 2)
        if self.camera_mounts:
            self.colourist = Colourist({}, delay=self.colour_delay)
            self.get_logger().info(f'{len(self.camera_mounts)} dome cameras take pictures: '
                                   'the 3D points get their colours')

    def on_calibration(self, i, msg):
        """Keep a camera's calibration: from now on its pictures colour the points."""
        if self.colourist is not None and i not in self.colourist.cameras:
            self.colourist.cameras[i] = (self.camera_mounts[i], list(msg.k))

    def on_picture(self, i, msg):
        """Keep a picture until the profiles after it have arrived."""
        if self.colourist is None or i not in self.colourist.cameras:
            return
        rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
        image = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding.lower().startswith('bgr'):
            image = image[:, :, ::-1]
        self.colourist.add_picture(stamp_seconds(msg), i, image)
        self.pictures += 1

    def pose_of(self, t):
        """Return the pose (x, y, yaw) at time t from the kept poses, or None."""
        return pose_at(self.history, t)

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
        if self.colourist is not None:
            self.colourist.add_profile(t, place(points, pose), points[:, 2])
            for xy, height, colours in self.colourist.ready(self.pose_of):
                self.map.add_colours(xy, height, colours)
        self.check(t)
        if self.last['publish'] is None or t - self.last['publish'] >= self.period:
            self.last['publish'] = t
            self.publish(msg.header.stamp)
        if self.cloud_publisher is not None and (
                self.last['cloud'] is None or t - self.last['cloud'] >= self.cloud_period):
            self.last['cloud'] = t
            self.publish_cloud(msg.header.stamp)

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

    def publish_cloud(self, stamp):
        """Publish the 3D points of the latest plots, in their colours."""
        latest = sorted(self.map.changed, key=self.map.changed.get)[-self.cloud_plots:]
        for plot_id in latest:
            plot_cloud = self.map.clouds[plot_id]
            if self.shown.get(plot_id, (None,))[0] != plot_cloud.version:   # it has changed
                self.shown[plot_id] = (plot_cloud.version, display_rows(
                    *plot_cloud.world(self.display_gain)))
        self.shown = {plot_id: self.shown[plot_id] for plot_id in latest}
        rows = [rows for _, rows in self.shown.values()]
        self.cloud_publisher.publish(cloud(None, stamp, np.concatenate(rows) if rows else
                                           display_rows(np.zeros((0, 3)), np.zeros((0, 3)))))

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
                + f' ({self.profiles} profiles, {self.unplaced} without a pose'
                + (f', {self.pictures} pictures' if self.colourist is not None else '') + ')')
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
