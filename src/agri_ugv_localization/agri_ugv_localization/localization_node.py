"""
ROS node: fuse wheel odometry, gyroscope and dual-antenna GNSS into the robot's pose.

Subscribes:  /robot_description (latched)   antenna offsets
             /imu                           gyro (predict at 100 Hz) and accelerometer (roll)
             /wheel/odom                    body velocity
             /gnss/front/fix, /gnss/rear/fix, /gnss/heading   corrections at 10 Hz
             /rows/measurement (std_msgs/String, JSON)       the robot's place across the
                                            crop rows in the LiDAR (agri_ugv_navigation rows)
Publishes:   /localization/odometry (nav_msgs/Odometry) pose in the world frame 'map'
Parameters:  world_file (its <spherical_coordinates>), the GNSS error model:
             gnss_shared_sigma, gnss_shared_tau, gnss_own_sigma. The shared error's spread
             follows the accuracy each fix reports (its covariance) when it reports one:
             worse fixes let it drift further; better fixes, or fixes again after more than
             gnss_restart_gap [s] without, are a new solution: it starts afresh. Messages
             without a fix (status STATUS_NO_FIX) are ignored: wheels and gyro carry on.
             tilt_from_gyro (true): roll and pitch from the gyro, pulled towards gravity
             and the GNSS pitch with time constant tilt_gyro_time [s] (TiltEstimator);
             false: roll = accelerometer and pitch = GNSS pitch, each smoothed with
             tilt_time [s] (sideways acceleration then reads as a lean)
             latency_compensation: move each GNSS measurement forward by the robot's motion
             since it was taken (fixes arrive later than the IMU that drives the filter)
             wheel_slip_speed, wheel_slip_turn: trust the wheels less at speed and while
             turning (the gyro sees turns, the wheels miss the slides that come with them)
             use_rows (true, read at every row measurement): correct the position across
             the crop rows when the row was found; rows_sigma [m] (0.02) is a measurement's
             uncertainty, and one further than rows_gate (3) of the combined uncertainties
             from the estimate is refused

The IMU's own orientation output is not used: in Gazebo it is perfect, a real one is not.
"""

import json
import math

from agri_ugv_localization.ekf import (own_acceleration, PoseEkf, tilted_lever, TiltEstimator,
                                       wheel_speed_sigma)
from agri_ugv_localization.frames import antenna_levers, spherical_coordinates, world_from_fix
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from std_msgs.msg import String

ANTENNAS = {'front': 'gnss_front_link', 'rear': 'gnss_rear_link'}


def stamp_seconds(msg):
    """Return a message's time stamp in seconds."""
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


class LocalizationNode(Node):
    """Run the pose EKF on the robot's sensors."""

    def __init__(self):
        """Read parameters, the world origin, and connect the topics."""
        super().__init__('localization')
        p = {name: self.declare_parameter(name, default).value for name, default in [
            ('world_file', ''), ('gnss_shared_sigma', 0.010), ('gnss_shared_tau', 60.0),
            ('gnss_own_sigma', 0.003), ('gyro_sigma', 0.0005), ('tilt_time', 1.0)]}
        self.declare_parameter('latency_compensation', True)     # read at every fix
        self.declare_parameter('wheel_slip_speed', 0.5)          # read at every IMU sample
        self.declare_parameter('wheel_slip_turn', 1.0)
        self.declare_parameter('tilt_from_gyro', True)           # read at every use
        self.declare_parameter('gnss_restart_gap', 1.0)          # read at every fix
        self.declare_parameter('use_rows', True)                 # read at every row measurement
        self.declare_parameter('rows_sigma', 0.02)
        self.declare_parameter('rows_gate', 3.0)
        self.rows = {'used': 0, 'refused': 0, 'moved': 0.0}
        self.tilt = TiltEstimator(self.declare_parameter('tilt_gyro_time', 10.0).value)
        self.acceleration, self.last_wheels, self.last_heading = (0.0, 0.0), None, None
        self.rate, self.ages = 0.0, []
        self.params = p
        self.origin = (0.0, 0.0, 0.0, 0.0)
        if p['world_file']:
            with open(p['world_file']) as world:
                self.origin = spherical_coordinates(world.read())
        self.get_logger().info(f'GNSS origin and heading_deg: {self.origin}')
        self.levers, self.ekf, self.last_imu = None, None, None
        self.velocity, self.speed_sigma = (0.0, 0.0), 0.05
        self.pitch, self.roll, self.fixes = 0.0, 0.0, {}
        self.gnss_sigma, self.last_fix, self.no_fix_since = p['gnss_shared_sigma'], None, None
        self.publisher = self.create_publisher(Odometry, '/localization/odometry', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self.on_description, latched)
        self.create_subscription(Imu, '/imu', self.on_imu, 50)
        self.create_subscription(Odometry, '/wheel/odom', self.on_wheels, 50)
        self.create_subscription(Imu, '/gnss/heading', self.on_heading, 10)
        for name in ANTENNAS:
            self.create_subscription(NavSatFix, f'/gnss/{name}/fix',
                                     lambda msg, n=name: self.on_fix(n, msg), 10)
        self.create_subscription(String, '/rows/measurement', self.on_rows, 10)

    def on_description(self, msg):
        """Read where the antennas sit on the robot."""
        found = antenna_levers(msg.data, tuple(ANTENNAS.values()))
        self.levers = {name: found[link] for name, link in ANTENNAS.items()}

    def on_wheels(self, msg):
        """Keep the newest body velocity from the wheels, its change and its uncertainty."""
        velocity = (msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        time = stamp_seconds(msg)
        if self.last_wheels is not None and 0 < time - self.last_wheels < 0.1:
            dt = time - self.last_wheels
            self.acceleration = ((velocity[0] - self.velocity[0]) / dt,
                                 (velocity[1] - self.velocity[1]) / dt)
        else:
            self.acceleration = (0.0, 0.0)
        self.last_wheels = time
        self.velocity = velocity
        self.speed_sigma = math.sqrt(max(msg.twist.covariance[0], 1e-8))

    def tilt_angles(self):
        """Return the (pitch, roll) used for the antenna levers."""
        if self.get_parameter('tilt_from_gyro').value and self.tilt.ready:
            return self.tilt.pitch, self.tilt.roll
        return self.pitch, self.roll

    def smooth(self, old, new, dt):
        """Low-pass filter for the tilt angles, time constant tilt_time."""
        a = dt / (self.params['tilt_time'] + dt)
        return old + a * (new - old)

    def on_imu(self, msg):
        """Predict with the gyro and the newest wheel velocity; estimate roll and pitch."""
        time = stamp_seconds(msg)
        a, g = msg.linear_acceleration, msg.angular_velocity
        dt = 0.0 if self.last_imu is None else time - self.last_imu
        self.last_imu = time
        self.roll = self.smooth(self.roll, math.atan2(a.y, a.z), dt)
        self.tilt.predict(dt, (g.x, g.y, g.z))
        self.tilt.correct_roll((a.x, a.y, a.z),
                               own_acceleration(self.acceleration, self.velocity, g.z), dt)
        if self.ekf is None or dt <= 0:
            return
        sigma = wheel_speed_sigma(self.speed_sigma, math.hypot(*self.velocity),
                                  abs(msg.angular_velocity.z - self.ekf.x[3]),
                                  self.get_parameter('wheel_slip_speed').value,
                                  self.get_parameter('wheel_slip_turn').value)
        self.ekf.predict(dt, *self.velocity, msg.angular_velocity.z, sigma,
                         self.params['gyro_sigma'])
        self.rate = msg.angular_velocity.z - self.ekf.x[3]
        self.publish(msg.header.stamp, msg.angular_velocity.z)

    def on_heading(self, msg):
        """Correct the heading (and start the filter on the first heading with a fix)."""
        q = msg.orientation
        yaw_true_east = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        self.pitch = pitch if self.ekf is None else self.smooth(self.pitch, pitch, 0.1)
        time = stamp_seconds(msg)
        self.tilt.correct_pitch(pitch, 0.0 if self.last_heading is None
                                else time - self.last_heading)
        self.last_heading = time
        yaw = yaw_true_east - math.radians(self.origin[3])         # into world axes
        variance = msg.orientation_covariance[8]
        if self.ekf is not None:
            yaw += self.rate * self.age(msg)               # it turned on since the measurement
            self.ekf.update_yaw(yaw, variance)
        elif self.levers is not None and 'front' in self.fixes:
            fx, fy = tilted_lever(self.levers['front'], *self.tilt_angles())
            x, y = self.fixes['front']
            c, s = math.cos(yaw), math.sin(yaw)
            self.ekf = PoseEkf(x - c * fx + s * fy, y - s * fx - c * fy, yaw,
                               0.02, math.sqrt(variance),
                               gnss_sigma=self.gnss_sigma,
                               gnss_tau=self.params['gnss_shared_tau'])
            self.get_logger().info(f'started at x {self.ekf.x[0]:.3f} y {self.ekf.x[1]:.3f} '
                                   f'yaw {math.degrees(yaw):.2f} deg')

    def age(self, msg):
        """Return how old a measurement is compared with the filter's time (0 if switched off)."""
        if self.last_imu is None or not self.get_parameter('latency_compensation').value:
            return 0.0
        return max(0.0, self.last_imu - stamp_seconds(msg))

    def shared_sigma(self, msg):
        """Return the spread [m] of the shared GNSS error, from the accuracy a fix reports."""
        if msg.position_covariance_type == NavSatFix.COVARIANCE_TYPE_UNKNOWN:
            return self.params['gnss_shared_sigma']
        horizontal = (msg.position_covariance[0] + msg.position_covariance[4]) / 2
        return math.sqrt(max(horizontal - self.params['gnss_own_sigma'] ** 2, 1e-6))

    def follow_gnss(self, time):
        """Adapt the filter's shared GNSS error to the reported accuracy and to outages."""
        gap = None if self.last_fix is None else time - self.last_fix
        self.last_fix = time if gap is None else max(self.last_fix, time)
        sigma, current = self.gnss_sigma, self.ekf.gnss_sigma
        if gap is not None and gap > self.get_parameter('gnss_restart_gap').value:
            self.ekf.restart_gnss(sigma)
            self.get_logger().info(f'GNSS: fixes again after {gap:.1f} s: a new solution, '
                                   f'shared error {100 * sigma:.1f} cm')
        elif sigma < current / 1.05:
            self.ekf.restart_gnss(sigma)
            self.get_logger().info(f'GNSS: better fixes, a new solution: shared error '
                                   f'{100 * sigma:.1f} cm')
        elif sigma > current * 1.05:
            self.ekf.set_gnss_sigma(sigma)
            self.get_logger().info(f'GNSS: worse fixes: the shared error may drift to '
                                   f'{100 * sigma:.1f} cm')

    def on_fix(self, name, msg):
        """Correct with one antenna's position, moved forward by the motion since it was taken."""
        if msg.status.status < NavSatStatus.STATUS_FIX:
            self.fixes.pop(name, None)
            if self.no_fix_since is None:
                self.no_fix_since = stamp_seconds(msg)
                self.get_logger().info('GNSS: no fix: carrying on with wheels and gyro')
            return
        self.no_fix_since = None
        self.gnss_sigma = self.shared_sigma(msg)
        x, y, _ = world_from_fix(self.origin, (msg.latitude, msg.longitude, msg.altitude))
        self.fixes[name] = (x, y)
        if self.ekf is None or self.levers is None:
            return
        self.follow_gnss(stamp_seconds(msg))
        if self.last_imu is not None:
            self.ages.append(self.last_imu - stamp_seconds(msg))
            if len(self.ages) == 200:
                self.get_logger().info(f'GNSS fixes arrive {1000 * sum(self.ages) / 200:.0f} ms '
                                       f'after they were taken (mean of 200)')
                self.ages = []
        age = self.age(msg)
        yaw = self.ekf.x[2]
        vx, vy = self.velocity
        x += (math.cos(yaw) * vx - math.sin(yaw) * vy) * age
        y += (math.sin(yaw) * vx + math.cos(yaw) * vy) * age
        own = self.params['gnss_own_sigma'] ** 2
        self.ekf.update_position((x, y), tilted_lever(self.levers[name], *self.tilt_angles()),
                                 [[own, 0.0], [0.0, own]])

    def on_rows(self, msg):
        """Correct the position across the crop rows with a row measurement of the LiDAR."""
        if self.ekf is None or not self.get_parameter('use_rows').value:
            return
        m = json.loads(msg.data)
        if m['left_plot'] is None:                    # which row is not clear
            return
        (dx, dy), (x, y) = m['left_dir'], m['pose'][:2]
        measured = dx * x + dy * y + m['left_plot']   # the true robot, across the rows
        age = 0.0
        if self.last_imu is not None and self.get_parameter('latency_compensation').value:
            age = max(0.0, self.last_imu - m['t'])    # it has moved on since the scan
        yaw = self.ekf.x[2]
        vx, vy = self.velocity
        measured += (dx * (math.cos(yaw) * vx - math.sin(yaw) * vy)
                     + dy * (math.sin(yaw) * vx + math.cos(yaw) * vy)) * age
        value, variance = self.ekf.along((dx, dy))
        sigma = self.get_parameter('rows_sigma').value
        if (measured - value) ** 2 > self.get_parameter('rows_gate').value ** 2 * (
                variance + sigma ** 2):
            self.rows['refused'] += 1
        else:
            self.ekf.update_along((dx, dy), measured, sigma ** 2)
            self.rows['used'] += 1
            self.rows['moved'] += abs(measured - value)
        if self.rows['used'] + self.rows['refused'] == 100:
            self.get_logger().info(
                f'Rows: {self.rows["used"]} corrections across the crop rows (mean '
                f'{100 * self.rows["moved"] / max(1, self.rows["used"]):.1f} cm), '
                f'{self.rows["refused"]} refused as too far off')
            self.rows = {'used': 0, 'refused': 0, 'moved': 0.0}

    def publish(self, stamp, gyro):
        """Publish the current estimate with its uncertainty."""
        px, py, yaw, bias = self.ekf.x[:4]
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id, odom.child_frame_id = 'map', 'base_footprint'
        odom.pose.pose.position.x, odom.pose.pose.position.y = float(px), float(py)
        odom.pose.pose.orientation.z = math.sin(yaw / 2)
        odom.pose.pose.orientation.w = math.cos(yaw / 2)
        covariance = [0.0] * 36
        for i, row in ((0, 0), (1, 1), (5, 2)):
            for j, col in ((0, 0), (1, 1), (5, 2)):
                covariance[6 * i + j] = float(self.ekf.P[row, col])
        covariance[14] = covariance[21] = covariance[28] = 1e6     # z, roll, pitch: not estimated
        odom.pose.covariance = covariance
        odom.twist.twist.linear.x, odom.twist.twist.linear.y = self.velocity
        odom.twist.twist.angular.z = float(gyro - bias)
        self.publisher.publish(odom)


def main(args=None):
    """Run the node until Ctrl+C."""
    rclpy.init(args=args)
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
