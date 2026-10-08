"""
ROS node: Gazebo's ideal GNSS fixes in, realistic RTK fixes and a dual-antenna heading out.

Subscribes:  /gnss/front/fix_ideal, /gnss/rear/fix_ideal (sensor_msgs/NavSatFix)
Publishes:   /gnss/front/fix, /gnss/rear/fix   with errors; status and covariance tell
                                               how good they are
             /gnss/heading (sensor_msgs/Imu)   yaw and pitch of the antenna line
Parameters:  common_sigma_horizontal, common_sigma_vertical, common_tau, own_sigma_horizontal,
             own_sigma_vertical: the RTK fixed error model (see RtkErrors); seed
             quality (read at every fix): 'fixed' (default), 'float' (the base station's
             corrections stopped: the shared error drifts towards float_sigma_horizontal
             and float_sigma_vertical) or 'none' (no fix: status STATUS_NO_FIX, position
             NaN, no heading). Change it while running:
                 ros2 param set /gnss_errors quality float
"""

import math

from agri_ugv_localization.gnss_errors import (heading_and_pitch, offset, QUALITIES, quaternion,
                                               RtkErrors, shift)
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus

ANTENNAS = ('front', 'rear')
UNKNOWN = 1e6             # variance for an angle the two antennas cannot see (roll)


class GnssErrorsNode(Node):
    """Add RTK errors to /gnss/<antenna>/fix_ideal; publish /gnss/<antenna>/fix, /gnss/heading."""

    def __init__(self):
        """Read the error sizes from parameters and connect the topics."""
        super().__init__('gnss_errors')
        value = {name: self.declare_parameter(name, default).value for name, default in [
            ('common_sigma_horizontal', 0.010), ('common_sigma_vertical', 0.020),
            ('common_tau', 60.0), ('own_sigma_horizontal', 0.003),
            ('own_sigma_vertical', 0.006), ('seed', 1), ('float_sigma_horizontal', 0.20),
            ('float_sigma_vertical', 0.40)]}
        self.errors = RtkErrors(
            (value['common_sigma_horizontal'],) * 2 + (value['common_sigma_vertical'],),
            value['common_tau'],
            (value['own_sigma_horizontal'],) * 2 + (value['own_sigma_vertical'],), value['seed'],
            (value['float_sigma_horizontal'],) * 2 + (value['float_sigma_vertical'],))
        self.errors.set_quality(self.declare_parameter('quality', 'fixed').value)
        self.add_on_set_parameters_callback(self.check_parameters)
        self.latest, self.stamp = {}, None
        self.fix_out = {a: self.create_publisher(NavSatFix, f'/gnss/{a}/fix', 10)
                        for a in ANTENNAS}
        self.heading_out = self.create_publisher(Imu, '/gnss/heading', 10)
        for antenna in ANTENNAS:
            self.create_subscription(NavSatFix, f'/gnss/{antenna}/fix_ideal',
                                     lambda msg, a=antenna: self.on_fix(a, msg), 10)
        self.get_logger().info(f'GNSS quality: {self.errors.quality}')

    def check_parameters(self, parameters):
        """Refuse a quality that does not exist (ros2 param set then reports why)."""
        for parameter in parameters:
            if parameter.name == 'quality' and parameter.value not in QUALITIES:
                return SetParametersResult(
                    successful=False, reason=f'quality must be one of {", ".join(QUALITIES)}')
        return SetParametersResult(successful=True)

    def on_fix(self, antenna, ideal):
        """Publish one antenna's fix with errors; once both have the same time, the heading."""
        stamp = (ideal.header.stamp.sec, ideal.header.stamp.nanosec)
        if stamp != self.stamp:                  # both antennas of one time: the same quality
            self.stamp = stamp
            quality = self.get_parameter('quality').value
            if self.errors.set_quality(quality):
                sigma = self.errors.common.sigma[0]
                self.get_logger().info(f'GNSS quality: {quality}' + (
                    '' if quality == 'none' else f' (shared error {100 * sigma:.1f} cm)'))
        time = ideal.header.stamp.sec + 1e-9 * ideal.header.stamp.nanosec
        error = self.errors.error(time)
        fix = NavSatFix()
        fix.header = ideal.header
        fix.status.service = NavSatStatus.SERVICE_GPS
        if error is None:
            fix.status.status = NavSatStatus.STATUS_NO_FIX
            fix.latitude = fix.longitude = fix.altitude = math.nan
            fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
            self.fix_out[antenna].publish(fix)
            self.latest.clear()
            return
        east, north, up = error
        fix.status.status = NavSatStatus.STATUS_GBAS_FIX        # RTK: corrections applied
        fix.latitude, fix.longitude, fix.altitude = shift(
            ideal.latitude, ideal.longitude, ideal.altitude, east, north, up)
        e, n, u = self.errors.variance()
        fix.position_covariance = [e, 0.0, 0.0, 0.0, n, 0.0, 0.0, 0.0, u]
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self.fix_out[antenna].publish(fix)
        self.latest[antenna] = (stamp, (fix.latitude, fix.longitude, fix.altitude),
                                (ideal.latitude, ideal.longitude, ideal.altitude))
        if len(self.latest) == 2 and self.latest['front'][0] == self.latest['rear'][0]:
            self.publish_heading(ideal.header)

    def publish_heading(self, header):
        """Publish yaw (from true east) and pitch of the antenna line as an orientation."""
        yaw, pitch = heading_and_pitch(self.latest['rear'][1], self.latest['front'][1])
        east, north, _ = offset(self.latest['rear'][2], self.latest['front'][2])
        sigma_yaw, sigma_pitch = self.errors.heading_sigmas((east ** 2 + north ** 2) ** 0.5)
        msg = Imu()
        msg.header.stamp = header.stamp
        msg.header.frame_id = 'base_link'
        (msg.orientation.x, msg.orientation.y, msg.orientation.z,
         msg.orientation.w) = quaternion(0.0, pitch, yaw)
        msg.orientation_covariance = [UNKNOWN, 0.0, 0.0, 0.0, sigma_pitch ** 2, 0.0,
                                      0.0, 0.0, sigma_yaw ** 2]
        msg.angular_velocity_covariance[0] = -1.0        # -1: this message has no rates
        msg.linear_acceleration_covariance[0] = -1.0     # and no accelerations
        self.heading_out.publish(msg)


def main(args=None):
    """Run the node until Ctrl+C."""
    rclpy.init(args=args)
    node = GnssErrorsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
