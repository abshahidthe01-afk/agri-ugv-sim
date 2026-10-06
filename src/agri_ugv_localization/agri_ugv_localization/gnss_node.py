"""ROS node: Gazebo's ideal GNSS fixes in, realistic RTK fixes and a dual-antenna heading out."""

from agri_ugv_localization.gnss_errors import (heading_and_pitch, offset, quaternion, RtkErrors,
                                               shift)
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
            ('own_sigma_vertical', 0.006), ('seed', 1)]}
        self.errors = RtkErrors(
            (value['common_sigma_horizontal'],) * 2 + (value['common_sigma_vertical'],),
            value['common_tau'],
            (value['own_sigma_horizontal'],) * 2 + (value['own_sigma_vertical'],), value['seed'])
        self.latest = {}
        self.fix_out = {a: self.create_publisher(NavSatFix, f'/gnss/{a}/fix', 10)
                        for a in ANTENNAS}
        self.heading_out = self.create_publisher(Imu, '/gnss/heading', 10)
        for antenna in ANTENNAS:
            self.create_subscription(NavSatFix, f'/gnss/{antenna}/fix_ideal',
                                     lambda msg, a=antenna: self.on_fix(a, msg), 10)

    def on_fix(self, antenna, ideal):
        """Publish one antenna's fix with errors; once both have the same time, the heading."""
        time = ideal.header.stamp.sec + 1e-9 * ideal.header.stamp.nanosec
        east, north, up = self.errors.error(time)
        fix = NavSatFix()
        fix.header = ideal.header
        fix.status.status = NavSatStatus.STATUS_GBAS_FIX        # RTK: corrections applied
        fix.status.service = NavSatStatus.SERVICE_GPS
        fix.latitude, fix.longitude, fix.altitude = shift(
            ideal.latitude, ideal.longitude, ideal.altitude, east, north, up)
        e, n, u = self.errors.variance()
        fix.position_covariance = [e, 0.0, 0.0, 0.0, n, 0.0, 0.0, 0.0, u]
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self.fix_out[antenna].publish(fix)
        stamp = (ideal.header.stamp.sec, ideal.header.stamp.nanosec)
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
