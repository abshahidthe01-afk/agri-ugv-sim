// Four-wheel-steering kinematics: pure math, no ROS.
//
// Conventions (ROS REP 103): x forward, y left, z up. Angles in radians, counter-clockwise
// positive. The same model as agri_ugv_control's kinematics.py, in C++.

#ifndef AGRI_UGV_FOUR_WS__KINEMATICS_HPP_
#define AGRI_UGV_FOUR_WS__KINEMATICS_HPP_

#include <optional>
#include <string>
#include <vector>

namespace agri_ugv_four_ws
{

/// Position of one steerable, driven wheel relative to the robot centre.
struct WheelModule
{
  std::string name;  ///< e.g. front_left
  double x;          ///< steering axis, forward of the centre [m]
  double y;          ///< steering axis, left of the centre [m]
};

/// What one wheel module should do.
struct WheelCommand
{
  std::optional<double> steer_angle;  ///< [rad] in [-pi/2, pi/2]; empty: keep the current angle
  double wheel_speed;                 ///< [rad/s]; positive = rolling forward
};

/// A robot velocity: forward and to the left [m/s], turn rate counter-clockwise [rad/s].
struct Velocity
{
  double vx;
  double vy;
  double wz;
};

/// A robot velocity recovered from the wheels, with the wheels' disagreement.
struct Odometry
{
  Velocity velocity;
  double residual;  ///< RMS mismatch [m/s] between the wheels: zero when they agree
};

/// A pose in the plane: position [m] and heading [rad].
struct Pose2D
{
  double x;
  double y;
  double yaw;
};

/// Convert a robot velocity into a steering angle and spin speed per wheel.
///
/// Steering joints turn at most +-90 deg: beyond that a wheel points the opposite way and
/// spins backwards (the same motion on the ground). A wheel that should point at most
/// `margin` [rad] past its limit stays at the limit and rolls at the part of its speed
/// along it, so that a small correction while crabbing does not swing it round by 180 deg.
/// Throws std::invalid_argument for a negative margin.
std::vector<WheelCommand> inverse_kinematics(
  const Velocity & velocity, const std::vector<WheelModule> & modules, double wheel_radius,
  double margin = 0.0);

/// Recover the robot velocity from measured steering angles and wheel speeds (odometry).
///
/// Each wheel's ground velocity must equal (vx - wz * y, vy + wz * x): with four wheels that
/// is 8 equations for 3 unknowns, solved by least squares. Throws std::invalid_argument for
/// inputs of different lengths or fewer than 2 wheels, and for a layout that does not
/// determine the motion.
Odometry forward_kinematics(
  const std::vector<double> & steer_angles, const std::vector<double> & wheel_speeds,
  const std::vector<WheelModule> & modules, double wheel_radius);

/// Move a pose by a body velocity held for dt seconds (dead reckoning), with the heading at
/// the middle of the step.
Pose2D integrate_pose(const Pose2D & pose, const Velocity & velocity, double dt);

/// The speed [rad/s] of the fastest steering joint between two sets of measured angles dt
/// seconds apart; 0 when dt is not positive. Throws std::invalid_argument for sets of
/// different lengths.
double steering_rate(
  const std::vector<double> & before, const std::vector<double> & now, double dt);

/// How uncertain the odometry is: variances of the speeds and of the turn rate.
struct OdometryVariances
{
  double speed;  ///< of vx and of vy [(m/s)^2]
  double turn;   ///< of wz [(rad/s)^2]
};

/// The odometry's variances: its noise (speed_sigma [m/s], turn_sigma [rad/s]), plus the
/// wheels' disagreement (residual [m/s], and over the wheels' distance from the centre,
/// lever [m], for the turn rate), plus a slide while the wheels turn on the spot: the tyres
/// scrub and push the base, steer_slip [m/s] per rad/s of the fastest steering joint.
OdometryVariances odometry_variances(
  double residual, double lever, double steer_rate, double speed_sigma, double turn_sigma,
  double steer_slip);

}  // namespace agri_ugv_four_ws

#endif  // AGRI_UGV_FOUR_WS__KINEMATICS_HPP_
