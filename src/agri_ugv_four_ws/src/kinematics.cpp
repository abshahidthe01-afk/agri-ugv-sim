#include "agri_ugv_four_ws/kinematics.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace agri_ugv_four_ws
{

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfPi = kPi / 2.0;

using Matrix3 = std::array<std::array<double, 3>, 3>;

double determinant(const Matrix3 & m)
{
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
         m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
         m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

// Solve a 3 x 3 linear system with Cramer's rule.
std::array<double, 3> solve3(const Matrix3 & a, const std::array<double, 3> & b)
{
  const double d = determinant(a);
  if (std::abs(d) < 1e-12) {
    throw std::invalid_argument("wheel layout does not determine the robot motion");
  }
  std::array<double, 3> result{};
  for (std::size_t k = 0; k < 3; ++k) {
    Matrix3 replaced = a;
    for (std::size_t r = 0; r < 3; ++r) {
      replaced[r][k] = b[r];
    }
    result[k] = determinant(replaced) / d;
  }
  return result;
}

}  // namespace

std::vector<WheelCommand> inverse_kinematics(
  const Velocity & velocity, const std::vector<WheelModule> & modules, double wheel_radius,
  double margin)
{
  if (margin < 0.0) {
    throw std::invalid_argument("margin must not be negative, got " + std::to_string(margin));
  }
  std::vector<WheelCommand> commands;
  commands.reserve(modules.size());
  for (const auto & m : modules) {
    // Velocity of this wheel = robot motion + extra motion from the rotation
    const double wheel_vx = velocity.vx - velocity.wz * m.y;
    const double wheel_vy = velocity.vy + velocity.wz * m.x;
    double speed = std::hypot(wheel_vx, wheel_vy);  // [m/s]
    if (speed < 1e-6) {
      // This wheel is not moving: its direction is meaningless, keep it as it is
      commands.push_back({std::nullopt, 0.0});
      continue;
    }
    double angle = std::atan2(wheel_vy, wheel_vx);
    const double past = std::abs(angle) - kHalfPi;
    if (past > 0.0 && past <= margin) {
      angle = std::copysign(kHalfPi, angle);
      speed *= std::cos(past);
    } else if (past > 0.0) {
      angle -= std::copysign(kPi, angle);
      speed = -speed;
    }
    commands.push_back({angle, speed / wheel_radius});
  }
  return commands;
}

Odometry forward_kinematics(
  const std::vector<double> & steer_angles, const std::vector<double> & wheel_speeds,
  const std::vector<WheelModule> & modules, double wheel_radius)
{
  const std::size_t n = modules.size();
  if (steer_angles.size() != n || wheel_speeds.size() != n || n < 2) {
    throw std::invalid_argument(
            "need the same number (at least 2) of angles, speeds and modules");
  }
  std::vector<std::array<double, 2>> measured(n);
  double sx = 0.0, sy = 0.0, srr = 0.0;
  std::array<double, 3> b{0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < n; ++i) {
    const double ground = wheel_radius * wheel_speeds[i];
    measured[i] = {ground * std::cos(steer_angles[i]), ground * std::sin(steer_angles[i])};
    sx += modules[i].x;
    sy += modules[i].y;
    srr += modules[i].x * modules[i].x + modules[i].y * modules[i].y;
    b[0] += measured[i][0];
    b[1] += measured[i][1];
    b[2] += modules[i].x * measured[i][1] - modules[i].y * measured[i][0];
  }
  // Normal equations (A^T A) p = A^T b for the rows [1, 0, -y] and [0, 1, x]
  const double count = static_cast<double>(n);
  const Matrix3 a{{{count, 0.0, -sy}, {0.0, count, sx}, {-sy, sx, srr}}};
  const auto p = solve3(a, b);
  double squares = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double ex = p[0] - p[2] * modules[i].y - measured[i][0];
    const double ey = p[1] + p[2] * modules[i].x - measured[i][1];
    squares += ex * ex + ey * ey;
  }
  return {{p[0], p[1], p[2]}, std::sqrt(squares / (2.0 * count))};
}

Pose2D integrate_pose(const Pose2D & pose, const Velocity & velocity, double dt)
{
  const double middle = pose.yaw + velocity.wz * dt / 2.0;
  const double c = std::cos(middle), s = std::sin(middle);
  return {pose.x + (c * velocity.vx - s * velocity.vy) * dt,
    pose.y + (s * velocity.vx + c * velocity.vy) * dt, pose.yaw + velocity.wz * dt};
}

}  // namespace agri_ugv_four_ws
