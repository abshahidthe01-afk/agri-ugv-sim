#include "agri_ugv_four_ws/steering.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace agri_ugv_four_ws
{

namespace
{

std::vector<std::optional<double>> angles_of(const std::vector<WheelCommand> & commands)
{
  std::vector<std::optional<double>> angles;
  angles.reserve(commands.size());
  for (const auto & c : commands) {
    angles.push_back(c.steer_angle);
  }
  return angles;
}

double largest(const Velocity & v)
{
  return std::max({std::abs(v.vx), std::abs(v.vy), std::abs(v.wz)});
}

}  // namespace

double steering_misalignment(
  const std::vector<std::optional<double>> & targets, const std::vector<double> & measured)
{
  if (targets.size() != measured.size()) {
    throw std::invalid_argument(
            std::to_string(targets.size()) + " targets but " + std::to_string(measured.size()) +
            " measured angles");
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < targets.size(); ++i) {
    if (targets[i]) {
      worst = std::max(worst, std::abs(*targets[i] - measured[i]));
    }
  }
  return worst;
}

Step steer_first(
  const Velocity & target, const Velocity & velocity, const std::vector<double> & measured,
  const std::vector<WheelModule> & modules, double wheel_radius, const VelocityLimits & limits,
  double dt, double tolerance, bool together, double margin)
{
  const auto aim = inverse_kinematics(target, modules, wheel_radius, margin);
  if (steering_misalignment(angles_of(aim), measured) <= tolerance) {
    const auto next = limit_velocity(target, velocity, limits, dt, together);
    return {next, inverse_kinematics(next, modules, wheel_radius, margin)};
  }
  if (largest(velocity) > kStopped) {  // brake first
    const auto next = limit_velocity({0.0, 0.0, 0.0}, velocity, limits, dt, together);
    return {next, inverse_kinematics(next, modules, wheel_radius, margin)};
  }
  std::vector<WheelCommand> turning;
  turning.reserve(aim.size());
  for (const auto & c : aim) {
    turning.push_back({c.steer_angle, 0.0});
  }
  return {{0.0, 0.0, 0.0}, turning};
}

StallWatch::StallWatch(double stall_time, double creep_time, double progress)
: stall_time_(stall_time), creep_time_(creep_time), progress_(progress)
{
  if (stall_time <= 0.0 || creep_time <= 0.0 || progress <= 0.0) {
    throw std::invalid_argument("stall_time, creep_time and progress must be positive");
  }
}

bool StallWatch::update(double dt, double misalignment, double tolerance, bool standing)
{
  if (misalignment <= tolerance) {
    best_.reset();
    waited_ = 0.0;
    creep_left_ = 0.0;
    return false;
  }
  if (creep_left_ > 0.0) {
    creep_left_ -= dt;
    return true;
  }
  if (!standing) {
    best_.reset();
    waited_ = 0.0;
    return false;
  }
  if (!best_ || misalignment < *best_ - progress_) {
    best_ = misalignment;
    waited_ = 0.0;
    return false;
  }
  waited_ += dt;
  if (waited_ < stall_time_) {
    return false;
  }
  best_.reset();
  waited_ = 0.0;
  creep_left_ = creep_time_ - dt;
  ++count_;
  return true;
}

Step creep(
  const Velocity & target, const Velocity & velocity, const std::vector<double> & measured,
  const std::vector<WheelModule> & modules, double wheel_radius, const VelocityLimits & limits,
  double dt, double speed, double margin)
{
  const double size = std::hypot(target.vx, target.vy);
  Velocity aim_velocity{0.0, 0.0, 0.0};
  if (size >= 1e-9) {
    const double scale = speed / size;
    aim_velocity = {target.vx * scale, target.vy * scale, target.wz * scale};
  }
  const auto next = limit_velocity(aim_velocity, velocity, limits, dt);
  const auto aim = inverse_kinematics(target, modules, wheel_radius, margin);
  std::vector<WheelCommand> commands;
  commands.reserve(modules.size());
  for (std::size_t i = 0; i < modules.size(); ++i) {
    const double ground_x = next.vx - next.wz * modules[i].y;
    const double ground_y = next.vy + next.wz * modules[i].x;
    const double rolling = ground_x * std::cos(measured[i]) + ground_y * std::sin(measured[i]);
    commands.push_back({aim[i].steer_angle, rolling / wheel_radius});
  }
  return {next, commands};
}

}  // namespace agri_ugv_four_ws
