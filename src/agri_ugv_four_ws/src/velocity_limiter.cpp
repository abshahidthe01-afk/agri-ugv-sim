#include "agri_ugv_four_ws/velocity_limiter.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace agri_ugv_four_ws
{

namespace
{

double clamp(double value, double limit)
{
  return std::max(-limit, std::min(limit, value));
}

}  // namespace

Velocity limit_velocity(
  const Velocity & target, const Velocity & current, const VelocityLimits & limits, double dt,
  bool together)
{
  const std::array<double, 3> goal{target.vx, target.vy, target.wz};
  const std::array<double, 3> now{current.vx, current.vy, current.wz};
  const std::array<double, 3> max_speed{limits.max_vx, limits.max_vy, limits.max_wz};
  const std::array<double, 3> max_accel{limits.max_ax, limits.max_ay, limits.max_az};
  std::array<double, 3> change{};
  for (std::size_t k = 0; k < 3; ++k) {
    change[k] = clamp(goal[k], max_speed[k]) - now[k];
  }
  std::array<double, 3> step{};
  if (together) {
    double fraction = 1.0;
    for (std::size_t k = 0; k < 3; ++k) {
      if (change[k] != 0.0) {
        fraction = std::min(fraction, max_accel[k] * dt / std::abs(change[k]));
      }
    }
    for (std::size_t k = 0; k < 3; ++k) {
      step[k] = fraction * change[k];
    }
  } else {
    for (std::size_t k = 0; k < 3; ++k) {
      step[k] = clamp(change[k], max_accel[k] * dt);
    }
  }
  return {now[0] + step[0], now[1] + step[1], now[2] + step[2]};
}

}  // namespace agri_ugv_four_ws
