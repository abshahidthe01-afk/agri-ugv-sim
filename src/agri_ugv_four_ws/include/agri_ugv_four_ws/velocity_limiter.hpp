// Speed and acceleration limits for velocity commands: pure math, no ROS.

#ifndef AGRI_UGV_FOUR_WS__VELOCITY_LIMITER_HPP_
#define AGRI_UGV_FOUR_WS__VELOCITY_LIMITER_HPP_

#include "agri_ugv_four_ws/kinematics.hpp"

namespace agri_ugv_four_ws
{

/// Maximum speeds and accelerations, per motion component.
struct VelocityLimits
{
  double max_vx;  ///< forward / backward [m/s]
  double max_vy;  ///< sideways (crab) [m/s]
  double max_wz;  ///< turning [rad/s]
  double max_ax;  ///< forward / backward acceleration [m/s^2]
  double max_ay;  ///< sideways acceleration [m/s^2]
  double max_az;  ///< turning acceleration [rad/s^2]
};

/// Return the next velocity: one time step dt from current towards target.
///
/// The target is first capped at the maximum speeds; then each component moves towards it
/// by at most (maximum acceleration x dt). With `together`, all components move by the same
/// fraction of their remaining change (the one with the least room sets the pace), so the
/// velocity keeps its direction and the wheels their angles; otherwise each component ramps
/// on its own.
Velocity limit_velocity(
  const Velocity & target, const Velocity & current, const VelocityLimits & limits, double dt,
  bool together = true);

}  // namespace agri_ugv_four_ws

#endif  // AGRI_UGV_FOUR_WS__VELOCITY_LIMITER_HPP_
