// Steer first, then drive: never roll the wheels while they point the wrong way.

#ifndef AGRI_UGV_FOUR_WS__STEERING_HPP_
#define AGRI_UGV_FOUR_WS__STEERING_HPP_

#include <optional>
#include <vector>

#include "agri_ugv_four_ws/kinematics.hpp"
#include "agri_ugv_four_ws/velocity_limiter.hpp"

namespace agri_ugv_four_ws
{

/// Below this speed [m/s or rad/s] the robot counts as standing still.
constexpr double kStopped = 1e-3;

/// The velocity to drive now and the wheel commands for it.
struct Step
{
  Velocity velocity;
  std::vector<WheelCommand> commands;
};

/// Return the largest difference [rad] between target and measured steering angles; targets
/// without an angle (wheels that keep theirs) are left out. Throws std::invalid_argument for
/// lists of different lengths.
double steering_misalignment(
  const std::vector<std::optional<double>> & targets, const std::vector<double> & measured);

/// Return the next step for a commanded target velocity.
///
/// If the wheels point within `tolerance` [rad] of what the target needs, the robot drives as
/// usual (speed and acceleration limited). If not, it first brakes (limited) and, once
/// standing, turns the wheels to the new angles without rolling them: rolling wheels that
/// point the wrong way would skid and corrupt the wheel odometry.
Step steer_first(
  const Velocity & target, const Velocity & velocity, const std::vector<double> & measured,
  const std::vector<WheelModule> & modules, double wheel_radius, const VelocityLimits & limits,
  double dt, double tolerance, bool together = true, double margin = 0.0);

/// Notice steering that has stopped making progress while the robot stands.
///
/// A tyre turning on the spot can get stuck (e.g. in a groove of the ground): its steering
/// joint stops short of the target and steer_first would wait for ever. update() tells when
/// to creep instead: after stall_time [s] standing without the misalignment shrinking by
/// `progress` [rad], for creep_time [s] or until the wheels are aligned.
class StallWatch
{
public:
  /// Throws std::invalid_argument unless all three are positive.
  explicit StallWatch(double stall_time = 1.5, double creep_time = 2.0, double progress = 0.02);

  /// Return true while the robot should creep to free a stuck wheel.
  bool update(double dt, double misalignment, double tolerance, bool standing);

  double stall_time() const {return stall_time_;}
  double creep_left() const {return creep_left_;}
  int count() const {return count_;}  ///< how many stalls so far

private:
  double stall_time_;
  double creep_time_;
  double progress_;
  std::optional<double> best_;
  double waited_ = 0.0;
  double creep_left_ = 0.0;
  int count_ = 0;
};

/// Return the step that creeps in the target's direction at `speed` [m/s].
///
/// The wheels keep steering towards the target's angles, and each rolls at the part of its
/// needed ground speed along the direction it points now (measured): a wheel that is still
/// turned rolls less instead of being dragged sideways.
Step creep(
  const Velocity & target, const Velocity & velocity, const std::vector<double> & measured,
  const std::vector<WheelModule> & modules, double wheel_radius, const VelocityLimits & limits,
  double dt, double speed, double margin = 0.0);

}  // namespace agri_ugv_four_ws

#endif  // AGRI_UGV_FOUR_WS__STEERING_HPP_
