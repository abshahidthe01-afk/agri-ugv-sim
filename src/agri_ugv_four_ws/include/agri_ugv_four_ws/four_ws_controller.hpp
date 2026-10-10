// ros2_control controller: /cmd_vel in, steering angles and wheel speeds out, wheel odometry.

#ifndef AGRI_UGV_FOUR_WS__FOUR_WS_CONTROLLER_HPP_
#define AGRI_UGV_FOUR_WS__FOUR_WS_CONTROLLER_HPP_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "agri_ugv_four_ws/kinematics.hpp"
#include "agri_ugv_four_ws/steering.hpp"
#include "agri_ugv_four_ws/velocity_limiter.hpp"
#include "controller_interface/controller_interface.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#if __has_include("realtime_tools/realtime_buffer.hpp")
#include "realtime_tools/realtime_buffer.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#else  // older Humble releases have only the .h headers
#include "realtime_tools/realtime_buffer.h"
#include "realtime_tools/realtime_publisher.h"
#endif

namespace agri_ugv_four_ws
{

/// Four-wheel-steering controller, run by the controller manager at its update rate.
///
/// The same behaviour as agri_ugv_control's four_ws_driver and wheel_odometry nodes, inside
/// the control loop: it reads the steering angles and wheel speeds from the joints' state
/// interfaces and writes the steering joints' positions and the wheels' velocities.
///
/// Subscribes:  /cmd_vel (geometry_msgs/Twist)  motion commands; none for 0.5 s means stop
/// Publishes:   /wheel/odom (nav_msgs/Odometry)  body velocity and dead-reckoned pose
/// Parameters:  steering_joints, wheel_joints, module_x, module_y [m] (per module, in the
///              same order), wheel_radius [m]; max_linear_x, max_linear_y [m/s],
///              max_angular_z [rad/s], max_accel_x, max_accel_y [m/s^2], max_accel_z
///              [rad/s^2]; wait_for_steering, steer_tolerance [rad], ramp_together,
///              limit_margin [rad], stall_time [s], creep_speed [m/s], creep_time [s];
///              speed_sigma [m/s], turn_sigma [rad/s] (odometry noise); all read when the
///              controller is configured
class FourWsController : public controller_interface::ControllerInterface
{
public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  struct Command
  {
    Velocity velocity;
    unsigned long sequence;  ///< counts the commands received, to tell a new one
  };

  void publish_odometry(const rclcpp::Time & time, const Odometry & odometry);

  std::vector<std::string> steering_joints_;
  std::vector<std::string> wheel_joints_;
  std::vector<WheelModule> modules_;
  double wheel_radius_ = 0.0;
  double lever_ = 1.0;  ///< the wheels' RMS distance from the centre [m]
  VelocityLimits limits_{};
  bool wait_for_steering_ = true;
  bool ramp_together_ = true;
  double steer_tolerance_ = 0.05;
  double limit_margin_ = 0.35;
  double creep_speed_ = 0.05;
  double speed_sigma_ = 0.02;
  double turn_sigma_ = 0.02;
  std::optional<StallWatch> watch_;

  // the command interfaces: steering positions, then wheel velocities (module order)
  std::vector<std::size_t> steer_command_;
  std::vector<std::size_t> wheel_command_;
  // the state interfaces: steering positions, then wheel velocities (module order)
  std::vector<std::size_t> steer_state_;
  std::vector<std::size_t> wheel_state_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr command_subscription_;
  realtime_tools::RealtimeBuffer<Command> command_;
  unsigned long received_ = 0;  ///< commands received (subscription thread)
  unsigned long seen_ = 0;      ///< the last command the control loop has taken
  std::optional<rclcpp::Time> command_time_;

  Velocity velocity_{0.0, 0.0, 0.0};  ///< the limited velocity being driven
  std::vector<double> steer_angles_;   ///< the last commanded angle per module
  Pose2D pose_{0.0, 0.0, 0.0};
  std::optional<rclcpp::Time> last_time_;

  std::shared_ptr<rclcpp::Publisher<nav_msgs::msg::Odometry>> odometry_publisher_;
  std::unique_ptr<realtime_tools::RealtimePublisher<nav_msgs::msg::Odometry>> odometry_;
};

}  // namespace agri_ugv_four_ws

#endif  // AGRI_UGV_FOUR_WS__FOUR_WS_CONTROLLER_HPP_
