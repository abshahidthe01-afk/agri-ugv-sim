#include "agri_ugv_four_ws/four_ws_controller.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace agri_ugv_four_ws
{

namespace
{

constexpr double kCommandTimeout = 0.5;  // [s] no new command for this long: stop
constexpr double kUnused = 1e6;          // variance of what a ground robot's odometry misses

// Return the index of the loaned interface 'joint/interface', or -1.
template<typename Interfaces>
long find_interface(
  const Interfaces & interfaces, const std::string & joint, const std::string & type)
{
  for (std::size_t i = 0; i < interfaces.size(); ++i) {
    if (interfaces[i].get_prefix_name() == joint && interfaces[i].get_interface_name() == type) {
      return static_cast<long>(i);
    }
  }
  return -1;
}

}  // namespace

controller_interface::CallbackReturn FourWsController::on_init()
{
  try {
    auto_declare<std::vector<std::string>>("steering_joints", std::vector<std::string>());
    auto_declare<std::vector<std::string>>("wheel_joints", std::vector<std::string>());
    auto_declare<std::vector<double>>("module_x", std::vector<double>());
    auto_declare<std::vector<double>>("module_y", std::vector<double>());
    auto_declare<double>("wheel_radius", 0.0);
    auto_declare<double>("max_linear_x", 1.5);
    auto_declare<double>("max_linear_y", 1.0);
    auto_declare<double>("max_angular_z", 1.0);
    auto_declare<double>("max_accel_x", 1.0);
    auto_declare<double>("max_accel_y", 1.0);
    auto_declare<double>("max_accel_z", 1.0);
    auto_declare<bool>("wait_for_steering", true);
    auto_declare<double>("steer_tolerance", 0.05);
    auto_declare<bool>("ramp_together", true);
    auto_declare<double>("limit_margin", 0.35);
    auto_declare<double>("stall_time", 1.5);
    auto_declare<double>("creep_speed", 0.05);
    auto_declare<double>("creep_time", 2.0);
    auto_declare<double>("speed_sigma", 0.02);
    auto_declare<double>("turn_sigma", 0.02);
    auto_declare<double>("steer_slip", 0.5);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Could not declare the parameters: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
FourWsController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint : steering_joints_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
  }
  for (const auto & joint : wheel_joints_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
  }
  return config;
}

controller_interface::InterfaceConfiguration
FourWsController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint : steering_joints_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
  }
  for (const auto & joint : wheel_joints_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
  }
  return config;
}

controller_interface::CallbackReturn FourWsController::on_configure(
  const rclcpp_lifecycle::State &)
{
  const auto node = get_node();
  steering_joints_ = node->get_parameter("steering_joints").as_string_array();
  wheel_joints_ = node->get_parameter("wheel_joints").as_string_array();
  const auto xs = node->get_parameter("module_x").as_double_array();
  const auto ys = node->get_parameter("module_y").as_double_array();
  wheel_radius_ = node->get_parameter("wheel_radius").as_double();
  const std::size_t n = steering_joints_.size();
  if (n < 2 || wheel_joints_.size() != n || xs.size() != n || ys.size() != n ||
    wheel_radius_ <= 0.0)
  {
    RCLCPP_ERROR(
      node->get_logger(),
      "Need at least 2 wheel modules with steering_joints, wheel_joints, module_x and "
      "module_y of the same length, and a positive wheel_radius");
    return controller_interface::CallbackReturn::ERROR;
  }
  modules_.clear();
  double squares = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    modules_.push_back({steering_joints_[i], xs[i], ys[i]});
    squares += xs[i] * xs[i] + ys[i] * ys[i];
  }
  lever_ = std::sqrt(squares / static_cast<double>(n));

  limits_ = {
    node->get_parameter("max_linear_x").as_double(),
    node->get_parameter("max_linear_y").as_double(),
    node->get_parameter("max_angular_z").as_double(),
    node->get_parameter("max_accel_x").as_double(),
    node->get_parameter("max_accel_y").as_double(),
    node->get_parameter("max_accel_z").as_double()};
  wait_for_steering_ = node->get_parameter("wait_for_steering").as_bool();
  steer_tolerance_ = node->get_parameter("steer_tolerance").as_double();
  ramp_together_ = node->get_parameter("ramp_together").as_bool();
  limit_margin_ = node->get_parameter("limit_margin").as_double();
  creep_speed_ = node->get_parameter("creep_speed").as_double();
  speed_sigma_ = node->get_parameter("speed_sigma").as_double();
  turn_sigma_ = node->get_parameter("turn_sigma").as_double();
  steer_slip_ = node->get_parameter("steer_slip").as_double();
  try {
    watch_.emplace(
      node->get_parameter("stall_time").as_double(),
      node->get_parameter("creep_time").as_double());
  } catch (const std::invalid_argument & e) {
    RCLCPP_ERROR(node->get_logger(), "%s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  command_subscription_ = node->create_subscription<geometry_msgs::msg::Twist>(
    "/cmd_vel", rclcpp::QoS(10),
    [this](const std::shared_ptr<geometry_msgs::msg::Twist> msg) {
      command_.writeFromNonRT(
        Command{{msg->linear.x, msg->linear.y, msg->angular.z}, ++received_});
    });
  odometry_publisher_ = node->create_publisher<nav_msgs::msg::Odometry>(
    "/wheel/odom", rclcpp::QoS(10));
  odometry_ = std::make_unique<realtime_tools::RealtimePublisher<nav_msgs::msg::Odometry>>(
    odometry_publisher_);

  RCLCPP_INFO(
    node->get_logger(),
    "%zu wheel modules, wheel radius %.3f m; limits %.2f / %.2f m/s, %.2f rad/s", n,
    wheel_radius_, limits_.max_vx, limits_.max_vy, limits_.max_wz);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FourWsController::on_activate(
  const rclcpp_lifecycle::State &)
{
  steer_command_.clear();
  wheel_command_.clear();
  steer_state_.clear();
  wheel_state_.clear();
  for (std::size_t i = 0; i < modules_.size(); ++i) {
    const long sc = find_interface(
      command_interfaces_, steering_joints_[i], hardware_interface::HW_IF_POSITION);
    const long wc = find_interface(
      command_interfaces_, wheel_joints_[i], hardware_interface::HW_IF_VELOCITY);
    const long ss = find_interface(
      state_interfaces_, steering_joints_[i], hardware_interface::HW_IF_POSITION);
    const long ws = find_interface(
      state_interfaces_, wheel_joints_[i], hardware_interface::HW_IF_VELOCITY);
    if (sc < 0 || wc < 0 || ss < 0 || ws < 0) {
      RCLCPP_ERROR(
        get_node()->get_logger(), "Missing an interface of %s or %s",
        steering_joints_[i].c_str(), wheel_joints_[i].c_str());
      return controller_interface::CallbackReturn::ERROR;
    }
    steer_command_.push_back(static_cast<std::size_t>(sc));
    wheel_command_.push_back(static_cast<std::size_t>(wc));
    steer_state_.push_back(static_cast<std::size_t>(ss));
    wheel_state_.push_back(static_cast<std::size_t>(ws));
  }
  steer_angles_.clear();
  for (std::size_t i = 0; i < modules_.size(); ++i) {   // keep the wheels where they point
    steer_angles_.push_back(state_interfaces_[steer_state_[i]].get_value());
  }
  last_measured_ = steer_angles_;
  velocity_ = {0.0, 0.0, 0.0};
  const Command * last = command_.readFromRT();   // a command from before is not new
  seen_ = last != nullptr ? last->sequence : 0;
  command_time_.reset();
  last_time_.reset();
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FourWsController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  for (std::size_t i = 0; i < wheel_command_.size(); ++i) {
    command_interfaces_[wheel_command_[i]].set_value(0.0);
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type FourWsController::update(
  const rclcpp::Time & time, const rclcpp::Duration & period)
{
  const double dt = period.seconds();
  std::vector<double> measured(modules_.size()), speeds(modules_.size());
  for (std::size_t i = 0; i < modules_.size(); ++i) {
    measured[i] = state_interfaces_[steer_state_[i]].get_value();
    speeds[i] = state_interfaces_[wheel_state_[i]].get_value();
  }

  // Odometry first: what the wheels did since the last update
  const auto odometry = forward_kinematics(measured, speeds, modules_, wheel_radius_);
  double steer_rate = 0.0;
  if (last_time_ && time > *last_time_) {
    const double step = (time - *last_time_).seconds();
    pose_ = integrate_pose(pose_, odometry.velocity, step);
    steer_rate = steering_rate(last_measured_, measured, step);
  }
  last_time_ = time;
  last_measured_ = measured;
  publish_odometry(time, odometry, steer_rate);

  // Watchdog: a command that has not been renewed recently means stop
  const Command * command = command_.readFromRT();
  if (command != nullptr && command->sequence != seen_) {
    seen_ = command->sequence;
    command_time_ = time;
  }
  Velocity target{0.0, 0.0, 0.0};
  if (command != nullptr && command_time_ && (time - *command_time_).seconds() < kCommandTimeout) {
    target = command->velocity;
  }

  Step step;
  if (wait_for_steering_ && dt > 0.0) {
    std::vector<std::optional<double>> aim;
    for (const auto & c : inverse_kinematics(target, modules_, wheel_radius_, limit_margin_)) {
      aim.push_back(c.steer_angle);
    }
    const double misalignment = steering_misalignment(aim, measured);
    const bool standing = std::max(
      {std::abs(velocity_.vx), std::abs(velocity_.vy), std::abs(velocity_.wz)}) <= kStopped;
    const bool stuck_before = watch_->creep_left() > 0.0;
    if (watch_->update(dt, misalignment, steer_tolerance_, standing)) {
      if (!stuck_before) {
        RCLCPP_WARN(
          get_node()->get_logger(),
          "Steering stalled (%.2f rad from the target for %.1f s): creeping to free the "
          "wheel (stall %d)", misalignment, watch_->stall_time(), watch_->count());
      }
      step = creep(
        target, velocity_, measured, modules_, wheel_radius_, limits_, dt, creep_speed_,
        limit_margin_);
    } else {
      step = steer_first(
        target, velocity_, measured, modules_, wheel_radius_, limits_, dt, steer_tolerance_,
        ramp_together_, limit_margin_);
    }
  } else {
    step.velocity = limit_velocity(target, velocity_, limits_, dt, ramp_together_);
    step.commands = inverse_kinematics(step.velocity, modules_, wheel_radius_, limit_margin_);
  }
  velocity_ = step.velocity;
  for (std::size_t i = 0; i < modules_.size(); ++i) {
    if (step.commands[i].steer_angle) {
      steer_angles_[i] = *step.commands[i].steer_angle;
    }
    command_interfaces_[steer_command_[i]].set_value(steer_angles_[i]);
    command_interfaces_[wheel_command_[i]].set_value(step.commands[i].wheel_speed);
  }
  return controller_interface::return_type::OK;
}

void FourWsController::publish_odometry(
  const rclcpp::Time & time, const Odometry & odometry, double steer_rate)
{
  if (!odometry_->trylock()) {
    return;
  }
  auto & msg = odometry_->msg_;
  msg.header.stamp = time;
  msg.header.frame_id = "odom";
  msg.child_frame_id = "base_footprint";
  msg.pose.pose.position.x = pose_.x;
  msg.pose.pose.position.y = pose_.y;
  msg.pose.pose.orientation.z = std::sin(pose_.yaw / 2.0);
  msg.pose.pose.orientation.w = std::cos(pose_.yaw / 2.0);
  msg.twist.twist.linear.x = odometry.velocity.vx;
  msg.twist.twist.linear.y = odometry.velocity.vy;
  msg.twist.twist.angular.z = odometry.velocity.wz;
  // the noise, the wheels' disagreement (slip, scrubbing) and the slide while they steer
  const auto variances = odometry_variances(
    odometry.residual, lever_, steer_rate, speed_sigma_, turn_sigma_, steer_slip_);
  std::fill(msg.twist.covariance.begin(), msg.twist.covariance.end(), 0.0);
  msg.twist.covariance[0] = msg.twist.covariance[7] = variances.speed;
  msg.twist.covariance[14] = msg.twist.covariance[21] = msg.twist.covariance[28] = kUnused;
  msg.twist.covariance[35] = variances.turn;
  odometry_->unlockAndPublish();
}

}  // namespace agri_ugv_four_ws

PLUGINLIB_EXPORT_CLASS(
  agri_ugv_four_ws::FourWsController, controller_interface::ControllerInterface)
