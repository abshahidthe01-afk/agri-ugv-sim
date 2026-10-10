// The same cases as agri_ugv_control's test_steering.py.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

#include "agri_ugv_four_ws/steering.hpp"

using agri_ugv_four_ws::creep;
using agri_ugv_four_ws::inverse_kinematics;
using agri_ugv_four_ws::StallWatch;
using agri_ugv_four_ws::steer_first;
using agri_ugv_four_ws::steering_misalignment;
using agri_ugv_four_ws::Velocity;
using agri_ugv_four_ws::VelocityLimits;
using agri_ugv_four_ws::WheelCommand;
using agri_ugv_four_ws::WheelModule;

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kRadius = 0.205;
const std::vector<WheelModule> kModules{
  {"front_left", 0.675, 0.75}, {"front_right", 0.675, -0.75},
  {"rear_left", -0.675, 0.75}, {"rear_right", -0.675, -0.75}};
const VelocityLimits kLimits{1.5, 1.0, 1.0, 1.0, 1.0, 1.0};
const std::vector<double> kStraight{0.0, 0.0, 0.0, 0.0};

std::vector<double> angles(const std::vector<WheelCommand> & commands)
{
  std::vector<double> result;
  for (const auto & c : commands) {
    result.push_back(c.steer_angle.value_or(0.0));
  }
  return result;
}

// The steering angles for turning on the spot.
std::vector<double> spin_angles()
{
  return angles(inverse_kinematics({0.0, 0.0, 0.3}, kModules, kRadius));
}

void expect_angles(const std::vector<WheelCommand> & commands, const std::vector<double> & want)
{
  ASSERT_EQ(commands.size(), want.size());
  for (std::size_t i = 0; i < want.size(); ++i) {
    ASSERT_TRUE(commands[i].steer_angle.has_value());
    EXPECT_NEAR(*commands[i].steer_angle, want[i], 1e-9);
  }
}

void expect_velocity(const Velocity & v, double vx, double vy, double wz)
{
  EXPECT_NEAR(v.vx, vx, 1e-9);
  EXPECT_NEAR(v.vy, vy, 1e-9);
  EXPECT_NEAR(v.wz, wz, 1e-9);
}

}  // namespace

TEST(Steering, MisalignmentIsTheLargestDifferenceAndIgnoresKeptAngles)
{
  EXPECT_NEAR(
    steering_misalignment({0.1, std::nullopt, -0.2, 0.0}, {0.0, 1.0, 0.0, 0.05}), 0.2, 1e-12);
  EXPECT_THROW(steering_misalignment({0.0}, {0.0, 0.0}), std::invalid_argument);
}

TEST(Steering, AlignedWheelsDriveWithTheUsualRamp)
{
  const auto step = steer_first(
    {0.5, 0.0, 0.0}, {0.0, 0.0, 0.0}, kStraight, kModules, kRadius, kLimits, 0.02, 0.05);
  expect_velocity(step.velocity, 0.02, 0.0, 0.0);   // 1 m/s^2 for 20 ms
  for (const auto & c : step.commands) {
    EXPECT_NEAR(c.wheel_speed, 0.02 / kRadius, 1e-9);
  }
}

TEST(Steering, AStandingRobotTurnsItsWheelsBeforeRolling)
{
  const auto step = steer_first(
    {0.0, 0.0, 0.3}, {0.0, 0.0, 0.0}, kStraight, kModules, kRadius, kLimits, 0.02, 0.05);
  expect_velocity(step.velocity, 0.0, 0.0, 0.0);
  expect_angles(step.commands, spin_angles());
  for (const auto & c : step.commands) {
    EXPECT_EQ(c.wheel_speed, 0.0);
  }
  double biggest = 0.0;
  for (double a : spin_angles()) {
    biggest = std::max(biggest, std::abs(a));
  }
  EXPECT_GT(biggest, 40.0 * kPi / 180.0);   // a big change
}

TEST(Steering, OnceAlignedTheSpinRampsUp)
{
  const auto step = steer_first(
    {0.0, 0.0, 0.3}, {0.0, 0.0, 0.0}, spin_angles(), kModules, kRadius, kLimits, 0.02, 0.05);
  expect_velocity(step.velocity, 0.0, 0.0, 0.02);
  for (const auto & c : step.commands) {
    EXPECT_GT(std::abs(c.wheel_speed), 0.0);
  }
}

TEST(Steering, AMovingRobotBrakesGentlyBeforeItSteers)
{
  const auto step = steer_first(
    {0.0, 0.3, 0.0}, {0.5, 0.0, 0.0}, kStraight, kModules, kRadius, kLimits, 0.02, 0.05);
  expect_velocity(step.velocity, 0.48, 0.0, 0.0);   // braking at 1 m/s^2
  expect_angles(step.commands, kStraight);          // still straight
}

TEST(Steering, AMixedCommandSpeedsUpWithTheWheelsAtItsAngles)
{
  const Velocity target{0.5, 0.03, 0.0};
  const auto aim = angles(inverse_kinematics(target, kModules, kRadius));
  auto step = steer_first(
    target, {0.0, 0.0, 0.0}, aim, kModules, kRadius, kLimits, 0.02, 0.05);
  expect_angles(step.commands, aim);
  step = steer_first(
    target, {0.0, 0.0, 0.0}, aim, kModules, kRadius, kLimits, 0.02, 0.05, false);
  expect_angles(step.commands, std::vector<double>(4, kPi / 4));   // ramping apart: 45 deg
}

TEST(Steering, ASmallCorrectionWhileCrabbingDoesNotStopTheRobotWithAMargin)
{
  const std::vector<double> crab(4, kPi / 2);
  auto step = steer_first(
    {-0.01, 0.5, 0.0}, {0.0, 0.5, 0.0}, crab, kModules, kRadius, kLimits, 0.02, 0.05, true,
    0.15);
  expect_velocity(step.velocity, -0.01, 0.5, 0.0);
  expect_angles(step.commands, crab);
  step = steer_first(
    {-0.01, 0.5, 0.0}, {0.0, 0.5, 0.0}, crab, kModules, kRadius, kLimits, 0.02, 0.05);
  expect_velocity(step.velocity, 0.0, 0.48, 0.0);   // brakes to swing the wheels round
}

TEST(StallWatch, ASteeringJointThatStopsShortIsNoticedAndACreepTimed)
{
  StallWatch watch(1.5, 2.0, 0.02);
  EXPECT_FALSE(watch.update(0.25, 0.8, 0.05, true));   // notes where it stands
  for (int k = 0; k < 5; ++k) {
    EXPECT_FALSE(watch.update(0.25, 0.8, 0.05, true));   // 1.25 s without progress
  }
  EXPECT_TRUE(watch.update(0.25, 0.8, 0.05, true));    // 1.5 s: creep
  EXPECT_EQ(watch.count(), 1);
  for (int k = 0; k < 7; ++k) {
    EXPECT_TRUE(watch.update(0.25, 0.8, 0.05, false));
  }
  EXPECT_FALSE(watch.update(0.25, 0.8, 0.05, false));  // 2 s
}

TEST(StallWatch, SteeringThatMakesProgressOrIsAlignedNeverCreeps)
{
  StallWatch watch;
  double angle = 1.5;
  for (int k = 0; k < 200; ++k) {   // a slow but steady turn
    angle -= 0.004;
    EXPECT_FALSE(watch.update(0.02, angle, 0.05, true));
  }
  EXPECT_FALSE(watch.update(0.02, 0.01, 0.05, true));
  EXPECT_THROW(StallWatch(0.0), std::invalid_argument);
}

TEST(Creep, RollsEachWheelAlongWhereItPoints)
{
  const double turned = -54.0 * kPi / 180.0;
  const std::vector<double> stuck{0.0, turned, 0.0, 0.0};   // front right still turned
  const auto step = creep(
    {0.5, 0.0, 0.0}, {0.0, 0.0, 0.0}, stuck, kModules, kRadius, kLimits, 0.1, 0.05);
  expect_velocity(step.velocity, 0.05, 0.0, 0.0);   // 1 m/s^2 for 0.1 s, at most 5 cm/s
  expect_angles(step.commands, std::vector<double>(4, 0.0));   // still steering
  EXPECT_NEAR(step.commands[0].wheel_speed * kRadius, 0.05, 1e-9);
  EXPECT_NEAR(step.commands[1].wheel_speed * kRadius, 0.05 * std::cos(turned), 1e-9);
}
