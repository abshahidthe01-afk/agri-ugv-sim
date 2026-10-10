// The same cases as agri_ugv_control's test_kinematics.py.

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "agri_ugv_four_ws/kinematics.hpp"

using agri_ugv_four_ws::forward_kinematics;
using agri_ugv_four_ws::integrate_pose;
using agri_ugv_four_ws::inverse_kinematics;
using agri_ugv_four_ws::Pose2D;
using agri_ugv_four_ws::Velocity;
using agri_ugv_four_ws::WheelModule;

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kRadius = 0.205;
const std::vector<WheelModule> kModules{
  {"front_left", 0.675, 0.75}, {"front_right", 0.675, -0.75},
  {"rear_left", -0.675, 0.75}, {"rear_right", -0.675, -0.75}};

struct Measured
{
  std::vector<double> angles;
  std::vector<double> speeds;
};

// The steering angles and wheel speeds the inverse kinematics would command.
Measured measured(const Velocity & v)
{
  Measured m;
  for (const auto & c : inverse_kinematics(v, kModules, kRadius)) {
    m.angles.push_back(c.steer_angle.value_or(0.0));
    m.speeds.push_back(c.wheel_speed);
  }
  return m;
}

}  // namespace

TEST(InverseKinematics, StraightForward)
{
  for (const auto & c : inverse_kinematics({1.0, 0.0, 0.0}, kModules, kRadius)) {
    EXPECT_NEAR(*c.steer_angle, 0.0, 1e-12);
    EXPECT_NEAR(c.wheel_speed, 1.0 / kRadius, 1e-12);
  }
}

TEST(InverseKinematics, StraightBackwardDoesNotFlipTheWheels)
{
  for (const auto & c : inverse_kinematics({-1.0, 0.0, 0.0}, kModules, kRadius)) {
    EXPECT_NEAR(*c.steer_angle, 0.0, 1e-12);
    EXPECT_NEAR(c.wheel_speed, -1.0 / kRadius, 1e-12);
  }
}

TEST(InverseKinematics, SpinningInPlaceTheWheelsAreTangent)
{
  const double wz = 0.5;
  const auto commands = inverse_kinematics({0.0, 0.0, wz}, kModules, kRadius);
  for (std::size_t i = 0; i < kModules.size(); ++i) {
    const double angle = *commands[i].steer_angle;
    EXPECT_NEAR(std::cos(angle) * kModules[i].x + std::sin(angle) * kModules[i].y, 0.0, 1e-9);
    EXPECT_NEAR(
      std::abs(commands[i].wheel_speed) * kRadius,
      wz * std::hypot(kModules[i].x, kModules[i].y), 1e-12);
  }
}

TEST(InverseKinematics, CrabSideways)
{
  for (const auto & c : inverse_kinematics({0.0, 0.5, 0.0}, kModules, kRadius)) {
    EXPECT_NEAR(*c.steer_angle, kPi / 2, 1e-12);
    EXPECT_NEAR(c.wheel_speed, 0.5 / kRadius, 1e-12);
  }
}

TEST(InverseKinematics, StoppedKeepsTheCurrentAngle)
{
  for (const auto & c : inverse_kinematics({0.0, 0.0, 0.0}, kModules, kRadius)) {
    EXPECT_FALSE(c.steer_angle.has_value());
    EXPECT_EQ(c.wheel_speed, 0.0);
  }
}

TEST(InverseKinematics, RandomMotionsReproduceTheNeededWheelVelocity)
{
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> any(-2.0, 2.0);
  for (int k = 0; k < 1000; ++k) {
    const Velocity v{any(rng), any(rng), any(rng)};
    const auto commands = inverse_kinematics(v, kModules, kRadius);
    for (std::size_t i = 0; i < kModules.size(); ++i) {
      const double angle = *commands[i].steer_angle;
      const double ground = commands[i].wheel_speed * kRadius;
      EXPECT_NEAR(ground * std::cos(angle), v.vx - v.wz * kModules[i].y, 1e-9);
      EXPECT_NEAR(ground * std::sin(angle), v.vy + v.wz * kModules[i].x, 1e-9);
      EXPECT_LE(std::abs(angle), kPi / 2 + 1e-12);   // within the steering limit
    }
  }
}

TEST(InverseKinematics, AWheelJustPastItsLimitStaysThereWithAMargin)
{
  for (const auto & c : inverse_kinematics({-0.01, 0.5, 0.0}, kModules, kRadius, 0.15)) {
    EXPECT_NEAR(*c.steer_angle, kPi / 2, 1e-12);
    EXPECT_NEAR(c.wheel_speed, 0.5 / kRadius, 1e-12);   // the part along it
  }
  const double past = std::atan2(0.01, 0.5);
  for (const auto & c : inverse_kinematics({-0.01, 0.5, 0.0}, kModules, kRadius)) {
    EXPECT_NEAR(*c.steer_angle, -kPi / 2 + past, 1e-12);   // swung round
    EXPECT_LT(c.wheel_speed, 0.0);
  }
}

TEST(InverseKinematics, AWheelFurtherPastItsLimitThanTheMarginStillSwingsRound)
{
  const double angle = std::atan2(0.5, -0.2);
  for (const auto & c : inverse_kinematics({-0.2, 0.5, 0.0}, kModules, kRadius, 0.15)) {
    EXPECT_NEAR(*c.steer_angle, angle - kPi, 1e-12);
    EXPECT_NEAR(c.wheel_speed, -std::hypot(0.2, 0.5) / kRadius, 1e-12);
  }
  EXPECT_THROW(inverse_kinematics({0.5, 0.0, 0.0}, kModules, kRadius, -0.1),
    std::invalid_argument);
}

TEST(ForwardKinematics, UndoesTheInverse)
{
  const std::vector<Velocity> motions{{0.5, 0.0, 0.0}, {-0.3, 0.0, 0.0}, {0.0, 0.4, 0.0},
    {0.0, 0.0, 0.3}, {0.5, 0.2, -0.25}, {-0.4, -0.3, 0.6}};
  for (const auto & v : motions) {
    const auto m = measured(v);
    const auto odom = forward_kinematics(m.angles, m.speeds, kModules, kRadius);
    EXPECT_NEAR(odom.velocity.vx, v.vx, 1e-12);
    EXPECT_NEAR(odom.velocity.vy, v.vy, 1e-12);
    EXPECT_NEAR(odom.velocity.wz, v.wz, 1e-12);
    EXPECT_NEAR(odom.residual, 0.0, 1e-12);
  }
}

TEST(ForwardKinematics, RandomMotionsRoundTrip)
{
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> fast(-1.5, 1.5), slow(-1.0, 1.0);
  for (int k = 0; k < 200; ++k) {
    const Velocity v{fast(rng), slow(rng), slow(rng)};
    const auto m = measured(v);
    const auto odom = forward_kinematics(m.angles, m.speeds, kModules, kRadius);
    EXPECT_NEAR(odom.velocity.vx, v.vx, 1e-9);
    EXPECT_NEAR(odom.velocity.vy, v.vy, 1e-9);
    EXPECT_NEAR(odom.velocity.wz, v.wz, 1e-9);
  }
}

TEST(ForwardKinematics, ASlippingWheelShowsInTheResidualAndIsAveragedOut)
{
  auto m = measured({0.5, 0.0, 0.0});
  m.speeds[0] *= 1.2;   // one wheel spins 20 % too fast
  const auto odom = forward_kinematics(m.angles, m.speeds, kModules, kRadius);
  EXPECT_NEAR(odom.velocity.vx, 0.525, 1e-12);   // 0.5 + 0.1 / 4: shared by 4 wheels
  EXPECT_GT(odom.residual, 0.02);
}

TEST(ForwardKinematics, RejectsMismatchedInput)
{
  EXPECT_THROW(forward_kinematics({0.0}, {1.0, 1.0}, kModules, kRadius), std::invalid_argument);
}

TEST(IntegratePose, AConstantTurnClosesTheCircle)
{
  Pose2D pose{0.0, 0.0, 0.0};
  const double dt = 2 * kPi / 0.5 / 1257;   // 1257 steps of about 10 ms: one full turn
  for (int k = 0; k < 1257; ++k) {
    pose = integrate_pose(pose, {1.0, 0.0, 0.5}, dt);
  }
  EXPECT_NEAR(pose.x, 0.0, 1e-9);
  EXPECT_NEAR(pose.y, 0.0, 1e-9);
  EXPECT_NEAR(pose.yaw, 2 * kPi, 1e-9);
}

TEST(IntegratePose, SidewaysMotionFollowsTheHeading)
{
  const auto pose = integrate_pose({1.0, 2.0, kPi / 2}, {0.0, 0.5, 0.0}, 2.0);
  EXPECT_NEAR(pose.x, 0.0, 1e-12);
  EXPECT_NEAR(pose.y, 2.0, 1e-12);
  EXPECT_NEAR(pose.yaw, kPi / 2, 1e-12);
}

TEST(SteeringRate, IsTheFastestJoint)
{
  using agri_ugv_four_ws::steering_rate;
  EXPECT_NEAR(steering_rate({0.0, 0.1, -0.2, 0.0}, {0.01, 0.1, -0.23, 0.0}, 0.01), 3.0, 1e-9);
  EXPECT_EQ(steering_rate({0.0}, {1.0}, 0.0), 0.0);   // no time: no speed
  EXPECT_THROW(steering_rate({0.0}, {0.0, 0.0}, 0.01), std::invalid_argument);
}

TEST(OdometryVariances, AddTheNoiseTheDisagreementAndTheSlideWhileSteering)
{
  using agri_ugv_four_ws::odometry_variances;
  auto v = odometry_variances(0.0, 1.0, 0.0, 0.02, 0.02, 0.5);   // rolling, wheels agree
  EXPECT_NEAR(v.speed, 0.02 * 0.02, 1e-15);
  EXPECT_NEAR(v.turn, 0.02 * 0.02, 1e-15);
  v = odometry_variances(0.03, 1.5, 0.0, 0.02, 0.02, 0.5);       // one wheel slips
  EXPECT_NEAR(v.speed, 0.02 * 0.02 + 0.03 * 0.03, 1e-15);
  EXPECT_NEAR(v.turn, 0.02 * 0.02 + 0.02 * 0.02, 1e-15);
  v = odometry_variances(0.0, 1.0, 1.0, 0.02, 0.02, 0.5);        // re-steering on the spot
  EXPECT_NEAR(std::sqrt(v.speed), std::hypot(0.02, 0.5), 1e-12);
  EXPECT_NEAR(v.turn, 0.02 * 0.02, 1e-15);
}
