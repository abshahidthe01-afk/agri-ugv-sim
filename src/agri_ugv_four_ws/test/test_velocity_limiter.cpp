// The same cases as agri_ugv_control's test_velocity_limiter.py.

#include <gtest/gtest.h>

#include <cmath>

#include "agri_ugv_four_ws/velocity_limiter.hpp"

using agri_ugv_four_ws::limit_velocity;
using agri_ugv_four_ws::Velocity;
using agri_ugv_four_ws::VelocityLimits;

namespace
{

const VelocityLimits kLimits{1.5, 1.0, 1.0, 1.0, 1.0, 1.0};
constexpr double kDt = 0.02;   // [s] one step at 50 Hz

// Apply the limiter repeatedly for the given time and return the final velocity.
Velocity run(const Velocity & target, const Velocity & start, double seconds)
{
  Velocity v = start;
  const int steps = static_cast<int>(std::lround(seconds / kDt));
  for (int k = 0; k < steps; ++k) {
    v = limit_velocity(target, v, kLimits, kDt);
  }
  return v;
}

}  // namespace

TEST(VelocityLimiter, SpeedIsCapped)
{
  const auto v = run({5.4, 3.0, -4.0}, {0.0, 0.0, 0.0}, 10.0);
  EXPECT_NEAR(v.vx, 1.5, 1e-9);
  EXPECT_NEAR(v.vy, 1.0, 1e-9);
  EXPECT_NEAR(v.wz, -1.0, 1e-9);
}

TEST(VelocityLimiter, AccelerationIsGradual)
{
  EXPECT_NEAR(run({1.5, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0.5).vx, 0.5, 1e-9);
}

TEST(VelocityLimiter, BrakingIsGradual)
{
  EXPECT_NEAR(run({0.0, 0.0, 0.0}, {1.5, 0.0, 0.0}, 1.0).vx, 0.5, 1e-9);
  EXPECT_NEAR(run({0.0, 0.0, 0.0}, {1.5, 0.0, 0.0}, 1.5).vx, 0.0, 1e-9);
}

TEST(VelocityLimiter, NoOvershoot)
{
  Velocity v{0.0, 0.0, 0.0};
  for (int k = 0; k < 500; ++k) {
    v = limit_velocity({0.7, -0.3, 0.2}, v, kLimits, kDt);
    EXPECT_LE(v.vx, 0.7 + 1e-12);
    EXPECT_GE(v.vy, -0.3 - 1e-12);
    EXPECT_LE(v.wz, 0.2 + 1e-12);
  }
  EXPECT_NEAR(v.vx, 0.7, 1e-12);
  EXPECT_NEAR(v.vy, -0.3, 1e-12);
  EXPECT_NEAR(v.wz, 0.2, 1e-12);
}

TEST(VelocityLimiter, ComponentsRampTogetherAndKeepTheDirection)
{
  const Velocity target{0.5, 0.02, 0.01};
  const auto v = limit_velocity(target, {0.0, 0.0, 0.0}, kLimits, kDt);
  EXPECT_NEAR(v.vx, 0.02, 1e-12);       // 4 % of the way
  EXPECT_NEAR(v.vy, 0.0008, 1e-12);
  EXPECT_NEAR(v.wz, 0.0004, 1e-12);
  const auto later = run(target, {0.0, 0.0, 0.0}, 0.5);
  EXPECT_NEAR(later.vx, 0.5, 1e-9);
  EXPECT_NEAR(later.vy, 0.02, 1e-9);
  EXPECT_NEAR(later.wz, 0.01, 1e-9);
}

TEST(VelocityLimiter, ComponentsOnTheirOwnReachASmallPartAtOnce)
{
  const auto v = limit_velocity({0.5, 0.02, 0.01}, {0.0, 0.0, 0.0}, kLimits, kDt, false);
  EXPECT_NEAR(v.vx, 0.02, 1e-12);
  EXPECT_NEAR(v.vy, 0.02, 1e-12);
  EXPECT_NEAR(v.wz, 0.01, 1e-12);
}
