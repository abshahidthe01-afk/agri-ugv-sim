"""Tests for the RTK error model and the heading from two antennas."""

import math

from agri_ugv_localization.gnss_errors import (GaussMarkov, heading_and_pitch,
                                               metres_per_degree, offset, quaternion,
                                               RtkErrors, shift)
import numpy as np
import pytest

FIELD = (50.626127482, 6.984883225, 221.845)       # the must_c_field world origin


def test_gauss_markov_keeps_its_spread_and_forgets_after_tau():
    process = GaussMarkov([0.01], 10.0, np.random.default_rng(1))
    values = np.array([process.step(1.0)[0] for _ in range(200000)])
    assert values.std() == pytest.approx(0.01, rel=0.05)
    lag = np.corrcoef(values[:-10], values[10:])[0, 1]       # 10 steps of 1 s = tau
    assert lag == pytest.approx(math.exp(-1), abs=0.05)


@pytest.mark.parametrize('sigma, tau, dt', [([-0.1], 1.0, 1.0), ([0.1], 0.0, 1.0),
                                            ([0.1], 1.0, -1.0)])
def test_gauss_markov_rejects_bad_numbers(sigma, tau, dt):
    with pytest.raises(ValueError):
        GaussMarkov(sigma, tau, np.random.default_rng(1)).step(dt)


def test_metres_per_degree_at_the_field_match_geodesic_distances():
    per_lat, per_lon = metres_per_degree(FIELD[0])
    assert (per_lat, per_lon) == pytest.approx((111241.1085, 70760.3179), abs=0.001)


def test_shift_and_offset_undo_each_other():
    moved = shift(*FIELD, 1.0, 2.0, 3.0)
    assert offset(FIELD, moved) == pytest.approx((1.0, 2.0, 3.0), abs=1e-6)


def test_heading_and_pitch_of_a_known_baseline():
    rear = FIELD
    front = shift(*rear, 1.2 * math.cos(math.radians(30)), 1.2 * math.sin(math.radians(30)),
                  -0.012)                          # front 1.2 cm lower: nose down
    yaw, pitch = heading_and_pitch(rear, front)
    assert math.degrees(yaw) == pytest.approx(30.0, abs=1e-5)
    assert pitch == pytest.approx(math.atan2(0.012, 1.2), abs=1e-7)


def test_shared_error_cancels_in_the_heading():
    errors = RtkErrors(own_sigma=(0.0, 0.0, 0.0), seed=3)
    for time in (0.0, 0.1, 5.0):
        front, rear = errors.error(time), errors.error(time)
        assert np.allclose(front, rear) and np.linalg.norm(front) > 0


def test_heading_noise_matches_the_own_noise_of_the_antennas():
    errors = RtkErrors(seed=4)
    yaws = []
    for k in range(4000):
        rear, front = errors.error(0.1 * k), errors.error(0.1 * k)
        front_fix = shift(*FIELD, 1.2 + front[0], front[1], front[2])
        yaws.append(heading_and_pitch(shift(*FIELD, *rear), front_fix)[0])
    expected_yaw, _ = errors.heading_sigmas(1.2)
    assert np.std(yaws) == pytest.approx(expected_yaw, rel=0.1)
    assert math.degrees(expected_yaw) == pytest.approx(0.2026, abs=1e-4)


def test_variance_adds_the_shared_and_own_errors():
    assert RtkErrors().variance() == pytest.approx([0.0001 + 0.000009] * 2 + [0.0004 + 0.000036])


def test_quaternion_turns_the_x_axis_to_yaw_and_pitch():
    x, y, z, w = quaternion(0.0, math.radians(5), math.radians(30))
    forward = (1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w))
    expected = (math.cos(math.radians(5)) * math.cos(math.radians(30)),
                math.cos(math.radians(5)) * math.sin(math.radians(30)), -math.sin(math.radians(5)))
    assert forward == pytest.approx(expected)
    assert quaternion(0.0, 0.0, 0.0) == pytest.approx((0.0, 0.0, 0.0, 1.0))
