"""RTK GNSS errors for simulated receivers, and the heading from two antennas."""

import math

import numpy as np

A, F = 6378137.0, 1 / 298.257223563     # WGS84 ellipsoid
E2 = F * (2 - F)


class GaussMarkov:
    """
    First-order Gauss-Markov process: a random error that drifts slowly.

    Its standard deviation stays sigma (one value per axis); values tau seconds apart are
    correlated by 1/e. It starts at a random value with that same spread.
    """

    def __init__(self, sigma, tau, rng):
        """Start the process; rng is a numpy random generator."""
        self.sigma = np.asarray(sigma, dtype=float)
        if np.any(self.sigma < 0) or tau <= 0:
            raise ValueError(f'need sigma >= 0 and tau > 0, got {sigma}, {tau}')
        self.tau, self.rng = float(tau), rng
        self.value = self.sigma * rng.standard_normal(self.sigma.shape)

    def step(self, dt):
        """Move the process dt seconds ahead and return its new value."""
        if dt < 0:
            raise ValueError(f'time must not go backwards, got dt = {dt}')
        a = math.exp(-dt / self.tau)
        noise = self.rng.standard_normal(self.sigma.shape)
        self.value = a * self.value + self.sigma * math.sqrt(1 - a * a) * noise
        return self.value


class RtkErrors:
    """
    Position errors of RTK-fixed receivers on one robot, as (east, north, up) in metres.

    All antennas share a slowly drifting error (same satellites, same corrections), plus
    small independent noise each. The shared part cancels in the heading between antennas.
    """

    def __init__(self, common_sigma=(0.010, 0.010, 0.020), common_tau=60.0,
                 own_sigma=(0.003, 0.003, 0.006), seed=None):
        """Set the error sizes (metres) and the drift's correlation time (seconds)."""
        self.rng = np.random.default_rng(seed)
        self.common = GaussMarkov(common_sigma, common_tau, self.rng)
        self.own_sigma = np.asarray(own_sigma, dtype=float)
        self.time = None

    def error(self, time):
        """Return one antenna's (east, north, up) error for a fix taken at time (seconds)."""
        if self.time is None:
            self.time = time
        elif time > self.time:
            self.common.step(time - self.time)
            self.time = time
        return self.common.value + self.own_sigma * self.rng.standard_normal(3)

    def variance(self):
        """Return the squared standard deviations (m^2) of a fix's east, north, up error."""
        return self.common.sigma ** 2 + self.own_sigma ** 2

    def heading_sigmas(self, baseline):
        """Return the standard deviations (rad) of yaw and pitch from two antennas."""
        return (math.sqrt(2) * self.own_sigma[0] / baseline,
                math.sqrt(2) * self.own_sigma[2] / baseline)


def metres_per_degree(lat):
    """Return (metres per degree of latitude, metres per degree of longitude) at lat."""
    s = math.sin(math.radians(lat))
    w = math.sqrt(1 - E2 * s * s)
    north_south = A * (1 - E2) / w ** 3           # radii of curvature of the ellipsoid
    east_west = A / w
    return (math.radians(1) * north_south,
            math.radians(1) * east_west * math.cos(math.radians(lat)))


def shift(lat, lon, alt, east, north, up):
    """Return (lat, lon, alt) moved by small offsets in metres (exact to well below 1 mm)."""
    per_lat, per_lon = metres_per_degree(lat)
    return lat + north / per_lat, lon + east / per_lon, alt + up


def offset(start, end):
    """Return the small (east, north, up) offset in metres between two (lat, lon, alt)."""
    per_lat, per_lon = metres_per_degree((start[0] + end[0]) / 2)
    return ((end[1] - start[1]) * per_lon, (end[0] - start[0]) * per_lat, end[2] - start[2])


def heading_and_pitch(rear, front):
    """
    Return (yaw, pitch) in radians of the line from the rear to the front antenna.

    yaw is counter-clockwise from true east (the ROS convention); pitch is positive when
    the front is lower (nose down, as in ROS).
    """
    east, north, up = offset(rear, front)
    return math.atan2(north, east), math.atan2(-up, math.hypot(east, north))


def quaternion(roll, pitch, yaw):
    """Return the (x, y, z, w) quaternion of ROS roll, pitch and yaw angles (radians)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)
