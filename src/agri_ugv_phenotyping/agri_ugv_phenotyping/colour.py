"""
Colours for the scanners' 3D points from the dome cameras' pictures (camera-laser fusion).

The line scanners measure where the plants are, the cameras see their colour. A camera sits
at a known place on the robot (its optical link in the robot model: x along its view, y to
the left of the picture, z up) and the robot's pose at the picture's time is known, so every
recent scanner point can be projected into the picture with the camera's calibration (focal
lengths and centre, camera_info). A point takes the picture's colour there only if the camera
sees it:
- it lies inside the enclosure (outside, the curtains and side panels hide it), and
- no other recent point lies in front of it: a depth test against all of them, projected
  into the same picture, in blocks of 4 pixels and closed over 7 blocks (about 7 cm at the
  plants, more than the 5 cm between two profiles at 10 per second and 0.5 m/s), so that
  the gaps between profiles do not let a hidden point through. Plot 198 (sugar beet): 57 %
  of the points get a colour; of those on the true leaves 97 % are green, of those on the
  soil 1 % (without the closing 84 %: soil under leaves seen through the gaps).
A picture is used only 'delay' seconds after it was taken, once the profiles just ahead of
the scan line (which may hide points from the camera) have been measured.
"""

from collections import deque
import math

import numpy as np

INSIDE = 0.72           # [m] the enclosure's inner walls: points this far from the centre
GREEN = 0.1             # excess green above this: a plant's colour (excess_green)


def project(points, mounting, k):
    """
    Return pixel columns u, rows v and depths of robot-frame points (N x 3) in a picture.

    mounting is (rotation, translation) of the camera's optical link in the robot frame
    (x along its view); k is the calibration matrix as camera_info gives it (row by row).
    Points behind the camera get a depth of zero or less.
    """
    rotation, translation = mounting
    q = (np.asarray(points, dtype=float) - translation) @ np.asarray(rotation)
    fx, cx, fy, cy = k[0], k[2], k[4], k[5]
    depth = q[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = cx - fx * q[:, 1] / depth
        v = cy - fy * q[:, 2] / depth
    return u, v, depth


def nearest_in_front(u, v, depth, shape, block=4, window=7, tolerance=0.05):
    """
    Return which points are seen: none of the others lies more than 'tolerance' in front.

    The nearest depth is kept per block of 'block' x 'block' pixels, then the smallest of
    'window' x 'window' blocks around each block: a point is hidden when anything within that
    neighbourhood is nearer to the camera.
    """
    rows, cols = shape[0] // block + 1, shape[1] // block + 1
    bu, bv = (u // block).astype(int), (v // block).astype(int)
    nearest = np.full((rows, cols), np.inf)
    np.minimum.at(nearest, (bv, bu), depth)
    half = window // 2
    padded = np.pad(nearest, half, constant_values=np.inf)
    closed = np.full((rows, cols), np.inf)
    for dv in range(window):
        for du in range(window):
            np.minimum(closed, padded[dv:dv + rows, du:du + cols], out=closed)
    return depth <= closed[bv, bu] + tolerance


def robot_points(xy, height, pose):
    """Return world points (xy N x 2, heights N) in the robot frame at pose (x, y, yaw)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(xy, dtype=float) - [x, y]
    return np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1],
                            np.asarray(height, dtype=float)])


class Colourist:
    """Keep the recent scanner points and pictures; colour the points the pictures see."""

    def __init__(self, cameras, keep=3.0, delay=1.0, block=4, window=7, tolerance=0.05):
        """
        cameras: {name: (mounting, k)} as for project(); keep: seconds of scanner points kept.

        delay: seconds a picture waits for the profiles after it; block, window and
        tolerance: the depth test (nearest_in_front).
        """
        self.cameras = dict(cameras)
        self.keep, self.delay = keep, delay
        self.test = {'block': block, 'window': window, 'tolerance': tolerance}
        self.profiles = deque()               # (t, xy N x 2, heights N)
        self.pictures = deque()               # (t, camera, image H x W x 3)
        self.latest = None                    # the latest profile's time

    def add_profile(self, t, xy, height):
        """Keep one profile's world points (xy N x 2, heights N) measured at time t."""
        if len(xy):
            self.profiles.append((t, np.asarray(xy, dtype=float),
                                  np.asarray(height, dtype=float)))
        self.latest = t if self.latest is None else max(self.latest, t)
        while self.profiles and self.profiles[0][0] < self.latest - self.keep:
            self.profiles.popleft()

    def add_picture(self, t, camera, image):
        """Keep a picture (H x W x 3, RGB) of a known camera, taken at time t."""
        if camera in self.cameras:
            self.pictures.append((t, camera, np.asarray(image)))
        while self.pictures and self.pictures[0][0] < t - self.keep - self.delay:
            self.pictures.popleft()            # no profiles came for it: it can never colour

    def ready(self, pose_at):
        """
        Colour the points seen in the pictures old enough; return a list of samples.

        pose_at(t) returns the robot's pose (x, y, yaw) at time t, or None. Each sample is
        (xy N x 2, heights N, colours N x 3) of the points one picture sees.
        """
        samples = []
        while self.pictures and self.latest is not None and \
                self.pictures[0][0] <= self.latest - self.delay:
            t, camera, image = self.pictures.popleft()
            pose = pose_at(t)
            if pose is None or not self.profiles:
                continue
            sample = self.colour(image, self.cameras[camera], pose)
            if sample is not None:
                samples.append(sample)
        return samples

    def colour(self, image, camera, pose):
        """Return (xy, heights, colours) of the kept points one picture sees, or None."""
        mounting, k = camera
        xy = np.vstack([p[1] for p in self.profiles])
        height = np.concatenate([p[2] for p in self.profiles])
        local = robot_points(xy, height, pose)
        inside = np.flatnonzero(np.all(np.abs(local[:, :2]) <= INSIDE, axis=1))
        u, v, depth = project(local[inside], mounting, k)
        rows, cols = image.shape[:2]
        on = (depth > 0.05) & (u >= 0) & (u < cols) & (v >= 0) & (v < rows)
        inside, u, v, depth = inside[on], u[on], v[on], depth[on]
        if len(inside) == 0:
            return None
        seen = nearest_in_front(u, v, depth, (rows, cols), **self.test)
        index = inside[seen]
        colours = image[v[seen].astype(int), u[seen].astype(int)]
        return xy[index], height[index], colours


def excess_green(colours):
    """Return 2g - r - b of RGB colours (N x 3), with r + g + b = 1: plants about 0.3, soil 0."""
    rgb = np.asarray(colours, dtype=float)
    total = rgb.sum(axis=1)
    total[total <= 0] = 1.0
    r, g, b = (rgb / total[:, None]).T
    return 2 * g - r - b
