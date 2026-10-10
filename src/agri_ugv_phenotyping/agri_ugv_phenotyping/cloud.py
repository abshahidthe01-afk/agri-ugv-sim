"""
3D point cloud of each plot from the line scanners: the plants' shape, not only their top.

The plant map (plant_map) keeps per cell only the highest point; here every profile point is
kept, in the plot's own frame (x along the rows, y across them, z the height above the plane
the wheels stand on, as in plant_map), merged per 'voxel' (a cube: the mean of the points in
it and how many there were). A plot then holds about 150 000 points (1 cm voxels) however
long the robot scans it. Saved as a PLY file per plot, which point cloud viewers open.

compare() checks a cloud against the true leaves (truth.true_leaves): how far its plant
points lie from the leaves (accuracy) and how much of the leaves' top surface it has
points on (completeness), without SciPy: nearest_distances searches a grid.
"""

import itertools

from agri_ugv_field.plants import plot_frame
import numpy as np

HEIGHTS = (-0.5, 3.0)   # [m] points lower or higher than this are not plants or soil


def to_plot_frame(plot, xy, height):
    """Return world points (xy N x 2, height N) in a plot's frame (along, across, height)."""
    centre, along, across = plot_frame(plot)
    d = np.asarray(xy, dtype=float) - centre
    return np.column_stack([d @ along, d @ across, np.asarray(height, dtype=float)])


class PlotCloud:
    """The 3D points of one plot in its own frame, merged per voxel."""

    def __init__(self, plot, voxel=0.01, batch=100000):
        """Start an empty cloud; points wait in batches of 'batch' before they are merged."""
        if voxel <= 0:
            raise ValueError(f'voxel must be positive, got {voxel}')
        self.plot, self.voxel, self.batch = plot, voxel, batch
        self.half = np.array([plot['length'] / 2, plot['width'] / 2])
        self.size = np.ceil(np.array([plot['length'], plot['width'],
                                      HEIGHTS[1] - HEIGHTS[0]]) / voxel).astype(np.int64) + 1
        self.codes = np.zeros(0, dtype=np.int64)
        self.sums = np.zeros((0, 3), dtype=np.float32)    # float32: a field of plots fits
        self.counts = np.zeros(0, dtype=np.int32)
        self.waiting = []
        self.waiting_points = 0

    def add(self, xy, height):
        """Add world points (xy N x 2, heights N); return how many fell inside the plot."""
        points = to_plot_frame(self.plot, xy, height)
        inside = np.all(np.abs(points[:, :2]) <= self.half, axis=1) & \
            (points[:, 2] >= HEIGHTS[0]) & (points[:, 2] < HEIGHTS[1])
        if inside.any():
            self.waiting.append(points[inside])
            self.waiting_points += int(inside.sum())
            if self.waiting_points >= self.batch:
                self._merge()
        return int(inside.sum())

    def _code(self, points):
        """Return each point's voxel as one number."""
        k = np.floor((points - [-self.half[0], -self.half[1], HEIGHTS[0]]) /
                     self.voxel).astype(np.int64)
        k = np.clip(k, 0, self.size - 1)
        return (k[:, 0] * self.size[1] + k[:, 1]) * self.size[2] + k[:, 2]

    def _merge(self):
        """Merge the waiting points into the voxels."""
        if not self.waiting:
            return
        new = np.vstack(self.waiting)
        self.waiting, self.waiting_points = [], 0
        codes = np.concatenate([self.codes, self._code(new)])
        sums = np.vstack([self.sums, new])
        counts = np.concatenate([self.counts, np.ones(len(new), dtype=np.int32)])
        self.codes, index = np.unique(codes, return_inverse=True)
        self.sums = np.column_stack([np.bincount(index, sums[:, k], len(self.codes))
                                     for k in range(3)]).astype(np.float32)
        self.counts = np.bincount(index, counts, len(self.codes)).astype(np.int32)

    def points(self):
        """Return each voxel's mean point (N x 3, plot frame) and how many points it holds."""
        self._merge()
        return self.sums.astype(float) / self.counts[:, None], self.counts.astype(np.int64)

    def save(self, path):
        """Save the cloud as a PLY file (plot frame); return the number of points."""
        points, counts = self.points()
        centre, along, _ = plot_frame(self.plot)
        write_ply(path, points, counts, comments=[
            f'plot {self.plot["plot_id"]} ({self.plot["crop"]}), voxel {self.voxel} m',
            'x along the rows, y across them, z height above the ground [m], from the '
            f'plot centre ({centre[0]:.3f}, {centre[1]:.3f}) in the world frame, x towards '
            f'({along[0]:.6f}, {along[1]:.6f})'])
        return len(points)


def write_ply(path, points, counts=None, colors=None, comments=()):
    """Write points (N x 3) as a binary PLY, with point counts and RGB colours if given."""
    points = np.asarray(points, dtype=float)
    fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    header = ['ply', 'format binary_little_endian 1.0', *[f'comment {c}' for c in comments],
              f'element vertex {len(points)}', 'property float x', 'property float y',
              'property float z']
    if counts is not None:
        fields.append(('count', '<u4'))
        header.append('property uint count')
    if colors is not None:
        fields += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        header += ['property uchar red', 'property uchar green', 'property uchar blue']
    rows = np.zeros(len(points), dtype=fields)
    rows['x'], rows['y'], rows['z'] = points.T
    if counts is not None:
        rows['count'] = counts
    if colors is not None:
        colors = np.asarray(colors)
        rows['red'], rows['green'], rows['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, 'wb') as out:
        out.write(('\n'.join(header + ['end_header']) + '\n').encode('ascii'))
        out.write(rows.tobytes())


def read_ply(path):
    """Read a binary PLY written by write_ply; return (points N x 3, fields dict, comments)."""
    types = {'float': '<f4', 'uint': '<u4', 'uchar': 'u1'}
    with open(path, 'rb') as text:
        fields, comments, count = [], [], 0
        while True:
            line = text.readline().decode('ascii').strip()
            if line.startswith('comment '):
                comments.append(line[8:])
            elif line.startswith('element vertex'):
                count = int(line.split()[2])
            elif line.startswith('property'):
                _, kind, name = line.split()
                fields.append((name, types[kind]))
            elif line == 'end_header':
                break
        rows = np.frombuffer(text.read(), dtype=fields, count=count)
    points = np.column_stack([rows['x'], rows['y'], rows['z']]).astype(float)
    return points, {name: rows[name] for name, _ in fields}, comments


def single_codes(keys):
    """Return integer keys (N x d) as one number each, equal where the rows are equal."""
    keys = np.asarray(keys, dtype=np.int64)
    low = keys.min(axis=0)
    span = keys.max(axis=0) - low + 1
    codes = np.zeros(len(keys), dtype=np.int64)
    for column in range(keys.shape[1]):
        codes = codes * span[column] + (keys[:, column] - low[column])
    return codes


def voxel_means(points, voxel):
    """Return one point per occupied voxel: the mean of the points in it."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points.reshape(0, 3)
    _, index = np.unique(single_codes(np.floor(points / voxel)), return_inverse=True)
    counts = np.bincount(index)
    return np.column_stack([np.bincount(index, points[:, k]) / counts for k in range(3)])


def nearest_distances(queries, points, radius, stages=(0.25, 1.0)):
    """
    Return the distance from each query (N x 3) to its nearest point; np.inf beyond radius.

    The points are sorted into cubes as wide as the search distance: a query's nearest point
    within it lies in its own cube or one of the 26 around it. Most queries have a point
    close by, so the search first looks 'stages[0]' of radius around them, and only the
    queries without a point that close look further.
    """
    queries = np.asarray(queries, dtype=float)
    points = np.asarray(points, dtype=float)
    best = np.full(len(queries), np.inf)
    todo = np.arange(len(queries))
    for fraction in stages:
        if len(todo) == 0 or len(points) == 0:
            break
        found = _search(queries[todo], points, radius * fraction)
        best[todo] = found
        todo = todo[np.isinf(found)]
    return best


def _search(queries, points, distance):
    """Return each query's distance to its nearest point, np.inf beyond 'distance'."""
    best = np.full(len(queries), np.inf)
    keys = np.floor(points / distance).astype(np.int64)
    low = keys.min(axis=0) - 1
    span = keys.max(axis=0) - low + 2

    def encode(k):
        return ((k[:, 0] - low[0]) * span[1] + (k[:, 1] - low[1])) * span[2] + (k[:, 2] - low[2])

    order = np.argsort(encode(keys), kind='stable')
    codes, sorted_points = encode(keys)[order], points[order]
    base = np.floor(queries / distance).astype(np.int64)
    for offset in itertools.product((-1, 0, 1), repeat=3):
        k = base + offset
        q = np.flatnonzero(np.all((k >= low) & (k < low + span), axis=1))
        cube = encode(k[q])
        start = np.searchsorted(codes, cube, 'left')
        count = np.searchsorted(codes, cube, 'right') - start
        q, start, count = q[count > 0], start[count > 0], count[count > 0]
        for j in range(int(count.max()) if len(count) else 0):
            m = count > j
            d = np.linalg.norm(sorted_points[start[m] + j] - queries[q[m]], axis=1)
            best[q[m]] = np.minimum(best[q[m]], d)
    best[best > distance] = np.inf
    return best


def top_surface(points, cell=0.01):
    """Return the highest point of each 'cell'-wide column (seen from above)."""
    points = np.asarray(points, dtype=float)
    keys = np.floor(points[:, :2] / cell).astype(np.int64)
    order = np.lexsort((-points[:, 2], keys[:, 1], keys[:, 0]))
    k = keys[order]
    first = np.ones(len(k), dtype=bool)
    first[1:] = np.any(k[1:] != k[:-1], axis=1)
    return points[order][first]


def compare(cloud, leaves, low=0.06, seen_cell=0.05, radius=0.05):
    """
    Compare a plot's cloud with its true leaves (both N x 3, plot frame).

    accuracy: distances from the cloud's plant points (higher than 'low') to the nearest
    leaf point: median, 95th percentile (np.inf: beyond 'radius') and the share within
    1 cm. completeness: of the leaves' top surface (seen from above, plant parts
    higher than 'low', in the 'seen_cell' columns the cloud has points in), the share with a
    cloud point within 2 cm and within 5 cm. The leaves should be sampled more finely than a
    centimetre (truth.true_leaves).
    """
    cloud, leaves = np.asarray(cloud, dtype=float), np.asarray(leaves, dtype=float)
    plants = cloud[cloud[:, 2] > low]
    distances = nearest_distances(plants, leaves, radius)
    top = top_surface(leaves)
    top = top[top[:, 2] > low]
    columns = single_codes(np.floor(np.vstack([cloud[:, :2], top[:, :2]]) / seen_cell))
    top = top[np.isin(columns[len(cloud):], columns[:len(cloud)])]
    back = nearest_distances(top, cloud, radius)
    ranked = np.sort(distances)
    return {'plant_points': int(len(plants)),
            'accuracy_median': float(ranked[(len(ranked) - 1) // 2]) if len(plants) else None,
            'accuracy_95': float(ranked[int(np.ceil(0.95 * len(ranked))) - 1])
            if len(plants) else None,
            'within_1cm': float(np.mean(distances <= 0.01)) if len(plants) else None,
            'top_points': int(len(top)),
            'complete_2cm': float(np.mean(back <= 0.02)) if len(top) else None,
            'complete_5cm': float(np.mean(back <= 0.05)) if len(top) else None}
