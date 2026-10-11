"""
3D reconstruction of the plants under the robot from the dome cameras' depth images.

The real robot (Esser et al. 2023) stops, takes a picture with each of its 20 dome cameras
at the same moment, finds the cameras' poses, fits a signed distance field to the pictures
and cuts a textured mesh out of it with Marching Cubes. Here the cameras measure depth
themselves (RGB-D cameras, sim.launch.py camera_depth:=true): perfect depth replaces the
matching of the pictures, the rest follows the same idea:
- fuse(): each depth image gives the voxels near the surface it sees their signed distance
  to that surface along its view (truncated at 'trunc' voxels), averaged over the cameras.
  Only voxels within that band are changed: the free space one camera sees past the edge of
  a thin leaf would otherwise erase the leaf another camera sees.
- mesh(): Marching Cubes cuts the surface where the distance is zero, in the cubes whose
  eight corners all have a distance.
- colour(): each vertex takes the mean colour of the pictures that see it (its depth there
  within 'tolerance' of the depth image); faces no camera sees (left-overs of the distance
  field under the soil and inside leaves) are dropped.

capture.py takes the pictures (ros2 run agri_ugv_phenotyping capture_depth); then
    ros2 run agri_ugv_phenotyping reconstruct ~/agri_data/runs/depth_198.npz [--truth]
writes <name>_mesh.ply (the coloured mesh, robot frame: x forward, z up from the plane the
wheels stand on) and <name>_mesh.png; with --truth (run from the workspace folder) it
compares the mesh with the true leaves (truth.grown_leaves) at the robot's true pose.
"""

import argparse
import math
from pathlib import Path
import time

from agri_ugv_phenotyping.cloud import nearest_distances, top_surface
from agri_ugv_phenotyping.colour import excess_green, GREEN, project
import numpy as np

LOW = (-0.6, -0.6, -0.05)       # [m] the volume, robot frame: inside the enclosure's walls
HIGH = (0.6, 0.6, 0.55)


def fuse(depths, cameras, low=LOW, high=HIGH, voxel=0.004, trunc=3):
    """
    Return the truncated signed distances (in units of trunc voxels) and their weights.

    depths: depth images (metres along each camera's view; non-finite: nothing measured);
    cameras: (mounting, k) per image as for colour.project. The volume reaches from 'low'
    to 'high' (robot frame) in cubes of 'voxel'; each value is the voxel's distance in front
    of (+) or behind (-) the surface, clipped to +-1, and the weight how many cameras gave
    it. Voxels no camera saw near a surface keep weight 0.
    """
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    shape = np.ceil((high - low) / voxel - 1e-9).astype(int)
    band = trunc * voxel
    distance = np.ones(shape, dtype=np.float32)
    weight = np.zeros(shape, dtype=np.float32)
    centres = [low[a] + (np.arange(shape[a]) + 0.5) * voxel for a in range(3)]
    y, z = np.meshgrid(centres[1], centres[2], indexing='ij')
    for depth, (mounting, k) in zip(depths, cameras):
        rows, cols = depth.shape
        for a, x in enumerate(centres[0]):          # one slice of constant x at a time
            points = np.column_stack([np.full(y.size, x), y.ravel(), z.ravel()])
            u, v, along = project(points, mounting, k)
            with np.errstate(invalid='ignore'):
                u, v = np.round(u), np.round(v)
                inside = (along > 0.05) & (u >= 0) & (u < cols) & (v >= 0) & (v < rows)
            index = np.flatnonzero(inside)
            measured = depth[v[index].astype(int), u[index].astype(int)]
            gap = measured - along[index]
            near = np.isfinite(gap) & (np.abs(gap) < band)
            index, gap = index[near], gap[near]
            if len(index) == 0:
                continue
            i, j = np.unravel_index(index, y.shape)
            w = weight[a, i, j]
            distance[a, i, j] = (distance[a, i, j] * w + gap / band) / (w + 1)
            weight[a, i, j] = w + 1
    return distance, weight


def mesh(distance, weight, low=LOW, voxel=0.004):
    """Return the surface (vertices N x 3 in the robot frame, faces M x 3) of fuse()."""
    from skimage.measure import marching_cubes
    seen = weight > 0
    cube = seen.copy()                    # a cube counts only if all its corners were seen
    cube[:-1] &= seen[1:]
    cube[:, :-1] &= cube[:, 1:]
    cube[:, :, :-1] &= cube[:, :, 1:]
    cube[-1], cube[:, -1], cube[:, :, -1] = False, False, False
    values = distance[seen]
    if not cube.any() or values.min() >= 0 or values.max() <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    vertices, faces, _, _ = marching_cubes(np.where(seen, distance, 1.0), 0.0,
                                           spacing=(voxel,) * 3, mask=cube)
    return vertices + np.asarray(low) + voxel / 2, faces


def colour(vertices, faces, pictures, depths, cameras, tolerance=0.006):
    """
    Return the vertices' mean colours (N x 3, 0-255), how many pictures saw each, the faces.

    A picture sees a vertex where its depth image there lies within 'tolerance' of the
    vertex's depth. Only the faces whose three vertices some picture sees are returned.
    """
    sums = np.zeros((len(vertices), 3))
    views = np.zeros(len(vertices), dtype=int)
    for picture, depth, (mounting, k) in zip(pictures, depths, cameras):
        rows, cols = depth.shape
        u, v, along = project(vertices, mounting, k)
        with np.errstate(invalid='ignore'):
            u, v = np.round(u), np.round(v)
            inside = np.flatnonzero((along > 0.05) & (u >= 0) & (u < cols) & (v >= 0) &
                                    (v < rows))
        uu, vv = u[inside].astype(int), v[inside].astype(int)
        seen = inside[np.abs(depth[vv, uu] - along[inside]) < tolerance]
        uu, vv = u[seen].astype(int), v[seen].astype(int)
        sums[seen] += picture[vv, uu]
        views[seen] += 1
    kept = faces[(views[faces] > 0).all(axis=1)] if len(faces) else faces
    return sums / np.maximum(views, 1)[:, None], views, kept


def write_mesh_ply(path, vertices, faces, colours, comments=()):
    """Write a mesh with vertex colours as a binary PLY (only the vertices the faces use)."""
    used = np.unique(faces) if len(faces) else np.zeros(0, dtype=int)
    index = np.full(len(vertices), -1)
    index[used] = np.arange(len(used))
    header = ['ply', 'format binary_little_endian 1.0', *[f'comment {c}' for c in comments],
              f'element vertex {len(used)}', 'property float x', 'property float y',
              'property float z', 'property uchar red', 'property uchar green',
              'property uchar blue', f'element face {len(faces)}',
              'property list uchar int vertex_indices', 'end_header']
    rows = np.zeros(len(used), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    rows['x'], rows['y'], rows['z'] = np.asarray(vertices, dtype=float)[used].T
    shown = np.clip(np.round(np.asarray(colours, dtype=float)[used]), 0, 255).astype(np.uint8)
    rows['red'], rows['green'], rows['blue'] = shown.T
    polygons = np.zeros(len(faces), dtype=[('n', 'u1'), ('a', '<i4'), ('b', '<i4'),
                                           ('c', '<i4')])
    polygons['n'] = 3
    if len(faces):
        polygons['a'], polygons['b'], polygons['c'] = index[faces].T
    with open(path, 'wb') as out:
        out.write(('\n'.join(header) + '\n').encode('ascii'))
        out.write(rows.tobytes())
        out.write(polygons.tobytes())
    return len(used)


def area(vertices, faces):
    """Return the area of a mesh's faces [m2]."""
    if len(faces) == 0:
        return 0.0
    corners = np.asarray(vertices)[faces]
    return float(0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                               corners[:, 2] - corners[:, 0]), axis=1).sum())


def leaf_scores(vertices, faces, colours, leaves, reach=0.55, low=0.06):
    """
    Compare a mesh (robot frame) with the true leaf points (robot frame).

    accuracy: distances of the vertices with a plant's colour to the nearest true leaf;
    completeness: of the true leaves' top surface higher than 'low' (the highest leaf point
    per 1 cm column), the share with a vertex within 5 mm, 1 cm and 2 cm. Only within
    'reach' of the robot's centre (x and y).
    """
    used = np.zeros(len(vertices), dtype=bool)
    used[faces] = True
    used &= np.all(np.abs(vertices[:, :2]) <= reach, axis=1)
    green = used & (excess_green(colours) > GREEN)
    leaves = leaves[np.all(np.abs(leaves[:, :2]) <= reach, axis=1)]
    gaps = nearest_distances(vertices[green], leaves, 0.05)
    top = top_surface(leaves, 0.01)
    top = top[top[:, 2] > low]
    reached = nearest_distances(top, vertices[used], 0.05)
    finite = gaps[np.isfinite(gaps)]
    return {'plant_vertices': int(green.sum()),
            'accuracy_median': float(np.median(finite)) if len(finite) else math.inf,
            'accuracy_95': float(np.sort(gaps)[int(0.95 * (len(gaps) - 1))])
            if len(gaps) else math.inf,
            'within_5mm': float(np.mean(gaps < 0.005)) if len(gaps) else 0.0,
            'top_cells': len(top),
            'complete_5mm': float(np.mean(reached < 0.005)) if len(top) else 0.0,
            'complete_1cm': float(np.mean(reached < 0.01)) if len(top) else 0.0,
            'complete_2cm': float(np.mean(reached < 0.02)) if len(top) else 0.0}


def rotation_rpy(roll, pitch, yaw):
    """Return the rotation matrix of roll, pitch and yaw (applied in that order)."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def true_leaves_here(pose, layout, terrain_model, seed=1, step=0.003, reach=0.62):
    """Return the true leaf points (robot frame) around a robot pose (x, y, z, roll, ...)."""
    from agri_ugv_field.ground import ground_height, read_obj_grid
    from agri_ugv_field.layout import read_layout_csv
    from agri_ugv_field.plants import plot_frame
    from agri_ugv_phenotyping.truth import grown_leaves

    plots = [p for p in read_layout_csv(Path(layout).read_text()) if p['type'] == 'plot']
    terrain = Path(terrain_model)
    grid = read_obj_grid((terrain / 'meshes' / f'{terrain.name}.obj').read_text())

    def ground(x, y):
        return ground_height(grid, x, y)

    def distance(plot):
        centre, along, across = plot_frame(plot)
        d = np.asarray(pose[:2]) - centre
        return max(abs(d @ along) - plot['length'] / 2, abs(d @ across) - plot['width'] / 2)

    near = {p['plot_id'] for p in plots if distance(p) < reach + 0.5}
    turn, origin = rotation_rpy(*pose[3:6]), np.asarray(pose[:3], dtype=float)
    found = [(np.asarray(points) - origin) @ turn
             for _, points in grown_leaves(plots, ground, near, seed, step)]
    points = np.vstack(found) if found else np.zeros((0, 3))
    return points[np.all(np.abs(points[:, :2]) <= reach, axis=1)]


def render(ax, vertices, faces, values, eye=(-0.55, -0.75, 1.25), target=(0.0, 0.0, 0.12),
           size=(900, 700), focal=900.0):
    """Draw a mesh's faces into a matplotlib axis, far ones first, values: RGB 0-1 per face."""
    from matplotlib.collections import PolyCollection
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    ahead = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(ahead, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, ahead)
    corners = np.asarray(vertices)[faces] - eye
    depth = corners @ ahead
    screen = np.stack([size[0] / 2 + focal * (corners @ right) / depth,
                       size[1] / 2 - focal * (corners @ up) / depth], axis=-1)
    order = np.argsort(-depth.mean(axis=1))
    ax.add_collection(PolyCollection(screen[order], facecolors=values[order],
                                     edgecolors='none', antialiased=False))
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)
    ax.set_aspect('equal')
    ax.set_facecolor((0.12, 0.12, 0.12))
    ax.set_xticks([])
    ax.set_yticks([])


def shading(vertices, faces):
    """Return each face's brightness under a light from above (0.45 to 1)."""
    corners = np.asarray(vertices)[faces]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    light = np.array([0.3, -0.4, 1.0]) / math.sqrt(1.25)
    return 0.45 + 0.55 * np.abs(normal @ light)


def main(argv=None):
    """Reconstruct the plants of one capture (capture_depth) as a coloured mesh."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('capture', help='a capture of capture_depth (.npz)')
    parser.add_argument('--voxel', type=float, default=0.004, help='voxel [m] (0.004)')
    parser.add_argument('--trunc', type=float, default=3, help='band, in voxels (3)')
    parser.add_argument('--tolerance', type=float, default=0.006,
                        help='depth tolerance [m] for a picture to see a vertex (0.006)')
    parser.add_argument('--gain', type=float, default=2.0,
                        help='brightness of the picture and mesh drawn in the PNG (2.0)')
    parser.add_argument('--truth', action='store_true',
                        help='compare with the true leaves (run from the workspace folder)')
    parser.add_argument('--layout', default='src/agri_ugv_field/data/must_c_field_plots.csv')
    parser.add_argument('--terrain-model', default='src/agri_ugv_gazebo/models/must_c_field')
    args = parser.parse_args(argv)
    path = Path(args.capture).expanduser()
    data = np.load(path, allow_pickle=False)
    mounts = data['mounts']
    cameras = [((m[:3, :3], m[:3, 3]), list(k)) for m, k in zip(mounts, data['k'])]
    start = time.monotonic()
    distance, weight = fuse(data['depths'], cameras, voxel=args.voxel, trunc=args.trunc)
    vertices, faces = mesh(distance, weight, voxel=args.voxel)
    fused = time.monotonic() - start
    colours, views, kept = colour(vertices, faces, data['pictures'], data['depths'], cameras,
                                  args.tolerance)
    pose = [float(v) for v in data['truth'][1:7]]
    out = path.with_name(path.stem + '_mesh')
    comments = [f'3D reconstruction from {len(cameras)} dome cameras, voxel {args.voxel} m',
                'robot frame: x forward, y left, z up from the plane the wheels stand on; '
                'robot (true pose) x {:.3f} y {:.3f} z {:.3f} roll {:.5f} pitch {:.5f} '
                'yaw {:.5f}'.format(*pose)]
    count = write_mesh_ply(f'{out}.ply', vertices, kept, colours, comments)
    lines = [f'{len(cameras)} cameras, {np.prod(distance.shape) / 1e6:.1f} million voxels of '
             f'{1000 * args.voxel:.0f} mm: {len(faces)} faces in {fused:.0f} s; seen by the '
             f'pictures: {len(kept)} faces ({100 * len(kept) / max(1, len(faces)):.0f} %), '
             f'{count} vertices, {area(vertices, kept):.2f} m2']
    if args.truth:
        leaves = true_leaves_here(pose, args.layout, args.terrain_model)
        s = leaf_scores(vertices, kept, colours, leaves)
        lines.append(f'Against the true leaves ({len(leaves)} points): {s["plant_vertices"]} '
                     f'vertices with a plant colour, median {1000 * s["accuracy_median"]:.1f} '
                     f'mm from the leaves, 95 % within {1000 * s["accuracy_95"]:.1f} mm '
                     f'({100 * s["within_5mm"]:.0f} % within 5 mm); the leaves\' top surface '
                     f'({s["top_cells"]} cells of 1 cm): {100 * s["complete_5mm"]:.0f} % has a '
                     f'vertex within 5 mm, {100 * s["complete_1cm"]:.0f} % within 1 cm, '
                     f'{100 * s["complete_2cm"]:.0f} % within 2 cm')
    print('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shown = kept[np.all(np.abs(vertices[kept].mean(axis=1)[:, :2]) <= 0.55, axis=1)] \
        if len(kept) else kept
    light = shading(vertices, shown)
    face_colour = np.clip(colours[shown].mean(axis=1) * args.gain / 255, 0, 1) * light[:, None]
    height = plt.cm.viridis(np.clip(vertices[shown][:, :, 2].mean(axis=1) / 0.32, 0, 1))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    top = int(np.argmax(mounts[:, 2, 3]))                       # the highest camera
    axes[0].imshow(np.clip(data['pictures'][top].astype(float) * args.gain, 0, 255)
                   .astype(np.uint8))
    axes[0].set_title(f'One of the {len(cameras)} pictures (brightened)')
    axes[0].axis('off')
    render(axes[1], vertices, shown, face_colour)
    axes[1].set_title(f'3D reconstruction from all {len(cameras)} cameras (its colours)')
    render(axes[2], vertices, shown, height[:, :3] * light[:, None])
    axes[2].set_title('The same mesh coloured by height (0 to 32 cm)')
    fig.tight_layout()
    fig.savefig(f'{out}.png', dpi=90)
    print(f'Saved {out}.ply and {out}.png')


if __name__ == '__main__':
    main()
