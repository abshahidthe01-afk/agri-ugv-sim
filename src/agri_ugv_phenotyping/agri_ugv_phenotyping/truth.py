"""
The plots' true canopy, from the crop plants that agri_ugv_field grows for Gazebo.

make_plants grows every plot's plants from one random generator (agri_ugv_field.plants.grow),
so the same layout, terrain and seed grow the same leaves again here. Seen from above, each
cell of a plot's raster (as in plant_map: 'cell' metres in the plot's frame) gets the highest
leaf point above the terrain, leaving out the textures' transparent parts: the leaves as a
camera shows them. Traits as in plant_map: cover (share of the cells higher than 'low') and
canopy height (95th percentile and median of those cells' heights).

Run from the workspace folder (the defaults are those of make_plants):
    ros2 run agri_ugv_phenotyping canopy_truth --plots 198,197
    ros2 run agri_ugv_phenotyping canopy_truth --map ~/plant_maps/plant_map.npz
    ros2 run agri_ugv_phenotyping canopy_truth --cloud ~/plant_maps/plant_map_cloud_198.ply
With --map it compares a saved plant map with the truth, on the cells the map has seen; with
--cloud a plot's saved 3D point cloud (cloud.PlotCloud) with the true leaves (cloud.compare).
"""

import argparse
import json
import math
from pathlib import Path

from agri_ugv_field.ground import ground_height, read_obj_grid
from agri_ugv_field.layout import read_layout_csv
from agri_ugv_field.plants import CROPS, grow, PARTS, plot_frame
from agri_ugv_phenotyping.cloud import compare, read_ply, to_plot_frame, voxel_means
import numpy as np


def leaf_points(vertices, uvs, faces, alpha=None, step=0.01):
    """
    Return points (N x 3) spread over a mesh's triangles, about 'step' apart.

    alpha (2D boolean image, True = opaque) leaves out the points on transparent texels;
    texture coordinates repeat every 1 (u to the right, v up, as in OBJ files).
    """
    vertices, uvs, faces = (np.asarray(a) for a in (vertices, uvs, faces))
    tris, tex = vertices[faces].astype(float), uvs[faces].astype(float)
    edges = np.linalg.norm(tris[:, [1, 2, 0]] - tris, axis=2).max(axis=1)
    splits = np.clip(np.ceil(edges / step).astype(int), 1, 200)
    out = []
    for n in np.unique(splits):
        a, b = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
        keep = a + b <= n
        weights = np.column_stack([1 - (a[keep] + b[keep]) / n, a[keep] / n, b[keep] / n])
        pick = splits == n
        points = np.einsum('kc,tcd->tkd', weights, tris[pick]).reshape(-1, 3)
        if alpha is not None:
            uv = np.einsum('kc,tcd->tkd', weights, tex[pick]).reshape(-1, 2)
            whole = np.floor(uv)
            uv = np.where((uv > 0) & (uv == whole), 1.0, uv - whole)    # 1 is the far edge
            rows, cols = alpha.shape
            col = np.minimum((uv[:, 0] * cols).astype(int), cols - 1)
            row = np.minimum(((1.0 - uv[:, 1]) * rows).astype(int), rows - 1)
            points = points[alpha[row, col]]
        out.append(points)
    return np.vstack(out) if out else np.zeros((0, 3))


def canopy_top(plot, points, ground, cell=0.05):
    """Return the raster of a plot's highest point per cell above the ground (-inf: none)."""
    centre, along, across = plot_frame(plot)
    shape = (int(math.ceil(plot['length'] / cell - 1e-9)),
             int(math.ceil(plot['width'] / cell - 1e-9)))
    top = np.full(shape, -np.inf)
    d = points[:, :2] - centre
    i = np.floor((d @ along + plot['length'] / 2) / cell).astype(int)
    j = np.floor((d @ across + plot['width'] / 2) / cell).astype(int)
    inside = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
    height = points[inside, 2] - ground(points[inside, 0], points[inside, 1])
    np.maximum.at(top, (i[inside], j[inside]), height)
    return top


def traits(top, low=0.06, cells=None):
    """Return cover, height (95th percentile) and height_median of a raster, on 'cells'."""
    cells = np.ones(top.shape, dtype=bool) if cells is None else cells
    plants = cells & (top > low)
    heights = top[plants]
    return {'cover': float(plants.sum() / max(1, cells.sum())),
            'height': float(np.percentile(heights, 95)) if len(heights) else None,
            'height_median': float(np.median(heights)) if len(heights) else None}


def grown_leaves(plots, ground, ids, seed=1, step=0.01):
    """Yield (plot, its leaf points N x 3 in the world) for the plots 'ids', in field order."""
    from agri_ugv_field.textures import draw_texture, leaf_outlines, opaque   # Pillow

    alpha, left = {}, set(ids)
    for plot, meshes, _ in grow(plots, ground, seed, leaf_outlines(seed)):
        if not left:
            return                       # the generator grows the plots one after another
        if plot['plot_id'] not in left:
            continue
        left.discard(plot['plot_id'])
        points = []
        for part, (vertices, uvs, faces) in meshes.items():
            texture = PARTS[CROPS[plot['crop']]['shape']][part]
            if texture not in alpha:
                alpha[texture] = opaque(draw_texture(texture, seed))
            points.append(leaf_points(vertices, uvs, faces, alpha[texture], step))
        yield plot, np.vstack(points)


def true_canopy(plots, ground, ids, seed=1, cell=0.05, step=0.01):
    """Return {plot_id: highest leaf per cell above the ground} for the plots 'ids'."""
    return {plot['plot_id']: canopy_top(plot, points, ground, cell)
            for plot, points in grown_leaves(plots, ground, ids, seed, step)}


def true_leaves(plots, ground, ids, seed=1, step=0.007, voxel=0.01):
    """
    Return {plot_id: leaf points (N x 3) in the plot's frame} for the plots 'ids'.

    As a PlotCloud holds them: x along the rows, y across, z the height above the ground;
    one point per 'voxel' (the mean of the leaf points in it).
    """
    leaves = {}
    for plot, points in grown_leaves(plots, ground, ids, seed, step):
        local = to_plot_frame(plot, points[:, :2],
                              points[:, 2] - ground(points[:, 0], points[:, 1]))
        leaves[plot['plot_id']] = voxel_means(local, voxel)
    return leaves


def map_traits(saved, plot_id):
    """Return (seen cells, traits) of one plot of a saved plant map (FieldMap.save)."""
    count, plant_count = saved[f'count_{plot_id}'], saved[f'plant_count_{plot_id}']
    seen = count > 0
    heights = saved[f'top_{plot_id}'][plant_count > 0].astype(float)
    return seen, {'cover': float((plant_count > 0).sum() / max(1, seen.sum())),
                  'height': float(np.percentile(heights, 95)) if len(heights) else None,
                  'height_median': float(np.median(heights)) if len(heights) else None,
                  'seen': float(seen.mean())}


def cloud_text(plot, result):
    """Return the comparison of a plot's cloud with its true leaves as text."""
    lines = [f'plot {plot["plot_id"]} ({plot["crop"]}), 3D cloud: {result["plant_points"]} '
             'plant points']
    if result['accuracy_median'] is not None:
        far = result['accuracy_95']
        lines.append('  accuracy: to the true leaves median '
                     f'{100 * result["accuracy_median"]:.1f} cm, 95 % '
                     + (f'{100 * far:.1f} cm' if np.isfinite(far) else 'more than 5 cm')
                     + f'; {100 * result["within_1cm"]:.0f} % within 1 cm')
    if result['complete_2cm'] is not None:
        lines.append(f'  completeness: of the true top surface (where the cloud has points) '
                     f'{100 * result["complete_2cm"]:.0f} % has a point within 2 cm, '
                     f'{100 * result["complete_5cm"]:.0f} % within 5 cm')
    return '\n'.join(lines)


def text(t):
    """Return traits as 'cover 43 %, height 32.4 cm (median 23.1 cm)'."""
    line = f'cover {100 * t["cover"]:3.0f} %'
    if t['height'] is not None:
        line += f', height {100 * t["height"]:4.1f} cm (median {100 * t["height_median"]:4.1f} cm)'
    return line


def main(argv=None):
    """Print the true canopy of plots, or compare a saved plant map with it."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--layout', default='src/agri_ugv_field/data/must_c_field_plots.csv',
                        help='plot layout CSV (as for make_plants)')
    parser.add_argument('--terrain-model', default='src/agri_ugv_gazebo/models/must_c_field',
                        help='terrain model folder the plants stand on')
    parser.add_argument('--seed', type=int, default=1, help='make_plants seed (default 1)')
    parser.add_argument('--plots', default='', help='plot_IDs separated by commas')
    parser.add_argument('--map', help='a saved plant map (.npz) to compare with the truth')
    parser.add_argument('--cloud', help='a saved 3D cloud of one plot (.ply) to compare')
    parser.add_argument('--cell', type=float, default=0.05, help='cell size [m] (default 0.05)')
    parser.add_argument('--low', type=float, default=0.06, help='plant height [m] (default 0.06)')
    args = parser.parse_args(argv)
    plots = [p for p in read_layout_csv(Path(args.layout).read_text()) if p['type'] == 'plot']
    terrain = Path(args.terrain_model)
    grid = read_obj_grid((terrain / 'meshes' / f'{terrain.name}.obj').read_text())

    def ground(x, y):
        return ground_height(grid, x, y)

    if args.cloud:
        points, _, comments = read_ply(Path(args.cloud).expanduser())
        plot_id = int(comments[0].split()[1])          # 'plot <plot_id> (<crop>), ...'
        [plot] = [p for p in plots if p['plot_id'] == plot_id]
        leaves = true_leaves(plots, ground, {plot_id}, args.seed)[plot_id]
        print(cloud_text(plot, compare(points, leaves, args.low)))
        return
    saved = np.load(Path(args.map).expanduser()) if args.map else None
    if saved is not None:
        ids = [p['plot_id'] for p in json.loads(str(saved['plots']))]
        cell, low = float(saved['cell']), float(saved['low'])
    else:
        ids = [int(p) for p in args.plots.split(',') if p.strip()] or \
            [p['plot_id'] for p in plots]
        cell, low = args.cell, args.low
    tops = true_canopy(plots, ground, set(ids), args.seed, cell)
    crop = {p['plot_id']: p['crop'] for p in plots}
    for plot_id in ids:
        if plot_id not in tops:
            print(f'plot {plot_id}: not in the layout')
            continue
        if saved is None:
            print(f'plot {plot_id} ({crop[plot_id]}): {text(traits(tops[plot_id], low))}')
            continue
        seen, measured = map_traits(saved, plot_id)
        truth = traits(tops[plot_id], low, seen)
        line = (f'plot {plot_id} ({crop[plot_id]}), {100 * measured["seen"]:.0f} % seen:\n'
                f'  truth {text(truth)}\n  map   {text(measured)}\n  map minus truth: cover '
                f'{100 * (measured["cover"] - truth["cover"]):+.0f} points')
        if measured['height'] is not None and truth['height'] is not None:
            line += (f', height {100 * (measured["height"] - truth["height"]):+.1f} cm '
                     f'(median {100 * (measured["height_median"] - truth["height_median"]):+.1f}'
                     ' cm)')
        print(line)
