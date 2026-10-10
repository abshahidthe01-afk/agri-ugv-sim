"""Command-line tool: grow crop plants in the plots of a layout CSV, as a Gazebo model."""

import argparse
from pathlib import Path

from agri_ugv_field.ground import ground_height, read_obj_grid
from agri_ugv_field.layout import read_layout_csv
from agri_ugv_field.plants import BANDS, CROPS, grow, leaf_outline, obj_text, PARTS, \
    vertex_normals
from agri_ugv_field.textures import draw_texture, opaque
from agri_ugv_terrain.heightfield import model_config


def plants_sdf(name, meshes):
    """
    Write a static model with one visual per mesh and no collision.

    meshes maps a mesh name to its texture name. Each material shows the texture's picture
    and cuts away its transparent parts. Every leaf has a face for each side (two_sided), so
    each face is drawn from its front only. Plants are visual only: the robot's physics
    ignores them (real plants are soft), cameras and LiDARs see them. One mesh per plot: a
    sensor draws only the plots in its view.
    """
    visuals = ''.join(f"""
      <visual name="{mesh}">
        <cast_shadows>false</cast_shadows>
        <geometry>
          <mesh><uri>meshes/{mesh}.obj</uri></mesh>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <double_sided>false</double_sided>
          <pbr>
            <metal>
              <albedo_map>textures/{texture}.png</albedo_map>
              <roughness>0.8</roughness>
              <metalness>0</metalness>
            </metal>
          </pbr>
        </material>
      </visual>""" for mesh, texture in meshes.items())
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="plants">{visuals}
    </link>
  </model>
</sdf>
"""


def main(argv=None):
    """Read plots and terrain, place the plants, write <output-dir>/<name>/ as a model."""
    parser = argparse.ArgumentParser(
        description='Make a Gazebo model of crop plants. The defaults grow the MuST-C field '
        'when run from the workspace folder.')
    parser.add_argument('--layout', default='src/agri_ugv_field/data/must_c_field_plots.csv',
                        help='plot layout CSV (make_field_layout)')
    parser.add_argument('--terrain-model', default='src/agri_ugv_gazebo/models/must_c_field',
                        help='terrain model folder; plants stand on its mesh')
    parser.add_argument('--name', default='must_c_plants', help='model name (letters, digits, _)')
    parser.add_argument('--output-dir', default='src/agri_ugv_gazebo/models',
                        help='folder to write the model into')
    parser.add_argument('--seed', type=int, default=1, help='random seed (same seed, same field)')
    args = parser.parse_args(argv)

    plots = [r for r in read_layout_csv(Path(args.layout).read_text()) if r['type'] == 'plot']
    terrain = Path(args.terrain_model)
    grid = read_obj_grid((terrain / 'meshes' / f'{terrain.name}.obj').read_text())

    def ground(x, y):
        return ground_height(grid, x, y)

    unknown = sorted({p['crop'] for p in plots} - set(CROPS))
    if unknown:
        bad = next(p for p in plots if p['crop'] == unknown[0])
        raise ValueError(f'plot {bad["plot_id"]}: no parameters for crop {bad["crop"]!r}')
    used = {texture for p in plots for texture in PARTS[CROPS[p['crop']]['shape']].values()}
    pictures = {texture: draw_texture(texture, args.seed) for texture in sorted(used)}
    outlines = {texture: leaf_outline(opaque(picture), BANDS)
                for texture, picture in pictures.items()
                if any(parts.get('') == texture for parts in PARTS.values())}
    model_dir = Path(args.output_dir) / args.name
    for folder in ('meshes', 'textures'):
        (model_dir / folder).mkdir(parents=True, exist_ok=True)
    for old in (model_dir / 'meshes').glob('*.obj'):     # meshes of an earlier run
        old.unlink()
    for texture, picture in pictures.items():
        picture.save(model_dir / 'textures' / f'{texture}.png')
    print(f'{model_dir}: {len(plots)} plots, seed {args.seed}')
    textures, stats = {}, {}
    for plot, meshes, count in grow(plots, ground, args.seed, outlines):
        crop = CROPS[plot['crop']]
        plots_done, items, triangles, size = stats.get(crop['key'], (0, 0, 0, 0))
        for part, (vertices, uvs, faces) in meshes.items():
            mesh = f'{crop["key"]}{part}_{plot["plot_id"]}'
            path = model_dir / 'meshes' / f'{mesh}.obj'
            path.write_text(obj_text(vertices, uvs, faces, vertex_normals(vertices, faces)))
            textures[mesh] = PARTS[crop['shape']][part]
            triangles, size = triangles + len(faces), size + path.stat().st_size
        stats[crop['key']] = (plots_done + 1, items + count, triangles, size)
    for crop_key, (plots_done, items, triangles, size) in sorted(stats.items()):
        unit = 'row strips' if crop_key in ('wheat', 'mixtures') else 'plants'
        print(f'  {crop_key:<10} {plots_done:2d} plots, {items:5d} {unit}, {triangles:8d} '
              f'triangles, {size / 1e6:6.1f} MB')
    (model_dir / 'model.sdf').write_text(plants_sdf(args.name, dict(sorted(textures.items()))))
    (model_dir / 'model.config').write_text(model_config(
        args.name, f'crop plants for the plots of {Path(args.layout).name} on the '
        f'{terrain.name} terrain, generated by agri_ugv_field (seed {args.seed}); '
        'row spacings and plant sizes are assumed typical values for late June'))
