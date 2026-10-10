# Plant map

The two laser line scanners on the side panels (see the [robot spec](robot_spec.md)) measure
the plants while the robot drives its coverage passes. The `plant_map` node
(`agri_ugv_phenotyping`) places every scanner profile in the field with the robot's pose at
the profile's time and fills one raster per plot, 5 cm cells in the plot's own frame (along
the rows, across them). A point's height is its height above the plane the wheels stand on.
Per plot it reports:

- **seen**: share of the plot's cells with scanner points
- **cover**: share of the seen cells with plant points (more than 6 cm high)
- **canopy height**: 95th percentile (and median) of the plant cells' heights
- **rows off the field map**: where the rows lie in the map, across the plot, compared with
  where the field map puts them; and how sharply the plants line up on rows (0 to 1). A
  wrong pose moves the rows in the map, and passes that disagree blur them.

```bash
ros2 launch agri_ugv_phenotyping survey.launch.py plots:=198
ros2 service call /mission/start std_srvs/srv/Trigger
# when the mission is complete: ~/plant_maps/plant_map.npz and .json
ros2 run agri_ugv_phenotyping canopy_truth --map ~/plant_maps/plant_map.npz
```

`canopy_truth` grows the plots' plants again exactly as `make_plants` grew them for Gazebo
(same generator and seed) and compares the map with the true canopy: the highest drawn leaf
per cell, seen from above, on the cells the map has seen.

## Map and true canopy

RTK fixed GNSS, plots 198, 197, 188 and 187 (12726 profiles, about 630 s simulated), map placed
with Gazebo's true pose:

| Plot | Crop | Seen | Cover: truth / map | Canopy height: truth / map |
|---|---|---|---|---|
| 198 | Sugar beet | 73 % | 90 % / 96 % | 32.2 / 33.9 cm |
| 197 | Summer wheat | 63 % | 97 % / 97 % | 70.0 / 75.7 cm |
| 188 | Soybean | 76 % | 59 % / 57 % | 31.8 / 33.7 cm |
| 187 | Potato | 72 % | 81 % / 85 % | 42.3 / 43.3 cm |

Cover is within -1 to +6 points of the truth, canopy height within +1.0 to +5.7 cm. The map
placed with the robot's own pose estimate (0.9 cm mean error across the rows) is the same
within 3 points of cover and 0.1 cm of height. The scanners see the plants at a slant, so
part of each plot stays hidden behind leaves (seen 63-76 %).

The heights read high partly because the ground under the scanned strip lies up to 1.5 cm
above the plane of the wheels (median per plot and scanner -0.1 to +1.5 cm, measured in the
profiles). Wheat reads highest: its rows are picture strips whose top edge stands above the
drawn ears, and the scanners see the whole strip (below).

### Why the leaves are shaped in the mesh

Gazebo's GPU lidar ignores the textures' transparency and the back sides of faces; the
cameras show neither. With leaves as plain textured cards, a ray cast of the two scanners
against the same meshes and terrain, at Gazebo's recorded true poses (550 profiles), matched
Gazebo only when it did the same:

| Plant points per profile, leaf cards | Sugar beet | Summer wheat | Soybean | Potato |
|---|---|---|---|---|
| Gazebo | 42 | 293 | 77 | 43 |
| Ray cast: whole cards, front side only | 43 | 293 | 79 | 43 |
| Ray cast: drawn leaves, both sides (as the cameras show them) | 154 | 256 | 94 | 105 |

The map then had cover 22 points too low in sugar beet and 13 in potato. So each leaf is
now four quads, each as wide as the drawn leaf in its part of the texture, and every face
is in the mesh twice, once facing each way (2.34 million plant triangles in the field, 0.65
million before; one mesh per plot, so a sensor draws only the plots in its view). Wheat's
row strips and canopy are still whole pictures.

## 3D point cloud

The raster keeps the highest point per 5 cm cell. The node also keeps every scanner point in
3D: per plot, in its own frame (x along the rows, y across them, z the height above the plane
of the wheels), merged per 1 cm voxel (`agri_ugv_phenotyping/cloud.py`, parameter `voxel`),
about 140 000 points per plot, saved with the map as `plant_map_cloud_<plot_id>.ply` (binary
PLY, which point cloud viewers open).

```bash
ros2 run agri_ugv_phenotyping canopy_truth --cloud ~/plant_maps/plant_map_cloud_198.ply
```

compares a cloud with the plot's true leaves (grown again, without the textures' transparent
parts, one point per 1 cm voxel): **accuracy**, how far the cloud's plant points lie from the
nearest leaf; **completeness**, how much of the leaves' top surface (seen from above, where
the cloud has points) has a point within 2 cm and within 5 cm.

Plot 198 (sugar beet), RTK fixed GNSS:

| Cloud placed with | Points to the true leaves: median / 95 % | Within 1 cm | Top surface within 2 / 5 cm |
|---|---|---|---|
| Gazebo's true pose | 0.9 / 3.5 cm | 52 % | 51 / 98 % |
| the robot's estimate, earlier localization | 1.4 / 4.3 cm | 37 % | 41 / 97 % |
| the robot's estimate, current localization | 1.0 / 3.6 cm | 48 % | 52 / 98 % |

The first two rows are the same profiles of one survey placed with two poses (the estimate
was 2.0 cm off along the rows and 0.9 cm across them on average); the third is a later
survey, after the localization learnt the row measurements' bias per lane and the slides
while the wheels steer on the spot (1.2 cm mean error in a run of the same plot). Placing
the profiles with the estimate adds its error to the scan: with the earlier localization
the median grew from 0.9 to 1.4 cm, with the current one only to 1.0 cm. Within 2 cm only
about half of the top surface has a point: at 10 profiles per second and 0.5 m/s the
profiles lie 5 cm apart along the rows (the real scanners take 200 per second).

## Localization and the map

Float GNSS (the base station's corrections stopped: shared error drifting towards 20 cm),
plot 198, 102.6 s mission, 2249 profiles, with the earlier leaf cards. The same profiles,
placed with four poses, recorded at the same time (the robot drove with its own estimate):

| Pose | Error across the rows (mean) | Rows off the field map | Row sharpness |
|---|---|---|---|
| Gazebo's true pose | - | +0.2 cm | 0.55 |
| RTK fixed, no row corrections | 0.8 cm | +0.2 cm | 0.55 |
| Float, with row corrections (the robot's own) | 0.8 cm | -0.0 cm | 0.55 |
| Float, no row corrections | 6.8 cm (-17.2 to +5.2 cm) | -5.1 cm | 0.26 |

Without the row corrections the map puts the rows 5.1 cm off (the pose was 5.3 cm off on
average, to the same side) and the passes disagree by up to 22 cm, which halves the rows'
sharpness. With them the map is as good as with the true pose. Along the rows the float
error stays (10.1 cm mean with the row corrections, 9.5 cm without): rows do not show where
along them the robot is. Cover and height changed by at most 1 point and 0.1 cm: they do
not show a wrong pose, the rows do.

With the shaped leaves the LiDAR sees both sides of the leaves, and the row measurement's
near-side shift comes out negative (about -3 cm; before, it could not go below 0). With that
allowed, the robot's pose is 0.9 cm off across the rows on average with RTK fixed GNSS (plots
198, 197, 188 and 187; on plot 198, 1.4 cm on the two passes along the plot's sides).

Limits: one float run over one plot; 10 profiles per second per scanner (the real scanners
take 200; more overloaded the simulation).
