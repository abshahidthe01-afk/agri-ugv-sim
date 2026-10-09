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

RTK fixed GNSS, plots 198, 197, 188 and 187 (14678 profiles, about 730 s simulated), map placed
with Gazebo's true pose:

| Plot | Crop | Seen | Cover: truth / map | Canopy height: truth / map |
|---|---|---|---|---|
| 198 | Sugar beet | 91 % | 66 % / 43 % | 31.9 / 32.4 cm |
| 197 | Summer wheat | 83 % | 98 % / 98 % | 70.1 / 75.1 cm |
| 188 | Soybean | 94 % | 47 % / 53 % | 31.7 / 35.0 cm |
| 187 | Potato | 87 % | 65 % / 52 % | 42.2 / 40.7 cm |

The map placed with the robot's own pose estimate is the same within 1 point of cover and
0.1 cm of height.

**Cover is off because of how Gazebo's GPU lidar sees the leaves.** The leaves are textured
cards whose transparent parts the cameras cut away, drawn from both sides. A ray cast of the
two scanners against the same plant meshes and terrain, at Gazebo's recorded true poses
(550 profiles), shows what the lidar does:

| Plant points per profile | Sugar beet | Summer wheat | Soybean | Potato |
|---|---|---|---|---|
| Gazebo | 42 | 293 | 77 | 43 |
| Ray cast: whole leaf cards, front side only | 43 | 293 | 79 | 43 |
| Ray cast: drawn leaves, both sides (as the cameras show them) | 154 | 256 | 94 | 105 |

The lidar ignores the textures' transparency and the back sides of the leaves. Real
scanners see both sides of real leaves, so the cover measured here is a property of the
simulation, not of the scanners.

The heights are within -1.5 to +4.9 cm of the truth. The leaf cards stand a little above the
drawn leaves, and the ground under the scanned strip lies up to 1.5 cm above the plane of
the wheels (median per plot and scanner -0.1 to +1.5 cm, measured in the profiles); both
raise the heights.

## Localization and the map

Float GNSS (the base station's corrections stopped: shared error drifting towards 20 cm),
plot 198, 102.6 s mission, 2249 profiles. The same profiles, placed with four poses, recorded
at the same time (the robot drove with its own estimate):

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

Limits: one float run over one plot; 10 profiles per second per scanner (the real scanners
take 200; more overloaded the simulation).
