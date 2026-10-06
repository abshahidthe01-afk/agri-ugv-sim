# Agricultural UGV simulation (work in progress)

[![CI](https://github.com/abshahidthe01-afk/agri-ugv-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/abshahidthe01-afk/agri-ugv-sim/actions/workflows/ci.yml)

A four-wheel-steering field robot in simulation, built to drive between crop
rows on real terrain. ROS 2 Humble and Gazebo Fortress.

## Status

- Thorvald-style robot model with suspension ([robot spec](docs/robot_spec.md))
- Four-wheel-steering kinematics, unit-tested
- Gazebo simulation with ros2_control, speed limits and a watchdog
- Ground-truth pose and velocity from Gazebo
- Terrain generator: test terrains and the real MuST-C field from drone elevation data
- Field: the 80 plots of the real MuST-C trial from its official shapefile, with
  generated crop plants (textured leaves, about 22,500 plants)

Next: GNSS/IMU localization, coverage planning and LiDAR row following.

## Quick start

```bash
git clone https://github.com/abshahidthe01-afk/agri-ugv-sim.git agri_ugv_ws
cd agri_ugv_ws
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash
ros2 run agri_ugv_field make_plants   # grows the crop plants (about 20 s, 90 MB)
colcon build --symlink-install --packages-select agri_ugv_gazebo
ros2 launch agri_ugv_gazebo sim.launch.py world:=must_c_field
```

Worlds: `flat` (default), `flat_mesh`, `waves`, `must_c_field`.

Drive it from a second terminal:

```bash
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Data and sources

The `must_c_field` terrain is derived from the MuST-C drone elevation map (DEM) and
aerial photo (orthophoto) of 17 May 2023: heights resampled to a 0.45 m grid using the
10th height percentile per cell, gaps outside the field filled, heights relative to the
field centre; the photo is used as the terrain's texture at 6 cm per pixel.
The plot layout (`src/agri_ugv_field/data/must_c_field_plots.csv`) comes from the
dataset's field shapefile (`md_FieldSHP`), placed in the same world frame. The crop
plants in the plots are generated: row spacings and plant sizes are assumed typical
values, not measured; leaf pictures and plant shapes are drawn by the generator.

- Dataset: Chong, Yue Linn, 2025, "MuST-C Dataset: The Multi-Sensor and Multi-Temporal
  Data Set of Multiple Crops for In-Field Phenotyping and Monitoring",
  https://doi.org/10.60507/FK2/OX9XTM, bonndata, V3. License: CC BY 4.0.
- Paper: Chong et al., Scientific Data, 2026, https://doi.org/10.1038/s41597-025-06462-y

Robot dimensions follow the Thorvald II (Grimstad & From, 2017) and the Bonn field
phenotyping robot (Esser et al., 2023).

## License

Code: MIT, see [LICENSE](LICENSE). The terrain and field layout files derived from
MuST-C are shared under CC BY 4.0, like their source.
