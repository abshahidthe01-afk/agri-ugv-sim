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

Next: GNSS/IMU localization, field layout, coverage planning and LiDAR row following.

## Quick start

```bash
git clone https://github.com/abshahidthe01-afk/agri-ugv-sim.git agri_ugv_ws
cd agri_ugv_ws
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash
ros2 launch agri_ugv_gazebo sim.launch.py world:=must_c_field
```

Worlds: `flat` (default), `flat_mesh`, `waves`, `must_c_field`.

Drive it from a second terminal:

```bash
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Data and sources

The `must_c_field` terrain is derived from the MuST-C drone elevation map (DEM) of
17 May 2023: resampled to a 0.45 m grid using the 10th height percentile per cell,
gaps outside the field filled, heights relative to the field centre.

- Dataset: Chong, Yue Linn, 2025, "MuST-C Dataset: The Multi-Sensor and Multi-Temporal
  Data Set of Multiple Crops for In-Field Phenotyping and Monitoring",
  https://doi.org/10.60507/FK2/OX9XTM, bonndata, V3. License: CC BY 4.0.
- Paper: Chong et al., Scientific Data, 2026, https://doi.org/10.1038/s41597-025-06462-y

Robot dimensions follow the Thorvald II (Grimstad & From, 2017) and the Bonn field
phenotyping robot (Esser et al., 2023).

## License

Code: MIT, see [LICENSE](LICENSE). The terrain files derived from MuST-C are shared
under CC BY 4.0, like their source.
