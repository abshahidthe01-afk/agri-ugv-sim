# Agricultural UGV simulation (work in progress)

[![CI](https://github.com/abshahidthe01-afk/agri-ugv-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/abshahidthe01-afk/agri-ugv-sim/actions/workflows/ci.yml)

A four-wheel-steering field robot in simulation, built to drive between crop
rows on real terrain. ROS 2 Humble and Gazebo Fortress.

## Status

- Thorvald-style robot model with suspension ([robot spec](docs/robot_spec.md))
- Four-wheel-steering kinematics, unit-tested
- Gazebo simulation with ros2_control, speed limits and a watchdog
- Ground-truth pose and velocity from Gazebo

Next: terrain, GNSS/IMU localization, field layout, coverage planning and
LiDAR row following.

## Quick start

```bash
git clone https://github.com/abshahidthe01-afk/agri-ugv-sim.git agri_ugv_ws
cd agri_ugv_ws
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash
ros2 launch agri_ugv_gazebo sim.launch.py
```

Drive it from a second terminal:

```bash
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Data and sources

Terrain and field layout will be derived from the MuST-C dataset (PhenoRob,
University of Bonn). Robot dimensions follow the Thorvald II (Grimstad & From,
2017) and the Bonn field phenotyping robot (Esser et al., 2023).

## License

MIT, see [LICENSE](LICENSE).
