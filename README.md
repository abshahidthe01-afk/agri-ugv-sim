# Agricultural UGV simulation (work in progress)

[![CI](https://github.com/abshahidthe01-afk/agri-ugv-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/abshahidthe01-afk/agri-ugv-sim/actions/workflows/ci.yml)

A four-wheel-steering ground robot that navigates a real crop field's
terrain in simulation, driving between the crop rows instead of over them.

Built with ROS 2 Humble and Gazebo Fortress.

## What works so far

- **Robot model** (`agri_ugv_description`): a Thorvald-style robot with four
  steering / suspension / wheel modules. Dimensions come from published
  sources, and every assumption is labelled ([robot spec](docs/robot_spec.md)).
- **Four-wheel-steering kinematics** (`agri_ugv_control`): one formula for
  straight driving, spinning in place, crab (sideways) motion and arcs. The
  wheel geometry is read from the robot model, never typed into the code.
  Unit-tested, including 1000 random motions.
- **Kinematic simulation**: drive the ideal robot in RViz from the keyboard,
  with a watchdog that stops the robot when commands stop.

## Quick start

```bash
git clone https://github.com/abshahidthe01-afk/agri-ugv-sim.git agri_ugv_ws
cd agri_ugv_ws
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash
ros2 launch agri_ugv_control kinematic_sim.launch.py
```

In a second terminal, drive with the keyboard:

```bash
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Run the tests:

```bash
colcon test && colcon test-result --verbose
```

## Data

Terrain and field layout are derived from the MuST-C dataset
(PhenoRob, University of Bonn).

## License

MIT, see [LICENSE](LICENSE).
