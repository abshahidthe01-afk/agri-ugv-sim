# Robot specification

The robot is modelled on the Thorvald II-based phenotyping robot used at the
University of Bonn [1, 2]. Every value is marked as published, estimated from
photos, or an engineering assumption.

| Property | Value | Basis |
|---|---|---|
| Track (wheel centres, left-right) | 1.50 m | Published [1] |
| Wheelbase (steering axes, front-rear) | 1.35 m | Published, standard Thorvald II [1] |
| Sensor enclosure footprint | 1.50 x 1.50 m | Published [2] |
| Enclosure vertical extent | 0.45 m to 1.85 m above ground | Estimated from photo |
| Overall height | ~2.0 m | Published [2] |
| Wheel | diameter 0.41 m, width 0.165 m (16x6.50-8 tyre) | Estimated from photo |
| Suspension | passive, 0.10 m travel | Type published [1]; travel assumed |
| Base platform mass | 180 kg | Assumed; published as under 200 kg [3] |
| Enclosure, sensors, computers | 100 kg | Assumed |
| Total mass | 280 kg | Sum |
| Line scanners | 1.2 m high, 1.4 m apart, pointing down | Published [2] |
| GNSS antennas | front and rear on the roof (dual-antenna heading) | Published [2]; positions from photo |
| IMU | under the roof, rear | Published [2] |

## Mass distribution (assumed)

| Part | Mass | Centre-of-mass height |
|---|---|---|
| 4 wheel modules (motor 4 kg, leg 6 kg, wheel 10 kg) | 4 x 20 kg | 0.38 m |
| Frame and batteries | 100 kg | 0.80 m |
| Enclosure and equipment | 100 kg | 1.40 m |

Resulting centre of mass: 0.894 m above ground (computed from the robot model, checked by a unit test).
Static tip-over angle: about 40 deg sideways, about 37 deg forward/backward.

## Suspension (derived from the masses)

Each spring carries a quarter of the sprung mass (frame, enclosure and steering
motors: 216 kg, so 54 kg per wheel) and is sized so the robot rests at
mid-travel: stiffness 10.6 kN/m, damping 605 N s/m (damping ratio 0.4, assumed).

In Gazebo the robot rests about 2.5 mm below mid-travel. The physics engine
(DART) applies joint damping before it resolves ground contacts, so the dampers
of the standing robot push with a small phantom force (about 26 N per wheel with
1 ms time steps). Measured: the offset doubles with twice the damping and halves
with half the time step.

## Soil profiles (assumed)

Gazebo is rigid, so soil is approximated by wheel friction and wheel slip (Gazebo's
WheelSlip system), chosen with the launch argument `soil:=`; `rigid` (default) keeps
Gazebo's default contact without a slip model. Slip compliance is the wheel slip ratio per
unit of tangential/normal force; the static wheel load (686.7 N) comes from the masses.
The values are order-of-magnitude assumptions from typical tyre-on-soil traction curves,
not measurements. Sinkage and rolling resistance are not modelled.

| Profile | Friction coefficient | Slip compliance |
|---|---|---|
| firm | 0.65 | 0.2 |
| soft | 0.45 | 0.55 |
| wet | 0.3 | 1.0 |

## Sensors (simulated)

| Sensor | Where | Output |
|---|---|---|
| IMU (accelerometer, gyroscope) | `imu_link`, under the roof at the back (0.60 m behind the centre, 1.79 m above the ground), like the real robot's inertial unit [2] | `/imu`, 100 Hz, with noise and biases |
| 2 GNSS antennas (NavSat) | `gnss_front_link` / `gnss_rear_link`, on the roof 1.90 m above the ground, 1.20 m apart front to rear | `/gnss/front/fix`, `/gnss/rear/fix` (`fix_ideal` without errors), `/gnss/heading`, 10 Hz |
| 3D LiDAR, like an Ouster OS0-64 (an addition: the robot of [2] is driven by hand) | `lidar_link`, on a mast at the centre of the roof, 2.70 m above the ground: its lowest beams (45° down) pass over the roof edges ahead, behind and to the sides | `/lidar/points`, 10 Hz, 1024 x 64 points, ±45° vertical, 0.3-50 m, range noise 1 cm (assumed); `lidar:=false` leaves it out |

Gazebo uses the SDF default gravity of 9.8 m/s² (the suspension above was sized with 9.81; 0.1 % apart).

IMU errors per axis (assumed, typical industrial MEMS): gyroscope white noise 0.0005 rad/s per sample, start-up bias 0.0003 rad/s, drifting bias 2e-5 rad/s (4°/h, correlation time 300 s); accelerometer 0.005 m/s², 0.01 m/s², 2e-4 m/s². Gazebo's `dynamic_bias_stddev` is a density: long-run spread = value × sqrt(τ/2). The IMU's orientation output stays perfect in Gazebo and is not used for localization.

GNSS errors (assumed, typical RTK fixed): a slowly drifting error shared by both antennas (1.0 cm horizontal, 2.0 cm vertical, correlation time 60 s) plus independent noise per antenna (0.3 cm, 0.6 cm). The shared part cancels in the heading: 0.20° for the 1.20 m baseline (agri_ugv_localization).

Worse GNSS on demand (parameter `quality` of `gnss_errors`, can be changed while running): `float` = the base station's corrections stop arriving; the shared error drifts from where it is towards 20 cm horizontal, 40 cm vertical (assumed, same 60 s correlation time). The two antennas still measure against each other, so the heading keeps its 0.20°. `none` = no fix: the fixes carry status `STATUS_NO_FIX` (position NaN) and there is no heading. Fixed again, or a fix after `none`, starts a new solution. Each fix reports its accuracy in its covariance; the localization follows it. After every change of quality the random errors start again from the seed, so two test runs that switch at the same moment meet the same errors.

In `must_c_field`, `heading_deg` = 1.558: world x/y follow the UTM grid, which here is turned 1.558° from true east/north (meridian convergence). With it, the simulated GNSS reports each terrain point's real coordinates (within 2 cm over the field).

## Phenotyping payload (visual only)

After [2]; no extra mass (it is part of the enclosure's 100 kg) and no collision.

| Part | From the paper | Assumed here |
|---|---|---|
| Camera dome | 20 cameras around the robot's centre, aimed at the plant | 3 rings (8 at 0.95 m, 8 at 1.40 m, 4 at 1.72 m) inside the enclosure, aimed at (0, 0, 0.30) |
| 2 laser line scanners | on the side panels, 1.2 m high, 1.4 m apart, tilted about 50° | 1.34 m apart (inside our 1.5 m box), 50° from vertical, looking across |
| LED panels | in the enclosure | 4 panels under the roof |
| Curtains | motorized, front and rear openings | rolled up at the top |
| 2 computers | on board | on the roof |

## References

1. L. Grimstad and P. J. From, "The Thorvald II Agricultural Robotic System",
   Robotics 6(4), 24, 2017.
2. F. Esser et al., "Field Robot for High-throughput and High-resolution 3D
   Plant Phenotyping", arXiv:2310.11516, 2023.
3. DEVELOP3D, "Saga Robotics: Fields of the future".
