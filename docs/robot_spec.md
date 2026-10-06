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
| IMU (accelerometer, gyroscope) | `imu_link`, on the body 0.90 m above the ground, near the centre of mass (0.894 m) | `/imu`, 100 Hz, no noise yet |

Gazebo uses the SDF default gravity of 9.8 m/s² (the suspension above was sized with 9.81; 0.1 % apart).

## References

1. L. Grimstad and P. J. From, "The Thorvald II Agricultural Robotic System",
   Robotics 6(4), 24, 2017.
2. F. Esser et al., "Field Robot for High-throughput and High-resolution 3D
   Plant Phenotyping", arXiv:2310.11516, 2023.
3. DEVELOP3D, "Saga Robotics: Fields of the future".
