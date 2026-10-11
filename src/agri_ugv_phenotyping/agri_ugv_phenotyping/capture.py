"""
Take one set of pictures with depth from all dome cameras while the robot stands in a plot.

For 3D reconstruction (reconstruct.py), like the real robot: it stops and all its cameras
take a picture at the same moment. Start the mission with all dome cameras measuring depth
(one picture per second is enough; the line scanners are not needed):
    ros2 launch agri_ugv_navigation mission.launch.py plots:=198 cameras:=all
        camera_depth:=true camera_rate:=1 scanners:=false
then, before /mission/start:
    ros2 run agri_ugv_phenotyping capture_depth [--stop-after 3.5] [--output ~/plant_maps/depth]
It waits until the robot has driven --stop-after metres (into the plot), stops the mission
(/mission/stop), waits until the robot stands still, and keeps the first set of pictures
that every camera with depth took at the same moment afterwards: the colour pictures, the
depth images (metres along the view, 32FC1), the calibrations, the cameras' places on the
robot (their optical links in base_footprint, from /robot_description), and the robot's true
(/ground_truth/odom) and estimated pose at that time. It prints per camera the share of
pixels with a depth, the nearest, median and farthest depth and the picture's brightness,
and saves everything as <output>.npz.
"""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

CAMERAS = range(20)
STILL = 0.01            # [m/s] the robot stands still below this speed (true velocity)


def stamp(msg):
    """Return a message's time stamp in seconds."""
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


def picture(msg):
    """Return a colour picture as (height, width, 3) RGB."""
    rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    image = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    return image[:, :, ::-1].copy() if msg.encoding.lower().startswith('bgr') else image.copy()


def depth_image(msg):
    """Return a depth image (32FC1) as (height, width) float32 metres."""
    rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    return rows[:, :msg.width * 4].copy().view('<f4').reshape(msg.height, msg.width)


def pose_of(msg):
    """Return (x, y, z, roll, pitch, yaw) of an Odometry message."""
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x ** 2 + q.y ** 2))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
    return (p.x, p.y, p.z, roll, pitch, yaw)


def capture(seconds, stop_after, still_time):
    """Follow the robot and the cameras; return the captured set (dict) or None."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    state = {'start': None, 'stopped': False, 'still_since': None, 'description': None,
             'truth': [], 'estimate': [], 'clock': []}
    pictures = {i: {} for i in CAMERAS}       # camera: {stamp: picture}
    depths = {i: {} for i in CAMERAS}         # camera: {stamp: depth image}
    calibration = {}

    def keep(store, i, t, value):
        store[i][t] = value
        for old in sorted(store[i])[:-3]:      # the latest 3 per camera
            del store[i][old]

    rclpy.init()
    node = Node('capture_depth')
    stop = node.create_client(Trigger, '/mission/stop')
    for i in CAMERAS:
        node.create_subscription(Image, f'/cameras/dome_{i}/image',
                                 lambda m, i=i: keep(pictures, i, stamp(m), picture(m)), 2)
        node.create_subscription(Image, f'/cameras/dome_{i}/depth_image',
                                 lambda m, i=i: keep(depths, i, stamp(m), depth_image(m)), 2)
        node.create_subscription(CameraInfo, f'/cameras/dome_{i}/camera_info',
                                 lambda m, i=i: calibration.__setitem__(i, list(m.k)), 2)

    def on_truth(msg):
        pose, t = pose_of(msg), stamp(msg)
        state['truth'] = (state['truth'] + [(t, *pose)])[-400:]
        if state['start'] is None:
            state['start'] = pose[:2]
        moved = math.hypot(pose[0] - state['start'][0], pose[1] - state['start'][1])
        if not state['stopped'] and moved >= stop_after:
            state['stopped'] = True
            stop.call_async(Trigger.Request())
            print(f'Moved {moved:.2f} m: mission stopped, waiting until the robot stands still')
        speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        if state['stopped'] and speed < STILL:
            state['still_since'] = state['still_since'] if state['still_since'] else t
        else:
            state['still_since'] = None

    def on_estimate(msg):
        state['estimate'] = (state['estimate'] + [(stamp(msg), *pose_of(msg))])[-400:]

    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String, '/robot_description',
                             lambda m: state.__setitem__('description', m.data), latched)
    node.create_subscription(Odometry, '/ground_truth/odom', on_truth, 50)
    node.create_subscription(Odometry, '/localization/odometry', on_estimate, 50)
    node.create_subscription(
        Clock, '/clock', lambda m: state['clock'].append(
            (time.monotonic(), m.clock.sec + 1e-9 * m.clock.nanosec)), 10)
    print(f'Waiting for the robot to drive {stop_after:.1f} m (start the mission) ...')
    begin, found = time.monotonic(), None
    while time.monotonic() - begin < seconds and found is None:
        rclpy.spin_once(node, timeout_sec=0.05)
        still = state['still_since']
        if still is None or not state['truth'] or state['truth'][-1][0] - still < still_time:
            continue
        with_depth = [i for i in CAMERAS if depths[i]]
        common = set.intersection(*[set(pictures[i]) & set(depths[i]) for i in with_depth]) \
            if with_depth else set()
        common = sorted(t for t in common if t >= still)
        if common and all(i in calibration for i in with_depth):
            found = (common[0], with_depth)
    description, clock = state['description'], state['clock']
    node.destroy_node()
    rclpy.shutdown()
    if found is None:
        return None
    t, cameras = found
    truth = min(state['truth'], key=lambda r: abs(r[0] - t))
    estimate = min(state['estimate'], key=lambda r: abs(r[0] - t)) if state['estimate'] \
        else (t,) + (np.nan,) * 6
    rtf = None
    if len(clock) > 20 and clock[-1][0] - clock[0][0] > 1.0:
        rtf = (clock[-1][1] - clock[0][1]) / (clock[-1][0] - clock[0][0])
    return {'t': t, 'cameras': cameras, 'pictures': [pictures[i][t] for i in cameras],
            'depths': [depths[i][t] for i in cameras], 'k': [calibration[i] for i in cameras],
            'description': description, 'truth': truth, 'estimate': estimate, 'rtf': rtf}


def mounts(description, cameras):
    """Return each camera's optical link in the robot frame as a 4 x 4 matrix."""
    from agri_ugv_navigation.obstacles import mount
    result = []
    for i in cameras:
        rotation, translation = mount(description, link=f'dome_camera_{i}_optical_link')
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rotation, translation
        result.append(matrix)
    return np.array(result)


def main(argv=None):
    """Take the pictures and save them."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--seconds', type=float, default=600.0,
                        help='wall-clock seconds to wait at most (default 600)')
    parser.add_argument('--stop-after', type=float, default=3.5,
                        help='metres the robot drives before the mission is stopped (3.5)')
    parser.add_argument('--still', type=float, default=1.0,
                        help='seconds the robot stands still before the pictures (1.0)')
    parser.add_argument('--output', default=str(Path.home() / 'plant_maps' / 'depth'),
                        help='file name without .npz (default ~/plant_maps/depth)')
    args = parser.parse_args(argv)
    data = capture(args.seconds, args.stop_after, args.still)
    if data is None:
        print('No set of pictures with depth: is the simulation on with cameras:=all '
              'camera_depth:=true, and did the mission start?')
        return
    x, y, z, roll, pitch, yaw = data['truth'][1:]
    ex, ey = data['estimate'][1:3]
    rtf = f'real-time factor {data["rtf"]:.2f}' if data['rtf'] else 'no /clock'
    lines = [f'Pictures with depth at {data["t"]:.2f} s from {len(data["cameras"])} cameras; '
             f'{rtf}',
             f'Robot (true pose): x {x:.3f} y {y:.3f} z {z:.3f}, roll {math.degrees(roll):.2f} '
             f'pitch {math.degrees(pitch):.2f} heading {math.degrees(yaw):.2f} deg; estimate '
             f'{100 * math.hypot(ex - x, ey - y):.1f} cm away']
    for i, image, depth, k in zip(data['cameras'], data['pictures'], data['depths'], data['k']):
        valid = np.isfinite(depth) & (depth > 0)
        d = depth[valid]
        text = (f'  dome_{i:<2} {image.shape[1]} x {image.shape[0]}, focal {k[0]:.1f} px, '
                f'brightness {image.mean():.0f}; depth on {100 * valid.mean():.1f} % of the '
                f'pixels')
        if d.size:
            text += f', {d.min():.3f} / {np.median(d):.3f} / {d.max():.3f} m'
        lines.append(text)
    report = '\n'.join(lines)
    print(report)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        f'{output}.npz', t=data['t'], cameras=np.array(data['cameras']),
        pictures=np.array(data['pictures']), depths=np.array(data['depths']),
        k=np.array(data['k']), mounts=mounts(data['description'], data['cameras']),
        truth=np.array(data['truth']), estimate=np.array(data['estimate']),
        description=data['description'], report=report,
        fields=json.dumps({'truth': 't x y z roll pitch yaw', 'mounts': 'optical link in '
                           'base_footprint (x along the view)', 'depths': 'metres'}))
    print(f'Saved {output}.npz')


if __name__ == '__main__':
    main()
