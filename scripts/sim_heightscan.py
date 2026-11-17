#!/usr/bin/env python3
"""Publish a ground-truth height scan by ray-casting a MuJoCo world at the simulated robot pose."""
import math
import sys

import mujoco
import numpy as np
import rclpy
import yaml
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

RAY_CASTER = 'kyon_isaac.sensors.ray_caster:KyonRayCaster'
DOWN = np.array([0.0, 0.0, -1.0])


class _Loader(yaml.SafeLoader):
    pass


# isaac metadata carries python tags: keep sequences, drop everything else
_Loader.add_multi_constructor(
    '', lambda loader, suffix, node: loader.construct_sequence(node) if isinstance(node, yaml.SequenceNode) else None)


def load_scanner(metadata_path):
    with open(metadata_path) as f:
        scene = yaml.load(f, Loader=_Loader)['scene']
    for cfg in scene.values():
        if isinstance(cfg, dict) and cfg.get('class_type') == RAY_CASTER:
            return cfg
    raise RuntimeError(f'No {RAY_CASTER} sensor in {metadata_path}')


def grid_pattern(cfg):
    """isaaclab grid_pattern: (N, 2) xy offsets, x fastest for ordering 'xy'."""
    res, (sx, sy) = cfg['resolution'], cfg['size']
    x = -sx / 2 + res * np.arange(round(sx / res) + 1)
    y = -sy / 2 + res * np.arange(round(sy / res) + 1)
    gx, gy = np.meshgrid(x, y, indexing='xy' if cfg['ordering'] == 'xy' else 'ij')
    return np.stack([gx.ravel(), gy.ravel()], axis=-1)


def scan_points(model, data, groups, grid, offset, pos, rot):
    """Ray-cast the yaw-aligned grid; returns hit points (N, 3) in the base frame."""
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    xy = pos[:2] + (grid + offset[:2]) @ np.array([[c, s], [-s, c]])
    start_z = pos[2] + offset[2]
    geomid = np.zeros(1, np.int32)
    hits = np.empty((len(grid), 3))
    for i, (x, y) in enumerate(xy):
        dist = mujoco.mj_ray(model, data, np.array([x, y, start_z]), DOWN, groups, 1, -1, geomid)
        # ponytail: no hit falls back to world z=0, fine while worlds have a floor plane
        hits[i] = (x, y, start_z - dist if dist >= 0 else 0.0)
    return (hits - pos) @ rot


def main():
    rclpy.init(args=sys.argv)
    node = rclpy.create_node('sim_heightscan')
    world_xml = node.declare_parameter('world_xml', '').value
    metadata = node.declare_parameter('metadata', '').value
    pose_topic = node.declare_parameter('pose_topic', '/pelvis/odom').value
    frame_id = node.declare_parameter('frame_id', 'base_link').value
    rate = node.declare_parameter('rate', 50.0).value

    scanner = load_scanner(metadata)
    grid = grid_pattern(scanner['pattern_cfg'])
    offset = np.array(scanner['offset']['pos'], dtype=float)

    model = mujoco.MjModel.from_xml_path(world_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # geoms not welded to the world (the robot) go to group 5, which the rays skip
    model.geom_group[model.body_weldid[model.geom_bodyid] != 0] = 5
    groups = np.array([1, 1, 1, 1, 1, 0], np.uint8)

    pub = node.create_publisher(PointCloud2, 'heightscan', 1)
    latest = {}
    node.create_subscription(Odometry, pose_topic, lambda msg: latest.update(pose=msg), 1)

    def publish():
        msg = latest.get('pose')
        if msg is None:
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.array([q.w, q.x, q.y, q.z]))
        points = scan_points(model, data, groups, grid, offset, np.array([p.x, p.y, p.z]), rot.reshape(3, 3))
        msg.header.frame_id = frame_id
        pub.publish(point_cloud2.create_cloud_xyz32(msg.header, points))

    node.create_timer(1.0 / rate, publish)
    node.get_logger().info(f'Ray-casting {len(grid)} points on {world_xml} from {pose_topic}')
    rclpy.spin(node)


if __name__ == '__main__':
    main()
