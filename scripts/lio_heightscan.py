#!/usr/bin/env python3
"""Publish the policy height scan sampled from the LIO elevation map at the LIO robot pose.

Same output as sim_heightscan.py: the scanner grid points (isaac order) in the base frame. The
position and yaw are the odom bridge odometry (~10 Hz), extrapolated with its twist to each publish
time; roll and pitch are the IMU's, the same the policy node uses to level the scan back.
"""
import math
import sys

import numpy as np
import rclpy
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from sim_heightscan import grid_pattern, load_scanner


def quat_to_mat(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec_to_mat(r):
    angle = np.linalg.norm(r)
    if angle < 1e-12:
        return np.eye(3)
    k = r / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def yaw_of(rot):
    return math.atan2(rot[1, 0], rot[0, 0])


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def output_rotation(lio_rot, imu_base_rot):
    """Base orientation in the map frame with the LIO yaw and the IMU roll/pitch (imu_base_rot: base in
    the IMU world frame), so the policy's IMU leveling cancels exactly."""
    return rot_z(yaw_of(lio_rot)) @ rot_z(-yaw_of(imu_base_rot)) @ imu_base_rot


def extrapolate(pos, rot, v_base, w_base, dt):
    """Constant base twist (base frame) applied for dt seconds."""
    return pos + rot @ v_base * dt, rot @ rotvec_to_mat(w_base * dt)


class HeightLayer:
    """One GridMap layer: cell (i, j) has row i along -x and column j along -y from the max corner."""

    def __init__(self, msg, layer):
        self.res = msg.info.resolution
        self.rows = round(msg.info.length_x / self.res)
        self.cols = round(msg.info.length_y / self.res)
        self.cx, self.cy = msg.info.pose.position.x, msg.info.pose.position.y
        data = msg.data[msg.layers.index(layer)].data
        self.h = np.asarray(data, dtype=float).reshape(self.cols, self.rows)  # [j, i], column-major

    def lookup(self, x, y):
        """Nearest-cell heights at (x, y); NaN outside the map or in unknown cells."""
        i = np.floor((self.cx + self.rows * self.res / 2 - x) / self.res).astype(int)
        j = np.floor((self.cy + self.cols * self.res / 2 - y) / self.res).astype(int)
        inside = (i >= 0) & (i < self.rows) & (j >= 0) & (j < self.cols)
        z = np.full(len(x), np.nan)
        z[inside] = self.h[j[inside], i[inside]]
        return z

    def lookup_max(self, x, y, radius):
        """Max height over the containing cell and the cells whose centres are within radius of (x, y)."""
        i0 = np.floor((self.cx + self.rows * self.res / 2 - x) / self.res).astype(int)
        j0 = np.floor((self.cy + self.cols * self.res / 2 - y) / self.res).astype(int)
        k = int(np.ceil(radius / self.res))
        z = np.full(len(x), -np.inf)
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                i, j = i0 + di, j0 + dj
                centre_x = self.cx + self.rows * self.res / 2 - (i + 0.5) * self.res
                centre_y = self.cy + self.cols * self.res / 2 - (j + 0.5) * self.res
                near = (di == 0 and dj == 0) | (np.hypot(x - centre_x, y - centre_y) <= radius)
                inside = near & (i >= 0) & (i < self.rows) & (j >= 0) & (j < self.cols)
                h = np.full(len(x), np.nan)
                h[inside] = self.h[j[inside], i[inside]]
                z = np.fmax(z, h)
        z[np.isneginf(z)] = np.nan
        return z


def fill_unknown(xy, z, k=4):
    """Unknown heights from the inverse-square-distance mean of the k nearest known samples."""
    known = np.isfinite(z)
    k = min(k, int(known.sum()))
    d = np.linalg.norm(xy[~known, None] - xy[None, known], axis=-1)
    nearest = np.argsort(d, axis=1)[:, :k]
    w = 1.0 / np.maximum(np.take_along_axis(d, nearest, axis=1), 1e-6) ** 2
    z[~known] = (w * z[known][nearest]).sum(axis=1) / w.sum(axis=1)


def scan_points(layer, grid, offset, pos, rot, nominal_height, max_radius=0.0):
    """Height scan of the yaw-aligned grid at (pos, rot) in the map frame; (N, 3) in the base frame.
    max_radius > 0: max over the cells within max_radius of each point, else the containing cell."""
    yaw = yaw_of(rot)
    c, s = math.cos(yaw), math.sin(yaw)
    xy = pos[:2] + (grid + offset[:2]) @ np.array([[c, s], [-s, c]])
    if max_radius > 0:
        z = layer.lookup_max(xy[:, 0], xy[:, 1], max_radius)
    else:
        z = layer.lookup(xy[:, 0], xy[:, 1])
    known = np.isfinite(z)
    if not known.any():
        z[:] = pos[2] - nominal_height
    elif not known.all():
        fill_unknown(xy, z)
    hits = np.column_stack([xy, z])
    return (hits - pos) @ rot, int(known.sum())


def main():
    rclpy.init(args=sys.argv)
    node = rclpy.create_node('lio_heightscan')
    metadata = node.declare_parameter('metadata', '').value
    map_topic = node.declare_parameter('map_topic', '/perception/elevation_mapping_node/elevation_map').value
    layer_names = node.declare_parameter('layers', ['inpaint', 'elevation']).value
    odom_topic = node.declare_parameter('odom_topic', '/localization/fast_lio_odom_bridge/odometry').value
    frame_id = node.declare_parameter('frame_id', 'base_link').value
    rate = node.declare_parameter('rate', 50.0).value
    max_extrapolation = node.declare_parameter('max_extrapolation', 0.2).value
    odom_timeout = node.declare_parameter('odom_timeout', 0.5).value
    nominal_height = node.declare_parameter('nominal_height', 0.562).value
    imu_topic = node.declare_parameter('imu_topic', '/vectornav/imu_uncompensated').value
    base_frame = node.declare_parameter('base_frame', 'pelvis').value
    max_radius = node.declare_parameter('max_radius', 0.0).value

    scanner = load_scanner(metadata)
    grid = grid_pattern(scanner['pattern_cfg'])
    offset = np.array(scanner['offset']['pos'], dtype=float)
    log = node.get_logger()

    latest = {}

    def on_map(msg):
        name = next((n for n in layer_names if n in msg.layers), None)
        if name is None:
            log.warn(f'No layer of {layer_names} in {map_topic} (has {list(msg.layers)})', throttle_duration_sec=5.0)
            return
        latest['map'] = HeightLayer(msg, name)
        latest['map_frame'] = msg.header.frame_id

    def on_odom(msg):
        latest['odom'] = msg

    def on_imu(msg):
        latest['imu'] = msg

    from tf2_ros import Buffer, TransformListener
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    def imu_base_rot():
        """Base orientation from the latest IMU sample, or None until the IMU and its static TF arrive."""
        imu = latest.get('imu')
        if imu is None:
            return None
        if 'base_R_imu' not in latest:
            try:
                q = tf_buffer.lookup_transform(base_frame, imu.header.frame_id, rclpy.time.Time()).transform.rotation
            except Exception as error:
                log.warn(f'Waiting for TF {base_frame} <- {imu.header.frame_id}: {error}', throttle_duration_sec=5.0)
                return None
            latest['base_R_imu'] = quat_to_mat(q.x, q.y, q.z, q.w)
        q = imu.orientation
        return quat_to_mat(q.x, q.y, q.z, q.w) @ latest['base_R_imu'].T

    node.create_subscription(GridMap, map_topic, on_map, 1)
    node.create_subscription(Odometry, odom_topic, on_odom, 10)
    node.create_subscription(Imu, imu_topic, on_imu, qos_profile_sensor_data)
    pub = node.create_publisher(PointCloud2, 'heightscan', 1)

    def publish():
        layer, odom = latest.get('map'), latest.get('odom')
        if layer is None or odom is None:
            log.info('Waiting for elevation map and odometry', throttle_duration_sec=5.0)
            return
        if odom.header.frame_id != latest['map_frame']:
            log.error(f'Odometry in {odom.header.frame_id}, map in {latest["map_frame"]}', throttle_duration_sec=5.0)
            return
        now = node.get_clock().now()
        age = (now - rclpy.time.Time.from_msg(odom.header.stamp)).nanoseconds * 1e-9
        if age > odom_timeout:
            log.warn(f'Odometry {age:.2f} s old, not publishing', throttle_duration_sec=2.0)
            return
        p, q = odom.pose.pose.position, odom.pose.pose.orientation
        v, w = odom.twist.twist.linear, odom.twist.twist.angular
        pos, rot = extrapolate(np.array([p.x, p.y, p.z]), quat_to_mat(q.x, q.y, q.z, q.w),
                               np.array([v.x, v.y, v.z]), np.array([w.x, w.y, w.z]),
                               min(max(age, 0.0), max_extrapolation))
        imu_rot = imu_base_rot()
        if imu_rot is None:
            log.warn(f'No IMU on {imu_topic} yet: roll/pitch from LIO', throttle_duration_sec=5.0)
        else:
            rot = output_rotation(rot, imu_rot)
        points, known = scan_points(layer, grid, offset, pos, rot, nominal_height, max_radius)
        if known < len(grid):
            log.warn(f'{len(grid) - known}/{len(grid)} scan cells unknown, filled', throttle_duration_sec=2.0)
        pub.publish(point_cloud2.create_cloud_xyz32(Header(stamp=now.to_msg(), frame_id=frame_id), points))

    node.create_timer(1.0 / rate, publish)
    log.info(f'Sampling {len(grid)} points from {map_topic} at the {odom_topic} pose, {rate:g} Hz')
    rclpy.spin(node)


def selftest():
    """Index convention, yaw alignment, base-frame output and fallback on a synthetic step map."""
    from types import SimpleNamespace as NS
    res, rows, cols = 0.1, 40, 30  # 4 m along x, 3 m along y, centred at (1, 0)
    xs = 1.0 + rows * res / 2 - (np.arange(rows) + 0.5) * res
    h = np.where(xs > 1.5, 0.2, 0.0)[None, :].repeat(cols, axis=0)  # [j, i]: 0.2 m step beyond x=1.5
    h[cols // 2, rows // 2] = np.nan
    msg = NS(info=NS(resolution=res, length_x=rows * res, length_y=cols * res,
                     pose=NS(position=NS(x=1.0, y=0.0))),
             layers=['elevation'], data=[NS(data=h.ravel().tolist())])
    layer = HeightLayer(msg, 'elevation')
    assert layer.lookup(np.array([1.6, 1.4, 9.0]), np.array([0.0, 0.0, 0.0]))[:2].tolist() == [0.2, 0.0]
    assert np.isnan(layer.lookup(np.array([9.0]), np.array([0.0])))[0]

    grid = grid_pattern({'resolution': 0.1, 'size': [1.6, 1.0], 'ordering': 'xy'})
    assert len(grid) == 187
    pos, yaw = np.array([1.0, 0.0, 0.55]), math.pi / 2
    rot = rotvec_to_mat(np.array([0.0, 0.0, yaw]))
    pts, known = scan_points(layer, grid, np.zeros(3), pos, rot, 0.562)
    # yawed 90 deg: grid +x points along world +y, so no cell reaches the step at world x > 1.5
    assert known == 186 and np.allclose(pts[:, :2], grid) and np.allclose(pts[:, 2], -0.55)
    pts, _ = scan_points(layer, grid, np.zeros(3), pos, np.eye(3), 0.562)
    ahead = grid[:, 0] > 0.55  # cell centres past x=1.5 in the world
    assert np.allclose(pts[ahead, 2], 0.2 - 0.55) and np.allclose(pts[grid[:, 0] < 0.45, 2], -0.55)

    # step at x=1.5 (cell edge), 0.1 m cells: from x=1.46 the 0.2 m cell centre at 1.55 is 0.09 m away
    x, y = np.array([1.46, 1.3]), np.array([-0.05, -0.05])  # on the cell centres' y
    assert layer.lookup(x, y)[0] == 0.0
    assert layer.lookup_max(x, y, 0.1).tolist() == [0.2, 0.0]
    assert layer.lookup_max(x, y, 0.05).tolist() == [0.0, 0.0]  # containing cells only
    assert np.isnan(layer.lookup_max(np.array([9.0]), np.array([0.0]), 0.1)[0])

    # unknown gap between a 0 m and a 0.2 m step: filled in between, nearer side weighted more
    xy = np.column_stack([np.arange(5) * 0.1, np.zeros(5)])
    z = np.array([0.0, 0.0, np.nan, np.nan, 0.2])
    fill_unknown(xy, z)
    assert 0.0 < z[2] < z[3] < 0.2 and z[2] < 0.1 < z[3]

    p, r = extrapolate(np.zeros(3), rotvec_to_mat(np.array([0, 0, yaw])), np.array([1.0, 0, 0]),
                       np.array([0, 0, 1.0]), 0.1)
    assert np.allclose(p, [0, 0.1, 0]) and np.isclose(math.atan2(r[1, 0], r[0, 0]), yaw + 0.1)

    # policy side: points rotated back by its IMU base orientation, then its yaw removed
    imu_base = rot_z(0.7) @ rotvec_to_mat(np.array([0.0, 0.1, 0.0])) @ rotvec_to_mat(np.array([-0.08, 0.0, 0.0]))
    lio = rot_z(0.3) @ rotvec_to_mat(np.array([0.0, 0.05, 0.0]))  # other world yaw, stale tilt
    out = output_rotation(lio, imu_base)
    assert np.isclose(yaw_of(out), 0.3)
    v = np.array([[0.5, -0.2, -0.5], [-0.8, 0.5, 0.1]])  # hits - pos in the map frame
    leveled = (rot_z(-yaw_of(imu_base)) @ imu_base @ (v @ out).T).T
    assert np.allclose(leveled, (rot_z(-0.3) @ v.T).T)  # world z kept, xy in the LIO heading frame
    print('selftest ok')


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main()
