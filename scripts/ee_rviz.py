#!/usr/bin/env python3
"""RViz markers of the arm EE tracking in the ee_task frame: reference (green) and measured (red) poses and trails."""
from collections import deque

import rclpy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray

REFERENCE = ColorRGBA(r=0.1, g=0.85, b=0.2, a=1.0)
MEASURED = ColorRGBA(r=0.95, g=0.15, b=0.1, a=1.0)


def main():
    rclpy.init()
    node = rclpy.create_node('ee_rviz')
    ee_node = node.declare_parameter('ee_node', 'arms_policy_node').value
    frame = node.declare_parameter('frame', 'ee_task').value
    trail_length = node.declare_parameter('trail_length', 200).value

    pub = node.create_publisher(MarkerArray, '~/markers', 1)
    # reference per arm: target pos, quat, goal pos, quat (14 values)
    state = {'reference': None}
    trails = {(kind, arm): deque(maxlen=trail_length) for kind in ('reference', 'measured') for arm in range(2)}

    def marker(ns, mid, mtype, color, scale, alpha=1.0):
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = node.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color = ColorRGBA(r=color.r, g=color.g, b=color.b, a=alpha)
        return m

    def sphere(ns, mid, color, scale, pos, alpha=1.0):
        m = marker(ns, mid, Marker.SPHERE, color, scale, alpha)
        m.pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
        return m

    def strip(ns, mid, color, points):
        m = marker(ns, mid, Marker.LINE_STRIP, color, 0.005, 0.8)
        m.points = [Point(x=p[0], y=p[1], z=p[2]) for p in points]
        return m

    def on_reference(msg):
        if len(msg.data) == 28:
            state['reference'] = [msg.data[14 * arm:14 * arm + 14] for arm in range(2)]

    def on_measured(msg):
        # measured per arm: pos, quat (7 values); then the pelvis height
        if len(msg.data) < 14:
            return
        ref = state['reference']
        markers = MarkerArray()
        for arm in range(2):
            meas = msg.data[7 * arm:7 * arm + 3]
            trails['measured', arm].append(meas)
            markers.markers.append(sphere('measured', arm, MEASURED, 0.025, meas))
            # line strips need at least two points
            if len(trails['measured', arm]) > 1:
                markers.markers.append(strip('measured_trail', arm, MEASURED, trails['measured', arm]))
            if ref is None:
                continue
            trails['reference', arm].append(ref[arm][0:3])
            markers.markers.append(sphere('reference', arm, REFERENCE, 0.03, ref[arm][0:3]))
            markers.markers.append(sphere('goal', arm, REFERENCE, 0.06, ref[arm][7:10], 0.35))
            if len(trails['reference', arm]) > 1:
                markers.markers.append(strip('reference_trail', arm, REFERENCE, trails['reference', arm]))
        pub.publish(markers)

    node.create_subscription(Float64MultiArray, f'/{ee_node}/commands/ee_pose/reference', on_reference, 1)
    node.create_subscription(Float64MultiArray, f'/{ee_node}/ee_measured', on_measured, 1)
    node.get_logger().info(f'markers on {pub.topic_name} in frame "{frame}"')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
