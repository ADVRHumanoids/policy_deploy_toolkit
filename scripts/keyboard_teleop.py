#!/usr/bin/env python3
"""Keyboard teleop for the deploy node: velocity (teleop_twist_keyboard keys) and posture."""
import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

# teleop_twist_keyboard bindings: key -> (x, y, yaw) direction, key -> (speed, turn) scale
MOVE_KEYS = {
    'i': (1, 0, 0), 'o': (1, 0, -1), 'j': (0, 0, 1), 'l': (0, 0, -1),
    'u': (1, 0, 1), ',': (-1, 0, 0), '.': (-1, 0, 1), 'm': (-1, 0, -1),
    'O': (1, -1, 0), 'I': (1, 0, 0), 'J': (0, 1, 0), 'L': (0, -1, 0),
    'U': (1, 1, 0), '<': (-1, 0, 0), '>': (-1, -1, 0), 'M': (-1, 1, 0),
}
SPEED_KEYS = {
    'q': (1.1, 1.1), 'z': (0.9, 0.9), 'w': (1.1, 1.0),
    'x': (0.9, 1.0), 'e': (1.0, 1.1), 'c': (1.0, 0.9),
}
# key -> (posture index, direction); posture = [roll, pitch, height]
POSTURE_KEYS = {'b': (0, -1), 'n': (0, +1), 'y': (1, +1), 'h': (1, -1), 't': (2, +1), 'g': (2, -1)}
DT = 0.02

HELP = """keyboard teleop (esc/ctrl-c: quit)
velocity (as teleop_twist_keyboard)
     u    i    o           shift: U    I    O   (strafe)
     j    k    l                  J    K    L
     m    ,    .                  M    <    >
  k or any other key: stop
  q/z: speed and turn x1.1/x0.9   w/x: speed only   e/c: turn only
posture   t/g: height +/-   y/h: pitch +/- (nose down/up)   b/n: roll -/+   p: nominal"""


def main():
    rclpy.init()
    node = rclpy.create_node('keyboard_teleop')
    param = lambda name, default: node.declare_parameter(name, default).value
    speed = param('speed', 0.5)
    turn = param('turn', 1.0)
    # limits match the trained VelocityCommand / PostureCommand ranges
    vel_max = param('vel_max', [1.0, 1.0, 1.5])
    posture_step = param('posture_step', [0.05, 0.05, 0.02])
    posture_min = param('posture_min', [-0.3, -0.3, 0.30])
    posture_max = param('posture_max', [0.3, 0.3, 0.65])
    nominal = param('nominal', [0.0, 0.0, 0.562])

    vel_pub = node.create_publisher(Twist, '/policy_deploy_node/commands/base_velocity', 1)
    posture_pub = node.create_publisher(Float64MultiArray, '/policy_deploy_node/commands/base_posture/raw', 1)

    direction = (0, 0, 0)
    posture = list(nominal)

    def velocity():
        raw = (direction[0] * speed, direction[1] * speed, direction[2] * turn)
        return [min(max(v, -m), m) for v, m in zip(raw, vel_max)]

    def publish_vel(vel):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = vel
        vel_pub.publish(msg)

    if not sys.stdin.isatty():
        node.get_logger().error('keyboard_teleop needs a terminal: run it with ros2 run')
        return

    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)
    print(HELP)
    try:
        while rclpy.ok():
            # republish every tick: the deploy node drops commands older than command_timeout_s
            if select.select([sys.stdin], [], [], DT)[0]:
                key = sys.stdin.read(1)
                if key in ('\x1b', '\x03'):
                    break
                if key in MOVE_KEYS:
                    direction = MOVE_KEYS[key]
                elif key in SPEED_KEYS:
                    speed *= SPEED_KEYS[key][0]
                    turn *= SPEED_KEYS[key][1]
                elif key in POSTURE_KEYS:
                    i, sign = POSTURE_KEYS[key]
                    posture[i] = min(max(posture[i] + sign * posture_step[i], posture_min[i]), posture_max[i])
                elif key == 'p':
                    posture = list(nominal)
                else:
                    direction = (0, 0, 0)
                vel = velocity()
                print(f'speed {speed:.2f} turn {turn:.2f}  '
                      f'vel [{vel[0]:+.2f} {vel[1]:+.2f} {vel[2]:+.2f}]  '
                      f'posture [{posture[0]:+.2f} {posture[1]:+.2f} {posture[2]:.3f}]')
            publish_vel(velocity())
            posture_pub.publish(Float64MultiArray(data=posture))
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        publish_vel([0.0, 0.0, 0.0])
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
