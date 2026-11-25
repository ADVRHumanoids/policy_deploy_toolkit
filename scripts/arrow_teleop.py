#!/usr/bin/env python3
"""Drive a policy_deploy_toolkit policy with the arrow keys (ROS 2 node, run inside the XBot2 container).

Publishes geometry_msgs/Twist on /policy_deploy_node/commands/base_velocity 20 times per second, so the command
never times out while this runs:
    up / down      forward speed +/- 0.1 m/s
    left / right   turn left / right (yaw rate +/- 0.1 rad/s)
    a / d          sideways left / right (+/- 0.1 m/s)
    space          stop (all zero)
    1..4           switch policy (policy_switcher_node / kyon_ferk_policies.launch.xml):
                   1 = 4-leg walking, 2 = stand-up, 3 = 2-leg walking, 4 = get-down;  0 = release
    q              quit (sends a stop first)

    ros2 run policy_deploy_toolkit arrow_teleop.py [--topic /policy_deploy_node/commands/base_velocity]
"""

import argparse
import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

STEP = 0.1
LIMITS = {"vx": 1.0, "vy": 0.5, "wz": 1.0}  # the lab's example locomotion command ranges are +-1


def read_key(timeout):
    """One key press (arrow keys as 'up'/'down'/'left'/'right'), or None after timeout seconds."""
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x1b":  # arrow keys arrive as ESC [ A..D
        seq = sys.stdin.read(2)
        return {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(seq)
    return ch


def main():
    parser = argparse.ArgumentParser(description="Arrow-key velocity commands for policy_deploy_node.")
    parser.add_argument("--topic", default="/policy_deploy_node/commands/base_velocity")
    args = parser.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])  # also when run with --ros-args

    rclpy.init()
    node = rclpy.create_node("arrow_teleop")
    pub = node.create_publisher(Twist, args.topic, 1)
    select_pub = node.create_publisher(String, "/policy_switcher/select", 10)
    cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}

    def clamp(key, value):
        cmd[key] = round(max(-LIMITS[key], min(LIMITS[key], value)), 2)

    def publish():
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = cmd["vx"], cmd["vy"], cmd["wz"]
        pub.publish(msg)

    print(__doc__.split("\n\n")[1])
    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            key = read_key(0.05)
            if key == "up":
                clamp("vx", cmd["vx"] + STEP)
            elif key == "down":
                clamp("vx", cmd["vx"] - STEP)
            elif key == "left":
                clamp("wz", cmd["wz"] + STEP)
            elif key == "right":
                clamp("wz", cmd["wz"] - STEP)
            elif key == "a":
                clamp("vy", cmd["vy"] + STEP)
            elif key == "d":
                clamp("vy", cmd["vy"] - STEP)
            elif key == " ":
                cmd.update(vx=0.0, vy=0.0, wz=0.0)
            elif key == "q":
                break
            elif key is not None and key.isdigit():
                select_pub.publish(String(data=key))
                print(f"\rswitch to policy {key} requested{' ' * 40}")
            if key is not None:
                print(f"\rforward {cmd['vx']:+.1f} m/s   sideways {cmd['vy']:+.1f} m/s   turn {cmd['wz']:+.1f} rad/s   ", end="", flush=True)
            publish()  # every 50 ms, also without key presses
    finally:
        cmd.update(vx=0.0, vy=0.0, wz=0.0)
        publish()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
