#!/usr/bin/env python3
"""Toggle simulated joint power loss from the keyboard (see deploy node ~/fault_joints)."""
import select
import sys
import termios
import tty

import rclpy
from std_msgs.msg import String

# columns = legs 1..4
KEYMAP = {
    '1': 'hip_roll_1', '2': 'hip_roll_2', '3': 'hip_roll_3', '4': 'hip_roll_4',
    'q': 'hip_pitch_1', 'w': 'hip_pitch_2', 'e': 'hip_pitch_3', 'r': 'hip_pitch_4',
    'a': 'knee_pitch_1', 's': 'knee_pitch_2', 'd': 'knee_pitch_3', 'f': 'knee_pitch_4',
}

HELP = """fault keyboard: key toggles power loss on joint
         leg1 leg2 leg3 leg4
  roll    1    2    3    4
  pitch   q    w    e    r
  knee    a    s    d    f
  +/-: fault level +-10% (100% = full power loss)
  space: clear all   esc/ctrl-c: quit"""


def main():
    rclpy.init()
    node = rclpy.create_node('fault_keyboard')
    pub = node.create_publisher(String, '/policy_deploy_node/fault_joints', 1)
    faulted = set()
    level = 100  # % of gain removed, shared by all faulted joints

    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)
    print(HELP)
    try:
        while rclpy.ok():
            if not select.select([sys.stdin], [], [], 0.1)[0]:
                continue
            key = sys.stdin.read(1)
            if key in ('\x1b', '\x03'):
                break
            if key == ' ':
                faulted.clear()
            elif key in KEYMAP:
                faulted ^= {KEYMAP[key]}
            elif key in '+=':
                level = min(level + 10, 100)
            elif key in '-_':
                level = max(level - 10, 0)
            else:
                continue
            scale = 1.0 - level / 100.0
            pub.publish(String(data=','.join(f'{j}:{scale:.2f}' for j in sorted(faulted))))
            print(f"level {level}%  faulted: {sorted(faulted) or 'none'}")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        pub.publish(String(data=''))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
