#!/usr/bin/env python3
"""Toggle simulated actuator faults from the keyboard (see deploy node ~/fault_joints)."""
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
# kyon_isaac fault_velocity types: Kp/Kd x fcl, effort limit x fcl, target window lock_pos +- scale * fcl
TYPEMAP = {'z': 'scale', 'x': 'clip', 'c': 'lock'}

HELP = """fault keyboard: key toggles a fault on joint (with the current type)
         leg1 leg2 leg3 leg4
  roll    1    2    3    4
  pitch   q    w    e    r
  knee    a    s    d    f
  z/x/c: type scale / clip / lock for the next toggled joints
  +/-: fcl +-0.05 on all faulted joints (1 = nominal, 0 = complete fault)
  space: clear all   esc/ctrl-c: quit"""


def main():
    rclpy.init()
    node = rclpy.create_node('fault_keyboard')
    pub = node.create_publisher(String, '/policy_deploy_node/fault_joints', 1)
    faulted = {}  # joint -> type
    ftype = 'scale'
    fcl = 0.0

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
                j = KEYMAP[key]
                if faulted.pop(j, None) is None:
                    faulted[j] = ftype
            elif key in TYPEMAP:
                ftype = TYPEMAP[key]
            elif key in '+=':
                fcl = min(round(fcl + 0.05, 2), 1.0)
            elif key in '-_':
                fcl = max(round(fcl - 0.05, 2), 0.0)
            else:
                continue
            pub.publish(String(data=','.join(f'{j}:{t}:{fcl:.2f}' for j, t in sorted(faulted.items()))))
            print(f"type {ftype}  fcl {fcl:.2f}  faulted: {dict(sorted(faulted.items())) or 'none'}")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        pub.publish(String(data=''))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
