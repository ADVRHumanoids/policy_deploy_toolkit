#!/usr/bin/env python3
"""Keyboard teleop for the arms only: EE goals (keys or typed poses) and EE shapes; the base runs from the joystick."""
import os
import select
import sys
import termios
import time
import tty

import numpy as np
import yaml

from keyboard_teleop import (ARM_NAMES, AXIS_NAMES, DT, EE_KEYS, default_shapes_file, parse_shape_command,
                             qconj, qmul, quat_from_rotvec, rotvec_from_quat, shape_waypoint, step_towards)

HELP = """arms teleop (esc/ctrl-c: quit), base from the joystick
EE goal (task frame)   x -1 +2   y -3 +4   z -5 +6   roll -7 +8   pitch -9 +0   yaw -- +=
  a: arm left/right/both (mirrored: y, roll, yaw flipped on the right)   d: goals to home   f: print goal vs measured
commands: type ':' then a command and Enter (Esc cancels the line)
  :pose <left|right|both> x y z [roll pitch yaw]   task frame [m, rad]; both: right arm mirrored
  :<small|medium|big> <yz|xz|xy> <circle|square|triangle> <left|right|both> [slow|fast|both]   e.g. :medium xz circle both
  :stop"""


def quat_from_rpy(r, p, y):
    """wxyz quaternion of Rz(y) Ry(p) Rx(r)."""
    return qmul(quat_from_rotvec(np.array([0.0, 0.0, y])),
                qmul(quat_from_rotvec(np.array([0.0, p, 0.0])), quat_from_rotvec(np.array([r, 0.0, 0.0]))))


def mirror_quat(q):
    """Mirror of a wxyz quaternion about the sagittal (xz) plane: roll and yaw flipped."""
    return np.array([q[0], -q[1], q[2], -q[3]])


def parse_pose_command(words, goal):
    """'left|right|both x y z [r p y]' -> new goals [(pos, quat)] * 2; both: right arm mirrored.

    Raises ValueError on a bad command.
    """
    if len(words) not in (4, 7) or words[0] not in ARM_NAMES:
        raise ValueError('usage: pose <left|right|both> x y z [roll pitch yaw]')
    vals = [float(v) for v in words[1:]]
    pos = np.array(vals[:3])
    arms = (0, 1) if words[0] == 'both' else (ARM_NAMES.index(words[0]),)
    new = [(p.copy(), q.copy()) for p, q in goal]
    for i in arms:
        mirrored = words[0] == 'both' and i == 1
        p = pos * np.array([1.0, -1.0, 1.0]) if mirrored else pos.copy()
        q = goal[i][1].copy()
        if len(vals) == 6:
            q = quat_from_rpy(*vals[3:])
            q = mirror_quat(q) if mirrored else q
        new[i] = (p, q)
    return new


def check():
    q = quat_from_rpy(0.3, -0.2, 0.5)
    assert np.allclose(mirror_quat(q), quat_from_rpy(-0.3, -0.2, -0.5)), 'mirror flips roll and yaw'
    goal = [(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])) for _ in range(2)]
    new = parse_pose_command('both 0.6 0.3 0.9 0.3 -0.2 0.5'.split(), goal)
    assert np.allclose(new[0][0], [0.6, 0.3, 0.9]) and np.allclose(new[1][0], [0.6, -0.3, 0.9]), 'mirrored position'
    assert np.allclose(new[0][1], q) and np.allclose(new[1][1], mirror_quat(q)), 'mirrored orientation'
    new = parse_pose_command('right 0.5 -0.2 0.8'.split(), goal)
    assert np.allclose(new[1][0], [0.5, -0.2, 0.8]) and np.allclose(new[1][1], goal[1][1]), 'right only, orientation kept'
    assert np.allclose(new[0][0], goal[0][0]), 'left untouched'
    for bad in ('top 1 2 3', 'left 1 2', 'both 1 2 3 4'):
        try:
            parse_pose_command(bad.split(), goal)
            raise AssertionError(f'accepted "{bad}"')
        except ValueError:
            pass
    print('arms_teleop check: ok')


def main():
    import rclpy
    from std_msgs.msg import Float64MultiArray

    rclpy.init()
    node = rclpy.create_node('arms_teleop')
    param = lambda name, default: node.declare_parameter(name, default).value
    ee_node = param('ee_node', 'arms_policy_node')
    ee_step = param('ee_step', 0.02)
    ee_angle_step = param('ee_angle_step', 0.1)
    ee_speed = param('ee_speed', 0.5)
    ee_ang_speed = param('ee_ang_speed', 1.5)
    shapes_file = param('shapes_file', '') or default_shapes_file()

    shapes = None
    if os.path.isfile(shapes_file):
        with open(shapes_file) as f:
            shapes = yaml.safe_load(f)
        print(f'[shape] {len(shapes["shapes"])} shapes from {shapes_file}')
    else:
        print(f'[shape] no shape tables at "{shapes_file}", shape commands disabled')
    shape_options, speed_values = {}, {}
    if shapes is not None:
        speeds = shapes['timing']['speeds_mps']
        speed_values = {'slow': [speeds[0]], 'fast': [speeds[-1]], 'both': speeds}
        shape_options = {name: list(dict.fromkeys(e[name] for e in shapes['shapes'])) for name in ('size', 'plane', 'shape')}
        shape_options.update(speed=list(speed_values), arm=list(ARM_NAMES))

    ee_pub = node.create_publisher(Float64MultiArray, f'/{ee_node}/commands/ee_pose/reference', 1)
    ee = {'measured': None}
    node.create_subscription(Float64MultiArray, f'/{ee_node}/ee_measured',
                             lambda msg: ee.__setitem__('measured', np.array(msg.data)), 1)

    ee_arm = 2
    home = goal = target = None
    # running shape: (table entry, arms, speeds, start wall time, start goals), None when idle
    shape_run = None
    # typed command after ':', None in single-key mode
    line = None

    def measured_poses():
        m = ee['measured']
        return [(m[7 * i:7 * i + 3].copy(), m[7 * i + 3:7 * i + 7].copy()) for i in range(2)]

    def ee_nudge(axis, sign):
        for i in ((0, 1) if ee_arm == 2 else (ee_arm,)):
            pos, quat = goal[i]
            # both: right arm mirrored about the sagittal plane (y, roll, yaw flipped)
            s = -sign if ee_arm == 2 and i == 1 and axis in (1, 3, 5) else sign
            if axis < 3:
                pos[axis] += s * ee_step
            else:
                goal[i] = (pos, qmul(quat_from_rotvec(s * ee_angle_step * np.eye(3)[axis - 3]), quat))
        print(f'[ee] {ARM_NAMES[ee_arm]} goal {AXIS_NAMES[axis]} {"+" if sign > 0 else "-"}')

    def ee_print():
        for i, (pos, quat) in enumerate(measured_poses()):
            g, gq = goal[i]
            ang = np.linalg.norm(rotvec_from_quat(qmul(gq, qconj(quat))))
            print(f'[ee] {ARM_NAMES[i]:>5s}  goal ({g[0]:+.3f}, {g[1]:+.3f}, {g[2]:.3f})'
                  f'  measured ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f})  ori err {ang:.3f} rad')
        print(f'[ee] pelvis height {ee["measured"][14]:.3f} m')

    def run_command(text):
        nonlocal shape_run, goal
        words = text.split()
        if words == ['stop']:
            if shape_run is not None:
                shape_run, goal = None, [(p.copy(), q.copy()) for p, q in target]
                print('[shape] stopped, goals held at the current references')
            return
        if home is None:
            print(f'[ee] no EE pose yet from /{ee_node}/ee_measured')
            return
        if shape_run is not None:
            print('[shape] running: ":stop" first')
            return
        if words[:1] == ['pose']:
            try:
                goal = parse_pose_command(words[1:], goal)
            except ValueError as e:
                print(f'[ee] {e}')
                return
            print(f'[ee] {words[1]} goal set')
            return
        if shapes is None:
            print('[shape] no shape tables loaded')
            return
        try:
            sel = parse_shape_command(text, shape_options)
        except ValueError as e:
            print(f'[shape] {e}; usage: <{"|".join(shape_options["size"])}> <{"|".join(shape_options["plane"])}>'
                  f' <{"|".join(shape_options["shape"])}> <left|right|both> [slow|fast|both]')
            return
        entry = next(e for e in shapes['shapes'] if all(e[k] == sel[k] for k in ('size', 'plane', 'shape')))
        arms = (0, 1) if sel['arm'] == 'both' else (ARM_NAMES.index(sel['arm']),)
        shape_run = (entry, arms, speed_values[sel['speed']], time.monotonic(), [(p.copy(), q.copy()) for p, q in goal])
        print(f'[shape] start: {sel["shape"]}, {sel["size"]} (r {entry["radius_m"]} m), plane {sel["plane"]},'
              f' speed {sel["speed"]}, {sel["arm"]} arm(s)')

    if not sys.stdin.isatty():
        node.get_logger().error('arms_teleop needs a terminal: run it with ros2 run')
        return

    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)
    print(HELP)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            # EE goals start at the first measured EE poses
            if home is None and ee['measured'] is not None:
                home = measured_poses()
                goal = [(p.copy(), q.copy()) for p, q in home]
                target = [(p.copy(), q.copy()) for p, q in home]
                print('[ee] EE keys active, goals at the current EE poses')

            if select.select([sys.stdin], [], [], DT)[0]:
                key = sys.stdin.read(1)
                # command line: keys are text until Enter (run) or Esc (cancel)
                if line is not None and key != '\x03':
                    if key in ('\n', '\r'):
                        print()
                        text, line = line, None
                        run_command(text)
                    elif key == '\x1b':
                        line = None
                        print(' [cancelled]')
                    elif key in ('\x7f', '\b'):
                        if line:
                            line = line[:-1]
                            sys.stdout.write('\b \b')
                    elif key.isprintable():
                        line += key
                        sys.stdout.write(key)
                    sys.stdout.flush()
                    continue
                if key in ('\x1b', '\x03'):
                    break
                if key == ':':
                    line = ''
                    sys.stdout.write(':')
                    sys.stdout.flush()
                    continue
                if key in EE_KEYS or key in 'adf':
                    if home is None:
                        print(f'[ee] no EE pose yet from /{ee_node}/ee_measured')
                    elif shape_run is not None and key != 'f':
                        print('[shape] running: ":stop" first')
                    elif key in EE_KEYS:
                        ee_nudge(*EE_KEYS[key])
                    elif key == 'a':
                        ee_arm = (ee_arm + 1) % 3
                        print(f'[ee] keys move: {ARM_NAMES[ee_arm]}')
                    elif key == 'd':
                        goal = [(p.copy(), q.copy()) for p, q in home]
                        print('[ee] goals back to home')
                    else:
                        ee_print()
            if shape_run is not None:
                # waypoints for the next control step; arms not drawing hold their start goal
                entry, arms, run_speeds, start, start_goal = shape_run
                goal = [(p.copy(), q.copy()) for p, q in start_goal]
                wps = {i: shape_waypoint(shapes, entry, start_goal[i], i, time.monotonic() - start + DT, run_speeds)
                       for i in arms}
                if all(wp is None for wp in wps.values()):
                    shape_run = None
                    print('[shape] done')
                else:
                    for i, wp in wps.items():
                        if wp is not None:
                            goal[i] = wp
            if home is not None:
                target = [step_towards(*target[i], *goal[i], ee_speed * DT, ee_ang_speed * DT) for i in range(2)]
                ee_pub.publish(Float64MultiArray(data=np.concatenate(
                    [np.r_[target[i][0], target[i][1], goal[i][0], goal[i][1]] for i in range(2)]).tolist()))
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    if '--check' in sys.argv:
        check()
    else:
        main()
