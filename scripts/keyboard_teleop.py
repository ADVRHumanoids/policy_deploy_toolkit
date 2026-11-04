#!/usr/bin/env python3
"""Keyboard teleop for the deploy nodes: velocity (teleop_twist_keyboard keys), posture, arm EE goals and EE shapes."""
import os
import select
import sys
import termios
import time
import tty

import numpy as np
import yaml

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
# key -> (EE axis, direction); axes x y z roll pitch yaw of the task frame
EE_KEYS = {'1': (0, -1), '2': (0, +1), '3': (1, -1), '4': (1, +1), '5': (2, -1), '6': (2, +1),
           '7': (3, -1), '8': (3, +1), '9': (4, -1), '0': (4, +1), '-': (5, -1), '=': (5, +1)}
ARM_NAMES = ('left', 'right', 'both')
AXIS_NAMES = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
DT = 0.02

HELP = """keyboard teleop (esc/ctrl-c: quit)
velocity (as teleop_twist_keyboard)
     u    i    o           shift: U    I    O   (strafe)
     j    k    l                  J    K    L
     m    ,    .                  M    <    >
  k or any other key: stop
  q/z: speed and turn x1.1/x0.9   w/x: speed only   e/c: turn only
posture   t/g: height +/-   y/h: pitch +/- (nose down/up)   b/n: roll -/+   p: nominal
EE goal (task frame)   x -1 +2   y -3 +4   z -5 +6   roll -7 +8   pitch -9 +0   yaw -- +=
  a: arm left/right/both   d: goals to home   f: print goal vs measured
EE shapes: type ':' then a command and Enter (Esc cancels the line)
  :<small|medium|big> <yz|xz|xy> <circle|square|triangle> <left|right> [slow|fast|both]   e.g. :medium xz square right
  :stop"""


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_from_rotvec(v):
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * v / angle]


def rotvec_from_quat(q):
    """Rotation vector of q, shortest way."""
    q = q if q[0] >= 0 else -q
    n = np.linalg.norm(q[1:])
    return np.zeros(3) if n < 1e-12 else 2 * np.arctan2(n, q[0]) * q[1:] / n


def step_towards(pos, quat, goal_pos, goal_quat, max_lin, max_ang):
    """Moves (pos, quat) towards the goal by at most max_lin [m] and max_ang [rad]."""
    dp = goal_pos - pos
    pos = pos + dp * min(1.0, max_lin / max(np.linalg.norm(dp), 1e-9))
    aa = rotvec_from_quat(qmul(goal_quat, qconj(quat)))
    aa = aa * min(1.0, max_ang / max(np.linalg.norm(aa), 1e-9))
    quat = qmul(quat_from_rotvec(aa), quat)
    return pos, quat / np.linalg.norm(quat)


def nlerp(q0, q1, w):
    """Normalized linear interpolation of wxyz quaternions, shortest way."""
    q1 = -q1 if np.dot(q0, q1) < 0 else q1
    q = (1.0 - w) * q0 + w * q1
    return q / np.linalg.norm(q)


def shape_waypoint(shapes, entry, arm, t, speeds=None):
    """Waypoint (pos, quat wxyz) of the drawing arm at time t since the start, None once finished.

    Hold at home, then per speed: min-jerk blend to the shape start, laps along the table, blend back to home.
    speeds [m/s] defaults to all the table speeds, in order.
    """
    timing = shapes['timing']
    home_p, home_q = np.array(shapes['home']['pos'][arm]), np.array(shapes['home']['quat_wxyz'][arm])
    table = entry[('left', 'right')[arm]]
    pos, quat = np.array(table['pos']), np.array(table['quat_wxyz'])
    n, move, tl = len(pos), timing['move_s'], t - timing['hold_s']
    if tl < 0.0:
        return home_p, home_q
    for speed in speeds or timing['speeds_mps']:
        period = entry['perimeter_m'] / speed
        laps = timing['laps'] * period
        if tl < move:
            tau = tl / move
        elif tl < move + laps:
            s = ((tl - move) / period % 1.0) * n
            i, u = int(s) % n, s - int(s)
            j = (i + 1) % n
            return (1.0 - u) * pos[i] + u * pos[j], nlerp(quat[i], quat[j], u)
        elif tl < 2 * move + laps:
            tau = 1.0 - (tl - move - laps) / move
        else:
            tl -= 2 * move + laps
            continue
        w = tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)
        return home_p + w * (pos[0] - home_p), nlerp(home_q, quat[0], w)
    return None


def parse_shape_command(text, options):
    """'medium xz square fast right' -> {setting: value}; words in any order, speed defaults to slow.

    Raises ValueError on unknown, repeated or missing settings.
    """
    sel = {}
    for word in text.split():
        hits = [name for name, values in options.items() if word in values]
        if not hits:
            raise ValueError(f'unknown "{word}"')
        if hits[0] in sel:
            raise ValueError(f'two values for {hits[0]}')
        sel[hits[0]] = word
    sel.setdefault('speed', 'slow')
    missing = [name for name in options if name not in sel]
    if missing:
        raise ValueError('missing ' + ', '.join(missing))
    return sel


def default_shapes_file():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('policy_deploy_toolkit'), 'examples', 'kyon_eetracker', 'ee_shapes.yaml')
    except Exception:
        return ''


def check():
    """Timeline boundaries on a two-sample table."""
    shapes = {'timing': {'hold_s': 2.0, 'move_s': 2.0, 'laps': 2, 'speeds_mps': [0.1, 0.25]},
              'home': {'pos': [[0.0, 0.0, 1.0]] * 2, 'quat_wxyz': [[1.0, 0.0, 0.0, 0.0]] * 2}}
    entry = {'perimeter_m': 0.4, 'left': {'pos': [[0.1, 0.0, 1.0], [0.0, 0.1, 1.0]], 'quat_wxyz': [[1.0, 0.0, 0.0, 0.0]] * 2}}
    p, _ = shape_waypoint(shapes, entry, 0, 1.0)
    assert np.allclose(p, [0.0, 0.0, 1.0]), 'hold at home'
    p, _ = shape_waypoint(shapes, entry, 0, 3.0)
    assert np.allclose(p, [0.05, 0.0, 1.0]), 'min-jerk blend midpoint'
    p, _ = shape_waypoint(shapes, entry, 0, 4.0)
    assert np.allclose(p, [0.1, 0.0, 1.0]), 'shape start after the blend'
    p, _ = shape_waypoint(shapes, entry, 0, 5.0)
    assert np.allclose(p, [0.05, 0.05, 1.0]), 'quarter lap at 0.1 m/s'
    total = 2.0 + sum(2 * 2.0 + 2 * 0.4 / v for v in (0.1, 0.25))
    assert shape_waypoint(shapes, entry, 0, total - 1e-6) is not None, 'running until the end'
    assert shape_waypoint(shapes, entry, 0, total + 1e-6) is None, 'finished'
    slow_total = 2.0 + 2 * 2.0 + 2 * 0.4 / 0.1
    assert shape_waypoint(shapes, entry, 0, slow_total - 1e-6, [0.1]) is not None, 'slow only: running'
    assert shape_waypoint(shapes, entry, 0, slow_total + 1e-6, [0.1]) is None, 'slow only: finished'

    options = {'size': ['small', 'big'], 'plane': ['xz'], 'shape': ['circle'], 'speed': ['slow', 'fast'],
               'arm': ['left', 'right']}
    assert parse_shape_command('right circle xz big', options) == \
        {'size': 'big', 'plane': 'xz', 'shape': 'circle', 'speed': 'slow', 'arm': 'right'}, 'any order, slow default'
    for bad in ('big xz circle', 'big xz circle left fats', 'big small xz circle left'):
        try:
            parse_shape_command(bad, options)
            raise AssertionError(f'accepted "{bad}"')
        except ValueError:
            pass
    print('keyboard_teleop check: ok')


def main():
    import rclpy
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Float64MultiArray

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
    # velocity and posture go to every node; EE references to ee_node
    nodes = param('nodes', ['policy_deploy_node', 'arms_policy_node'])
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
        shape_options.update(speed=list(speed_values), arm=['left', 'right'])
    # running shape: (table entry, arm index, speeds, start wall time), None when idle
    shape_run = None
    # typed command after ':', None in single-key mode
    line = None

    def shapes_home():
        return [(np.array(shapes['home']['pos'][i]), np.array(shapes['home']['quat_wxyz'][i])) for i in range(2)]

    def run_command(text):
        nonlocal shape_run, goal
        if text.strip() == 'stop':
            if shape_run is not None:
                shape_run, goal = None, shapes_home()
                print('[shape] stopped, goals to the shape home')
            return
        if shapes is None or home is None:
            print('[shape] needs the shape tables and a measured EE pose')
            return
        if shape_run is not None:
            print('[shape] running: ":stop" first')
            return
        try:
            sel = parse_shape_command(text, shape_options)
        except ValueError as e:
            print(f'[shape] {e}; usage: <{"|".join(shape_options.get("size", []))}> <{"|".join(shape_options.get("plane", []))}>'
                  f' <{"|".join(shape_options.get("shape", []))}> <left|right> [slow|fast|both]')
            return
        entry = next(e for e in shapes['shapes'] if all(e[k] == sel[k] for k in ('size', 'plane', 'shape')))
        shape_run = (entry, ('left', 'right').index(sel['arm']), speed_values[sel['speed']], time.monotonic())
        print(f'[shape] start: {sel["shape"]}, {sel["size"]} (r {entry["radius_m"]} m), plane {sel["plane"]},'
              f' speed {sel["speed"]}, {sel["arm"]} arm')

    vel_pubs = [node.create_publisher(Twist, f'/{n}/commands/base_velocity', 1) for n in nodes]
    posture_pubs = [node.create_publisher(Float64MultiArray, f'/{n}/commands/base_posture/raw', 1) for n in nodes]
    ee_pub = node.create_publisher(Float64MultiArray, f'/{ee_node}/commands/ee_pose/reference', 1)

    # measured EE poses [left pos, quat, right pos, quat, h] from the arms node
    ee = {'measured': None}
    node.create_subscription(Float64MultiArray, f'/{ee_node}/ee_measured',
                             lambda msg: ee.__setitem__('measured', np.array(msg.data)), 1)

    direction = (0, 0, 0)
    posture = list(nominal)
    ee_arm = 2
    home = goal = target = None

    def velocity():
        raw = (direction[0] * speed, direction[1] * speed, direction[2] * turn)
        return [min(max(v, -m), m) for v, m in zip(raw, vel_max)]

    def publish_vel(vel):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = vel
        for pub in vel_pubs:
            pub.publish(msg)

    def measured_poses():
        m = ee['measured']
        return [(m[7 * i:7 * i + 3].copy(), m[7 * i + 3:7 * i + 7].copy()) for i in range(2)]

    def ee_nudge(axis, sign):
        for i in ((0, 1) if ee_arm == 2 else (ee_arm,)):
            pos, quat = goal[i]
            if axis < 3:
                pos[axis] += sign * ee_step
            else:
                goal[i] = (pos, qmul(quat_from_rotvec(sign * ee_angle_step * np.eye(3)[axis - 3]), quat))
        print(f'[ee] {ARM_NAMES[ee_arm]} goal {AXIS_NAMES[axis]} {"+" if sign > 0 else "-"}')

    def ee_print():
        for i, (pos, quat) in enumerate(measured_poses()):
            g, gq = goal[i]
            ang = np.linalg.norm(rotvec_from_quat(qmul(gq, qconj(quat))))
            print(f'[ee] {ARM_NAMES[i]:>5s}  goal ({g[0]:+.3f}, {g[1]:+.3f}, {g[2]:.3f})'
                  f'  measured ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f})  ori err {ang:.3f} rad')
        print(f'[ee] pelvis height {ee["measured"][14]:.3f} m')

    if not sys.stdin.isatty():
        node.get_logger().error('keyboard_teleop needs a terminal: run it with ros2 run')
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

            # republish every tick: the deploy node drops commands older than command_timeout_s
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
                elif key in EE_KEYS or key in 'adf':
                    if home is None:
                        print(f'[ee] no EE pose yet from /{ee_node}/ee_measured')
                    elif shape_run is not None and key != 'f':
                        print('[ee] shape running: ":stop" first')
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
                    continue
                else:
                    direction = (0, 0, 0)
                vel = velocity()
                print(f'speed {speed:.2f} turn {turn:.2f}  '
                      f'vel [{vel[0]:+.2f} {vel[1]:+.2f} {vel[2]:+.2f}]  '
                      f'posture [{posture[0]:+.2f} {posture[1]:+.2f} {posture[2]:.3f}]')
            publish_vel(velocity())
            for pub in posture_pubs:
                pub.publish(Float64MultiArray(data=posture))
            if shape_run is not None:
                # waypoint for the next control step; the other arm holds the shape home
                entry, arm, run_speeds, start = shape_run
                wp = shape_waypoint(shapes, entry, arm, time.monotonic() - start + DT, run_speeds)
                goal = shapes_home()
                if wp is None:
                    shape_run = None
                    print('[shape] done')
                else:
                    goal[arm] = wp
            if home is not None:
                target = [step_towards(*target[i], *goal[i], ee_speed * DT, ee_ang_speed * DT) for i in range(2)]
                ee_pub.publish(Float64MultiArray(data=np.concatenate(
                    [np.r_[target[i][0], target[i][1], goal[i][0], goal[i][1]] for i in range(2)]).tolist()))
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        publish_vel([0.0, 0.0, 0.0])
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    if '--check' in sys.argv:
        check()
    else:
        main()
