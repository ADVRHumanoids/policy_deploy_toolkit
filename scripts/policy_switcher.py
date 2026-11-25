#!/usr/bin/env python3
"""Run several policies on Kyon through XBot2 and switch between them live, without a gap in control.

The lab's policy_deploy_node runs one policy per process, so switching means stopping one node and starting another:
while no node runs, XBot2 only holds the last joint references, and upright on point feet the robot falls. This
node loads all the policies up front and runs the selected one every control period, with the toolkit's own
policy code (policy_deploy_toolkit Python bindings: same observations, actions and gains as policy_deploy_node,
from each policy's deploy_metadata.yaml). Switching happens between two control periods: the new policy starts
from a clean state (no previous action, its own gains), like a freshly started policy_deploy_node.

Keys (in this terminal):
    1..9           switch to that policy (listed at start)      0   release: stop commanding (XBot2 holds)
    up / down      forward speed +/- 0.1 m/s                     space   zero the velocity command
    left / right   turn +/- 0.1 rad/s        a / d   sideways +/- 0.1 m/s        q   quit

With xbot2-core running (and MuJoCo in simulation), no other policy node:
    ros2 run policy_deploy_toolkit policy_switcher.py --policy NAME=DIR --policy NAME=DIR ...
    ros2 run policy_deploy_toolkit policy_switcher.py --policies_dir DIR   # the four kyon ferk policy folders in DIR
    ros2 launch policy_deploy_toolkit kyon_ferk_policies.launch.xml        # the same, driven by ROS topics
Each DIR holds a policy's policy.onnx and deploy_metadata.yaml. Needs the toolkit's Python bindings
(BUILD_PYTHON_BINDINGS=ON, see the README); a separate bindings build can be given with POLICY_DEPLOY_TOOLKIT_PYTHONPATH.

--ros_input (used by kyon_ferk_policies.launch.xml) takes the commands from ROS topics instead of this
terminal: std_msgs/String on /policy_switcher/select (a policy name, its number, or "release"/"0") and
geometry_msgs/Twist on /policy_deploy_node/commands/base_velocity (the topic of policy_deploy_node, so joystick,
GUI and arrow_teleop.py work unchanged; dropped after 0.5 s without messages, like policy_deploy_node). The
active policy is published on /policy_switcher/active.
Only for policies with the lab's standard terms (the ones policy_deploy_node can run).
"""

import argparse
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

# toolkit Python bindings: installed with BUILD_PYTHON_BINDINGS=ON, or a separate build given by
# POLICY_DEPLOY_TOOLKIT_PYTHONPATH (e.g. cmake -DBUILD_ROS_NODE=OFF -DBUILD_PYTHON_BINDINGS=ON -B <dir>: <dir>/python)
if os.environ.get("POLICY_DEPLOY_TOOLKIT_PYTHONPATH"):
    sys.path.insert(0, os.environ["POLICY_DEPLOY_TOOLKIT_PYTHONPATH"])

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.qos import QoSDurabilityPolicy, QoSProfile  # noqa: E402
from rclpy.wait_for_message import wait_for_message  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from xbot2_interface import pyxbot2_interface as xb  # noqa: E402

import policy_deploy_toolkit as pdt  # noqa: E402

# the kyon ferk policies: name -> folder (in the package's examples/, or in --policies_dir)
RELEASE_FOLDERS = [
    ("ferk4", "kyon_ferk_locomotion_flat"),
    ("standup", "kyon_ferk_standup"),
    ("biped", "kyon_ferk_biped_locomotion_flat"),
    ("getdown", "kyon_ferk_getdown"),
]
COMMAND_TIMEOUT_S = 0.5  # --ros_input: like policy_deploy_node, a velocity command older than this is dropped
STEP = 0.1
LIMITS = np.array([1.0, 0.5, 1.0])  # vx, vy, wz
GAINS_SECONDS = 1.0  # like policy_deploy_node: send each policy's stiffness/damping during its first second
CTRL_POS_VEL_IMPEDANCE = 25  # XBot2 control mode the toolkit uses for actuated joints (position+velocity+stiffness+damping)


def as_matrix(pose):
    """XBot2 poses (pyaffine3.Affine3) as a 4x4 array."""
    return np.asarray(pose.matrix(), dtype=float)


class Switcher:
    def __init__(self, node, policies, imu_name):
        qos = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        _, urdf = wait_for_message(String, node, "/xbotcore/robot_description", qos_profile=qos, time_to_wait=10.0)
        _, srdf = wait_for_message(String, node, "/xbotcore/robot_description_semantic", qos_profile=qos, time_to_wait=10.0)
        if urdf is None or srdf is None:
            raise RuntimeError("no /xbotcore/robot_description: is xbot2-core running?")
        self.robot = xb.RobotInterface2(urdf.data, srdf.data)
        imus = self.robot.getImu()  # name -> ImuSensor
        if imu_name not in imus:
            raise RuntimeError(f"IMU '{imu_name}' not found (robot has {list(imus)})")
        self.imu = imus[imu_name]
        self.wait_sense()

        # robot description for the policies, as policy_deploy_node builds it
        info = pdt.RobotInfo()
        info.joint_names = list(self.robot.getVNames())
        info.base_T_imu = as_matrix(self.robot.getPose(imu_name, "base_link"))
        qmin, qmax = self.robot.getJointLimits()
        info.joint_pos_min, info.joint_pos_max = np.asarray(qmin, float), np.asarray(qmax, float)
        info.joint_vel_max = np.asarray(self.robot.getVelocityLimits(), float)
        self.robot_info = info
        self.vnames = info.joint_names
        # control modes are per joint, the policy outputs are per velocity index (as in policy_deploy_node)
        self.vid_to_jid = [-1] * len(self.vnames)
        for jid, jname in enumerate(self.robot.getJointNames()):
            vid = self.robot.getVIndexFromVName(jname)
            if vid >= 0:
                self.vid_to_jid[vid] = jid
        self.ctrl_mode = np.zeros(self.robot.getJointNum(), dtype=np.uint8)

        self.policies = {}  # name -> (onnx path, metadata path)
        for name, exported in policies:
            onnx, meta = exported / "policy.onnx", exported / "deploy_metadata.yaml"
            policy = pdt.OnnxPolicy(str(onnx), str(meta), "", info)  # load once now: fail early on a bad policy
            pinfo = policy.policyInfo()
            self.policies[name] = (str(onnx), str(meta))
            print(f"[INFO] {len(self.policies)}: {name:8s} {pinfo.action_size} actions, control period "
                  f"{pinfo.control_dt * 1000:.0f} ms ({exported})")
        self.active = None  # name of the running policy
        self.command = np.zeros(3)

    def wait_sense(self):
        while rclpy.ok() and not self.robot.sense():
            time.sleep(0.001)

    def activate(self, name):
        """Start a policy from a clean state, like a fresh policy_deploy_node."""
        onnx, meta = self.policies[name]
        self.policy = pdt.OnnxPolicy(onnx, meta, "", self.robot_info)
        self.pinfo = self.policy.policyInfo()
        self.inputs = pdt.Inputs()
        self.inputs.command = self.policy.default_commands()
        self.inputs.last_action = np.zeros(self.pinfo.action_size)
        self.outputs = pdt.Outputs(self.pinfo.action_size, len(self.vnames))
        self.command_name = next((n for n in self.inputs.command if len(self.inputs.command[n]) == 3), None)
        # policy joints the policy does not act on (e.g. the front legs of the 2-leg policy): in training they were
        # held at the default pose by their PD; the toolkit leaves them uncontrolled, so hold them here
        robot_ids = np.asarray(self.pinfo.joint_id_policy_to_robot)
        self.hold_ids = robot_ids  # filtered after the first run, when the control modes are known
        self.hold_q = np.asarray(self.pinfo.joint_default_pos, float)
        self.hold_k = np.asarray(self.pinfo.stiffness, float)
        self.hold_d = np.asarray(self.pinfo.damping, float)
        self.t_active = time.monotonic()
        self.active = name

    def step(self):
        """One control period of the active policy (same order as policy_deploy_node's timer callback)."""
        self.wait_sense()
        r, inputs, outputs = self.robot, self.inputs, self.outputs
        inputs.q_ref = np.asarray(r.positionToMinimal(r.getPositionReferenceFeedback()), float)
        inputs.w_R_imu = self.imu_quaternion()
        inputs.last_action = outputs.raw_action.copy()
        if self.command_name is not None:
            commands = dict(inputs.command)
            ok, sanitized, _ = self.policy.sanitize_command(inputs, self.command_name, self.command.copy())
            commands[self.command_name] = sanitized if ok else np.zeros(3)
            inputs.command = commands
        inputs.time_sec = time.monotonic() - self.t_active
        inputs.q = np.asarray(r.getJointPositionMinimal(), float)
        inputs.v = np.asarray(r.getJointVelocity(), float)
        inputs.tau = np.asarray(r.getJointEffort(), float)
        inputs.v_ref = np.asarray(r.getVelocityReferenceFeedback(), float)
        inputs.tau_ref = np.asarray(r.getEffortReferenceFeedback(), float)
        inputs.k = np.asarray(r.getStiffness(), float)
        inputs.d = np.asarray(r.getDamping(), float)
        inputs.imu_omega = np.asarray(self.imu.getAngularVelocity(), float)
        inputs.imu_acc = np.asarray(self.imu.getLinearAcceleration(), float)
        if inputs.step == 0:
            outputs.q_des = inputs.q_ref.copy()
        self.policy.run(inputs, outputs)
        inputs.step = inputs.step + 1
        if self.hold_ids is not None:
            held = np.asarray(outputs.ctrl_mode)[self.hold_ids] == 0
            if inputs.step == 1 and held.any():
                print(f"\n[INFO] {self.active}: holding {int(held.sum())} joints it does not act on at its default pose")
            self.hold_ids, self.hold_q = self.hold_ids[held], self.hold_q[held]
            self.hold_k, self.hold_d = self.hold_k[held], self.hold_d[held]
            if not len(self.hold_ids):
                self.hold_ids = None
        if self.hold_ids is not None:
            q_des, k_des, d_des, modes = (np.array(outputs.q_des), np.array(outputs.k_des),
                                          np.array(outputs.d_des), np.array(outputs.ctrl_mode))
            q_des[self.hold_ids], k_des[self.hold_ids], d_des[self.hold_ids] = self.hold_q, self.hold_k, self.hold_d
            modes[self.hold_ids] = CTRL_POS_VEL_IMPEDANCE
            outputs.q_des, outputs.k_des, outputs.d_des, outputs.ctrl_mode = q_des, k_des, d_des, modes

        for vi, jid in enumerate(self.vid_to_jid):
            if jid >= 0:
                self.ctrl_mode[jid] = outputs.ctrl_mode[vi]
        r.setControlMode(self.ctrl_mode)
        r.setPositionReference(r.minimalToPosition(outputs.q_des))
        r.setVelocityReference(outputs.v_des)
        r.setEffortReference(outputs.tau_des)
        if time.monotonic() - self.t_active < GAINS_SECONDS:
            r.setStiffness(outputs.k_des)
            r.setDamping(outputs.d_des)
        r.move()

    def imu_quaternion(self):
        """IMU orientation as [x, y, z, w]: XBot2's getOrientation (Eigen quaternion coefficients) and the
        toolkit's Inputs.w_R_imu use the same order."""
        return np.asarray(self.imu.getOrientation(), float).reshape(4)

    def release(self):
        self.active = None

    def pitch_deg(self):
        """Body pitch from the IMU: 0 = flat (ferk), ~85 = upright on the hind legs."""
        from scipy.spatial.transform import Rotation

        w_R_imu = Rotation.from_quat(self.imu_quaternion()).as_matrix()
        w_R_base = w_R_imu @ self.robot_info.base_T_imu[:3, :3].T
        g = w_R_base.T @ np.array([0.0, 0.0, -1.0])
        return float(np.degrees(np.arctan2(-g[0], -g[2])))


def read_key(timeout):
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        return {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(sys.stdin.read(2))
    return ch


def run_ros_input(node, sw, names, dt, cmd_topic):
    """Main loop driven by ROS topics instead of the keyboard (for ros2 launch, joystick, GUI)."""
    from geometry_msgs.msg import Twist

    active_pub = node.create_publisher(String, "/policy_switcher/active", 10)
    last_cmd = {"t": 0.0}

    def publish_active():
        active_pub.publish(String(data=sw.active or "released"))
        print(f"[INFO] active policy: {sw.active or 'released'}", flush=True)

    def on_select(msg):
        choice = msg.data.strip()
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            choice = names[int(choice) - 1]
        if choice == sw.active:
            return  # already running: a repeated message must not restart it mid-motion
        if choice in ("0", "release", "released"):
            sw.release()
        elif choice in names:
            sw.activate(choice)
        else:
            node.get_logger().warn(f"unknown policy '{choice}' (known: {names}, or 'release')")
            return
        publish_active()

    def on_cmd(msg):
        sw.command = np.clip(np.array([msg.linear.x, msg.linear.y, msg.angular.z]), -LIMITS, LIMITS)
        last_cmd["t"] = time.monotonic()

    node.create_subscription(String, "/policy_switcher/select", on_select, 10)
    node.create_subscription(Twist, cmd_topic, on_cmd, 1)
    print("[INFO] policies: " + "  ".join(f"{i + 1}={n}" for i, n in enumerate(names)), flush=True)
    print('[INFO] switch with: ros2 topic pub --once /policy_switcher/select std_msgs/msg/String "data: standup"', flush=True)
    publish_active()
    next_status = time.monotonic()
    try:
        while rclpy.ok():
            t0 = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0.0)
            if t0 - last_cmd["t"] > COMMAND_TIMEOUT_S:
                sw.command = np.zeros(3)
            if sw.active:
                sw.step()
            if t0 >= next_status:  # status every 2 s
                pitch = f"pitch {sw.pitch_deg():5.1f} deg, " if sw.active else ""  # the IMU is read only when active
                print(f"[STATUS] {sw.active or 'released'}: {pitch}command "
                      f"{sw.command[0]:+.1f} {sw.command[1]:+.1f} {sw.command[2]:+.1f}", flush=True)
                next_status = t0 + 2.0
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


def main():
    p = argparse.ArgumentParser(description="Switch between policies live on Kyon through XBot2.")
    p.add_argument("--policy", action="append", default=[], metavar="NAME=EXPORTED_DIR",
                   help="A policy: name and its exported folder (policy.onnx + deploy_metadata.yaml). Repeatable; keys 1..9 in order.")
    p.add_argument("--policies_dir", type=Path, default=None,
                   help="Load ferk4, standup, biped, getdown from their folders in this directory (e.g. the package's examples).")
    p.add_argument("--ros_input", action="store_true", help="Take policy selection and velocity from ROS topics (no keyboard).")
    p.add_argument("--cmd_topic", default="/policy_deploy_node/commands/base_velocity")
    p.add_argument("--imu", default="imu_link")
    p.add_argument("--start", default=None, help="Policy to start right away (default: none, wait for a key).")
    p.add_argument("--script", default=None,
                   help="Non-interactive test: 'name:seconds,...' runs each policy for that long, then exits.")
    # a ros2 launch <node> appends --ros-args ...: keep only our arguments
    args = p.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    policies = []
    if args.policies_dir:
        policies += [(n, args.policies_dir / folder) for n, folder in RELEASE_FOLDERS]
    for item in args.policy:
        name, path = item.split("=", 1)
        policies.append((name, Path(path)))
    if not policies or len(policies) > 9:
        p.error("give 1..9 policies (--policy NAME=DIR or --policies_dir DIR)")

    rclpy.init()
    node = rclpy.create_node("policy_switcher")
    sw = Switcher(node, policies, args.imu)
    names = list(sw.policies)
    dt = min(pdt.OnnxPolicy(*sw.policies[n], "", sw.robot_info).policyInfo().control_dt for n in names)

    def status():
        print(f"\r[{sw.active or 'released'}] forward {sw.command[0]:+.1f}  sideways {sw.command[1]:+.1f}  "
              f"turn {sw.command[2]:+.1f}      ", end="", flush=True)

    if args.script:  # e.g. "standup:8,getdown:8"
        for item in args.script.split(","):
            name, seconds = item.split(":")
            sw.activate(name)
            print(f"[SCRIPT] {name} for {seconds} s")
            t_start = time.monotonic()
            next_print = t_start
            while time.monotonic() < t_start + float(seconds):
                t0 = time.monotonic()
                sw.step()
                if t0 >= next_print:
                    print(f"[SCRIPT]   t={t0 - t_start:4.1f} s  pitch {sw.pitch_deg():6.1f} deg")
                    next_print += 1.0
                time.sleep(max(0.0, dt - (time.monotonic() - t0)))
        rclpy.shutdown()
        return

    if args.start:
        sw.activate(args.start)
    if args.ros_input:
        run_ros_input(node, sw, names, dt, args.cmd_topic)
        return
    print(__doc__.split("\n\n")[2])
    print("Policies: " + "  ".join(f"{i + 1}={n}" for i, n in enumerate(names)))
    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        status()
        while rclpy.ok():
            t0 = time.monotonic()
            key = read_key(0.0)
            if key and key.isdigit():
                i = int(key)
                if i == 0:
                    sw.release()
                elif i <= len(names) and names[i - 1] != sw.active:
                    sw.activate(names[i - 1])
            elif key == "up":
                sw.command[0] += STEP
            elif key == "down":
                sw.command[0] -= STEP
            elif key == "left":
                sw.command[2] += STEP
            elif key == "right":
                sw.command[2] -= STEP
            elif key == "a":
                sw.command[1] += STEP
            elif key == "d":
                sw.command[1] -= STEP
            elif key == " ":
                sw.command[:] = 0.0
            elif key == "q":
                break
            sw.command = np.round(np.clip(sw.command, -LIMITS, LIMITS), 2)
            if key:
                status()
            if sw.active:
                sw.step()
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
