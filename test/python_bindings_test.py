"""Exercise NumPy conversion, callbacks, and inference with a bundled policy."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
import policy_deploy_toolkit as pdt

robot = pdt.RobotInfo()
np.testing.assert_array_equal(robot.base_T_imu, np.eye(4))
pose = np.eye(4)
pose[0, 3] = 0.5
robot.body_pose_in_base = lambda name: pose if name == "wheel" else np.eye(4)
np.testing.assert_array_equal(robot.body_pose_in_base("wheel"), pose)
robot.body_pose_in_base = None
assert robot.body_pose_in_base is None

inputs = pdt.Inputs()
np.testing.assert_array_equal(inputs.w_R_imu, [0, 0, 0, 1])
inputs.w_R_imu = [0, 0, 1, 1]
np.testing.assert_allclose(inputs.w_R_imu, [0, 0, 2**-0.5, 2**-0.5])
try:
    inputs.w_R_imu = np.zeros(4)
except ValueError:
    pass
else:
    raise AssertionError("zero quaternion accepted")
inputs.w_R_imu = [0, 0, 0, 1]

example = Path(sys.argv[2]) / "examples/kyon_full_locomotion_flat"
metadata = example / "deploy_metadata.yaml"
# Read the top-level joint list without introducing a PyYAML test dependency.
joint_list = metadata.read_text().split("\njoint_names:\n", 1)[1].split("\nfixed_joints:", 1)[0]
robot.joint_names = [line.removeprefix("- ") for line in joint_list.splitlines()]
n = len(robot.joint_names)
robot.joint_pos_min = np.full(n, -np.pi)
robot.joint_pos_max = np.full(n, np.pi)
robot.joint_vel_max = np.full(n, 10.0)
policy = pdt.OnnxPolicy(str(example / "policy.onnx"), str(metadata), "policy", robot)
info = policy.policyInfo()
assert info.joint_names == robot.joint_names
assert policy.command_specs()
assert isinstance(policy.sensor_specs(), list)
assert isinstance(policy.recurrent_state_names(), list)
inputs.command = policy.default_commands()
for name, command in inputs.command.items():
    ok, sanitized, reason = policy.sanitize_command(inputs, name, command)
    assert ok, reason
    np.testing.assert_allclose(sanitized, command)
ok, _, reason = policy.sanitize_command(inputs, "unknown", np.zeros(3))
assert not ok and reason
inputs.q = np.array(info.joint_default_pos)
inputs.q_ref = inputs.q.copy()
for field in ("v", "tau", "v_ref", "tau_ref", "k", "d"):
    setattr(inputs, field, np.zeros(n))
inputs.last_action = np.zeros(info.action_size)
outputs = pdt.Outputs(info.action_size, n)
for step in range(2):
    inputs.step = step
    inputs.time_sec = step * info.control_dt
    assert policy.run(inputs, outputs)
    assert outputs.raw_action.shape == (info.action_size,)
    assert outputs.q_des.shape == (n,)
    assert outputs.ctrl_mode.dtype == np.uint8
    assert np.all(np.isfinite(outputs.raw_observation))
    assert np.all(np.isfinite(outputs.raw_action))
    inputs.last_action = outputs.raw_action.copy()
