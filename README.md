# policy_deploy_toolkit

Minimal ONNX policy deployment for XBot2.

The package wraps an exported policy as a small runtime pipeline:

1. Load an ONNX model with ONNX Runtime.
2. Load the matching deployment metadata from YAML.
3. Map policy joint names to the robot joints exposed by `RobotInterface`.
4. Build observations from the live robot state, commands, IMU data, and previous action.
5. Run the policy and translate the output into XBot2 position, velocity, effort, stiffness, damping, and control-mode commands.

The metadata drives most of the wiring: observation terms, action terms, joint ordering, default joint positions, gains, command dimensions, and the control period. The executable expects the robot configuration to be available from the ROS 2 parameter tree under `xbotcore/`.

## Build

From this package directory:

```bash
cmake -S . -B build
cmake --build build
```

Or with forest:

```bash
forest grow xbot2_policy_toolkit -j4
```

## Optional Python bindings

Python bindings are disabled by default. To enable them, install pybind11 headers
and CMake configuration (for example, `pybind11-dev` on Ubuntu), Python development
headers, `ament_cmake_python`, and NumPy for the selected Python interpreter. Then:

```bash
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build
cmake --install build
source install/share/policy_deploy_toolkit/local_setup.bash
python3 -c "import policy_deploy_toolkit"
```

With colcon, pass `--cmake-args -DBUILD_PYTHON_BINDINGS=ON`. Set
`-DPython3_EXECUTABLE=/path/to/python3` to select an interpreter. The extension
installs under the prefix's Python package directory; sourcing the prefix adds
it to `PYTHONPATH`.

The Python package contains the internal `_bindings` extension and the MuJoCo
runner. Its top-level API exposes `OnnxPolicy`, `RobotInfo`, `PolicyInfo`, `Inputs`, `Outputs`,
and command/sensor specifications. Eigen vectors and matrices use NumPy arrays.
`RobotInfo.base_T_imu` and the optional `body_pose_in_base(name)` callback use
4×4 transformation matrices. `Inputs.w_R_imu` uses `[x, y, z, w]` quaternion
coefficients and normalizes assigned values. The Python defaults use identity
orientation and zero IMU velocity/acceleration.

```python
import numpy as np
import policy_deploy_toolkit as pdt

robot = pdt.RobotInfo()
robot.joint_names = joint_names  # robot joint order
robot.joint_pos_min = joint_position_lower_limits
robot.joint_pos_max = joint_position_upper_limits
robot.joint_vel_max = joint_velocity_limits
policy = pdt.OnnxPolicy(model_path, metadata_path, "policy", robot)
info = policy.policyInfo()
inputs = pdt.Inputs()
inputs.command = policy.default_commands()
inputs.last_action = np.zeros(info.action_size)
outputs = pdt.Outputs(info.action_size, len(robot.joint_names))
# Populate inputs.q, v, tau, q_ref, v_ref, tau_ref, k, d, and IMU/sensor
# fields from the robot before each call; advance step and time_sec as well.
policy.run(inputs, outputs)
inputs.last_action = outputs.raw_action.copy()
```

`run(inputs, outputs)` fills the supplied output object and returns a boolean.
`sanitize_command(inputs, name, raw_command)` returns `(ok, sanitized, reason)`.
STL containers are converted to Python copies: to update one command, modify a
local dictionary and assign it back to `inputs.command`.

## Evaluate directly in MuJoCo

Build with `BUILD_PYTHON_BINDINGS=ON` as above and install `xbot2_py_mujoco`
(from the `xbot2_mujoco` repository's `python` directory) in the selected Python
environment, along with its dependencies. Run an existing robot/scene MJCF with:

```bash
python3 -m policy_deploy_toolkit.simulate \
    --mjcf /path/to/scene.xml \
    --policy examples/kyon_full_locomotion_flat/policy.onnx \
    --command base_velocity=0.3,0,0
```

The runner uses `SimulatorWrapper` from `xbot2_py_mujoco` with an in-process ONNX
bridge. It requires no XBot2 transport, RobotInterface, or ROS node; the default
XBot2 bridge is imported only when selected. It reads `deploy_metadata.yaml`
beside the policy unless `--metadata` is supplied. `--base-body` and `--imu-site`
default to `base_link` and `imu_link`; the site must have a gyro and an
accelerometer. The MJCF must already contain the robot, actuators, and terrain.

Joints start at the policy defaults. The runner maps joints by name and updates
native actuator controls, stiffness, damping, and feedforward effort once per
policy period, retaining MuJoCo actuator dynamics and limits. It supports one
unit-gear position or `intvelocity` actuator per controlled scalar joint;
velocity actions with nonzero stiffness require `intvelocity`. Other actuated
joints on the robot hold their initial positions with their existing gains.
MuJoCo's timestep must divide the metadata's `ctrl_dt` exactly. Joint speed
limits default to unbounded because MJCF has no joint velocity limit; use
`--joint-velocity-limit` to supply a uniform limit for action processing.

Use `--headless --duration 10` for a finite rollout at maximum speed, recording
to `rollout.mp4`. Headless mode ignores `--target-rtf` and never sleeps for pacing.
Video encoding requires `imageio` and `imageio-ffmpeg` (`pip install 'imageio[ffmpeg]'`).
Choose a path with `--video /path/to/rollout.mp4`, a frame rate with
`--video-fps 30`, and even image dimensions with `--video-size 640 480`.
Frames follow simulation time, so playback speed stays consistent regardless
of computation speed. Frames are streamed to disk and the video is finalized
on exit, including Ctrl+C. Use `MUJOCO_GL=egl` for offscreen rendering without
a display, or `MUJOCO_GL=osmesa` for software rendering when available; set it
before starting Python.

Add `--profile` to print a timing breakdown on exit and throughput updates every
5 wall-clock seconds. RTF uses a rolling 5-second window of loop wall time
(or the available shorter interval), excluding setup and finalization. The final
report keeps the last loop window. The report includes call counts,
total time, wall-time percentage, and mean/maximum latency for physics, policy
inference, height scans, rendering/readback, video writes, setup, and video
finalization. Timings measure wall time; video writes include waits on FFmpeg's
pipe, while finalization includes waiting for the encoder to finish. Stage totals
and wall-time percentages still cover the full run. Profiling adds
timer overhead, especially for small physics steps.

In viewer mode, `--target-rtf 2` runs at twice real time; `--video` also enables
recording. Repeat `--command NAME=V1,V2,...` to override commands.
Use `--track-body base_link` to make both the viewer and video cameras follow a named body.
Set `--camera-distance 2.5` to place either camera 2.5 meters from its target;
omitting it preserves MuJoCo's default distance.
Use `--geom-group-blacklist 0 2` to hide geometry groups in both the viewer and
video. Groups range from 0 to 5; the option can be repeated. This only controls
visibility, leaving collisions and terrain scanning unchanged.
Commands pass through the policy's sanitization. `--obs-group` overrides the
metadata's observation group. `MujocoPolicyRunner` in the `simulate` module
also exposes `step()` for one control period and its `model`, `data`, `inputs`,
and `outputs` for Python experiments.

### Use from your own script

`Simulation` handles stepping, the viewer, video recording, pacing, profiling,
and cleanup. `Simulation.from_cli()` inherits every simulator CLI option:

```python
from policy_deploy_toolkit.simulate import Simulation

with Simulation.from_cli() as sim:
    while sim.running:
        sim.set_velocity_command(vx=0.3, vy=0.0, wz=0.0)
        if sim.data.time >= 5.0:
            sim.reset()
        sim.step()
```

Run that script with the same arguments as the module, including `--mjcf`,
`--policy`, `--headless`, `--duration`, camera options, and video options.
To extend the CLI, call `create_arg_parser()`, add your own arguments, then
construct `Simulation(parser.parse_args())`. Alternatively,
`add_arguments(parser)` adds simulator options to an existing parser.
See `examples/scripted_rollout.py` for an executable example.

The example uses action definitions from the main Python library:

```python
from policy_deploy_toolkit.rollout_actions import (
    ActionSequence, VelocityCommand, ApplyForce, EmptyAction, Respawn,
)

actions = [
    VelocityCommand(vx=1.0, duration=2.0),
    VelocityCommand(wz=1.0, duration=2.0),
    VelocityCommand(vx=1.0, duration=10.0, sub_actions=[
        ApplyForce(body='base_link', fy=1000.0, duration=0.2),
        EmptyAction(duration=0.5),
        ApplyForce(body='base_link', fy=-1000.0, duration=0.2),
    ]),
    Respawn(position=[0.0, 0.0, 0.5], yaw=0.0),
]
with Simulation.from_cli() as sim:
    sim.run(ActionSequence(actions))
```

Siblings run sequentially, and `start_offset` adds a delay after the preceding
sibling (or before the first sibling). Child sequences run while their parent
remains active and must fit within its duration. In this example, the two forces
start at 4.0 s and 4.7 s while the forward command stays active until 14.0 s.
`EmptyAction` changes no commands. `VelocityCommand` restores the previous named
command on completion, including nested overrides; `wz` is yaw rate in rad/s,
while `vx`/`vy` are m/s. Set `command_name=` for policies using another name.

`ApplyForce` uses the main simulator's timed wrench API, so overlapping forces
add and expire without clearing unrelated disturbances. It accepts `body`,
`fx`/`fy`/`fz` in N, optional `torque=[tx, ty, tz]` in Nm, and `frame='global'`
or `'local'` (also `'world'`/`'body'`). `Respawn` is instantaneous, takes world
position and yaw in radians, and clears disturbances through the simulator's
normal reset path. Action durations and offsets are finite, non-negative seconds;
velocity and force actions require positive durations.

`ActionSequence` uses session `elapsed_time`, so respawning does not restart the
timeline. Effects start on the next policy callback at or after their scheduled
time; late force starts use only the remaining duration. Fully missed timed
actions are skipped, and instantaneous actions execute once when their time is
crossed. At equal boundaries, children finish before their parent, and the next
sibling starts after cleanup. Definitions are immutable and copy `sub_actions`;
construct a new `ActionSequence` to reuse them in another rollout.

The active parent/child actions, parameters and remaining time appear in a small
top-left overlay in both the viewer and recorded video. Scripted forces draw
the same red arrows as `mjctl`; payload markers are shared as well. These cues
do not require a remote endpoint and remain visible when model geometry groups
are hidden. Custom scripts can set or clear the corner text with
`sim.set_status_text('...')` / `sim.set_status_text('')`. Viewer text requires
MuJoCo's `set_texts` viewer extension; video text uses Pillow. Force cues and
text are included from the first recorded frame.

The example stops when its sequence ends. `--duration` remains an optional upper
limit, and can truncate the sequence. The old periodic `--reset-every` logic is
replaced by explicit `Respawn` actions. `--speed`, `--force-body` (defaulting to
`--base-body`), and `--respawn-position` customize the example. Subclass library
`Action` and override `start(sim, remaining_duration)` to define more actions;
it may return a cleanup callback to run when that action finishes.

For callback control, use `sim.run(before_step=control)`, where `control(sim)`
runs before each policy step. Return `False` or call `sim.stop()` to end the
rollout. Call command updates and resets between steps on the simulation thread.

- `sim.set_command(name, values)` sets a named command with policy validation.
- `sim.set_commands({...})` updates several commands atomically.
- `sim.set_velocity_command(vx, vy, wz)` sets `base_velocity`; pass `name=` for
  policies using a different command name. `sim.commands` returns copies.
- `sim.reset()` restores the startup robot state and clears previous actions,
  observation history, command filters, and recurrent policy state. Commands
  are preserved by default; use `preserve_commands=False` for policy defaults.
- `sim.respawn(position=[x, y, z], quaternion=[w, x, y, z])` resets at a world
  pose; an explicit pose requires a floating root joint. Omitted pose components
  use their startup values. `sim.respawn()` is equivalent to `sim.reset()`.

`sim.data.time` and `sim.inputs.step` restart on reset. `sim.elapsed_time` and
`sim.runner.control_steps` count across all episodes, so duration limits and
video frame timing continue through resets. The context manager finalizes video
on normal completion, Ctrl+C, or an exception. Reset currently reloads the policy
because the bindings have no history-reset API, so it includes ONNX loading cost.
`sim.inputs` and `sim.outputs` are replaced on reset; access them through the
simulation instead of retaining references across episodes. The same command
and reset methods are available on the lower-level `MujocoPolicyRunner`.

Apply external forces and torques by body name, using world axes or the body's
rotating local axes:

```python
sim.apply_wrench('base_link', force=[100, 0, 0], frame='global')
sim.apply_wrench('base_link', force=[100, 0, 0], torque=[0, 0, 5], frame='local', duration=.25)
sim.clear_wrenches('base_link')  # One body
sim.clear_wrenches()             # All bodies
```

Force is in newtons and torque in newton-meters, applied at the body's center
of mass. Force and torque default to zero. Without `duration`, a call replaces
the persistent wrench on that body until cleared or reset/respawn. With a
positive duration in simulation seconds, it adds a timed wrench. Timed and
persistent wrenches sum. Local wrenches follow body orientation every physics
step. `world`/`body` are aliases for `global`/`local`.
Call these methods between control steps on the simulation thread. Managed
wrenches overwrite that body's `sim.data.xfrc_applied` row each step; raw forces
on other bodies remain untouched. `clear_wrenches()` without a name clears all
external body forces, including ones set through raw `MjData`. These methods are
also available on `MujocoPolicyRunner`. Wrenches are owned by the main
`SimulatorWrapper`, exposed as `sim.simulator`, rather than by `MjOnnxBridge`.

Every `mjctl` control is also available from `sim` and `sim.simulator`:

```python
sim.set_friction('.*_foot', [.8, .005, .001], regex=True)
sim.set_contact_parameters('base_link', solref=[.01, 1])
sim.set_body_properties('base_link', mass=25, com=[.02, 0, .08])
sim.add_payload('base_link', mass=5, position=[.2, 0, .1])
sim.set_joint_torque_limits('.*_knee', 40, regex=True)
sim.set_actuator_delay(.02)
sim.set_velocity_feedback_noise('.*_knee', .05, regex=True)
sim.restore()
```

Selectors are exact names by default; use sequences for multiple body/joint
selectors or `regex=True` for full-name regex matching. Payload and wrench
selectors are a single name or regex. Commands use the same validation and
startup snapshot as `mjctl`. `restore()` clears disturbances and restores
API-managed model properties, delay and noise without resetting robot state or
episode time. Delay and noise work with the ONNX bridge: delay queues complete
actuator targets, and noise changes policy velocity feedback without changing
true MuJoCo velocities. Pending actuator commands keep their original deadlines
when delay changes; resetting an episode flushes pending commands.

For live `mjctl` access, add `--remote-control-endpoint tcp://127.0.0.1:5555`
to the policy CLI or your script. Local calls require no endpoint, and share
one control state with remote requests when enabled. `sim.control_request({...})`
also accepts any `mjctl` JSON request directly.

### Inject the ONNX bridge directly

`MjOnnxBridge` uses the model and data owned by `SimulatorWrapper`. Its
`receive()` evaluates only when the metadata control period is due, while
`send_state()` is a no-op. The wrapper owns all physics stepping. Existing
`Simulation` and `MujocoPolicyRunner` APIs delegate to this setup:

```python
from xbot2_py_mujoco.simulator_wrapper import SimulatorWrapper
from policy_deploy_toolkit.simulate import MjOnnxBridge

sim = SimulatorWrapper(
    model, viewer=False, ros=False, enable_pacing=False,
    bridge_factory=lambda model, data: MjOnnxBridge(
        model, data, 'policy.onnx', 'deploy_metadata.yaml'),
)
try:
    sim.xbot2_bridge.set_velocity_command(vx=0.3)
    for _ in range(1000):
        sim.run()  # One physics step; inference is internally decimated.
finally:
    sim.close()
```

The factory replaces the default bridge regardless of `enable_xbot2_bridge`.
The optional bridge `reset()` hook clears controller state after initial joint
placement and on wrapper respawn/reset. Script-level `reset()` still restores
the full episode state. Video, profiling, CLI handling, and the scripting facade
remain in this package for this first refactor; actuator and sensor code is
still local to each bridge.

Policies with height scan sensors use downward MuJoCo rays in a yaw-aligned
grid. Set `--height-scan-shape N M` and `--height-scan-spacing DX DY` to match
training (defaults: 17×11 at 0.1 m spacing). Scan values are terrain heights
relative to the base, in forward-major order. Rays start 2 m above the base;
a missing terrain hit stops the rollout. Height scanning reserves geom group
5 for robot geometry; environment geometry must use another group.

## Run the Example

Start the robot or simulator that provides the XBot2 interface and `xbotcore/` parameters, then run:

```bash
ros2 launch policy_deploy_toolkit run_example.launch.xml  # [enable_joy:=true] [example:=kyon_full_locomotion_rough] (defaults to kyon_full_locomotion_flat)
```

The executable runs a control loop, feeds the policy from the current robot state, and sends the resulting command back through `RobotInterface`.
Commands are received through the `/policy_deploy_node/commands/<command-name>` topic. 

## Height scans

Height scan PointCloud2 messages on `~/sensors/<sensor-name>/points` must contain
finite `x`, `y`, and `z` fields (FLOAT32 or FLOAT64), the expected number of points,
and a nonempty source frame. Point ordering is
preserved; the receiver does not resample the policy's scan grid.

The receiver treats `base_link <- header.frame_id` as static extrinsics. It looks
up the latest available TF without waiting and caches it per source frame for
its lifetime; `base_link` clouds need no TF. Both translation and rotation are
applied before leveling. Restart the node if these extrinsics change.

Each new scan uses the latest valid XBot IMU orientation, without timestamp
matching or interpolation. Cloud timestamps are used only for delay diagnostics.
The IMU mounting rotation is removed, then each point is expressed in a horizontal
frame at the base origin with the robot's yaw heading. Its z coordinate feeds the
policy height scan.

Only the newest cloud per sensor waits for an initial valid IMU reading or static
TF. `sensor_timeout_s` bounds the wait from receipt; zero requires both to be
immediately available. Invalid or expired scans retain the last valid scan,
including existing stale-scan reuse behavior. Once consumed, a scan is not
retransformed when the IMU orientation changes.
