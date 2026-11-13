"""Evaluate an ONNX policy directly in MuJoCo: python -m policy_deploy_toolkit.simulate."""

import argparse
from collections import deque
from contextlib import ExitStack, contextmanager
from pathlib import Path
import time

import mujoco
import numpy as np

from . import HeightScanSpec, Inputs, OnnxPolicy, Outputs, RobotInfo


class _Profile:
    """Exclusive wall-clock timings, including waits for rendering and encoding."""

    def __init__(self):
        self.started = time.perf_counter()
        self.last_progress = self.started
        self.timings = {}
        self._rtf_samples = deque()
        self._rtf_window_seconds = 5.

    def start_loop(self, sim_time, wall_time):
        self._rtf_samples.clear()
        self._rtf_samples.append((wall_time, sim_time))
        self.last_progress = wall_time

    def _window_rtf(self):
        if len(self._rtf_samples) < 2:
            return None, 0.
        end_wall, end_sim = self._rtf_samples[-1]
        start_wall, start_sim = self._rtf_samples[0]
        cutoff = end_wall - self._rtf_window_seconds
        if start_wall < cutoff:
            next_wall, next_sim = self._rtf_samples[1]
            start_sim += (next_sim - start_sim) * (cutoff - start_wall) / (next_wall - start_wall)
            start_wall = cutoff
        elapsed = end_wall - start_wall
        return ((end_sim - start_sim) / elapsed, elapsed) if elapsed > 0 else (None, 0.)

    def _rtf_text(self):
        rtf, elapsed = self._window_rtf()
        return (f'RTF {rtf:.2f} (last {elapsed:.2f} s of loop wall time)'
                if rtf is not None else 'RTF n/a (no completed loop steps)')

    def call(self, name, function, *args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            count, total, maximum = self.timings.get(name, (0, 0., 0.))
            self.timings[name] = (count + 1, total + elapsed, max(maximum, elapsed))

    def progress(self, sim_time):
        now = time.perf_counter()
        self._rtf_samples.append((now, sim_time))
        cutoff = now - self._rtf_window_seconds
        # Keep the sample bracketing the boundary for exact window interpolation.
        while len(self._rtf_samples) > 2 and self._rtf_samples[1][0] <= cutoff:
            self._rtf_samples.popleft()
        if now - self.last_progress >= 5:
            print(f'[profile] {self._rtf_text()}; simulated {sim_time:.2f} s total', flush=True)
            self.last_progress = now

    def report(self, sim_time):
        elapsed = time.perf_counter() - self.started
        print(f'Profile: {sim_time:.3f} s simulated / {elapsed:.3f} s wall; '
              f'{self._rtf_text()}')
        print('Stage totals include setup and finalization; window RTF excludes both.')
        print(f'{"Stage":<26} {"Calls":>8} {"Total s":>10} {"Wall %":>8} {"Mean ms":>10} {"Max ms":>10}')
        for name, (count, total, maximum) in sorted(self.timings.items(), key=lambda item: -item[1][1]):
            print(f'{name:<26} {count:8d} {total:10.3f} {100 * total / elapsed:7.1f}% '
                  f'{1000 * total / count:10.3f} {1000 * maximum:10.3f}')
        other = max(0., elapsed - sum(total for _, total, _ in self.timings.values()))
        print(f'{"Other Python / overhead":<26} {"":>8} {other:10.3f} {100 * other / elapsed:7.1f}%')


def _profile_call(profile, name, function, *args, **kwargs):
    if profile is None:
        return function(*args, **kwargs)
    return profile.call(name, function, *args, **kwargs)


class MjOnnxBridge:
    """Connect scalar joint actuators and a site IMU to the policy bindings.

    The model/data are owned by SimulatorWrapper. receive() evaluates at the
    policy period and holds native actuator targets between evaluations.
    """

    def __init__(self, model, data, policy_path, metadata_path, *, base_body='base_link',
                 imu_site='imu_link', obs_group='', joint_velocity_limit=np.inf,
                 commands=None, height_scan_shape=(17, 11),
                 height_scan_spacing=(0.1, 0.1)):
        self.model = model
        self.profile = None
        self.data = data
        self._next_evaluation_time = data.time
        self._has_evaluated = False
        self._startup_finalized = False
        self.base_id = model.body(base_body).id
        if self.base_id == 0:
            raise ValueError('base_body must be a robot body, not world')
        self.imu_id = model.site(imu_site).id
        if np.isnan(joint_velocity_limit) or joint_velocity_limit <= 0:
            raise ValueError('joint_velocity_limit must be positive')
        robot_bodies = np.zeros(model.nbody, dtype=bool)
        root_id = self.base_id
        while model.body_parentid[root_id] != 0:
            root_id = int(model.body_parentid[root_id])
        free_joints = np.flatnonzero((model.jnt_bodyid == root_id)
                                    & (model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
        self._free_joint = int(free_joints[0]) if len(free_joints) else None
        robot_bodies[root_id] = True
        for body in range(1, model.nbody):
            robot_bodies[body] |= robot_bodies[model.body_parentid[body]]
        if not robot_bodies[model.site_bodyid[self.imu_id]]:
            raise ValueError('IMU site must belong to the same robot as the base body')

        joint_ids, actuator_ids = [], []
        for actuator in range(model.nu):
            if model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError('Only scalar joint-transmission actuators are supported')
            joint = int(model.actuator_trnid[actuator, 0])
            if not robot_bodies[model.jnt_bodyid[joint]]:
                continue
            if (int(model.jnt_type[joint]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                                  int(mujoco.mjtJoint.mjJNT_SLIDE))
                    or not np.array_equal(model.actuator_gear[actuator], [1, 0, 0, 0, 0, 0])
                    or joint in joint_ids):
                raise ValueError('Each controlled scalar joint needs one actuator with unit gear')
            if (model.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED
                    or model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE
                    or int(model.actuator_dyntype[actuator]) not in
                    (int(mujoco.mjtDyn.mjDYN_NONE), int(mujoco.mjtDyn.mjDYN_INTEGRATOR))):
                raise ValueError('Use position or intvelocity actuators with fixed gain and affine bias')
            joint_ids.append(joint)
            actuator_ids.append(actuator)
        if not joint_ids:
            raise ValueError('No controlled joints found in the base body subtree')
        self.joint_ids = np.array(joint_ids)
        self.actuator_ids = np.array(actuator_ids)
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.dof_ids = model.jnt_dofadr[self.joint_ids]
        self.integral = model.actuator_dyntype[self.actuator_ids] == mujoco.mjtDyn.mjDYN_INTEGRATOR
        mujoco.mj_forward(model, self.data)

        robot = RobotInfo()
        robot.joint_names = [model.joint(j).name for j in joint_ids]
        robot.joint_pos_min = np.where(model.jnt_limited[self.joint_ids],
                                       model.jnt_range[self.joint_ids, 0], -np.inf)
        robot.joint_pos_max = np.where(model.jnt_limited[self.joint_ids],
                                       model.jnt_range[self.joint_ids, 1], np.inf)
        robot.joint_vel_max = np.full(len(joint_ids), joint_velocity_limit)
        robot.body_names = [model.body(i).name for i in range(1, model.nbody)
                            if robot_bodies[i] and model.body(i).name]
        robot.body_pose_in_base = self.body_pose_in_base
        robot.base_T_imu = self._pose_in_base(
            self.data.site_xpos[self.imu_id], self.data.site_xmat[self.imu_id])
        self._policy_args = (str(policy_path), str(metadata_path), obs_group, robot)
        self.policy = OnnxPolicy(*self._policy_args)
        self.info = self.policy.policyInfo()
        ratio = self.info.control_dt / model.opt.timestep
        if not np.isfinite(ratio) or ratio < 1 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-8):
            raise ValueError('Policy control_dt must be an integer multiple of the MuJoCo timestep')
        self.decimation = round(ratio)
        self.initial_actuator_delay_s = 0.
        self.actuator_delay_s = 0.
        self._pending_commands = deque()
        self._joint_name_to_index = {name: i for i, name in enumerate(robot.joint_names)}
        self.velocity_feedback_noise_std = np.zeros(len(joint_ids))
        self._noise_rng = np.random.default_rng()

        # Initialize policy joints in robot order; other actuated joints hold qpos0.
        for policy_id, robot_id in enumerate(self.info.joint_id_policy_to_robot):
            self.data.qpos[self.qpos_ids[robot_id]] = self.info.joint_default_pos[policy_id]
        self.data.ctrl[self.actuator_ids] = np.where(self.integral, 0, self.data.qpos[self.qpos_ids])
        for actuator in self.actuator_ids[self.integral]:
            joint = model.actuator_trnid[actuator, 0]
            self.data.act[model.actuator_actadr[actuator]] = self.data.qpos[model.jnt_qposadr[joint]]
        mujoco.mj_forward(model, self.data)

        self._initialize_io(commands or {})
        self.gyro = self._imu_sensor(mujoco.mjtSensor.mjSENS_GYRO)
        self.accelerometer = self._imu_sensor(mujoco.mjtSensor.mjSENS_ACCELEROMETER)

        scans = [spec for spec in self.policy.sensor_specs() if isinstance(spec, HeightScanSpec)]
        self.scan_shape = tuple(height_scan_shape)
        self.scan_spacing = tuple(height_scan_spacing)
        self.scan_enabled = bool(scans)
        if self.scan_enabled:
            if (len(self.scan_shape) != 2 or any(n <= 0 or int(n) != n for n in self.scan_shape)
                    or len(self.scan_spacing) != 2 or not np.isfinite(self.scan_spacing).all()
                    or any(s <= 0 for s in self.scan_spacing)):
                raise ValueError('Height scan shape and spacing must contain two positive values')
            if any(spec.size != np.prod(self.scan_shape) for spec in scans):
                raise ValueError('Height scan shape must match the policy sensor size')
            # Reserve geom group 5 for the robot so rays exclude its entire subtree.
            robot_geoms = robot_bodies[model.geom_bodyid]
            if np.any((model.geom_group == 5) & ~robot_geoms):
                raise ValueError('Height scanning reserves geom group 5 for the robot')
            model.geom_group[robot_geoms] = 5

        self._state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._initial_state = np.empty(mujoco.mj_stateSize(model, self._state_spec))
        mujoco.mj_getState(model, self.data, self._initial_state, self._state_spec)
        self._initial_gains = model.actuator_gainprm[self.actuator_ids].copy()
        self._initial_biases = model.actuator_biasprm[self.actuator_ids].copy()

    @property
    def commands(self):
        """Copies of current commands; use set_command(s) to change them."""
        return {name: value.copy() for name, value in self.inputs.command.items()}

    def set_commands(self, commands):
        """Validate and update named commands atomically, preserving other commands."""
        values = self.commands
        for name, value in commands.items():
            ok, sanitized, reason = self.policy.sanitize_command(
                self.inputs, name, np.asarray(value, dtype=float))
            if not ok:
                raise ValueError(f'Invalid command {name!r}: {reason}')
            values[name] = sanitized
        self.inputs.command = values

    def set_command(self, name, values):
        """Set one command, using the policy's dimensional checks and limits."""
        self.set_commands({name: values})

    def set_velocity_command(self, vx=0., vy=0., wz=0., *, name='base_velocity'):
        """Set forward/lateral speed (m/s) and yaw rate (rad/s)."""
        self.set_command(name, [vx, vy, wz])

    def set_actuator_delay(self, seconds):
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not np.isfinite(seconds) or seconds < 0):
            raise ValueError('actuator delay must be finite and non-negative')
        self.actuator_delay_s = float(seconds)
        return self.actuator_delay_s

    def set_velocity_feedback_noise(self, joint_names, stddev):
        if (isinstance(stddev, bool) or not isinstance(stddev, (int, float))
                or not np.isfinite(stddev) or stddev < 0):
            raise ValueError('velocity feedback noise stddev must be finite and non-negative')
        indices = [self._joint_name_to_index[name] for name in joint_names]
        self.velocity_feedback_noise_std[indices] = stddev

    def reset_velocity_feedback_noise(self):
        self.velocity_feedback_noise_std[:] = 0

    def _apply_due_commands(self):
        newest = None
        for index, (deadline, _) in enumerate(self._pending_commands):
            if deadline <= self.data.time + 1e-12:
                newest = index
        if newest is None:
            return
        for _ in range(newest + 1):
            _, command = self._pending_commands.popleft()
        q, v, tau, k, d = command
        self.model.actuator_gainprm[self.actuator_ids, 0] = k
        self.model.actuator_biasprm[self.actuator_ids, 0] = d * v + tau
        self.model.actuator_biasprm[self.actuator_ids, 1] = -k
        self.model.actuator_biasprm[self.actuator_ids, 2] = -d
        self.data.ctrl[self.actuator_ids] = np.where(self.integral, v, q)
        self._applied_command = command

    def _initialize_io(self, commands):
        self.inputs = Inputs()
        self.outputs = Outputs(self.info.action_size, len(self.joint_ids))
        self.outputs.q_des = self.data.qpos[self.qpos_ids].copy()
        self.outputs.k_des = self.model.actuator_gainprm[self.actuator_ids, 0].copy()
        self.outputs.d_des = -self.model.actuator_biasprm[self.actuator_ids, 2].copy()
        self.inputs.last_action = np.zeros(self.info.action_size)
        self.inputs.command = self.policy.default_commands()
        self.set_commands(commands)
        self._pending_commands.clear()
        self._applied_command = tuple(getattr(self.outputs, field).copy()
                                      for field in ('q_des', 'v_des', 'tau_des', 'k_des', 'd_des'))

    def reset_episode(self, *, position=None, quaternion=None, preserve_commands=True):
        """Restore startup state and clear policy history/recurrent state.

        Optional position and quaternion override the floating root pose in world
        coordinates. Quaternions use MuJoCo's [w, x, y, z] ordering. Episode time
        and policy step reset to zero; elapsed_time and control_steps continue.
        Call between control steps, on the simulation thread.
        """
        if (position is not None or quaternion is not None) and self._free_joint is None:
            raise ValueError('Respawning at a pose requires a floating root joint')
        if position is not None:
            position = np.asarray(position, dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError('position must contain three finite world coordinates')
        if quaternion is not None:
            quaternion = np.asarray(quaternion, dtype=float)
            if (quaternion.shape != (4,) or not np.isfinite(quaternion).all()
                    or not np.isfinite(np.linalg.norm(quaternion)) or np.linalg.norm(quaternion) == 0):
                raise ValueError('quaternion must be a finite nonzero [w, x, y, z] vector')
            quaternion = quaternion / np.linalg.norm(quaternion)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, self._initial_state, self._state_spec)
        if self._free_joint is not None:
            address = self.model.jnt_qposadr[self._free_joint]
            if position is not None:
                self.data.qpos[address:address + 3] = position
            if quaternion is not None:
                self.data.qpos[address + 3:address + 7] = quaternion
        self.model.actuator_gainprm[self.actuator_ids] = self._initial_gains
        self.model.actuator_biasprm[self.actuator_ids] = self._initial_biases
        mujoco.mj_forward(self.model, self.data)
        self.reset(preserve_commands=preserve_commands)

    def reset(self, *, preserve_commands=True):
        """Bridge lifecycle hook: clear controller state after a simulator reset.

        The wrapper has already placed the robot. This hook preserves that pose
        and simulation time, and initializes targets from the current state.
        """
        commands = self.commands if preserve_commands else {}
        self.model.actuator_gainprm[self.actuator_ids] = self._initial_gains
        self.model.actuator_biasprm[self.actuator_ids] = self._initial_biases
        self.data.ctrl[self.actuator_ids] = np.where(self.integral, 0, self.data.qpos[self.qpos_ids])
        for actuator in self.actuator_ids[self.integral]:
            joint = self.model.actuator_trnid[actuator, 0]
            self.data.act[self.model.actuator_actadr[actuator]] = self.data.qpos[self.model.jnt_qposadr[joint]]
        # Reload only after inference: initial construction already has clean
        # observation history, action state, and recurrent ONNX buffers.
        if self._has_evaluated:
            self.policy = OnnxPolicy(*self._policy_args)
        self._has_evaluated = False
        self._initialize_io(commands)
        self._next_evaluation_time = self.data.time
        if not self._startup_finalized:
            # Include explicit q_init overrides applied after the factory returns.
            mujoco.mj_getState(self.model, self.data, self._initial_state, self._state_spec)
            self._startup_finalized = True

    def receive(self):
        """Called every physics step; inference runs only when control is due."""
        self._apply_due_commands()
        if self.data.time + 1e-12 >= self._next_evaluation_time:
            self.evaluate()
            self._next_evaluation_time = self.data.time + self.info.control_dt
            return True
        self._apply_due_commands()
        return False

    def send_state(self):
        """No transport is needed for an in-process policy."""

    def close(self):
        """No transport resources are owned by this bridge."""

    def _imu_sensor(self, sensor_type):
        ids = np.flatnonzero((self.model.sensor_type == sensor_type)
                             & (self.model.sensor_objtype == mujoco.mjtObj.mjOBJ_SITE)
                             & (self.model.sensor_objid == self.imu_id))
        if len(ids) != 1:
            raise ValueError('IMU site requires one gyro and one accelerometer sensor')
        address = int(self.model.sensor_adr[ids[0]])
        return slice(address, address + 3)

    def _pose_in_base(self, position, rotation):
        base_rotation = self.data.xmat[self.base_id].reshape(3, 3)
        pose = np.eye(4)
        pose[:3, :3] = base_rotation.T @ rotation.reshape(3, 3)
        pose[:3, 3] = base_rotation.T @ (position - self.data.xpos[self.base_id])
        return pose

    def body_pose_in_base(self, name):
        body = self.model.body(name).id
        return self._pose_in_base(self.data.xpos[body], self.data.xmat[body])

    def _height_scan(self):
        center = self.data.xpos[self.base_id]
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        x, y = np.meshgrid(
            (np.arange(self.scan_shape[0]) - (self.scan_shape[0] - 1) / 2) * self.scan_spacing[0],
            (np.arange(self.scan_shape[1]) - (self.scan_shape[1] - 1) / 2) * self.scan_spacing[1],
            indexing='ij')
        heights = []
        geom_id = np.empty(1, dtype=np.int32)
        for dx, dy in zip(x.flat, y.flat):
            origin = center + [c * dx - s * dy, s * dx + c * dy, 2.0]
            distance = mujoco.mj_ray(self.model, self.data, origin, np.array([0., 0., -1.]),
                                     np.array([1, 1, 1, 1, 1, 0], dtype=np.uint8), 1, -1, geom_id)
            if distance < 0:
                raise RuntimeError('Height scan ray missed terrain')
            # Leveled point z relative to the base, as expected by the runtime.
            heights.append(2.0 - distance)
        return np.asarray(heights)

    def evaluate(self):
        """Sense and evaluate once, then hold the result for one control period."""
        self._startup_finalized = True
        _profile_call(self.profile, 'State forward', mujoco.mj_forward, self.model, self.data)
        inputs, outputs = self.inputs, self.outputs
        inputs.time_sec = self.data.time
        inputs.q = self.data.qpos[self.qpos_ids].copy()
        velocity = self.data.qvel[self.dof_ids].copy()
        noisy = self.velocity_feedback_noise_std > 0
        if noisy.any():
            velocity[noisy] += self._noise_rng.normal(0., self.velocity_feedback_noise_std[noisy])
        inputs.v = velocity
        inputs.tau = self.data.qfrc_actuator[self.dof_ids].copy()
        inputs.q_ref, inputs.v_ref, inputs.tau_ref, inputs.k, inputs.d = (
            value.copy() for value in self._applied_command)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, self.data.site_xmat[self.imu_id])
        inputs.w_R_imu = np.roll(quaternion, -1)
        inputs.imu_omega = self.data.sensordata[self.gyro].copy()
        inputs.imu_acc = self.data.sensordata[self.accelerometer].copy()
        if self.scan_enabled:
            inputs.height_scan = _profile_call(self.profile, 'Height scan', self._height_scan)
        self._has_evaluated = True
        if not _profile_call(self.profile, 'Policy inference', self.policy.run, inputs, outputs):
            raise RuntimeError('Policy evaluation failed')
        for field in ('raw_action', 'q_des', 'v_des', 'tau_des', 'k_des', 'd_des'):
            if not np.isfinite(getattr(outputs, field)).all():
                raise RuntimeError(f'Policy produced non-finite {field}')
        position_mode = (outputs.ctrl_mode & 1) != 0
        velocity_mode = (outputs.ctrl_mode & 2) != 0
        if np.any(position_mode & self.integral):
            raise ValueError('Position actions require position actuators, not intvelocity')
        if np.any(velocity_mode & ~self.integral & (outputs.k_des != 0)):
            raise ValueError('Velocity actions with stiffness require intvelocity actuators')
        command = tuple(getattr(outputs, field).copy()
                        for field in ('q_des', 'v_des', 'tau_des', 'k_des', 'd_des'))
        self._pending_commands.append((self.data.time + self.actuator_delay_s, command))
        self._apply_due_commands()
        inputs.last_action = outputs.raw_action.copy()
        inputs.step += 1


class MujocoPolicyRunner:
    """Compatibility facade: an ONNX bridge running inside SimulatorWrapper.

    step() still advances one policy period. Only SimulatorWrapper allocates
    MjData and steps physics; the bridge reads and commands that shared state.
    """

    def __init__(self, model, policy_path, metadata_path, *, remote_control_endpoint=None, **kwargs):
        from xbot2_py_mujoco.simulator_wrapper import SimulatorWrapper

        self.simulator = SimulatorWrapper(
            model, viewer=False, ros=False, enable_pacing=False,
            remote_control_endpoint=remote_control_endpoint,
            bridge_factory=lambda model, data: MjOnnxBridge(
                model, data, policy_path, metadata_path, **kwargs))
        self.bridge = self.simulator.xbot2_bridge
        self.simulator.rtf_print_interval = float('inf')
        # Retain the policy startup posture for the wrapper's keyboard respawn.
        self.simulator.q_init = dict(zip(self.bridge._policy_args[3].joint_names,
                                         self.data.qpos[self.bridge.qpos_ids].copy()))
        self.control_steps = 0

    def __getattr__(self, name):
        try:
            return getattr(self.simulator, name)
        except AttributeError:
            return getattr(self.bridge, name)

    @property
    def profile(self):
        return self.bridge.profile

    @profile.setter
    def profile(self, value):
        self.bridge.profile = value

    @property
    def physics_steps(self):
        return self.simulator.physics_steps

    @property
    def elapsed_time(self):
        return self.simulator.elapsed_time

    def reset(self, *, position=None, quaternion=None, preserve_commands=True):
        self.bridge.reset_episode(position=position, quaternion=quaternion,
                                  preserve_commands=preserve_commands)
        self.simulator.clear_wrenches()
        self.simulator.last_sim_time = self.data.time
        self.simulator._reset_timing()

    def respawn(self, position=None, quaternion=None, *, preserve_commands=True):
        self.reset(position=position, quaternion=quaternion, preserve_commands=preserve_commands)

    def step(self, on_step=None):
        """Advance one policy period using the wrapper's physics phases."""
        completed = 0
        while completed < self.bridge.decimation and self.simulator.running:
            previous = self.simulator.physics_steps
            self.simulator.pre_step()
            _profile_call(self.profile, 'Physics step', self.simulator.step)
            self.simulator.post_step()
            if self.simulator.physics_steps != previous:
                completed += 1
                if on_step is not None:
                    on_step()
        if completed == self.bridge.decimation:
            self.control_steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError('MuJoCo state became non-finite')
        return completed == self.bridge.decimation

    def close(self):
        self.simulator.close()


def _positive(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Must be positive and finite')
    return number


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('Must be a positive integer')
    return number


@contextmanager
def _video_recorder(runner, path, fps, size, track_body_id, duration, camera_distance=None,
                    geom_group_blacklist=()):
    """Stream frames to disk at simulation-time intervals, without buffering a rollout."""
    import imageio.v2 as imageio

    width, height = size
    model, data = runner.model, runner.data
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    camera = mujoco.MjvCamera()
    scene_option = mujoco.MjvOption()
    for group in geom_group_blacklist:
        scene_option.geomgroup[group] = 0
    mujoco.mjv_defaultFreeCamera(model, camera)
    if camera_distance is not None:
        camera.distance = camera_distance
    if track_body_id is not None:
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = track_body_id
    path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        renderer = stack.enter_context(_profile_call(
            runner.profile, 'Renderer setup', mujoco.Renderer, model, height=height, width=width))
        writer = imageio.get_writer(
            str(path), format='FFMPEG', mode='I', fps=fps, codec='libx264', macro_block_size=1)
        # Waiting for FFmpeg to finish is timed separately from frame submission.
        stack.callback(_profile_call, runner.profile, 'Video finalization', writer.close)
        frame_index = 0

        def capture():
            nonlocal frame_index
            sim_time = runner.elapsed_time
            if duration is not None and sim_time >= duration - 1e-12:
                return
            if sim_time + 1e-12 < frame_index / fps:
                return
            # mj_step integrates positions after computing derived body poses.
            _profile_call(runner.profile, 'Video forward', mujoco.mj_forward, model, data)
            _profile_call(runner.profile, 'Scene update', renderer.update_scene, data,
                          camera=camera, scene_option=scene_option)
            runner.simulator.control.update_visuals(renderer.scene, clear=False)
            frame = _profile_call(runner.profile, 'Render / readback', renderer.render)
            frame = _profile_call(runner.profile, 'Status overlay', runner.simulator.draw_status_text, frame)
            while frame_index / fps <= sim_time + 1e-12:
                _profile_call(runner.profile, 'Video write / encode', writer.append_data, frame)
                frame_index += 1

        yield capture


def add_arguments(parser):
    """Add all simulator options to an existing ArgumentParser and return it."""
    parser.add_argument('--mjcf', required=True, type=Path, help='Compiled robot/scene MJCF XML')
    parser.add_argument('--policy', required=True, type=Path, help='ONNX policy file')
    parser.add_argument('--metadata', type=Path, help='Defaults to deploy_metadata.yaml beside the policy')
    parser.add_argument('--base-body', default='base_link')
    parser.add_argument('--imu-site', default='imu_link')
    parser.add_argument('--obs-group', default='')
    parser.add_argument('--joint-velocity-limit', type=_positive, default=np.inf,
                        help='Uniform joint speed limit; default: unbounded (MuJoCo has no joint velocity limits)')
    parser.add_argument('--command', action='append', default=[], metavar='NAME=V1,V2,...',
                        help='Override a policy command; repeat for multiple commands')
    parser.add_argument('--height-scan-shape', nargs=2, type=int, default=(17, 11))
    parser.add_argument('--height-scan-spacing', nargs=2, type=_positive, default=(0.1, 0.1))
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--remote-control-endpoint', metavar='ENDPOINT',
                        help='Enable mjctl access to the same controls as the local Python API')
    parser.add_argument('--video', type=Path, help='Video output path (headless default: rollout.mp4)')
    parser.add_argument('--video-fps', type=_positive_int, default=30, help='Video frames per simulation second')
    parser.add_argument('--video-size', nargs=2, type=_positive_int, default=(640, 480),
                        metavar=('WIDTH', 'HEIGHT'), help='Video resolution (default: 640 480)')
    parser.add_argument('--track-body', metavar='BODY', help='Make the viewer camera follow this body')
    parser.add_argument('--camera-distance', type=_positive, metavar='METERS',
                        help='Viewer and video camera distance from the target')
    parser.add_argument('--geom-group-blacklist', nargs='+', type=int, choices=range(6),
                        action='extend', default=[], metavar='GROUP',
                        help='Hide geometry groups in viewer and video (0–5; repeatable)')
    parser.add_argument('--duration', type=_positive, help='Stop after this many simulation seconds')
    parser.add_argument('--target-rtf', type=_positive, default=1.0, help='Viewer pacing; ignored in headless mode')
    parser.add_argument('--profile', action='store_true',
                        help='Report wall time by stage, with progress every 5 seconds')
    return parser


def create_arg_parser(**kwargs):
    """Create the CLI parser; callers can add their own options before parsing."""
    kwargs.setdefault('description', __doc__)
    return add_arguments(argparse.ArgumentParser(**kwargs))


def parse_args(argv=None):
    """Parse simulator options from a supplied list, or sys.argv when omitted."""
    return create_arg_parser().parse_args(argv)


class Simulation:
    """A scripted rollout with the same options and lifecycle as the CLI.

    Use as a context manager. Call step() in your own loop, or run(callback)
    with a callback invoked before each policy step. Commands, reset(), and
    respawn() must be called on that same thread, between steps.
    """

    def __init__(self, args):
        self.args = args
        self.video_path = args.video or (Path('rollout.mp4') if args.headless else None)
        if self.video_path is not None and any(dimension % 2 for dimension in args.video_size):
            raise ValueError('--video-size dimensions must be even for H.264 encoding')
        commands = {}
        for command in args.command:
            try:
                name, values = command.split('=', 1)
                commands[name] = [float(value) for value in values.split(',')]
            except ValueError as exc:
                raise ValueError('--command must have the form NAME=V1,V2,...') from exc
        self.profile = _Profile() if args.profile else None

        def create_runner():
            return MujocoPolicyRunner(
                mujoco.MjModel.from_xml_path(str(args.mjcf.resolve())), args.policy,
                args.metadata or args.policy.with_name('deploy_metadata.yaml'),
                base_body=args.base_body, imu_site=args.imu_site, obs_group=args.obs_group,
                joint_velocity_limit=args.joint_velocity_limit, commands=commands,
                remote_control_endpoint=args.remote_control_endpoint,
                height_scan_shape=args.height_scan_shape, height_scan_spacing=args.height_scan_spacing)

        self.runner = _profile_call(self.profile, 'Model / policy setup', create_runner)
        self.runner.profile = self.profile
        self._track_body_id = None
        if args.track_body is not None:
            self._track_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, args.track_body)
            if self._track_body_id < 0:
                raise ValueError(f'Unknown camera tracking body: {args.track_body!r}')
        self.viewer = None
        self._stack = None
        self._capture = None
        self._capture_started = False
        self._closed = False
        self._stopped = False

    @property
    def simulator(self):
        return self.runner.simulator

    def __getattr__(self, name):
        runner = self.__dict__.get('runner')
        if runner is None:
            raise AttributeError(name)
        return getattr(runner.simulator, name)

    @classmethod
    def from_cli(cls, argv=None):
        """Construct from the standard CLI options, optionally supplied as a list."""
        return cls(parse_args(argv))

    @property
    def model(self):
        return self.runner.model

    @property
    def data(self):
        return self.runner.data

    @property
    def inputs(self):
        return self.runner.inputs

    @property
    def outputs(self):
        return self.runner.outputs

    @property
    def commands(self):
        return self.runner.commands

    @property
    def elapsed_time(self):
        """Session simulation time; reset/respawn only reset data.time."""
        return self.runner.elapsed_time

    @property
    def running(self):
        return (self._stack is not None and not self._closed and not self._stopped
                and self.runner.simulator.running
                and (self.viewer is None or self.viewer.is_running())
                and (self.args.duration is None or self.elapsed_time < self.args.duration - 1e-12))

    def set_commands(self, commands):
        self.runner.set_commands(commands)

    def set_command(self, name, values):
        self.runner.set_command(name, values)

    def set_velocity_command(self, vx=0., vy=0., wz=0., *, name='base_velocity'):
        self.runner.set_velocity_command(vx, vy, wz, name=name)

    def apply_wrench(self, body_name, force=(0., 0., 0.), torque=(0., 0., 0.), *,
                     frame='global', duration=None, regex=False):
        return self.simulator.apply_wrench(body_name, force, torque, frame=frame,
                                           duration=duration, regex=regex)

    def clear_wrenches(self, body_name=None):
        """Clear one body's wrench, or all external body wrenches when omitted."""
        self.simulator.clear_wrenches(body_name)

    def reset(self, *, position=None, quaternion=None, preserve_commands=True):
        """Reset the robot and policy while keeping the session/video timeline."""
        _profile_call(self.profile, 'Reset / respawn', self.runner.reset,
                      position=position, quaternion=quaternion, preserve_commands=preserve_commands)

    def respawn(self, position=None, quaternion=None, *, preserve_commands=True):
        """Reset at a floating-root world pose (quaternion order: w, x, y, z)."""
        self.reset(position=position, quaternion=quaternion, preserve_commands=preserve_commands)

    def stop(self):
        """End run() or a loop using running; resources close on context exit."""
        self._stopped = True

    def __enter__(self):
        if self._closed or self._stack is not None:
            raise RuntimeError('Simulation contexts cannot be entered more than once')
        stack = ExitStack()
        stack.callback(self.runner.close)
        try:
            if not self.args.headless:
                from mujoco.viewer import launch_passive
                self.viewer = stack.enter_context(launch_passive(self.model, self.data))
                # The wrapper owns viewer synchronization and its force/payload cues.
                self.simulator.viewer = self.viewer
                stack.callback(setattr, self.simulator, 'viewer', None)
            if self.video_path is not None:
                self._capture = stack.enter_context(_video_recorder(
                    self.runner, self.video_path, self.args.video_fps, self.args.video_size,
                    self._track_body_id, self.args.duration, camera_distance=self.args.camera_distance,
                    geom_group_blacklist=self.args.geom_group_blacklist))
            if self.viewer is not None:
                with self.viewer.lock():
                    for group in self.args.geom_group_blacklist:
                        self.viewer.opt.geomgroup[group] = 0
                    if self._track_body_id is not None:
                        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                        self.viewer.cam.trackbodyid = self._track_body_id
                    if self.args.camera_distance is not None:
                        self.viewer.cam.distance = self.args.camera_distance
        except BaseException:
            self._closed = True
            stack.close()
            raise
        self._stack = stack
        self._wall_start = time.perf_counter()
        if self.profile is not None:
            self.profile.start_loop(self.elapsed_time, self._wall_start)
        return self

    def step(self):
        """Advance one policy period, recording and pacing as configured.

        Returns False if stopped, the viewer closed, or duration was reached.
        """
        if self._stack is None:
            raise RuntimeError('Use Simulation inside a with block before stepping')
        if not self.running:
            return False
        if self._capture is not None and not self._capture_started:
            # The first callback has now set the initial action and its status.
            self._capture()
            self._capture_started = True
        self.runner.step(on_step=self._capture)
        if not self.args.headless:
            delay = self.elapsed_time / self.args.target_rtf - (time.perf_counter() - self._wall_start)
            if delay > 0:
                _profile_call(self.profile, 'Pacing sleep', time.sleep, delay)
        if self.profile is not None:
            self.profile.progress(self.elapsed_time)
        return True

    def run(self, before_step=None):
        """Run until finished; before_step(simulation) may return False to stop."""
        if self._stack is None:
            raise RuntimeError('Use Simulation inside a with block before running')
        while self.running:
            if before_step is not None and before_step(self) is False:
                self.stop()
                break
            if not self.step():
                break

    def close(self):
        """Close viewer/video and report once; safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        stack, self._stack = self._stack, None
        if stack is not None:
            stack.close()
            print(f'Simulated {self.elapsed_time:.3f} s; '
                  f'evaluated {self.runner.control_steps} policy steps')
            if self.video_path is not None:
                print(f'Video saved to {self.video_path}')
            if self.profile is not None:
                self.profile.report(self.elapsed_time)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def main(argv=None):
    parser = create_arg_parser()
    args = parser.parse_args(argv)
    try:
        simulation = Simulation(args)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        with simulation:
            simulation.run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
