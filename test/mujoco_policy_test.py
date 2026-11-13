"""Integration tests using a real bundled ONNX policy and small MuJoCo models."""
from contextlib import redirect_stdout
import io
import importlib.util
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, sys.argv.pop(1))
ROOT = Path(sys.argv.pop(1))
try:
    import mujoco
except ImportError:
    mujoco = None
else:
    from policy_deploy_toolkit.simulate import MjOnnxBridge, MujocoPolicyRunner, Simulation, create_arg_parser, main


@unittest.skipIf(mujoco is None, 'Optional MuJoCo dependency is not installed')
class MujocoPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = ROOT / 'examples/kyon_full_locomotion_flat'
        cls.metadata = cls.example / 'deploy_metadata.yaml'
        names = cls.metadata.read_text().split('\njoint_names:\n', 1)[1].split('\nfixed_joints:', 1)[0]
        # Deliberately reverse robot ordering to exercise policy-to-robot mapping.
        cls.names = [line.removeprefix('- ') for line in names.splitlines()][::-1]
        bodies = ''.join(f'<body name="body_{i}"><joint name="{name}"/>'
                         '<geom type="sphere" size=".02" mass="1" contype="0" conaffinity="0"/>'
                         '</body>' for i, name in enumerate(cls.names))
        actuators = ''.join(f'<position joint="{name}" kp="10" kv="1" '
                            'forcerange="-50 50"/>' for name in cls.names)
        cls.xml = ('<mujoco><option timestep=".001"/><worldbody>'
                   '<body name="root" pos="0 0 1"><geom size=".1" mass="1"/>'
                   '<body name="base_link"><site name="imu_link" euler=".1 .2 .3"/></body>'
                   f'{bodies}</body><geom type="plane" size="10 10 .1"/></worldbody>'
                   f'<actuator>{actuators}</actuator><sensor>'
                   '<gyro site="imu_link"/><accelerometer site="imu_link"/>'
                   '</sensor></mujoco>')

    def runner(self, xml=None, **kwargs):
        return MujocoPolicyRunner(mujoco.MjModel.from_xml_string(xml or self.xml),
                                  self.example / 'policy.onnx', self.metadata, **kwargs)

    def test_mapping_inference_and_native_pd(self):
        runner = self.runner(commands={'base_velocity': [0.2, 0, 0]})
        initial_positions = runner.data.qpos[runner.qpos_ids].copy()
        for policy_id, robot_id in enumerate(runner.info.joint_id_policy_to_robot):
            self.assertEqual(initial_positions[robot_id], runner.info.joint_default_pos[policy_id])
        runner.evaluate()
        np.testing.assert_allclose(runner.inputs.command['base_velocity'], [0.2, 0, 0])
        mujoco.mj_forward(runner.model, runner.data)
        outputs = runner.outputs
        expected = (outputs.k_des * (outputs.q_des - initial_positions)
                    + outputs.d_des * outputs.v_des + outputs.tau_des)
        np.testing.assert_allclose(runner.data.qfrc_actuator[runner.dof_ids], np.clip(expected, -50, 50))
        np.testing.assert_array_equal(runner.inputs.last_action, outputs.raw_action)
        runner.step()
        self.assertAlmostEqual(runner.data.time, runner.info.control_dt)
        self.assertEqual(runner.inputs.step, 2)
        self.assertTrue(np.isfinite(runner.outputs.raw_observation).all())
        self.assertFalse(any(name.startswith(('xbot2_py_bridge', 'pyxbot2_diagnostics', 'rclpy'))
                             for name in sys.modules))

    def test_invalid_command_and_timestep(self):
        with self.assertRaisesRegex(ValueError, 'Invalid command'):
            self.runner(commands={'unknown': [0]})
        with self.assertRaisesRegex(ValueError, 'integer multiple'):
            self.runner(self.xml.replace('timestep=".001"', 'timestep=".003"'))

    def test_injected_bridge_cadence_and_respawn_hook(self):
        from xbot2_py_mujoco.simulator_wrapper import SimulatorWrapper

        model = mujoco.MjModel.from_xml_string(self.xml.replace(
            '<body name="root" pos="0 0 1">', '<body name="root" pos="0 0 1"><freejoint/>'))
        sim = SimulatorWrapper(
            model, viewer=False, ros=False, enable_pacing=False, send_state_decimation=7,
            bridge_factory=lambda m, d: MjOnnxBridge(m, d, self.example / 'policy.onnx', self.metadata))
        try:
            bridge = sim.xbot2_bridge
            self.assertIs(bridge.model, sim.model)
            self.assertIs(bridge.data, sim.data)
            for _ in range(25):
                sim.run()
            self.assertEqual(bridge.inputs.step, 2)
            self.assertEqual(sim.physics_steps, 25)
            bridge.set_velocity_command(0.2)
            sim.apply_wrench('root', [1, 0, 0])
            sim.spawn_locations = [(1, 2, 3)]
            sim._respawn()
            self.assertEqual(bridge.inputs.step, 0)
            np.testing.assert_array_equal(sim.data.qpos[:3], [1, 2, 3])
            np.testing.assert_array_equal(sim.data.xfrc_applied, 0)
            np.testing.assert_allclose(bridge.commands['base_velocity'], [0.2, 0, 0])
            sim.run()
            self.assertEqual(bridge.inputs.step, 1)
        finally:
            sim.close()

    def test_wrapper_rollout_matches_explicit_policy_and_physics(self):
        runner = self.runner()
        model = mujoco.MjModel.from_xml_string(self.xml)
        data = mujoco.MjData(model)
        reference = MjOnnxBridge(model, data, self.example / 'policy.onnx', self.metadata)
        self.assertIs(runner.data, runner.simulator.data)
        self.assertIs(runner.bridge, runner.simulator.xbot2_bridge)
        try:
            for _ in range(5):
                reference.evaluate()
                for _ in range(reference.decimation):
                    mujoco.mj_step(model, data)
                runner.step()
                np.testing.assert_allclose(runner.outputs.raw_observation, reference.outputs.raw_observation, atol=1e-7)
                np.testing.assert_allclose(runner.outputs.raw_action, reference.outputs.raw_action, atol=1e-7)
                np.testing.assert_allclose(runner.data.qpos, data.qpos, atol=1e-10)
                np.testing.assert_allclose(runner.data.qvel, data.qvel, atol=1e-10)
            self.assertEqual(runner.physics_steps, 5 * reference.decimation)
            self.assertEqual(runner.control_steps, 5)
        finally:
            runner.close()

    def test_injected_bridge_reset_keeps_explicit_startup_posture(self):
        from xbot2_py_mujoco.simulator_wrapper import SimulatorWrapper

        sim = SimulatorWrapper(
            mujoco.MjModel.from_xml_string(self.xml), viewer=False, ros=False, enable_pacing=False,
            q_init={self.names[0]: 0.25},
            bridge_factory=lambda m, d: MjOnnxBridge(m, d, self.example / 'policy.onnx', self.metadata))
        try:
            self.assertEqual(sim.data.joint(self.names[0]).qpos[0], 0.25)
            sim.run()
            sim.xbot2_bridge.reset_episode()
            self.assertEqual(sim.data.joint(self.names[0]).qpos[0], 0.25)
        finally:
            sim.close()

    def test_command_updates_and_reset_clear_history(self):
        runner = self.runner(commands={'base_velocity': [0.2, 0, 0]})
        initial_qpos = runner.data.qpos.copy()
        runner.step()
        first_observation = runner.outputs.raw_observation.copy()
        first_action = runner.outputs.raw_action.copy()
        runner.step()
        runner.set_velocity_command(-0.3, 0.1, 0.2)
        np.testing.assert_allclose(runner.commands['base_velocity'], [-0.3, 0.1, 0.2])
        with self.assertRaisesRegex(ValueError, 'Invalid command'):
            runner.set_commands({'base_velocity': [0, 0, 0], 'unknown': [0]})
        np.testing.assert_allclose(runner.commands['base_velocity'], [-0.3, 0.1, 0.2])
        runner.set_velocity_command(0.2)
        elapsed = runner.elapsed_time
        runner.reset()
        self.assertEqual(runner.inputs.step, 0)
        self.assertEqual(runner.data.time, 0)
        self.assertEqual(runner.elapsed_time, elapsed)
        self.assertEqual(runner.control_steps, 2)
        np.testing.assert_array_equal(runner.data.qpos, initial_qpos)
        np.testing.assert_array_equal(runner.inputs.last_action, 0)
        runner.step()
        np.testing.assert_allclose(runner.outputs.raw_observation, first_observation, atol=1e-6)
        np.testing.assert_allclose(runner.outputs.raw_action, first_action, atol=1e-6)
        runner.reset(preserve_commands=False)
        np.testing.assert_array_equal(runner.commands['base_velocity'], runner.policy.default_commands()['base_velocity'])

    def test_respawn_floating_root_and_validation(self):
        runner = self.runner(self.xml.replace('<body name="root" pos="0 0 1">',
                                             '<body name="root" pos="0 0 1"><freejoint/>'))
        runner.step()
        elapsed = runner.elapsed_time
        runner.data.qfrc_applied[:] = 1
        runner.respawn([1, 2, 3], [2, 0, 0, 0])
        np.testing.assert_array_equal(runner.data.qpos[:7], [1, 2, 3, 1, 0, 0, 0])
        np.testing.assert_array_equal(runner.data.qvel, 0)
        np.testing.assert_array_equal(runner.data.qfrc_applied, 0)
        self.assertEqual(runner.elapsed_time, elapsed)
        with self.assertRaisesRegex(ValueError, 'quaternion'):
            runner.respawn([1, 2, 3], [0, 0, 0, 0])
        np.testing.assert_array_equal(runner.data.qpos[:3], [1, 2, 3])
        with self.assertRaisesRegex(ValueError, 'floating root'):
            self.runner().respawn([0, 0, 1])

    def test_wrenches_follow_local_axes_and_persist(self):
        xml = self.xml.replace('<body name="root" pos="0 0 1">',
                               '<body name="root" pos="0 0 1"><freejoint/>')
        runner = self.runner(xml)
        body_id = runner.model.body('root').id
        runner.data.qpos[3:7] = [2**-0.5, 0, 0, 2**-0.5]
        force = np.array([10., 0, 0])
        runner.apply_wrench('root', force, [1, 0, 0], frame='local')
        force[:] = 0  # The API owns independent copies.
        np.testing.assert_allclose(runner.data.xfrc_applied[body_id], [0, 10, 0, 0, 1, 0], atol=1e-12)
        other_id = runner.model.body('body_0').id
        runner.data.xfrc_applied[other_id, 2] = 2
        runner.step()
        rotation = runner.data.xmat[body_id].reshape(3, 3)
        np.testing.assert_allclose(runner.data.xfrc_applied[body_id, :3], rotation @ [10, 0, 0])
        np.testing.assert_allclose(runner.data.xfrc_applied[body_id, 3:], rotation @ [1, 0, 0])
        self.assertEqual(runner.data.xfrc_applied[other_id, 2], 2)
        self.assertGreater(runner.data.qvel[1], 0)  # Force actually accelerates the free root.
        runner.apply_wrench('root', [2, 0, 0], [0, 0, 3], frame='global')
        runner.step()
        np.testing.assert_array_equal(runner.data.xfrc_applied[body_id], [2, 0, 0, 0, 0, 3])
        runner.clear_wrenches('root')
        runner.step()
        np.testing.assert_array_equal(runner.data.xfrc_applied[body_id], 0)
        self.assertEqual(runner.data.xfrc_applied[other_id, 2], 2)
        runner.clear_wrenches()
        np.testing.assert_array_equal(runner.data.xfrc_applied, 0)
        runner.apply_wrench('root', [1, 0, 0])
        runner.reset()
        runner.step()
        np.testing.assert_array_equal(runner.data.xfrc_applied, 0)

    def test_wrench_validation_is_atomic(self):
        runner = self.runner()
        runner.apply_wrench('root', [1, 2, 3])
        before = runner.data.xfrc_applied.copy()
        for kwargs in ({'body_name': 'missing'}, {'body_name': 'world'},
                       {'frame': 'invalid'}, {'force': [np.nan, 0, 0]}, {'torque': [1, 2]}):
            with self.assertRaises(ValueError):
                runner.apply_wrench(**({'body_name': 'root'} | kwargs))
            np.testing.assert_array_equal(runner.data.xfrc_applied, before)

    def test_onnx_delay_and_noise_use_main_wrapper_api(self):
        from types import SimpleNamespace

        runner = self.runner()
        sim = runner.simulator
        try:
            initial = sim.data.ctrl.copy()
            sim.set_actuator_delay(.003)
            sim.run()
            desired = runner.outputs.q_des.copy()
            np.testing.assert_array_equal(sim.data.ctrl, initial)
            sim.run()
            sim.run()
            np.testing.assert_array_equal(sim.data.ctrl, initial)
            sim.run()
            np.testing.assert_array_equal(sim.data.ctrl[runner.actuator_ids], desired)
            self.assertEqual(len(runner.bridge._pending_commands), 0)
            sim.set_velocity_feedback_noise(self.names[0], .1)
            runner.bridge._noise_rng = SimpleNamespace(normal=lambda mean, std: np.full(std.shape, .25))
            velocity = sim.data.qvel[runner.dof_ids].copy()
            runner.evaluate()
            expected = velocity.copy()
            expected[0] += .25
            np.testing.assert_allclose(runner.inputs.v, expected)
            np.testing.assert_array_equal(sim.data.qvel[runner.dof_ids], velocity)
            sim.restore()
            self.assertEqual(runner.bridge.actuator_delay_s, 0)
            np.testing.assert_array_equal(runner.bridge.velocity_feedback_noise_std, 0)
        finally:
            runner.close()

    def test_imported_session_inherits_args_and_handles_resets(self):
        from contextlib import nullcontext

        parser = create_arg_parser()
        parser.add_argument('--reverse-at', type=float, default=0.02)
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / 'scene.xml'
            scene.write_text(self.xml)
            args = parser.parse_args(['--mjcf', str(scene), '--policy', str(self.example / 'policy.onnx'),
                                      '--headless', '--duration', '0.08', '--camera-distance', '2'])
            capture_times = []
            calls = []

            def control(sim):
                calls.append(sim.elapsed_time)
                self.assertIs(sim.simulator, sim.runner.simulator)
                sim.set_friction('root', [.7, .01, .001])
                root_geom = np.flatnonzero(sim.model.geom_bodyid == sim.model.body('root').id)[0]
                np.testing.assert_array_equal(sim.model.geom_friction[root_geom], [.7, .01, .001])
                sim.apply_wrench('root', force=[1, 0, 0], frame='local')
                self.assertEqual(sim.data.xfrc_applied[sim.model.body('root').id, 0], 1)
                sim.clear_wrenches('root')
                sim.set_velocity_command(0.2 if sim.elapsed_time < args.reverse_at else -0.2)
                if len(calls) == 2:
                    sim.reset()
                if len(calls) == 3:
                    sim.respawn()

            def video(runner, *args, **kwargs):
                return nullcontext(lambda: capture_times.append(runner.elapsed_time))

            with patch('policy_deploy_toolkit.simulate._video_recorder', side_effect=video):
                with Simulation(args) as sim:
                    sim.run(control)
                    self.assertEqual(sim.runner.control_steps, 4)
                    self.assertAlmostEqual(sim.elapsed_time, 0.08)
                    self.assertAlmostEqual(sim.data.time, 0.04)
                    self.assertFalse(sim.step())
                    self.assertFalse(sim.running)
                    np.testing.assert_allclose(sim.commands['base_velocity'], [-0.2, 0, 0])
                sim.close()
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(capture_times), 81)  # Initial frame, then each physics step.
            self.assertTrue(np.all(np.diff(capture_times) > 0))

    def test_action_sequence_drives_real_policy_forces_and_respawns(self):
        from contextlib import nullcontext
        from policy_deploy_toolkit.rollout_actions import (
            ActionSequence, ApplyForce, Respawn, VelocityCommand,
        )

        xml = self.xml.replace('<body name="root" pos="0 0 1">',
                               '<body name="root" pos="0 0 1"><freejoint/>')
        sequence = ActionSequence([
            VelocityCommand(vx=.2, duration=.04, sub_actions=[
                ApplyForce(body='root', fx=5, duration=.02),
            ]),
            Respawn(position=[0, 0, 1.1]),
            VelocityCommand(wz=.1, duration=.04),
            Respawn(position=[1, 2, 1.2], yaw=np.pi / 2),
        ])
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / 'scene.xml'
            scene.write_text(xml)
            with patch('policy_deploy_toolkit.simulate._video_recorder', return_value=nullcontext(None)):
                with Simulation.from_cli(['--mjcf', str(scene), '--policy', str(self.example / 'policy.onnx'),
                                         '--headless']) as sim:
                    sim.run(sequence)
                    self.assertTrue(sequence.finished)
                    self.assertEqual(sim.runner.control_steps, 4)
                    self.assertAlmostEqual(sim.elapsed_time, .08)
                    self.assertEqual(sim.data.time, 0)
                    np.testing.assert_allclose(sim.data.qpos[:7], [1, 2, 1.2, 2**-.5, 0, 0, 2**-.5])
                    np.testing.assert_array_equal(sim.commands['base_velocity'], [0, 0, 0])
                    np.testing.assert_array_equal(sim.data.xfrc_applied, 0)

    @unittest.skipUnless(importlib.util.find_spec('imageio') and importlib.util.find_spec('imageio_ffmpeg'),
                         'Optional video dependencies are not installed')
    def test_cli_headless(self):
        import imageio.v2 as imageio

        renderer = MagicMock()
        renderer.__enter__.return_value = renderer
        renderer.render.return_value = np.full((48, 64, 3), 100, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / 'scene.xml'
            video = Path(directory) / 'rollout.mp4'
            scene.write_text(self.xml)
            output = io.StringIO()
            with (redirect_stdout(output), patch('mujoco.Renderer', return_value=renderer),
                  patch('policy_deploy_toolkit.simulate.time', wraps=time) as clock):
                clock.sleep.side_effect = AssertionError('Headless must not sleep for pacing')
                main(['--mjcf', str(scene), '--policy', str(self.example / 'policy.onnx'),
                      '--headless', '--duration', '0.04', '--video', str(video),
                      '--video-size', '64', '48', '--track-body', 'base_link',
                      '--camera-distance', '2.5', '--geom-group-blacklist', '0', '2', '--profile'])
            with imageio.get_reader(str(video)) as reader:
                self.assertEqual(reader.count_frames(), 2)
                self.assertEqual(reader.get_meta_data()['fps'], 30)
                self.assertEqual(reader.get_data(0).shape, (48, 64, 3))
            camera = renderer.update_scene.call_args.kwargs['camera']
            self.assertEqual(camera.type, mujoco.mjtCamera.mjCAMERA_TRACKING)
            self.assertEqual(camera.distance, 2.5)
            scene_option = renderer.update_scene.call_args.kwargs['scene_option']
            np.testing.assert_array_equal(scene_option.geomgroup[:3], [0, 1, 0])
            self.assertEqual(camera.trackbodyid, mujoco.MjModel.from_xml_string(self.xml).body('base_link').id)
            renderer.__exit__.assert_called_once()
        self.assertIn('Simulated 0.040 s; evaluated 2 policy steps', output.getvalue())
        for stage in ('Profile:', 'Policy inference', 'Physics step', 'Render / readback',
                      'Video write / encode', 'Video finalization', 'Wall %'):
            self.assertIn(stage, output.getvalue())

    def test_viewer_uses_shared_force_visuals_and_action_text(self):
        from contextlib import nullcontext
        from policy_deploy_toolkit.rollout_actions import ActionSequence, ApplyForce, VelocityCommand

        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / 'scene.xml'
            scene.write_text(self.xml)
            sim = Simulation.from_cli(['--mjcf', str(scene), '--policy', str(self.example / 'policy.onnx'),
                                       '--duration', '.04', '--target-rtf', '1000'])
            viewer = MagicMock()
            viewer.__enter__.return_value = viewer
            viewer.__exit__.return_value = False
            viewer.lock.side_effect = lambda: nullcontext()
            viewer.is_running.return_value = True
            viewer.user_scn = mujoco.MjvScene(sim.model, maxgeom=100)
            actions = ActionSequence([VelocityCommand(vx=.1, duration=.04, sub_actions=[
                ApplyForce(body='root', fx=5, duration=.04),
            ])])
            with patch('mujoco.viewer.launch_passive', return_value=viewer):
                with sim:
                    self.assertIs(sim.simulator.viewer, viewer)
                    actions(sim)
                    sim.step()
                    viewer.sync.assert_called_with(state_only=True)
                    self.assertEqual(viewer.user_scn.ngeom, 1)
                    self.assertEqual(viewer.user_scn.geoms[0].type, mujoco.mjtGeom.mjGEOM_ARROW)
                    text = viewer.set_texts.call_args.args[0][0]
                    self.assertEqual(text[1], mujoco.mjtGridPos.mjGRID_TOPLEFT)
                    self.assertIn('Velocity', text[2])
                    self.assertIn('Force root', text[2])
                    sim.set_status_text('')
                    sim.simulator._next_viewer_sync_wall = 0
                    sim.simulator._sync_viewer()
                    viewer.set_texts.assert_called_with([])
            self.assertIsNone(sim.simulator.viewer)
            viewer.__exit__.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec('imageio') and importlib.util.find_spec('imageio_ffmpeg'),
                         'Optional video dependencies are not installed')
    def test_video_keeps_scene_geometry_adds_force_and_action_text(self):
        import imageio.v2 as imageio
        from policy_deploy_toolkit.rollout_actions import ActionSequence, ApplyForce, VelocityCommand

        renderer = MagicMock()
        renderer.__enter__.return_value = renderer
        renderer.render.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / 'scene.xml'
            video = Path(directory) / 'actions.mp4'
            scene.write_text(self.xml)
            sim = Simulation.from_cli(['--mjcf', str(scene), '--policy', str(self.example / 'policy.onnx'),
                                       '--headless', '--duration', '.04', '--video', str(video),
                                       '--video-size', '320', '240'])
            renderer.scene = mujoco.MjvScene(sim.model, maxgeom=100)
            renderer.scene.geoms[0].type = mujoco.mjtGeom.mjGEOM_BOX
            renderer.update_scene.side_effect = lambda *args, **kwargs: setattr(renderer.scene, 'ngeom', 1)
            sequence = ActionSequence([VelocityCommand(vx=.1, duration=.04, sub_actions=[
                ApplyForce(body='root', fx=5, duration=.04),
            ])])
            with patch('mujoco.Renderer', return_value=renderer):
                with sim:
                    sim.run(sequence)
            self.assertEqual(renderer.scene.ngeom, 2)
            self.assertEqual(renderer.scene.geoms[0].type, mujoco.mjtGeom.mjGEOM_BOX)
            self.assertEqual(renderer.scene.geoms[1].type, mujoco.mjtGeom.mjGEOM_ARROW)
            with imageio.get_reader(str(video)) as reader:
                first = reader.get_data(0)
                self.assertGreater(first[:150].max(), 150)  # White action text in frame zero.
                self.assertLess(first[200:].max(), 5)
                self.assertEqual(reader.count_frames(), 2)

    def test_integrated_velocity_actuator_initialization(self):
        # Unused joints retain their native gains and integrated position targets.
        extra = '<body name="wheel"><joint name="wheel"/><geom size=".02" mass="1"/></body>'
        xml = self.xml.replace('</body><geom type="plane"', extra + '</body><geom type="plane"')
        xml = xml.replace('</actuator>', '<intvelocity joint="wheel" kp="5" kv="1" actrange="-10 10"/></actuator>')
        runner = self.runner(xml)
        self.assertTrue(runner.integral[-1])
        actuator = runner.actuator_ids[-1]
        self.assertEqual(runner.data.act[runner.model.actuator_actadr[actuator]], 0)
        runner.step()
        self.assertTrue(np.isfinite(runner.data.qpos).all())

    @unittest.skipUnless(importlib.util.find_spec('imageio') and importlib.util.find_spec('imageio_ffmpeg'),
                         'Optional video dependencies are not installed')
    def test_video_timeline_survives_repeated_resets(self):
        import imageio.v2 as imageio

        renderer = MagicMock()
        renderer.__enter__.return_value = renderer
        renderer.render.return_value = np.zeros((48, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / 'scene.xml'
            video = Path(directory) / 'episodes.mp4'
            scene.write_text(self.xml)
            with patch('mujoco.Renderer', return_value=renderer):
                with Simulation.from_cli([
                    '--mjcf', str(scene), '--policy', str(self.example / 'policy.onnx'),
                    '--headless', '--duration', '0.1', '--video', str(video),
                    '--video-size', '64', '48',
                ]) as sim:
                    def control(sim):
                        if sim.elapsed_time > 0:
                            sim.reset()
                    sim.run(control)
                    self.assertEqual(sim.runner.control_steps, 5)
                    self.assertAlmostEqual(sim.elapsed_time, 0.1)
                    self.assertAlmostEqual(sim.data.time, 0.02)
            with imageio.get_reader(str(video)) as reader:
                self.assertEqual(reader.count_frames(), 3)
            self.assertEqual(renderer.render.call_count, 3)

    def test_height_scan_policy(self):
        example = ROOT / 'examples/kyon_full_locomotion_rough_heightscan'
        runner = MujocoPolicyRunner(mujoco.MjModel.from_xml_string(self.xml),
                                    example / 'policy.onnx', example / 'deploy_metadata.yaml')
        runner.step()
        self.assertEqual(runner.info.height_scan_size, 187)
        np.testing.assert_allclose(runner.inputs.height_scan, -np.ones(187))
        self.assertTrue(np.isfinite(runner.outputs.raw_observation).all())


@unittest.skipIf(mujoco is None, 'Optional MuJoCo dependency is not installed')
class ProfileWindowTest(unittest.TestCase):
    def test_rtf_window_excludes_startup_earlier_speed_and_shutdown(self):
        from policy_deploy_toolkit.simulate import _Profile

        clock = SimpleNamespace(now=0.)
        with patch('policy_deploy_toolkit.simulate.time', SimpleNamespace(perf_counter=lambda: clock.now)):
            profile = _Profile()
            clock.now = 30.  # Slow setup is excluded.
            profile.start_loop(0., clock.now)
            with redirect_stdout(io.StringIO()):
                for second in range(1, 11):
                    clock.now = 30. + second
                    sim_time = second if second <= 5 else 5 + 2 * (second - 5)
                    profile.progress(sim_time)
            rtf, elapsed = profile._window_rtf()
            self.assertAlmostEqual(rtf, 2.)
            self.assertAlmostEqual(elapsed, 5.)
            clock.now = 90.  # Slow finalization does not dilute the last window.
            output = io.StringIO()
            with redirect_stdout(output):
                profile.report(15.)
            self.assertIn('RTF 2.00 (last 5.00 s of loop wall time)', output.getvalue())

    def test_rtf_interpolates_boundary_and_handles_short_runs(self):
        from policy_deploy_toolkit.simulate import _Profile

        clock = SimpleNamespace(now=0.)
        with patch('policy_deploy_toolkit.simulate.time', SimpleNamespace(perf_counter=lambda: clock.now)):
            profile = _Profile()
            profile.start_loop(0., 0.)
            self.assertEqual(profile._window_rtf(), (None, 0.))
            clock.now = 2.
            profile.progress(4.)
            self.assertEqual(profile._window_rtf(), (2., 2.))
            clock.now = 6.
            with redirect_stdout(io.StringIO()):
                profile.progress(6.)
            # At t=1 the interpolated simulation clock was 2; (6-2)/(6-1)=.8.
            rtf, elapsed = profile._window_rtf()
            self.assertAlmostEqual(rtf, .8)
            self.assertEqual(elapsed, 5.)


if __name__ == '__main__':
    unittest.main()
