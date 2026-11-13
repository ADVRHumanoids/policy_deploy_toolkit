"""Action scheduling, scopes, and episode resets on a deterministic session clock."""
from types import SimpleNamespace
import math
import sys
import unittest

sys.path.insert(0, sys.argv.pop(1))
from policy_deploy_toolkit.rollout_actions import (
    ActionSequence, ApplyForce, EmptyAction, Respawn, VelocityCommand,
)


class Simulation:
    def __init__(self):
        self.elapsed_time = 0.
        self.data = SimpleNamespace(time=0.)
        self.commands = {'base_velocity': [0., 0., 0.]}
        self.events = []
        self.status_text = ''

    def set_status_text(self, text):
        self.status_text = text

    def set_command(self, name, value):
        self.commands[name] = list(value)
        self.events.append(('velocity', self.elapsed_time, tuple(value)))

    def apply_wrench(self, body, **kwargs):
        self.events.append(('force', self.elapsed_time, body, kwargs))

    def respawn(self, **kwargs):
        self.events.append(('respawn', self.elapsed_time, kwargs))
        self.data.time = 0

    def tick(self, sequence, at):
        self.data.time += at - self.elapsed_time
        self.elapsed_time = at
        return sequence(self)


class RolloutActionsTest(unittest.TestCase):
    def test_example_flushes_completion_at_duration_limit(self):
        import importlib.util
        import runpy
        from pathlib import Path
        from unittest.mock import patch

        if importlib.util.find_spec('mujoco') is None:
            self.skipTest('Optional MuJoCo dependency is not installed')
        made = []

        class Session(Simulation):
            def __init__(self, args):
                super().__init__()
                self.args = args
                made.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def run(self, callback):
                self.sequence = callback
                while self.elapsed_time < self.args.duration - 1e-9:
                    if callback(self) is False:
                        break
                    self.elapsed_time = round(self.elapsed_time + .02, 9)
                    self.data.time += .02

        main = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'examples/scripted_rollout.py'))['main']
        with (patch.dict(main.__globals__, {'Simulation': Session}),
              patch.object(sys, 'argv', ['scripted_rollout.py', '--mjcf', 'scene.xml',
                                        '--policy', 'policy.onnx'])):
            main()
        self.assertEqual(made[0].elapsed_time, made[0].sequence.duration)
        self.assertTrue(made[0].sequence.finished)
        self.assertTrue(any(event[0] == 'respawn' for event in made[0].events))
        self.assertEqual(made[0].commands['base_velocity'], [0, 0, 0])

    def test_status_describes_parent_force_and_wait(self):
        sim = Simulation()
        sequence = ActionSequence([VelocityCommand(vx=1, duration=1, sub_actions=[
            ApplyForce(body='pelvis', fy=100, start_offset=.1, duration=.2),
            EmptyAction(duration=.2),
        ])])
        sim.tick(sequence, 0)
        self.assertIn('Velocity vx=1', sim.status_text)
        self.assertNotIn('Force', sim.status_text)
        sim.tick(sequence, .1)
        self.assertIn('Velocity vx=1', sim.status_text)
        self.assertIn('Force pelvis', sim.status_text)
        self.assertIn('100', sim.status_text)
        sim.tick(sequence, .3)
        self.assertNotIn('Force', sim.status_text)
        self.assertIn('Wait', sim.status_text)
        sim.tick(sequence, 1)
        self.assertEqual(sim.status_text, 'Actions complete')

    def test_drafted_sequence_children_run_during_parent(self):
        sim = Simulation()
        sequence = ActionSequence([
            VelocityCommand(vx=1, duration=2),
            VelocityCommand(wz=1, duration=2),
            VelocityCommand(vx=1, duration=10, sub_actions=[
                ApplyForce(fy=1000, duration=.2), EmptyAction(duration=.5),
                ApplyForce(fy=-1000, duration=.2),
            ]),
            Respawn(position=[0, 0, .5]),
        ])
        self.assertEqual(sequence.duration, 14)
        for step in range(701):
            running = sim.tick(sequence, step * .02)
            if step < 700:
                self.assertTrue(running)
        self.assertFalse(running)
        self.assertTrue(sequence.finished)
        forces = [item for item in sim.events if item[0] == 'force']
        self.assertEqual(len(forces), 2)
        self.assertAlmostEqual(forces[0][1], 4.)
        self.assertAlmostEqual(forces[1][1], 4.7)
        self.assertEqual(forces[0][3]['force'], [0, 1000, 0])
        self.assertEqual(forces[1][3]['force'], [0, -1000, 0])
        self.assertAlmostEqual(forces[0][3]['duration'], .2)
        self.assertAlmostEqual(forces[1][3]['duration'], .2)
        commands = [item for item in sim.events if item[0] == 'velocity']
        self.assertEqual(commands[0], ('velocity', 0., (1., 0., 0.)))
        self.assertIn(('velocity', 2., (0., 0., 1.)), commands)
        self.assertEqual(sim.commands['base_velocity'], [0, 0, 0])
        self.assertEqual(sim.events[-1][0], 'respawn')
        self.assertEqual(sim.events[-1][1], 14)

    def test_offsets_and_nested_command_restoration(self):
        sim = Simulation()
        sim.commands['base_velocity'] = [.2, 0, 0]
        sequence = ActionSequence([
            VelocityCommand(vx=1, duration=2, start_offset=.5, sub_actions=[
                VelocityCommand(wz=2, start_offset=.2, duration=.3),
                EmptyAction(duration=.1),
                VelocityCommand(vy=1, duration=.3, start_offset=.1),
            ]),
        ])
        sim.tick(sequence, 0)
        self.assertEqual(sim.events, [])
        sim.tick(sequence, .5)
        self.assertEqual(sim.commands['base_velocity'], [1, 0, 0])
        sim.tick(sequence, .7)
        self.assertEqual(sim.commands['base_velocity'], [0, 0, 2])
        sim.tick(sequence, 1.)
        self.assertEqual(sim.commands['base_velocity'], [1, 0, 0])
        sim.tick(sequence, 1.2)
        self.assertEqual(sim.commands['base_velocity'], [0, 1, 0])
        sim.tick(sequence, 1.5)
        self.assertEqual(sim.commands['base_velocity'], [1, 0, 0])
        self.assertFalse(sim.tick(sequence, 2.5))
        self.assertEqual(sim.commands['base_velocity'], [.2, 0, 0])

    def test_start_boundary_and_floating_point_cleanup_order(self):
        sim = Simulation()
        sequence = ActionSequence([
            VelocityCommand(vx=1, duration=.3, sub_actions=[
                EmptyAction(duration=.1), VelocityCommand(wz=1, duration=.2),
            ]),
            Respawn(position=[1, 2, 3]),
            VelocityCommand(vy=1, duration=.1),
        ])
        sim.tick(sequence, 0)
        sim.tick(sequence, .1)
        self.assertTrue(sim.tick(sequence, .3))
        self.assertEqual(sim.commands['base_velocity'], [0, 1, 0])
        self.assertFalse(sim.tick(sequence, .4))
        self.assertEqual(sim.commands['base_velocity'], [0, 0, 0])
        at_boundary = [item[0] for item in sim.events if item[1] == .3]
        self.assertEqual(at_boundary, ['velocity', 'velocity', 'respawn', 'velocity'])

    def test_session_clock_survives_instantaneous_respawn(self):
        sim = Simulation()
        sim.elapsed_time = 10.
        sequence = ActionSequence([
            EmptyAction(duration=1), Respawn(position=[1, 2, 3], yaw=math.pi / 2),
            VelocityCommand(vx=.5, duration=1),
        ])
        sim.tick(sequence, 10.)
        sim.tick(sequence, 11.)
        self.assertEqual(sim.data.time, 0)
        self.assertEqual(sim.commands['base_velocity'], [.5, 0, 0])
        quaternion = [item for item in sim.events if item[0] == 'respawn'][0][2]['quaternion']
        for actual, expected in zip(quaternion, [2**-.5, 0, 0, 2**-.5]):
            self.assertAlmostEqual(actual, expected)
        self.assertFalse(sim.tick(sequence, 12.))
        sim.tick(sequence, 12.)
        self.assertEqual(sum(item[0] == 'respawn' for item in sim.events), 1)

    def test_late_start_uses_remaining_force_duration_and_skips_expired_force(self):
        sim = Simulation()
        sequence = ActionSequence([
            ApplyForce(body='pelvis', fx=10, frame='local', duration=.1, start_offset=.01),
            ApplyForce(fx=20, duration=.01),
            Respawn(position=[0, 0, 1]),
        ])
        sim.tick(sequence, 0)
        sim.tick(sequence, .02)
        force = sim.events[0]
        self.assertEqual(force[2], 'pelvis')
        self.assertEqual(force[3]['frame'], 'local')
        self.assertAlmostEqual(force[3]['duration'], .09)
        self.assertFalse(sim.tick(sequence, .14))
        self.assertEqual(sum(item[0] == 'force' for item in sim.events), 1)
        self.assertEqual(sum(item[0] == 'respawn' for item in sim.events), 1)

    def test_definitions_are_immutable_copied_and_reusable(self):
        children = [VelocityCommand(vx=1, duration=.1)]
        action = EmptyAction(duration=.2, sub_actions=children)
        children.append(EmptyAction(duration=10))
        self.assertEqual(len(action.sub_actions), 1)
        self.assertEqual(EmptyAction().sub_actions, ())
        for _ in range(2):
            sim = Simulation()
            sequence = ActionSequence([action])
            sim.tick(sequence, 0)
            self.assertEqual(sim.commands['base_velocity'], [1, 0, 0])
            self.assertFalse(sim.tick(sequence, .2))
            self.assertEqual(sim.commands['base_velocity'], [0, 0, 0])

    def test_validation_and_empty_sequence(self):
        for kwargs in ({'duration': -1}, {'duration': float('nan')}, {'start_offset': -1},
                       {'duration': float('inf')}, {'duration': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EmptyAction(**kwargs)
        with self.assertRaises(ValueError):
            EmptyAction(duration=.1, sub_actions=[EmptyAction(duration=.2)])
        with self.assertRaises(ValueError):
            ActionSequence([EmptyAction(duration=1e308), EmptyAction(duration=1e308)])
        for create in (lambda: VelocityCommand(wz=float('inf')),
                       lambda: VelocityCommand(duration=0), lambda: ApplyForce(duration=0),
                       lambda: ApplyForce(frame='invalid'), lambda: ApplyForce(torque=[1, 2]),
                       lambda: Respawn(position=[0, 0]), lambda: Respawn(position=[0, 0, 1], yaw=float('nan'))):
            with self.assertRaises(ValueError):
                create()
        self.assertFalse(ActionSequence([])(Simulation()))
        sequence = ActionSequence([EmptyAction()])
        sim = Simulation()
        sim.tick(sequence, 5)
        with self.assertRaises(ValueError):
            sim.tick(sequence, 4)
        with self.assertRaises(ValueError):
            sequence(Simulation())


if __name__ == '__main__':
    unittest.main()
