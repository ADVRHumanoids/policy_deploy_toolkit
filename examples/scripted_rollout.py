"""Run a sequence of commands and disturbances using all simulator CLI options.

Example:
    python examples/scripted_rollout.py --mjcf scene.xml --policy policy.onnx \
        --headless --speed 1.0
"""

from policy_deploy_toolkit.simulate import Simulation, create_arg_parser
from policy_deploy_toolkit.rollout_actions import (
    ActionSequence, ApplyForce, EmptyAction, Respawn, VelocityCommand,
)


def main():
    parser = create_arg_parser(description=__doc__)
    args = parser.parse_args()

    try:
        actions = [
            # climb stairs
            Respawn(terrain='pyramid_stairs_inv_5', position=[0.8, 0, -0.3]),
            VelocityCommand(vx=0.5, duration=10.),

            # descend stairs with disturbances
            Respawn(terrain='pyramid_stairs_5', position=[0.8, 0, -0.3]),
            VelocityCommand(vx=1.0, duration=6., sub_actions=[
                ApplyForce(body='base_link', fy=1000., duration=.2, start_offset=4.0),
                EmptyAction(duration=1.0),
                ApplyForce(body='base_link', fy=-1000., duration=.2),
            ]),

            # climb hill
            Respawn(terrain='hf_pyramid_slope_inv_3', position=[0.8, 0, -0.3]),
            VelocityCommand(vx=1.0, duration=6., sub_actions=[
                ApplyForce(body='base_link', fy=1000., duration=.2, start_offset=4.0),
                EmptyAction(duration=1.0),
                ApplyForce(body='base_link', fy=-1000., duration=.2),
            ]),

            # rough boxes_1
            Respawn(terrain='boxes_5', position=[0.8, 0, -0.3]),
            VelocityCommand(vx=1.0, duration=6., sub_actions=[
                ApplyForce(body='base_link', fy=1000., duration=.2, start_offset=4.0),
                EmptyAction(duration=1.0),
                ApplyForce(body='base_link', fy=-1000., duration=.2),
            ]),
        ]
        sequence = ActionSequence(actions)
        args.duration = min(args.duration, sequence.duration) if args.duration is not None else sequence.duration
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    try:
        with Simulation(args) as sim:
            sim.run(sequence)
            # A duration limit can stop the loop just before its final callback.
            # Flush completion events (not another physics step) at that boundary.
            if not sequence.finished and sim.elapsed_time >= sequence.duration:
                sequence(sim)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
