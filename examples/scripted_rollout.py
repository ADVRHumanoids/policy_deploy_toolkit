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
    parser.add_argument('--speed', type=float, default=1., help='Forward command in m/s')
    parser.add_argument('--force-body', help='Body receiving disturbances (default: --base-body)')
    parser.add_argument('--respawn-position', nargs=3, type=float, default=(0., 0., .5),
                        metavar=('X', 'Y', 'Z'), help='Final floating-root world position')
    args = parser.parse_args()

    try:
        actions = [
            VelocityCommand(vx=args.speed, duration=2.),
            VelocityCommand(wz=1., duration=2.),
            VelocityCommand(vx=args.speed, duration=10., sub_actions=[
                ApplyForce(body=args.force_body or args.base_body, fy=1000., duration=.2),
                EmptyAction(duration=.5),
                ApplyForce(body=args.force_body or args.base_body, fy=-1000., duration=.2),
            ]),
            Respawn(position=args.respawn_position, yaw=0.),
            VelocityCommand(vx=args.speed, duration=5.),
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
