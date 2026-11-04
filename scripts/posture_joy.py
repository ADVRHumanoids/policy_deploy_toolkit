#!/usr/bin/env python3
"""Hold joystick buttons to ramp the posture command [roll, pitch, height] of the deploy node."""
import rclpy
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray

# (param name, default button index (ros2 joy, SDL game controller layout), field, direction)
BUTTONS = [
    ('button_height_up', 5, 2, +1),    # R1
    ('button_height_down', 4, 2, -1),  # L1
    ('button_roll_neg', 2, 0, -1),     # X
    ('button_roll_pos', 1, 0, +1),     # B
    ('button_pitch_pos', 3, 1, +1),    # Y
    ('button_pitch_neg', 0, 1, -1),    # A
]
DT = 0.02


def main():
    rclpy.init()
    node = rclpy.create_node('posture_joy')
    param = lambda name, default: node.declare_parameter(name, default).value
    bindings = [(param(name, b), field, sign) for name, b, field, sign in BUTTONS]
    reset_button = param('button_reset', -1)  # Start; -1 disables
    # half the training rate limit (0.5, 0.5 rad/s, 0.2 m/s); limits match the PostureCommand ranges
    rate = param('rate', [0.25, 0.25, 0.1])
    lo = param('min', [-0.3, -0.3, 0.30])
    hi = param('max', [0.3, 0.3, 0.65])
    nominal = param('nominal', [0.0, 0.0, 0.562])

    pub = node.create_publisher(Float64MultiArray, '/policy_deploy_node/commands/base_posture/raw', 1)
    posture = list(nominal)
    pressed = []

    def on_joy(msg):
        nonlocal pressed
        pressed = msg.buttons

    def is_pressed(b):
        return 0 <= b < len(pressed) and pressed[b]

    def tick():
        nonlocal posture
        if is_pressed(reset_button):
            posture = list(nominal)
        moved = False
        for b, field, sign in bindings:
            if is_pressed(b):
                posture[field] = min(max(posture[field] + sign * rate[field] * DT, lo[field]), hi[field])
                moved = True
        if moved or is_pressed(reset_button):
            node.get_logger().info(
                f'roll {posture[0]:+.2f} rad  pitch {posture[1]:+.2f} rad  height {posture[2]:.3f} m',
                throttle_duration_sec=0.3)
        pub.publish(Float64MultiArray(data=posture))

    node.create_subscription(Joy, '/joy', on_joy, 10)
    node.create_timer(DT, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
