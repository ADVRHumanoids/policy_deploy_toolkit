#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.wait_for_message import wait_for_message
from xbot_msgs.msg import JointState


TOPIC = "/xbotcore/joint_states"
TAG = "latest"
RELEASES_URL = "https://github.com/ADVRHumanoids/policy_deploy_toolkit/releases"


class LoadPolicyNode(Node):
    def __init__(self):
        super().__init__("load_policy")
        self.__steering_wheels = False
        self.__simple_wheels = False
        self.__legged = False
        self.__arms = False
        self.__is_callback_done = False
        self.__policy_name = str

    def callback(self, message):
        if self.__is_callback_done:
            return

        self.__is_callback_done = True
        self.__joint_names = message.name

        if 'ankle_yaw_1' in self.__joint_names:
            self.__steering_wheels = True
        elif 'wheel_joint_1' in self.__joint_names and 'ankle_yaw_1' not in self.__joint_names:
            self.__simple_wheels = True
        elif 'wheel_joint_1' not in self.__joint_names and 'ankle_yaw_1' not in self.__joint_names:
            self.__legged = True
        
        if 'shoulder_yaw_1' in self.__joint_names:
            self.__arms = True

        self.load_policy()

    def load_policy(self):
        if not self.__is_callback_done:
            raise RuntimeError("No joint states received yet")
        
        if not self.__steering_wheels and not self.__simple_wheels and not self.__legged:
            self.get_logger().error("No valid configuration found")
            raise RuntimeError("No valid configuration found")
        
        if self.__legged and not self.__arms:
            self.__policy_name = "kyon_legged"
        elif self.__legged and self.__arms:
            self.__policy_name = "kyon_legged_arms"
        elif self.__steering_wheels and not self.__arms:
            self.__policy_name = "kyon_steering_wheels"
        elif self.__steering_wheels and self.__arms:
            self.__policy_name = "kyon_steering_wheels_arms"
        elif self.__simple_wheels and not self.__arms:
            self.__policy_name = "kyon_simple_wheels"
        elif self.__simple_wheels and self.__arms:
            self.__policy_name = "kyon_simple_wheels_arms"

        examples_dir = (
            Path(get_package_share_directory("policy_deploy_toolkit")) / "examples"
        )
        policy_dir = examples_dir / self.__policy_name
        if not policy_dir.is_dir():
            
            archive = f"{self.__policy_name}.zip"
            if TAG == "latest":
                url = f"{RELEASES_URL}/latest/download/{archive}"
            else:
                url = f"{RELEASES_URL}/download/{TAG}/{archive}"

            print(f"Policy {self.__policy_name} not found, downloading from {url}...")

            subprocess.run(
                ["wget", url, "-O", archive], cwd=examples_dir, check=True
            )
            subprocess.run(
                ["unzip", archive, "-d", policy_dir],
                cwd=examples_dir,
                stdout=sys.stderr,
                check=True,
            )

    def get_policy_name(self):
        return self.__policy_name


def main():
    global TAG

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=TAG)
    args, ros_args = parser.parse_known_args()
    TAG = args.tag

    rclpy.init(args=ros_args)
    node = LoadPolicyNode()

    topics = dict(node.get_topic_names_and_types())
    if TOPIC not in topics:
        node.get_logger().error(f"Topic {TOPIC} does not exist")
        node.destroy_node()
        rclpy.shutdown()
        return 1
    else:
        _, message = wait_for_message(
            JointState,
            node,
            TOPIC,
            qos_profile=qos_profile_sensor_data,
        )
        node.callback(message)
        policy_name = node.get_policy_name()

    node.destroy_node()
    rclpy.shutdown()

    print(policy_name, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
