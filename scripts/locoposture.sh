#!/usr/bin/env bash
# sim_lio.sh (MuJoCo + xbot2 + lio_mapping) plus the policy with LIO height scan, in tmux session sim_lio.
# Run inside docker-dev-1. Env overrides: those of sim_lio.sh, EXAMPLE JOY=true|false KEYBOARD=1
# LOCOMAN=1: legs (EXAMPLE) + arms EE tracking (run_locoman) with keyboard_teleop.py
set -e
EXAMPLE=${EXAMPLE:-kyon_full_locoposture_rough}
JOY=${JOY:-true}
ATTACH=0 source "$(dirname "$(readlink -f "$0")")/sim_lio.sh"

if [ "${LOCOMAN:-0}" = 1 ]; then
    POLICY_CMD="ros2 launch policy_deploy_toolkit run_locoman.launch.xml legs:=$EXAMPLE lio_heightscan:=true"
else
    POLICY_CMD="ros2 launch policy_deploy_toolkit run_example.launch.xml example:=$EXAMPLE lio_heightscan:=true enable_joy:=$JOY"
fi
window policy "$(wait_msg /xbotcore/joint_states); $(wait_msg /localization/fast_lio_odom_bridge/odometry); $POLICY_CMD"

if [ "${LOCOMAN:-0}" = 1 ]; then
    window keyboard "$(wait_msg /arms_policy_node/ee_measured); ros2 run policy_deploy_toolkit keyboard_teleop.py"
elif [ "${KEYBOARD:-0}" = 1 ]; then
    window keyboard "ros2 run teleop_twist_keyboard teleop_twist_keyboard \
--ros-args -r cmd_vel:=/policy_deploy_node/commands/base_velocity"
fi

attach
