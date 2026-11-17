#!/usr/bin/env bash
# MuJoCo sim + xbot2 + lio_mapping, one window each in tmux session sim_lio; base layer for the policy launchers.
# Run inside docker-dev-1. Env overrides: WORLD LIO_MODE=localization|slam MAP_DIR ATTACH=0 (no tmux attach)
set -e
WS=/home/user/forest_ws
WORLD=${WORLD:-$WS/src/iit-kyon-ros-pkg/kyon_mujoco/world/rough_terrain/rough_terrain.xml}
LIO_MODE=${LIO_MODE:-localization}
S=sim_lio
ENV="source $WS/setup.bash; export CYCLONEDDS_URI=file://$WS/cyclonedds.xml"

# Shell snippet blocking until one message arrives on topic $1.
wait_msg() { echo "until ros2 topic echo --once $1 >/dev/null 2>&1; do sleep 1; done"; }

# New tmux window $1 running command $2 with the workspace environment.
window() {
    tmux new-window -t $S -n "$1"
    tmux send-keys -t "$S:$1" "$ENV; $2" C-m
}

# Brings the session to the foreground on the mujoco window.
attach() {
    tmux select-window -t $S:mujoco
    if [ -n "$TMUX" ]; then tmux switch-client -t $S; else tmux attach -t $S; fi
}

tmux kill-session -t $S 2>/dev/null || true
tmux new-session -d -s $S -n mujoco
tmux send-keys -t $S:mujoco "$ENV; ros2 launch kyon_mujoco kyon_world.launch lidar:=true imu:=true \
world:=$WORLD heightscan_viz_topic:=/heightscan enable_ee_markers:=true" C-m

window xbot2 "$(wait_msg /clock); rm -f /tmp/mj_simulator_server.sock.client.*; xbot2-core --hw mj"

if [ "$LIO_MODE" = localization ]; then
    LIO_ARGS="mode:=localization map_dir:=${MAP_DIR:-$WS/maps/rough} elevation_prior:=true"
else
    LIO_ARGS="mode:=slam map_dir:=${MAP_DIR:-$WS/maps/sim_$(date +%Y%m%d_%H%M%S)}"
fi
window lio "$(wait_msg /xbotcore/joint_states); ros2 launch lio_mapping bag.launch.py \
fast_lio_config:=fast_lio_hesai.yaml,fast_lio_sim.yaml $LIO_ARGS base_frame:=pelvis \
elevation_mapping:=true deskew_device:=gpu tmux:=false"

if [ "${ATTACH:-1}" = 1 ]; then attach; fi
