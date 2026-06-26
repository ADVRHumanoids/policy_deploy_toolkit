# kyon_full_locomotion_rough_heightscan

Robot configuration:
 - upper body: yes
 - wheels: no

Trained from:
 - Git repo `git@github.com:ADVRHumanoids/kyon_isaac.git` 
 - Branch `locomotion` at commit `2ac83fb1d7b15c0b89382996ec904c4256f5ab21`
 - Folder `kyon_isaac/scripts/rsl_rl/logs/rsl_rl/kyon_rough/mlp_with_heightfield/exported`
 - Task name `Isaac-Velocity-Rough-KyonFull-v0`

Deployment (SIM):
 - Simulator: gazebo
 - Launch simulator with: `ros2 launch kyon_gazebo kyon_world.launch world_name:=kyon_pyramid.world hesai_jt128:=true`
 - Run height scan generator from `ADVRHumanoids/isaaclab_height_scan_builder.git`
   ```bash
   cd isaaclab_height_scan_builder
   source launch/roslaunch.sh
   ```