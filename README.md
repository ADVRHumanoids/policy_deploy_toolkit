# policy_deploy_toolkit

Minimal ONNX policy deployment for XBot2.

The package wraps an exported policy as a small runtime pipeline:

1. Load an ONNX model with ONNX Runtime.
2. Load the matching deployment metadata from YAML.
3. Map policy joint names to the robot joints exposed by `RobotInterface`.
4. Build observations from the live robot state, commands, IMU data, and previous action.
5. Run the policy and translate the output into XBot2 position, velocity, effort, stiffness, damping, and control-mode commands.

The metadata drives most of the wiring: observation terms, action terms, joint ordering, default joint positions, gains, command dimensions, and the control period. The executable expects the robot configuration to be available from the ROS 2 parameter tree under `xbotcore/`.

## Build

From this package directory:

```bash
cmake -S . -B build
cmake --build build
```

Or with forest:

```bash
forest grow xbot2_policy_toolkit -j4
```

## Run the Example

Start the robot or simulator that provides the XBot2 interface and `xbotcore/` parameters, then run:

```bash
ros2 launch policy_deploy_toolkit run_example.launch.xml  # [enable_joy:=true] [example:=kyon_full_locomotion_rough] (defaults to kyon_full_locomotion_flat)
```

The executable runs a control loop, feeds the policy from the current robot state, and sends the resulting command back through `RobotInterface`.
Commands are received through the `/policy_deploy_node/commands/<command-name>` topic. 
