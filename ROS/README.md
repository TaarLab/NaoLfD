# ROS1 NAO Robot Communication

Two ROS1 (catkin / `rospy`) packages that expose the same robot interface. The existing HTTP server in `RealWorldScripts/nao_server.py` is unchanged.

| Version | Package | Backend | When to use |
|---------|---------|---------|-------------|
| A | `nao_ros_http_bridge` | HTTP `nao_server` on `:5001` | Safe with current Python 2 / NAOqi setup |
| B | `nao_ros_direct` | `ALProxy("ALMotion")` | Machine that can import `naoqi` + `almath` |

**Do not run A and B at the same time** against the same robot.

## Shared interface

**Services (one-shot / regular)**
- `/nao/move_arms` (`nao_lfd_msgs/MoveArms`)
- `/nao/get_angles` (`nao_lfd_msgs/GetAngles`)

**Topics (closed loop)**
- `/nao/joint_command` (`nao_lfd_msgs/ArmJoints`) — stream joint targets
- `/nao/joint_states` (`nao_lfd_msgs/ArmJoints`) — published ~10 Hz

## Build (ROS Melodic or Noetic)

```bash
cd ros_ws
source /opt/ros/<distro>/setup.bash
catkin_make
source devel/setup.bash
```

## Version A — HTTP bridge

Start HTTP first:

```bash
python RealWorldScripts/nao_server.py
```

Then:

```bash
roslaunch nao_ros_http_bridge nao_http_bridge.launch
```

Params: `nao_http_url` (default `http://127.0.0.1:5001`), `http_timeout`, `joint_state_rate`.

## Version B — direct ALProxy

```bash
roslaunch nao_ros_direct nao_direct.launch
```

Params: `nao_ip`, `nao_port`, `move_speed`, `joint_state_rate`, `set_stiffness_on_start`, `home_on_start`.

## Examples

```bash
# One-shot move
rosservice call /nao/move_arms "joints:
  left_shoulder_pitch: 0.0
  left_shoulder_roll: 0.0
  left_elbow_yaw: -42.0
  left_elbow_roll: -5.0
  right_shoulder_pitch: 0.0
  right_shoulder_roll: 0.0
  right_elbow_yaw: 0.0
  right_elbow_roll: 0.0"

# One-shot read
rosservice call /nao/get_angles "{}"

# Closed-loop command stream
rostopic pub /nao/joint_command nao_lfd_msgs/ArmJoints "{left_shoulder_pitch: 10.0, left_shoulder_roll: 0.0, left_elbow_yaw: -42.0, left_elbow_roll: -5.0, right_shoulder_pitch: 0.0, right_shoulder_roll: 0.0, right_elbow_yaw: 0.0, right_elbow_roll: 0.0}"

# Closed-loop feedback
rostopic echo /nao/joint_states
```

## Notes

- This is **ROS1**, not ROS2 (`rospy` / catkin only).
- C# Kinect RealWorld client continues to use HTTP `:5001` directly.
- Predictor Flask (`:5000`) is unchanged.
