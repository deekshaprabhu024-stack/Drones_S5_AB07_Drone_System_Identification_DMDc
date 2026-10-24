# Drone System Identification Using DMDc — Gazebo Edition

A Gazebo Harmonic + ROS 2 Jazzy implementation of the quadrotor system-identification workflow.

> **Package renamed** from `drone_dmdc_gazebo` to `quad_dmdc_sim`. Also includes a fix to
> the motor-mixing matrix in `pid_controller.py`: roll and pitch were cross-coupled and
> the pitch channel had inverted feedback polarity, which caused flips/spins/crashes
> during flight. Both axes are now independently and correctly wired.

The workflow is:

**PID flight → Gazebo flight data → X, X_next, U → SVD-DMDc → A, B → multi-step prediction → RMSE → plots**

The DMDc state is intentionally the same 6-state representation used by the original PyBullet part of the project:

```text
x = [x, y, z, vx, vy, vz]^T
u = [u1, u2, u3, u4]^T
```

The Gazebo controller internally uses the full pose state (position, velocity, attitude and angular rate) because a physically simulated quadrotor must tilt to translate. **Only the six translational states are stored in the DMDc dataset.**

## Requirements

- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Harmonic (the Jazzy vendor pairing)
- Python 3

Install the ROS/Gazebo packages:

```bash
sudo apt update
sudo apt install ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge ros-jazzy-actuator-msgs python3-numpy python3-matplotlib
```

Source ROS 2 in every terminal:

```bash
source /opt/ros/jazzy/setup.bash
```

## Build

Put this folder inside a ROS 2 workspace:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
# copy quad_dmdc_sim here
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select quad_dmdc_sim
source install/setup.bash
```

## 1. Test Gazebo visually

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch quad_dmdc_sim sim.launch.py gui:=true
```

Gazebo should open with the quadrotor above the ground.

In another terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 topic list | grep x500_dmdc
```

You should see the pose and motor-command topics.

## 2. Run the flight demo

With Gazebo still running:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run quad_dmdc_sim fly_manual
```

The intended sequence is:

```text
takeoff → hover → move X → move Y → return → controlled descent → stop motors
```

The PID controller uses attitude internally, but attitude is not part of the DMDc state.

## 3. Collect DMDc data

Start a fresh Gazebo simulation in headless mode:

```bash
ros2 launch quad_dmdc_sim sim.launch.py gui:=false
```

In a second terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run quad_dmdc_sim data_collection
```

The collector saves:

```text
data/collected_data.npz
```

Expected dimensions:

```text
X       : 6 × N
X_next  : 6 × N
U       : 4 × N
```

## 4. Run DMDc identification

From the package directory containing `data/`:

```bash
python3 -m quad_dmdc_sim.run_identification
```

Or, after sourcing the workspace:

```bash
ros2 run quad_dmdc_sim run_identification
```

The script computes:

```text
Omega = [ X ]
         [ U ]

Omega = U Sigma V^T

G = X_next Omega^dagger

G = [ A  B ]
```

Expected matrix dimensions:

```text
A : 6 × 6
B : 6 × 4
```

The identified model is:

```text
x(k+1) = A x(k) + B u(k)
```

## 5. Results

The identification script creates:

```text
models_out/dmdc_model.npz
plots/position.png
plots/velocity.png
plots/singular_values.png
plots/eigenvalues.png
```

It also prints the held-out multi-step prediction RMSE.

## Important implementation detail

Gazebo internally receives full pose so the cascaded PID controller can stabilize roll/pitch while moving horizontally. The DMDc pipeline deliberately slices only:

```python
state[:6]  # [x, y, z, vx, vy, vz]
```

Therefore the learned model is **6-state / 4-input**, matching the original PyBullet formulation.

## Troubleshooting

### No pose received

Check:

```bash
ros2 topic list | grep x500_dmdc
ros2 topic echo /x500_dmdc/odometry --once
```

Note: the model only has an `OdometryPublisher` plugin, not a `PosePublisher`
plugin, so pose is read out of `/x500_dmdc/odometry` (`nav_msgs/Odometry`)
rather than a dedicated `/pose` topic. (A `PosePublisher` plugin would be
the more "obvious" fix, but its `publish_model_pose` option is
[broken on Gazebo Harmonic](https://github.com/gazebosim/gz-sim/issues/2690),
so odometry is the reliable path here.)

### No motor response

Check the ROS motor topic:

```bash
ros2 topic info /x500_dmdc/command/motor_speed
```

The Gazebo motor plugins are configured to use the model-scoped motor command topic and the ROS bridge maps that topic to `actuator_msgs/msg/Actuators`.

### Gazebo cannot find the model

The launch file sets `GZ_SIM_RESOURCE_PATH` to the installed `models/` directory. Rebuild and source the workspace if the model was recently changed:

```bash
colcon build --symlink-install --packages-select quad_dmdc_sim
source install/setup.bash
```

### Headless mode still opens a GUI

Use:

```bash
ros2 launch quad_dmdc_sim sim.launch.py gui:=false
```

The launch file selects Gazebo server-only mode when `gui` is false.
