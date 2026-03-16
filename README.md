# SVDC Preception VLM

VLM-based driving-scene analysis project with two execution modes:

- Local webcam debug mode
- ROS 2 camera subscriber mode using `/camera/image_raw`

## Install

### 1. Create the conda environment

```powershell
conda env create -f environment.yml
```

### 2. Activate it

```powershell
conda activate svdc-vlm
```

The editable install is already handled by `environment.yml`, so you do not need a separate `pip install -e .` step.

### 3. Update the environment later if dependencies change

```powershell
conda env update -f environment.yml --prune
```

## Daily start

When you start working again later, you do not recreate the environment.
You only activate it again in the new terminal:

```powershell
cd C:\Users\yemoy\SVDC_preception_vlm
conda activate svdc-vlm
```

## Run

### Local webcam debug

```powershell
svdc-webcam
```

or

```powershell
python test_with_webcam.py
```

### ROS 2 camera subscriber

This mode subscribes to `/camera/image_raw` and publishes driving decisions to `/svdc/driving_decision`.

```powershell
svdc-ros-camera
```

or

```powershell
python test_with_ros_camera.py
```

## ROS topic map

### Output topic

Both execution modes publish the analyzed driving decision to:

```text
/svdc/driving_decision
```

Message type:

```text
std_msgs/msg/String
```

The published message is a JSON string like this:

```json
{
  "road_type": "unknown",
  "road_surface": "unknown",
  "hazard_present": false,
  "hazard_type": "none",
  "hazard_reason": "No forward hazard detected.",
  "driving_action": "maintain_speed",
  "decision_reason": "Scene certainty is limited, so maintaining speed is safer."
}
```

### Input topic

Only the ROS camera mode subscribes to:

```text
/camera/image_raw
```

Message type:

```text
sensor_msgs/msg/Image
```

### Mode summary

- `svdc-webcam`: uses the local webcam directly and publishes results to `/svdc/driving_decision`
- `svdc-ros-camera`: subscribes to `/camera/image_raw` and publishes results to `/svdc/driving_decision`

## ROS 2 note

The conda environment installs the Python packages from this repo through `pip install -e .` internally.

ROS 2 packages such as `rclpy`, `sensor_msgs`, and `std_msgs` are not installed by the conda environment file.
They must come from an existing ROS 2 installation that has been sourced in the shell before running ROS mode.

Example:

```powershell
call C:\dev\ros2\local_setup.bat
conda activate svdc-vlm
svdc-ros-camera
```

## Conda note

You do not need to recreate the conda environment every time.
You only need to run `conda activate svdc-vlm` again in each new terminal session.

If VS Code is using the same conda interpreter, the integrated terminal may auto-activate it for you.

## Model server

The VLM server is configured as:

```text
http://192.168.0.87:8000
```
