# SVDC Preception VLM

VLM-based driving-scene analysis project with two execution modes:

- usb_cam-based webcam debug mode using `/image/image_raw`
- ROS 2 camera subscriber mode using `/image/image_raw`

## ROS 2 package layout

The repository now follows a ROS 2 Python package layout:

```text
SVDC_preception_vlm/
├── package.xml
├── resource/
│   └── svdc_preception_vlm
├── setup.py
├── setup.cfg
├── svdc_preception_vlm/
│   ├── __init__.py
│   ├── vlm_driving_common.py
│   ├── test_with_webcam.py
│   └── test_with_ros_camera.py
├── test_with_webcam.py
├── test_with_ros_camera.py
└── README.md
```

The top-level `test_with_webcam.py` and `test_with_ros_camera.py` files are kept as thin wrappers for convenience.

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

### Local webcam debug through usb_cam

First start the usb camera publisher:

```powershell
ros2 run usb_cam usb_cam_node_exe
```

Then run the VLM debug node:

```powershell
svdc-webcam
```

or

```powershell
python test_with_webcam.py
```

If you build this as a ROS 2 package, you can also run:

```powershell
ros2 run svdc_preception_vlm webcam_node
```

### ROS 2 camera subscriber

This mode subscribes to `/image/image_raw` and publishes driving decisions to `/svdc/driving_decision`.

```powershell
svdc-ros-camera
```

or

```powershell
python test_with_ros_camera.py
```

If you build this as a ROS 2 package, you can also run:

```powershell
ros2 run svdc_preception_vlm ros_camera_node
```

## ROS topic map

### Output topic

Both execution modes publish the analyzed driving decision to:

```text
/drive_context
```

Message type:

```text
std_msgs/msg/String
```

The published message is a JSON string like this:

```json
{
  "scene_context": "simple",
  "hazard_present": false,
  "hazard_type": "none",
  "traffic_signal_red": false,
  "traffic_signal_green": false,
  "driving_action": "maintain_speed"
}
```

Traffic-signal detection accepts small experimental signals placed low near the
road, not only full-size road signals. Red/green flags are true only when the
illuminated lamp color is clearly visible.

Scene context is also published separately:

```text
/scene_context
/traffic_cone/present
```

`/traffic_cone/present` is kept for compatibility. It is true only when the
close-cone condition triggers `scene_context=complex`. A cone that is near the
forward path and close enough to affect the next driving decision should trigger
`complex`, even if the exact distance is uncertain. A cone merely visible far
ahead or clearly off to the side remains `scene_context=simple`.

### Input topic

Both execution modes subscribe to the usb_cam image topic:

```text
/image/image_raw
```

Message type:

```text
sensor_msgs/msg/Image
```

### Mode summary

- `svdc-webcam`: subscribes to `/image/image_raw` from `usb_cam` and publishes results to `/svdc/driving_decision`
- `svdc-ros-camera`: subscribes to `/image/image_raw` and publishes results to `/svdc/driving_decision`

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

The default OpenAI-compatible VLM server is configured as:

```text
Endpoint: http://192.168.64.1:11434
API: http://192.168.64.1:11434/v1/chat/completions
Model: qwen3-vl:8b-instruct
Auth: none
```

The ROS camera node already subscribes to the raw image topic:

```text
/image/image_raw
```

You can run it with:

```powershell
ros2 run svdc_preception_vlm ros_camera_node
```

Optional overrides:

```powershell
set SVDC_VLM_BASE_URL=http://192.168.64.1:11434
set SVDC_VLM_MODEL=qwen3-vl:8b-instruct
```

Gemini API mode:

```bash
source /home/yemo/ros2_ws/install/setup.bash
ros2 run svdc_preception_vlm ros_camera_node \
  --provider gemini \
  --gemini-api-key "$GEMINI_API_KEY"
```

Gemini mode uses `gemini-3-flash-preview` by default. You can override it with:

```bash
ros2 run svdc_preception_vlm ros_camera_node \
  --provider gemini \
  --gemini-api-key "$GEMINI_API_KEY" \
  --gemini-model gemini-3-flash-preview
```

For one-command startup on the ROS machine, create a local config file at the
repository root. This file is ignored by git:

```bash
cp svdc_vlm_config.example.json svdc_vlm_config.local.json
```

Edit `svdc_vlm_config.local.json`:

```json
{
  "provider": "gemini",
  "gemini_api_key": "PASTE_GEMINI_API_KEY_HERE",
  "gemini_model": "gemini-3-flash-preview",
  "image_max_dim": 640,
  "max_in_flight": 2,
  "capture_interval": 3.0
}
```

Lower `image_max_dim` to `512` or `448` if Gemini responses are too slow. The
debug camera window still uses the original ROS image; this only resizes the
image sent to the vision model.

Increase `max_in_flight` to `3` to send multiple VLM requests concurrently.
Completed responses are published in frame-number order, so downstream topics
stay FIFO even if later requests finish first. Lower `capture_interval` to send
frames more frequently, but watch API cost and rate limits.

Then rebuild/source once and run normally:

```bash
cd /home/yemo/ros2_ws
colcon build --packages-select svdc_preception_vlm --symlink-install
source install/setup.bash
ros2 run svdc_preception_vlm ros_camera_node
```
