# EduBot Autonomous Lane Following
ROS 2 lane detection and navigation nodes for the EduBot robot. The robot uses a downward-facing camera to detect lanes and follow them autonomously.

## How it works
- `lane_detection_node` — reads the downward camera, detects white/yellow/orange floor lines using a YOLOv8 segmentation model, and publishes a lateral error signal
- `navigation_node` — subscribes to that error and drives the robot using a PD controller with a built-in state machine for U-turns, obstacle stops, and intersection handling

---

## Requirements
- ROS 2
- Python 3 packages: `opencv-python`, `numpy`, `cv_bridge`, `ultralytics` (for YOLO)
- EduBot base package with `edubot.launch.py`
- YOLO model file on the robot

---

## Running the nodes
Everything is run over SSH into the robot. You need **three terminals**, each SSH'd in.

### Terminal 1 — launch the robot base
```bash
ros2 launch edubot edubot.launch.py nav2:=True slam:=True
```
Wait until the launch is fully up before continuing.

---

### Terminal 2 — lane detection
```bash
ros2 run edubot_autonomous lane_detection_node
```
This uses YOLO by default (`use_yolo` is already set to `True` in the code). If you need to point it to a different model path on the robot:
```bash
ros2 run edubot_autonomous lane_detection_node --ros-args -p yolo_model_path:=/path/to/model.pt
```
To fall back to HSV-only mode (no YOLO):
```bash
ros2 run edubot_autonomous lane_detection_node --ros-args -p use_yolo:=False
```

---

### Terminal 3 — navigation
> **Important:** `dry_run` is `True` by default, which means the robot will not actually move. You must set it to `False` to enable real movement.

```bash
ros2 run edubot_autonomous navigation_node --ros-args -p dry_run:=False
```

---

## Tuning parameters at runtime
You can adjust parameters live without restarting nodes. For example:
```bash
# Slow the robot down
ros2 param set /navigation_node max_linear 0.08

# Increase feed-forward gain on curves
ros2 param set /navigation_node kff 0.2

# Check what the lane detector is seeing
ros2 topic echo /lane/confidence
ros2 topic echo /lane/error
```
To view the debug camera feed (annotated with detected lanes):
```bash
ros2 run rqt_image_view rqt_image_view /lane/debug_image
```

---

## Key topics published by lane_detection_node

| Topic | Type | Description |
|---|---|---|
| `/lane/error` | Float32 | Lateral error (−1 to 1). Positive = drifting left |
| `/lane/heading` | Float32 | Feed-forward curve hint (−1 to 1) |
| `/lane/confidence` | Float32 | Detection confidence (0 = nothing visible, 1 = both lines) |
| `/lane/end_of_road` | Bool | True when orange stop line is detected |
| `/lane/white_detected` | Bool | True when right boundary is visible |
| `/lane/debug_image` | Image | Annotated camera view for tuning |

---

## Troubleshooting
**Robot not moving** — check that `dry_run:=False` was passed to navigation_node.

**YOLO model not loading** — confirm the `.pt` file exists at the path shown in the log output. Set `use_yolo:=False` to fall back to HSV detection.

**Lane not detected** — run `rqt_image_view /lane/debug_image` and watch the annotated feed. HSV thresholds can be tuned live with `ros2 param set /lane_detection_node white_v_min 180` etc.

**Robot spinning / not correcting** — check `/lane/confidence`. If it's near 0, the camera can't see the lines. Try adjusting lighting or HSV thresholds.
