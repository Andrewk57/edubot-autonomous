# EduBot Autonomous Lane Following

ROS 2 nodes for autonomous lane following on the EduBot, using a downward-facing
camera with a YOLOv8 segmentation model.

## Nodes

- `lane_detection_node` — detects white / yellow / orange lines, publishes
  lateral error and per-class lane point clouds.
- `navigation_node` — PD controller + state machine (U-turns, obstacle stops,
  intersections).
- `mapping_node` — accumulates lane points in the `/map` frame and publishes
  them as a colored marker array and a 2D occupancy grid for RViz.

## Requirements

- ROS 2
- Python: `opencv-python`, `numpy`, `cv_bridge`, `ultralytics`
- `edubot` base package
- YOLO model `.pt` on the robot

## Run

SSH into the robot. Before the first launch on a fresh boot:

```bash
sudo ln -s /dev/video2 /dev/edubot_camera_2
```

Three terminals, all on the robot.

**1. Base + SLAM**

```bash
ros2 launch edubot edubot.launch.py nav2:=True slam:=True
```

**2. Lane detection**

```bash
ros2 run edubot_autonomous lane_detection_node
```

Override model path or disable YOLO:

```bash
ros2 run edubot_autonomous lane_detection_node \
  --ros-args -p yolo_model_path:=/path/to/model.pt
ros2 run edubot_autonomous lane_detection_node --ros-args -p use_yolo:=False
```

**3. Navigation** — `dry_run` is `True` by default; set to `False` to move:

```bash
ros2 run edubot_autonomous navigation_node --ros-args -p dry_run:=False
```

**4. Mapping (optional, recommended)**

```bash
ros2 run edubot_autonomous mapping_node
```

Requires SLAM up (terminal 1) so the `map -> base_link` TF exists.

## Visualizing the map in RViz

Set **Fixed Frame** to `map` and add:

| Display       | Topic                | Notes                              |
|---------------|----------------------|------------------------------------|
| MarkerArray   | `/lane_map/markers`  | White + yellow cubes — main view   |
| Map           | `/lane_map/grid`     | Top-down 2D occupancy of lanes     |
| PointCloud2   | `/lane_map/points`   | Combined raw cloud (optional)      |

Per-frame raw points are on `/lane/points/white` and `/lane/points/yellow` in
the `base_link` frame.

## Live tuning

```bash
# Slow it down
ros2 param set /navigation_node max_linear 0.08

# Lane detector signals
ros2 topic echo /lane/confidence
ros2 topic echo /lane/error


# Limit how far away lane points are mapped (default 1.2 m)
ros2 param set /lane_detection_node max_lane_range_m 1.0

# Make map cubes bigger / smaller in RViz
ros2 param set /mapping_node marker_cube_size 0.10
```

## Troubleshooting

- **Robot won't move** — pass `dry_run:=False` to `navigation_node`.
- **YOLO won't load** — verify the `.pt` path in the log; fall back with
  `use_yolo:=False`.
- **No lanes detected** — open `/lane/debug_image`; tune HSV live, e.g.
  `ros2 param set /lane_detection_node white_v_min 180`.
- **Low `/lane/confidence`** — lighting or HSV problem; check the debug image.
- **Map empty in RViz** — confirm SLAM is up (`ros2 topic echo /tf` should show
  `map -> odom`); without it `mapping_node` can't transform points.
- **Map looks like noisy fog far from the robot** — lower `max_lane_range_m`
  on `lane_detection_node`.
