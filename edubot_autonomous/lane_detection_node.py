"""
lane_detection_node.py
======================
Downward-camera lane detection for EduBot.

Detects three colored features on the floor:
  * Solid WHITE  line  -> right boundary (robot must stay LEFT of it)
  * Dashed YELLOW line -> centre divider (may be crossed to dodge obstacles)
  * Solid ORANGE line  -> end-of-road marker (triggers a 180 turn upstream)

Outputs (consumed by navigation_node and mapping_node):
  /lane/error          std_msgs/Float32        normalised lateral error in [-1, 1]
                                               +ve = drift LEFT (steer right)
                                               -ve = drift RIGHT (steer left)
  /lane/heading        std_msgs/Float32        feed-forward heading hint in
                                               [-1, 1] from the yellow line
                                               slope; +ve = lane curves right
  /lane/confidence     std_msgs/Float32        detection confidence in [0, 1]
                                               1.0 both lanes, 0.7 white only,
                                               0.5 yellow live, 0.3 yellow memory,
                                               0.0 nothing visible
  /lane/end_of_road    std_msgs/Bool           True after orange line is sustained
  /lane/white_detected std_msgs/Bool           True if white contour was found
  /lane/debug_image    sensor_msgs/Image       annotated mask for tuning
  /lane/points         sensor_msgs/PointCloud2 lane points in base_link for mapping

Design notes
------------
The pipeline is intentionally minimal: crop -> HSV -> mask -> contour -> centroid.
All thresholds are ROS parameters so they can be tuned at runtime with
`ros2 param set ...` instead of recompiling.

Optionally, a YOLOv8n segmentation model can replace the HSV masks for white
and yellow detection. Enable with use_yolo:=True and point yolo_model_path at
the .pt file. The model is expected to have class 0 = White lane,
class 1 = Yellow Lane. Falls back to HSV if YOLO returns empty detections.

Dashed yellow is intermittent by definition; we hold the last centroid for
`yellow_memory_secs` so the controller does not see a square wave on it.

The 3D projection assumes a flat floor and uses the camera intrinsics from
/camera_2/camera_info plus the static base_link <- camera transform. If
camera_info or TF is not yet available we still publish 2D outputs so
navigation can run without mapping.
"""
import math
import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Bool, Float32

import tf2_ros
from tf2_ros import StaticTransformBroadcaster, TransformException


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # Camera / TF
        self.declare_parameter('image_topic', '/camera_2/image_raw')
        self.declare_parameter('camera_info_topic', '/camera_2/camera_info')
        self.declare_parameter('camera_frame', 'camera_2_optical_frame')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('publish_camera_tf', True)
        self.declare_parameter('camera_height_m', 0.15)
        self.declare_parameter('camera_pitch_deg', -45.0)

        # ROI - crop_top_ratio raised to 0.20 to cut out more of the upper
        # frame where window glare and floor reflections cause false detections.
        self.declare_parameter('crop_top_ratio', 0.10)
        self.declare_parameter('crop_bottom_ratio', 0.05)
        self.declare_parameter('crop_side_ratio', 0.0)
        self.declare_parameter('crop_top_orange_ratio', 0.10)

        # HSV gates - OpenCV uses H in [0,179], S/V in [0,255].
        # white_v_min lowered 200->180 to keep detecting white under shadows.
        # yellow hue range widened (15-50) for robustness under warm/cool lighting.
        self.declare_parameter('white_h_min', 0)
        self.declare_parameter('white_h_max', 179)
        self.declare_parameter('white_s_min', 0)
        self.declare_parameter('white_s_max', 40)
        self.declare_parameter('white_v_min', 200)
        self.declare_parameter('white_v_max', 255)

        self.declare_parameter('yellow_h_min', 20)
        self.declare_parameter('yellow_h_max', 45)
        self.declare_parameter('yellow_s_min', 80)
        self.declare_parameter('yellow_s_max', 255)
        self.declare_parameter('yellow_v_min', 100)
        self.declare_parameter('yellow_v_max', 255)

        self.declare_parameter('orange_h_min', 5)
        self.declare_parameter('orange_h_max', 20)
        self.declare_parameter('orange_s_min', 120)
        self.declare_parameter('orange_s_max', 255)
        self.declare_parameter('orange_v_min', 120)
        self.declare_parameter('orange_v_max', 255)

        # Geometry / behavior.
        self.declare_parameter('min_contour_area', 1500.0)
        self.declare_parameter('target_white_x_ratio', 0.80)
        self.declare_parameter('target_yellow_x_ratio', 0.30)
        self.declare_parameter('white_x_min_ratio', 0.45)
        self.declare_parameter('yellow_weight', 0.35)
        self.declare_parameter('yellow_memory_secs', 0.25)
        self.declare_parameter('min_orange_pixels', 4000)
        self.declare_parameter('min_orange_width_ratio', 0.45)
        self.declare_parameter('orange_consecutive_frames', 3)
        self.declare_parameter('confidence_smoothing', 0.5)
        self.declare_parameter('heading_smoothing', 0.6)
        self.declare_parameter('heading_gain', 1.5)
        self.declare_parameter('heading_min_points', 8)
        self.declare_parameter('use_clahe', True)
        self.declare_parameter('debug_image', True)

        # YOLO segmentation detector (optional).
        # Set use_yolo:=True to activate. Set yolo_model_path to the .pt file
        # on the robot. Falls back to HSV if a detection class is empty.
        # yolo_frame_skip runs inference every Nth frame to reduce CPU load;
        # cached masks are used for intermediate frames.
        self.declare_parameter('use_yolo', True)
        self.declare_parameter('yolo_model_path',
            '/home/developer/andrew_ws/src/edubot_autonomous/models/lane_yolov8n_seg.pt')
        self.declare_parameter('yolo_conf_threshold', 0.4)
        self.declare_parameter('yolo_frame_skip', 3)

        self.bridge = CvBridge()
        self.fx = self.fy = self.cx_pix = self.cy_pix = None

        self._last_yellow_cx = None
        self._last_yellow_stamp = 0.0
        self._orange_streak = 0
        self._confidence_smoothed = 0.0
        self._heading_smoothed = 0.0
        self._last_yellow_line = None

        # YOLO state
        self._yolo_model = None
        self._yolo_frame_count = 0
        self._yolo_white_mask_cache = None
        self._yolo_yellow_mask_cache = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        if self.get_parameter('publish_camera_tf').value:
            self._publish_camera_tf()

        if self.get_parameter('use_yolo').value:
            self._load_yolo_model()

        self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self._camera_info_cb,
            10,
        )
        self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self._image_cb,
            10,
        )

        self.pub_error = self.create_publisher(Float32, '/lane/error', 10)
        self.pub_heading = self.create_publisher(Float32, '/lane/heading', 10)
        self.pub_conf = self.create_publisher(Float32, '/lane/confidence', 10)
        self.pub_eor = self.create_publisher(Bool, '/lane/end_of_road', 10)
        self.pub_white_det = self.create_publisher(Bool, '/lane/white_detected', 10)
        self.pub_debug = self.create_publisher(Image, '/lane/debug_image', 10)
        self.pub_points = self.create_publisher(PointCloud2, '/lane/points', 10)

        self.get_logger().info('Lane detection node up.')

    # ------------------------------------------------------------------ TF
    def _publish_camera_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter('base_frame').value
        t.child_frame_id = self.get_parameter('camera_frame').value

        t.transform.translation.x = 0.10
        t.transform.translation.y = 0.0
        t.transform.translation.z = self.get_parameter('camera_height_m').value

        pitch = math.radians(self.get_parameter('camera_pitch_deg').value)
        cp, sp = math.cos(pitch), math.sin(pitch)
        rot = np.array([
            [0.0, -1.0, 0.0],
            [-sp, 0.0, -cp],
            [cp, 0.0, -sp],
        ])
        qw, qx, qy, qz = self._mat_to_quat(rot)
        t.transform.rotation.w = qw
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        self._static_tf_broadcaster.sendTransform(t)

    @staticmethod
    def _mat_to_quat(m):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
        return qw, qx, qy, qz

    def _camera_info_cb(self, msg: CameraInfo):
        if self.fx is not None:
            return
        k = msg.k
        self.fx, self.fy, self.cx_pix, self.cy_pix = k[0], k[4], k[2], k[5]
        self.get_logger().info(
            f'Camera intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} '
            f'cx={self.cx_pix:.1f} cy={self.cy_pix:.1f}'
        )

    # ------------------------------------------------------------------ YOLO
    def _load_yolo_model(self):
        try:
            from ultralytics import YOLO  # noqa: PLC0415
            path = os.path.expanduser(self.get_parameter('yolo_model_path').value)
            self._yolo_model = YOLO(path)
            self.get_logger().info(f'YOLO model loaded from {path}')
        except Exception as exc:
            self.get_logger().error(f'Failed to load YOLO model: {exc}')
            self._yolo_model = None

    def _yolo_detect(self, roi):
        """Run YOLO segmentation; return (white_mask, yellow_mask) uint8 or (None, None)."""
        if self._yolo_model is None:
            return None, None
        try:
            h, w = roi.shape[:2]
            results = self._yolo_model.predict(
                roi,
                conf=float(self.get_parameter('yolo_conf_threshold').value),
                verbose=False,
            )
            white_mask = np.zeros((h, w), dtype=np.uint8)
            yellow_mask = np.zeros((h, w), dtype=np.uint8)

            if results and results[0].masks is not None:
                masks_data = results[0].masks.data.cpu().numpy()  # (N, Hm, Wm)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                for i, cls_id in enumerate(classes):
                    if i >= len(masks_data):
                        break
                    m = cv2.resize(
                        masks_data[i], (w, h), interpolation=cv2.INTER_LINEAR
                    )
                    binary = (m > 0.5).astype(np.uint8) * 255
                    if cls_id == 0:
                        white_mask = cv2.bitwise_or(white_mask, binary)
                    elif cls_id == 1:
                        yellow_mask = cv2.bitwise_or(yellow_mask, binary)

            return white_mask, yellow_mask
        except Exception as exc:
            self.get_logger().warn(
                f'YOLO detection failed: {exc}', throttle_duration_sec=5.0
            )
            return None, None

    # ------------------------------------------------------------------ main
    def _image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'cv_bridge failure: {exc}')
            return

        h, w_full = frame.shape[:2]
        roi_top = int(h * self.get_parameter('crop_top_ratio').value)
        roi_bot = int(h * (1.0 - self.get_parameter('crop_bottom_ratio').value))
        side = self.get_parameter('crop_side_ratio').value
        x_lo = int(w_full * side)
        x_hi = w_full - x_lo
        if x_hi - x_lo < 4:
            x_lo, x_hi = 0, w_full
        roi = frame[roi_top:roi_bot, x_lo:x_hi]
        roi_h, roi_w = roi.shape[:2]

        orange_top = int(h * self.get_parameter('crop_top_orange_ratio').value)
        roi_orange = frame[orange_top:roi_bot, x_lo:x_hi]

        if self.get_parameter('use_clahe').value:
            roi = self._clahe(roi)
            roi_orange = self._clahe(roi_orange)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv_orange = cv2.cvtColor(roi_orange, cv2.COLOR_BGR2HSV)

        # Choose lane mask source: YOLO or HSV
        if self.get_parameter('use_yolo').value and self._yolo_model is not None:
            self._yolo_frame_count += 1
            skip = int(self.get_parameter('yolo_frame_skip').value)
            if self._yolo_frame_count % skip == 0 or self._yolo_white_mask_cache is None:
                yw, yy = self._yolo_detect(roi)
                if yw is not None:
                    self._yolo_white_mask_cache = yw
                    self._yolo_yellow_mask_cache = yy

            # Use YOLO masks, fall back to HSV per channel if empty
            if (
                self._yolo_white_mask_cache is not None
                and cv2.countNonZero(self._yolo_white_mask_cache) >= 100
            ):
                white_mask = self._yolo_white_mask_cache
            else:
                white_mask = self._color_mask(hsv, 'white', kernel=5)

            if (
                self._yolo_yellow_mask_cache is not None
                and cv2.countNonZero(self._yolo_yellow_mask_cache) >= 100
            ):
                yellow_mask = self._yolo_yellow_mask_cache
            else:
                yellow_mask = self._color_mask(hsv, 'yellow', kernel=5)
        else:
            white_mask = self._color_mask(hsv, 'white', kernel=5)
            yellow_mask = self._color_mask(hsv, 'yellow', kernel=5)

        orange_mask = self._color_mask(hsv_orange, 'orange', kernel=7)

        white_cnt = self._largest_contour(white_mask)
        yellow_cnt = self._largest_contour(yellow_mask)

        white_cx = self._centroid_x(white_cnt)
        

# Reject white detections on the wrong side of the frame
        if white_cx is not None:
            white_min_x = int(roi_w * self.get_parameter('white_x_min_ratio').value)
            if white_cx < white_min_x:
                white_cx = None
                white_cnt = None
        yellow_cx = self._centroid_x(yellow_cnt)

        now = self.get_clock().now().nanoseconds * 1e-9
        yellow_from_memory = False
        if yellow_cx is not None:
            self._last_yellow_cx = yellow_cx
            self._last_yellow_stamp = now
        elif (
            self._last_yellow_cx is not None
            and (now - self._last_yellow_stamp)
            < self.get_parameter('yellow_memory_secs').value
        ):
            yellow_cx = self._last_yellow_cx
            yellow_from_memory = True

        white_err = self._error_from_centroid(
            white_cx, roi_w, self.get_parameter('target_white_x_ratio').value
        )
        yellow_err = self._error_from_centroid(
            yellow_cx, roi_w, self.get_parameter('target_yellow_x_ratio').value
        )

        white_ok = white_cx is not None
        yellow_live = yellow_cx is not None and not yellow_from_memory
        yellow_any = yellow_cx is not None

        if white_ok and yellow_live:
            raw_conf = 1.0
        elif white_ok and yellow_any:
            raw_conf = 0.85
        elif white_ok:
            raw_conf = 0.7
        elif yellow_live:
            raw_conf = 0.5
        elif yellow_any:
            raw_conf = 0.3
        else:
            raw_conf = 0.0

        if white_ok and yellow_any:
            yw = self.get_parameter('yellow_weight').value
            final_err = (1.0 - yw) * white_err + yw * yellow_err
            src = 'BLEND' if yellow_live else 'BLEND_MEM'
        elif white_ok:
            final_err = white_err
            src = 'WHITE'
        elif yellow_any:
            final_err = yellow_err
            src = 'YELLOW_MEM' if yellow_from_memory else 'YELLOW'
        else:
            final_err = 0.0
            src = 'NONE'

        final_err = float(np.clip(final_err, -1.5, 1.5))

        raw_heading = self._yellow_heading(yellow_cnt)
        if raw_heading is None:
            target_h = 0.0
            beta = 0.85
        else:
            target_h = raw_heading
            beta = float(self.get_parameter('heading_smoothing').value)
            beta = max(0.0, min(1.0, beta))
        self._heading_smoothed = beta * self._heading_smoothed + (1.0 - beta) * target_h
        heading = float(np.clip(self._heading_smoothed, -1.0, 1.0))

        alpha = float(self.get_parameter('confidence_smoothing').value)
        alpha = max(0.0, min(1.0, alpha))
        self._confidence_smoothed = (
            alpha * self._confidence_smoothed + (1.0 - alpha) * raw_conf
        )
        confidence = float(np.clip(self._confidence_smoothed, 0.0, 1.0))

        orange_pixels = int(cv2.countNonZero(orange_mask))
        orange_wide_enough = False
        orange_largest = self._largest_contour(orange_mask)
        if orange_largest is not None:
            x, _y, w_box, _h_box = cv2.boundingRect(orange_largest)
            min_w_ratio = self.get_parameter('min_orange_width_ratio').value
            orange_wide_enough = w_box >= int(roi_w * min_w_ratio)
            _ = x
        if (
            orange_pixels >= self.get_parameter('min_orange_pixels').value
            and orange_wide_enough
        ):
            self._orange_streak += 1
        else:
            self._orange_streak = 0
        end_of_road = self._orange_streak >= self.get_parameter(
            'orange_consecutive_frames'
        ).value

        self.pub_error.publish(Float32(data=final_err))
        self.pub_heading.publish(Float32(data=heading))
        self.pub_conf.publish(Float32(data=confidence))
        self.pub_eor.publish(Bool(data=bool(end_of_road)))
        self.pub_white_det.publish(Bool(data=bool(white_ok)))

        self._maybe_publish_points(
            msg.header,
            roi_top,
            white_cnt,
            yellow_cnt,
        )

        if self.get_parameter('debug_image').value:
            self._publish_debug(
                roi,
                white_mask,
                yellow_mask,
                orange_mask,
                white_cx,
                yellow_cx,
                roi_w,
                white_ok,
                yellow_from_memory,
                orange_pixels,
                end_of_road,
                final_err,
                src,
                confidence,
                heading,
            )

        self.get_logger().info(
            f'src={src} err={final_err:+.2f} hdg={heading:+.2f} '
            f'conf={confidence:.2f} white={white_ok} yellow={yellow_any} '
            f'orange_px={orange_pixels} eor={end_of_road}',
            throttle_duration_sec=1.0,
        )

    # -------------------------------------------------------------- helpers
    def _clahe(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _color_mask(self, hsv, name, kernel=5):
        get = self.get_parameter
        lo = np.array([
            get(f'{name}_h_min').value,
            get(f'{name}_s_min').value,
            get(f'{name}_v_min').value,
        ])
        hi = np.array([
            get(f'{name}_h_max').value,
            get(f'{name}_s_max').value,
            get(f'{name}_v_max').value,
        ])
        mask = cv2.inRange(hsv, lo, hi)
        k = np.ones((kernel, kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def _largest_contour(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) < self.get_parameter('min_contour_area').value:
            return None
        return best

    @staticmethod
    def _centroid_x(cnt):
        if cnt is None:
            return None
        m = cv2.moments(cnt)
        if m['m00'] <= 0:
            return None
        return int(m['m10'] / m['m00'])

    def _yellow_heading(self, cnt):
        if cnt is None:
            self._last_yellow_line = None
            return None
        min_pts = int(self.get_parameter('heading_min_points').value)
        if len(cnt) < min_pts:
            self._last_yellow_line = None
            return None
        pts = cnt.reshape(-1, 2).astype(np.float32)
        line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = (
            float(line[0]), float(line[1]), float(line[2]), float(line[3])
        )
        if vy < 0.0:
            vx, vy = -vx, -vy
        if vy < 0.05:
            self._last_yellow_line = None
            return None
        slope_per_depth = -vx / vy
        gain = float(self.get_parameter('heading_gain').value)
        self._last_yellow_line = (vx, vy, x0, y0)
        return float(max(-1.0, min(1.0, slope_per_depth * gain)))

    @staticmethod
    def _error_from_centroid(cx, width, target_ratio):
        if cx is None:
            return 0.0
        target = width * target_ratio
        return float(target - cx) / float(width / 2.0)

    # --------------------------------------------------------- 3D mapping
    def _maybe_publish_points(self, header, roi_y_offset, white_cnt, yellow_cnt):
        if self.fx is None:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.get_parameter('base_frame').value,
                self.get_parameter('camera_frame').value,
                rclpy.time.Time(),
            )
        except TransformException:
            return

        pixels = []
        if white_cnt is not None:
            pixels.extend(self._sample_contour(white_cnt, roi_y_offset, n=8))
        if yellow_cnt is not None:
            pixels.extend(self._sample_contour(yellow_cnt, roi_y_offset, n=5))
        if not pixels:
            return

        T = self._tf_to_matrix(tf)
        height = self.get_parameter('camera_height_m').value
        pts = []
        for u, v in pixels:
            ray_cam = np.array([
                (u - self.cx_pix) / self.fx,
                (v - self.cy_pix) / self.fy,
                1.0,
            ])
            ray_base = T[:3, :3] @ ray_cam
            origin_base = T[:3, 3]
            if abs(ray_base[2]) < 1e-6:
                continue
            scale = (-origin_base[2]) / ray_base[2]
            if scale <= 0:
                continue
            world = origin_base + scale * ray_base
            if world[0] < 0.0 or world[0] > 4.0:
                continue
            pts.append((float(world[0]), float(world[1]), 0.0))
            _ = height

        if pts:
            self._publish_point_cloud(
                header, pts, self.get_parameter('base_frame').value
            )

    @staticmethod
    def _sample_contour(cnt, roi_y_offset, n=6):
        step = max(1, len(cnt) // n)
        out = []
        for i in range(0, len(cnt), step):
            x, y = cnt[i][0]
            out.append((float(x), float(y + roi_y_offset)))
        return out

    @staticmethod
    def _tf_to_matrix(tf):
        t = tf.transform.translation
        r = tf.transform.rotation
        qx, qy, qz, qw = r.x, r.y, r.z, r.w
        rot = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
        ])
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def _publish_point_cloud(self, header, points, frame_id):
        msg = PointCloud2()
        msg.header.stamp = header.stamp
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * len(points)
        msg.is_dense = True
        msg.data = np.array(points, dtype=np.float32).tobytes()
        self.pub_points.publish(msg)

    # ----------------------------------------------------------- debug img
    def _publish_debug(
        self,
        roi,
        white_mask,
        yellow_mask,
        orange_mask,
        white_cx,
        yellow_cx,
        roi_w,
        white_ok,
        yellow_from_memory,
        orange_pixels,
        end_of_road,
        final_err,
        src,
        confidence,
        heading,
    ):
        vis = roi.copy()
        vis[white_mask > 0] = (220, 220, 255)
        vis[yellow_mask > 0] = (0, 220, 220)

        target_white = int(roi_w * self.get_parameter('target_white_x_ratio').value)
        target_yellow = int(roi_w * self.get_parameter('target_yellow_x_ratio').value)
        cv2.line(vis, (target_white, 0), (target_white, vis.shape[0]), (0, 255, 0), 1)
        cv2.line(vis, (target_yellow, 0), (target_yellow, vis.shape[0]), (0, 180, 0), 1)

        if white_cx is not None:
            cv2.line(
                vis, (white_cx, 0), (white_cx, vis.shape[0]),
                (255, 255, 0), 3 if white_ok else 1,
            )
        if yellow_cx is not None:
            color = (180, 0, 180) if yellow_from_memory else (255, 0, 255)
            cv2.line(vis, (yellow_cx, 0), (yellow_cx, vis.shape[0]), color, 2)

        if self._last_yellow_line is not None:
            vx, vy, x0, y0 = self._last_yellow_line
            h = vis.shape[0]
            if abs(vy) > 1e-3:
                t_top = (0 - y0) / vy
                t_bot = (h - 1 - y0) / vy
                p1 = (int(x0 + t_top * vx), 0)
                p2 = (int(x0 + t_bot * vx), h - 1)
                cv2.line(vis, p1, p2, (0, 140, 255), 2)

        bar_h = 8
        y0 = vis.shape[0] - bar_h - 2
        cv2.rectangle(vis, (0, y0), (roi_w - 1, y0 + bar_h), (40, 40, 40), -1)
        if confidence >= 0.7:
            bar_color = (0, 200, 0)
        elif confidence >= 0.4:
            bar_color = (0, 220, 220)
        else:
            bar_color = (0, 0, 220)
        bar_w = int(max(0.0, min(1.0, confidence)) * (roi_w - 1))
        cv2.rectangle(vis, (0, y0), (bar_w, y0 + bar_h), bar_color, -1)

        yolo_tag = ' [YOLO]' if (
            self.get_parameter('use_yolo').value and self._yolo_model is not None
        ) else ''
        cv2.putText(
            vis,
            f'src={src}{yolo_tag} err={final_err:+.2f} hdg={heading:+.2f} '
            f'conf={confidence:.2f}',
            (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            vis, f'orange={orange_pixels}px  EOR={end_of_road}',
            (5, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 80, 255) if end_of_road else (180, 180, 180), 1, cv2.LINE_AA,
        )

        out = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        self.pub_debug.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()