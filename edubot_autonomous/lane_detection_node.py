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

Dashed yellow is intermittent by definition; we hold the last centroid for
`yellow_memory_secs` so the controller does not see a square wave on it.

The 3D projection assumes a flat floor and uses the camera intrinsics from
/camera_2/camera_info plus the static base_link <- camera transform. If
camera_info or TF is not yet available we still publish 2D outputs so
navigation can run without mapping.
"""
import math

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

        # ROI - cropping out sky / bumper makes everything else easier.
        # crop_top_ratio is intentionally low (0.28) so the lane stays in
        # the ROI through the curves; previously 0.45 lost the line mid-turn.
        # Going lower than this caused systematic perspective bias in the
        # slope fit on straightaways.
        self.declare_parameter('crop_top_ratio', 0.10)
        self.declare_parameter('crop_bottom_ratio', 0.05)
        self.declare_parameter('crop_side_ratio', 0.0)
        self.declare_parameter('crop_top_orange_ratio', 0.10)

        # HSV gates - OpenCV uses H in [0,179], S/V in [0,255]
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

        # Geometry / behavior. target_yellow=0.30 / yellow_weight=0.35 are
        # the values that worked best at the track; lowering them pulled
        # the robot off the yellow line so hard it drove onto the right
        # boundary. Tune these only via ros2 param at trackside.
        self.declare_parameter('min_contour_area', 1500.0)
        self.declare_parameter('target_white_x_ratio', 0.80)
        self.declare_parameter('target_yellow_x_ratio', 0.30)
        self.declare_parameter('yellow_weight', 0.35)
        # Yellow memory is short on purpose: dashed gaps still bridge, but a
        # real loss stops driving stale errors into the controller.
        self.declare_parameter('yellow_memory_secs', 0.25)
        self.declare_parameter('min_orange_pixels', 4000)
        # Orange must span most of the ROI width to count - otherwise small
        # corner blobs of red carpet were tripping the U-turn mid-track.
        self.declare_parameter('min_orange_width_ratio', 0.45)
        self.declare_parameter('orange_consecutive_frames', 3)
        self.declare_parameter('confidence_smoothing', 0.5)
        # Heading feed-forward (yellow-line slope). Smoothing damps single-
        # frame fits; gain scales the unitless slope into the [-1,1] range
        # the navigation node expects.
        self.declare_parameter('heading_smoothing', 0.6)
        self.declare_parameter('heading_gain', 1.5)
        self.declare_parameter('heading_min_points', 8)
        self.declare_parameter('use_clahe', True)
        self.declare_parameter('debug_image', True)

        self.bridge = CvBridge()
        self.fx = self.fy = self.cx_pix = self.cy_pix = None

        self._last_yellow_cx = None
        self._last_yellow_stamp = 0.0
        self._orange_streak = 0
        self._confidence_smoothed = 0.0
        self._heading_smoothed = 0.0
        self._last_yellow_line = None  # (vx, vy, x0, y0) for debug overlay

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        if self.get_parameter('publish_camera_tf').value:
            self._publish_camera_tf()

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
        # Provides a sane default base_link -> camera transform if the robot's
        # URDF has not already published one. Override with publish_camera_tf:=False
        # if your URDF takes care of it.
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter('base_frame').value
        t.child_frame_id = self.get_parameter('camera_frame').value

        t.transform.translation.x = 0.10
        t.transform.translation.y = 0.0
        t.transform.translation.z = self.get_parameter('camera_height_m').value

        # Optical frame: x=right, y=down, z=forward into the scene.
        # Build that orientation from a base_link-relative pitch (camera tilted
        # downward by camera_pitch_deg).
        pitch = math.radians(self.get_parameter('camera_pitch_deg').value)
        # base_link forward (x) -> optical z
        # base_link left (y) -> optical -x
        # base_link up (z) -> optical -y
        # ... rotated by pitch about base_link y axis.
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

        white_mask = self._color_mask(hsv, 'white', kernel=5)
        yellow_mask = self._color_mask(hsv, 'yellow', kernel=5)
        orange_mask = self._color_mask(hsv_orange, 'orange', kernel=7)

        white_cnt = self._largest_contour(white_mask)
        yellow_cnt = self._largest_contour(yellow_mask)

        # Lateral error from white (preferred) and yellow (fallback)
        white_cx = self._centroid_x(white_cnt)
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

        # Confidence rules: prefer to know "is the controller on solid ground"
        # over a precise probability. Live both -> 1.0. Live yellow alone is
        # weaker than white alone (white is the boundary we drive next to).
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
            # No right_bias: a saturating offset here was the source of the
            # max-angular spin-out on lane loss.
            final_err = yellow_err
            src = 'YELLOW_MEM' if yellow_from_memory else 'YELLOW'
        else:
            final_err = 0.0
            src = 'NONE'

        final_err = float(np.clip(final_err, -1.5, 1.5))

        # Heading feed-forward from the yellow contour shape. Curves push
        # the centroid only after the line has visibly slid sideways, which
        # is too late to steer in time. Slope-of-yellow tells us the curve
        # is coming several frames earlier.
        raw_heading = self._yellow_heading(yellow_cnt)
        if raw_heading is None:
            # Decay rather than snap to zero so we don't kill the FF on a
            # single dropped frame in a curve.
            target_h = 0.0
            beta = 0.85
        else:
            target_h = raw_heading
            beta = float(self.get_parameter('heading_smoothing').value)
            beta = max(0.0, min(1.0, beta))
        self._heading_smoothed = beta * self._heading_smoothed + (1.0 - beta) * target_h
        heading = float(np.clip(self._heading_smoothed, -1.0, 1.0))

        # Smooth confidence so a single dropped frame doesn't punish the
        # controller, but a sustained loss decays within a few frames.
        alpha = float(self.get_parameter('confidence_smoothing').value)
        alpha = max(0.0, min(1.0, alpha))
        self._confidence_smoothed = (
            alpha * self._confidence_smoothed + (1.0 - alpha) * raw_conf
        )
        confidence = float(np.clip(self._confidence_smoothed, 0.0, 1.0))

        # End-of-road - require a few consecutive frames so a single noisy
        # patch of red carpet does not flip us into U_TURN, AND the orange
        # blob must span most of the ROI width (orange is a perpendicular
        # line, so a narrow contour is almost always a false positive).
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

        # 3D projection for mapping (best-effort, safe to skip when not ready)
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
        # Apply CLAHE to the V (intensity) channel of HSV. Equalising V keeps
        # hue stable while flattening the dynamic range across patchy lighting,
        # which is exactly what HSV thresholding wants.
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
        # Fit a line to the yellow contour and convert its image-space slope
        # into a unitless heading hint in [-1, 1]: positive = lane curves
        # right ahead, negative = curves left. Enables steering BEFORE the
        # centroid has slid sideways.
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
        # cv2 picks an arbitrary direction along the line; force vy >= 0 so
        # the slope sign is consistent (vy > 0 == direction goes downward
        # in the image).
        if vy < 0.0:
            vx, vy = -vx, -vy
        if vy < 0.05:
            # Fitted line is nearly horizontal -> almost certainly garbage
            # (yellow contour spans a single image row).
            self._last_yellow_line = None
            return None
        # slope_per_depth: how much x changes per unit decrease in y (i.e.
        # per unit further into the scene). Right curve -> top of contour
        # is to the right of bottom -> slope > 0.
        slope_per_depth = -vx / vy
        gain = float(self.get_parameter('heading_gain').value)
        self._last_yellow_line = (vx, vy, x0, y0)
        return float(max(-1.0, min(1.0, slope_per_depth * gain)))

    @staticmethod
    def _error_from_centroid(cx, width, target_ratio):
        if cx is None:
            return 0.0
        target = width * target_ratio
        # +ve when the line is to the LEFT of where it should be -> robot
        # has drifted RIGHT relative to the line, so steer LEFT (turn left).
        # Wait: if the WHITE line (right boundary) shows up further LEFT in
        # the image than expected, the robot is too close to it on its right
        # side, so it must steer LEFT.
        # cx < target -> diff > 0 -> err > 0 -> nav steers LEFT. Correct.
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
            # Project (u,v) into the camera's optical frame as a ray, then
            # intersect that ray with the floor (z = -camera_height in base).
            ray_cam = np.array([
                (u - self.cx_pix) / self.fx,
                (v - self.cy_pix) / self.fy,
                1.0,
            ])
            ray_base = T[:3, :3] @ ray_cam
            origin_base = T[:3, 3]
            if abs(ray_base[2]) < 1e-6:
                continue
            scale = (-origin_base[2]) / ray_base[2]  # ground plane z=0 in base
            if scale <= 0:
                continue
            world = origin_base + scale * ray_base
            # Ignore anything behind the robot or absurdly far - those are
            # almost always projection artefacts.
            if world[0] < 0.0 or world[0] > 4.0:
                continue
            pts.append((float(world[0]), float(world[1]), 0.0))
            _ = height  # silence unused; height is encoded via the TF translation

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
        # Tint detected pixels so we can see masks against the source image.
        vis[white_mask > 0] = (220, 220, 255)
        vis[yellow_mask > 0] = (0, 220, 220)
        # orange_mask is from a different (taller) ROI - skip overlay here,
        # the navigation node only cares about the boolean.

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

        # Draw the fitted yellow line (orange) so the slope/heading source
        # is visible alongside the centroid bar.
        if self._last_yellow_line is not None:
            vx, vy, x0, y0 = self._last_yellow_line
            h = vis.shape[0]
            if abs(vy) > 1e-3:
                t_top = (0 - y0) / vy
                t_bot = (h - 1 - y0) / vy
                p1 = (int(x0 + t_top * vx), 0)
                p2 = (int(x0 + t_bot * vx), h - 1)
                cv2.line(vis, p1, p2, (0, 140, 255), 2)

        # Confidence bar across the bottom of the image. Color-coded:
        # green >= 0.7, yellow 0.4..0.7, red < 0.4.
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

        cv2.putText(
            vis,
            f'src={src} err={final_err:+.2f} hdg={heading:+.2f} '
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

#somehwat working version
if __name__ == '__main__':
    main()