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

        # ROI - cropping out sky / bumper makes everything else easier
        self.declare_parameter('crop_top_ratio', 0.45)
        self.declare_parameter('crop_bottom_ratio', 0.05)
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

        # Geometry / behavior
        self.declare_parameter('min_contour_area', 1500.0)
        self.declare_parameter('target_white_x_ratio', 0.80)   # white sits 80% across
        self.declare_parameter('target_yellow_x_ratio', 0.30)  # yellow sits 30% across
        self.declare_parameter('yellow_weight', 0.35)          # blend factor when both visible
        self.declare_parameter('right_bias', 0.20)             # nudge right when only yellow
        self.declare_parameter('yellow_memory_secs', 0.6)
        self.declare_parameter('min_orange_pixels', 4000)
        self.declare_parameter('orange_consecutive_frames', 3)
        self.declare_parameter('use_clahe', True)
        self.declare_parameter('debug_image', True)

        self.bridge = CvBridge()
        self.fx = self.fy = self.cx_pix = self.cy_pix = None

        self._last_yellow_cx = None
        self._last_yellow_stamp = 0.0
        self._orange_streak = 0

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
        roi = frame[roi_top:roi_bot, :]
        roi_h, roi_w = roi.shape[:2]

        orange_top = int(h * self.get_parameter('crop_top_orange_ratio').value)
        roi_orange = frame[orange_top:roi_bot, :]

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
        yellow_ok = yellow_cx is not None

        if white_ok and yellow_ok:
            yw = self.get_parameter('yellow_weight').value
            final_err = (1.0 - yw) * white_err + yw * yellow_err
            src = 'BLEND'
        elif white_ok:
            final_err = white_err
            src = 'WHITE'
        elif yellow_ok:
            final_err = yellow_err + self.get_parameter('right_bias').value
            src = 'YELLOW+BIAS'
        else:
            final_err = 0.0
            src = 'NONE'

        final_err = float(np.clip(final_err, -1.5, 1.5))

        # End-of-road - require a few consecutive frames so a single noisy
        # patch of red carpet does not flip us into U_TURN.
        orange_pixels = int(cv2.countNonZero(orange_mask))
        if orange_pixels >= self.get_parameter('min_orange_pixels').value:
            self._orange_streak += 1
        else:
            self._orange_streak = 0
        end_of_road = self._orange_streak >= self.get_parameter(
            'orange_consecutive_frames'
        ).value

        self.pub_error.publish(Float32(data=final_err))
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
            )

        self.get_logger().info(
            f'src={src} err={final_err:+.2f} white={white_ok} yellow={yellow_ok} '
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

    @staticmethod
    def _error_from_centroid(cx, width, target_ratio):
        if cx is None:
            return 0.0
        target = width * target_ratio
        # If the WHITE line (right boundary) shows up further LEFT in the
        # image than expected, the robot is too close to it on its right
        # side, so it must steer LEFT.
        # cx < target -> diff > 0 -> err > 0 -> nav steers LEFT.
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
            scale = (-origin_base[2]) / ray_base[2]  # ground plane z=0 in base
            if scale <= 0:
                continue
            world = origin_base + scale * ray_base
            if world[0] < 0.0 or world[0] > 4.0:
                continue
            pts.append((float(world[0]), float(world[1]), 0.0))

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

        cv2.putText(
            vis, f'src={src} err={final_err:+.2f}',
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