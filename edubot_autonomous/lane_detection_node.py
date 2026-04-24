import math
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PolygonStamped, Point32, TransformStamped
import tf2_ros
from tf2_ros import TransformException, StaticTransformBroadcaster


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # ── Geometry / TF ─────────────────────────────────────────
        self.declare_parameter('camera_height_m', 0.15)
        self.declare_parameter('camera_frame', 'camera_2_optical_frame')
        self.declare_parameter('base_frame', 'base_link')

        # ── ROI crop (bottom portion of image, robot-relevant area) ─
        self.declare_parameter('roi_top_ratio', 0.25)
        self.declare_parameter('roi_bottom_ratio', 0.05)

        # ── White line detection (Canny + Hough) ──────────────────
        self.declare_parameter('canny_low', 60)
        self.declare_parameter('canny_high', 180)
        self.declare_parameter('hough_threshold', 60)
        self.declare_parameter('hough_min_line_length', 80)
        self.declare_parameter('hough_max_line_gap', 10)

        # ── Yellow center line (tight LAB gate) ───────────────────
        self.declare_parameter('yellow_l_min', 150)
        self.declare_parameter('yellow_l_max', 255)
        self.declare_parameter('yellow_a_min', 80)
        self.declare_parameter('yellow_a_max', 120)
        self.declare_parameter('yellow_b_min', 160)
        self.declare_parameter('yellow_b_max', 220)
        self.declare_parameter('min_yellow_area', 200.0)
        self.declare_parameter('contour_sample_points', 5)

        self.fx = self.fy = self.cx = self.cy = None
        self.bridge = CvBridge()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_camera_2_tf()

        self.create_subscription(CameraInfo, '/camera_2/camera_info', self._camera_info_cb, 10)
        self.create_subscription(Image, '/camera_2/image_raw', self._image_cb, 10)

        self.pub_mask = self.create_publisher(Image, '/lane/mask', 10)
        self.pub_white = self.create_publisher(PolygonStamped, '/lane/white_lines', 10)
        self.pub_center = self.create_publisher(PolygonStamped, '/lane/center_line', 10)
        self.pub_points = self.create_publisher(PointCloud2, '/lane/points', 10)

        self.get_logger().info('Lane detection node started (Canny+Hough mode)')

    def _publish_camera_2_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_link'
        t.child_frame_id = 'camera_2_optical_frame'

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = -0.0254

        pitch = math.radians(-45.0)
        roll  = math.radians(90.0)
        yaw   = 0.0

        cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cr, sr = math.cos(roll/2),  math.sin(roll/2)

        t.transform.rotation.w = cr*cp*cy + sr*sp*sy
        t.transform.rotation.x = sr*cp*cy - cr*sp*sy
        t.transform.rotation.y = cr*sp*cy + sr*cp*sy
        t.transform.rotation.z = cr*cp*sy - sr*sp*cy

        self._static_broadcaster.sendTransform(t)

    def _camera_info_cb(self, msg: CameraInfo):
        if self.fx is not None:
            return
        k = msg.k
        self.fx, self.fy, self.cx, self.cy = k[0], k[4], k[2], k[5]
        self.get_logger().info(
            f'Camera intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} '
            f'cx={self.cx:.1f} cy={self.cy:.1f}'
        )

    def _image_cb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = bgr.shape[:2]

        top = int(h * self.get_parameter('roi_top_ratio').value)
        bot = int(h * (1.0 - self.get_parameter('roi_bottom_ratio').value))
        roi = bgr[top:bot, :]
        roi_y_offset = top

        white_lines = self._detect_white_lines(roi)
        yellow_mask, yellow_contours = self._detect_yellow(roi)

        self._publish_mask(msg.header, roi, white_lines, yellow_mask, yellow_contours)

        self.get_logger().info(
            f'white_lines={len(white_lines)} yellow_contours={len(yellow_contours)}',
            throttle_duration_sec=2.0,
        )

        if self.fx is None:
            self.get_logger().warn('Waiting for camera_info', throttle_duration_sec=5.0)
            return

        camera_frame = self.get_parameter('camera_frame').value
        base_frame = self.get_parameter('base_frame').value
        height = self.get_parameter('camera_height_m').value

        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                base_frame, camera_frame, rclpy.time.Time()
            )
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=2.0)
            return

        T = self._tf_to_matrix(tf_stamped)

        white_pixels = []
        for x1, y1, x2, y2 in white_lines:
            white_pixels.append((float(x1), float(y1 + roi_y_offset)))
            white_pixels.append((float(x2), float(y2 + roi_y_offset)))
            white_pixels.append((float((x1+x2)/2), float((y1+y2)/2 + roi_y_offset)))

        yellow_pixels = self._sample_contours(yellow_contours, roi_y_offset)

        white_pts = self._project(white_pixels, height, T)
        yellow_pts = self._project(yellow_pixels, height, T)

        self._publish_polygon(msg.header, self.pub_white, white_pts, base_frame)
        self._publish_polygon(msg.header, self.pub_center, yellow_pts, base_frame)
        self._publish_point_cloud(msg.header, white_pts + yellow_pts, base_frame)

    def _detect_white_lines(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(
            blur,
            self.get_parameter('canny_low').value,
            self.get_parameter('canny_high').value,
        )

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.get_parameter('hough_threshold').value,
            minLineLength=self.get_parameter('hough_min_line_length').value,
            maxLineGap=self.get_parameter('hough_max_line_gap').value,
        )

        if lines is None:
            return []

        filtered = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if 20 < angle < 160:
                filtered.append((x1, y1, x2, y2))
        return filtered

    def _detect_yellow(self, roi):
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        p = self.get_parameter

        mask = cv2.inRange(
            lab,
            (p('yellow_l_min').value, p('yellow_a_min').value, p('yellow_b_min').value),
            (p('yellow_l_max').value, p('yellow_a_max').value, p('yellow_b_max').value),
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        min_area = p('min_yellow_area').value
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
        return mask, filtered

    def _sample_contours(self, contours, y_offset):
        n = self.get_parameter('contour_sample_points').value
        pixels = []
        for cnt in contours:
            step = max(1, len(cnt) // n)
            for i in range(0, len(cnt), step):
                pt = cnt[i][0]
                pixels.append((float(pt[0]), float(pt[1] + y_offset)))
        return pixels

    def _project(self, pixels, height, T):
        points = []
        for u, v in pixels:
            p_cam = np.array([
                (u - self.cx) / self.fx * height,
                (v - self.cy) / self.fy * height,
                height,
                1.0,
            ])
            p_base = T @ p_cam
            points.append((float(p_base[0]), float(p_base[1]), float(p_base[2])))
        return points

    def _tf_to_matrix(self, tf_stamped):
        t = tf_stamped.transform.translation
        r = tf_stamped.transform.rotation
        qx, qy, qz, qw = r.x, r.y, r.z, r.w
        rot = np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ])
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def _publish_mask(self, header, roi, white_lines, yellow_mask, yellow_contours):
        vis = roi.copy()
        vis[yellow_mask > 0] = (0, 200, 0)
        for x1, y1, x2, y2 in white_lines:
            cv2.line(vis, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv2.drawContours(vis, yellow_contours, -1, (0, 255, 255), 2)

        out = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        out.header = header
        self.pub_mask.publish(out)

    def _publish_polygon(self, header, publisher, points_3d, frame_id):
        msg = PolygonStamped()
        msg.header = header
        msg.header.frame_id = frame_id
        msg.polygon.points = [Point32(x=x, y=y, z=z) for x, y, z in points_3d]
        publisher.publish(msg)

    def _publish_point_cloud(self, header, points_3d, frame_id):
        if not points_3d:
            return
        msg = PointCloud2()
        msg.header = header
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = len(points_3d)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * len(points_3d)
        msg.is_dense = True
        msg.data = np.array(points_3d, dtype=np.float32).tobytes()
        self.pub_points.publish(msg)


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