"""
cone_detection_node.py
======================
Front-camera traffic-cone detection for EduBot.

Detects bright orange cones with HSV color segmentation and publishes:
  /cone/detected       std_msgs/Bool      True when cone is visible
  /cone/center_x       std_msgs/Float32   normalized horizontal position [-1, 1]
                                        -1 = left, 0 = center, +1 = right
  /cone/area           std_msgs/Float32   largest cone contour area in pixels
  /cone/debug_image    sensor_msgs/Image  camera image with detection overlay

Navigation can use /cone/detected and /cone/center_x to avoid the cone.
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32


class ConeDetectionNode(Node):
    def __init__(self):
        super().__init__('cone_detection_node')

        # Use the front camera by default. If your robot uses another topic,
        # override it with: --ros-args -p image_topic:=/your/topic
        self.declare_parameter('image_topic', '/camera_1/image_raw')
        self.declare_parameter('debug_image', True)

        # HSV range for bright orange traffic cones.
        # Tune live using ros2 param set if lighting is different.
        self.declare_parameter('h_min', 5)
        self.declare_parameter('h_max', 25)
        self.declare_parameter('s_min', 90)
        self.declare_parameter('s_max', 255)
        self.declare_parameter('v_min', 90)
        self.declare_parameter('v_max', 255)

        # Filter small orange objects/noise.
        self.declare_parameter('min_area', 700.0)
        self.declare_parameter('min_width', 15)
        self.declare_parameter('min_height', 15)

        # Only look in the lower/middle part of image where a cone appears.
        self.declare_parameter('crop_top_ratio', 0.20)
        self.declare_parameter('crop_bottom_ratio', 0.02)

        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self._image_cb,
            10,
        )

        self.pub_detected = self.create_publisher(Bool, '/cone/detected', 10)
        self.pub_center_x = self.create_publisher(Float32, '/cone/center_x', 10)
        self.pub_area = self.create_publisher(Float32, '/cone/area', 10)
        self.pub_debug = self.create_publisher(Image, '/cone/debug_image', 10)

        self.get_logger().info(
            f'Cone detection node started. Subscribing to {self.get_parameter("image_topic").value}'
        )

    def _image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'cv_bridge image conversion failed: {exc}')
            return

        height, width = frame.shape[:2]
        y1 = int(height * float(self.get_parameter('crop_top_ratio').value))
        y2 = int(height * (1.0 - float(self.get_parameter('crop_bottom_ratio').value)))
        roi = frame[y1:y2, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([
            int(self.get_parameter('h_min').value),
            int(self.get_parameter('s_min').value),
            int(self.get_parameter('v_min').value),
        ], dtype=np.uint8)
        upper = np.array([
            int(self.get_parameter('h_max').value),
            int(self.get_parameter('s_max').value),
            int(self.get_parameter('v_max').value),
        ], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = False
        best_area = 0.0
        norm_center_x = 0.0
        best_box = None

        min_area = float(self.get_parameter('min_area').value)
        min_w = int(self.get_parameter('min_width').value)
        min_h = int(self.get_parameter('min_height').value)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < min_w or h < min_h:
                continue
            if area > best_area:
                best_area = area
                cx = x + w / 2.0
                norm_center_x = float((cx - width / 2.0) / (width / 2.0))
                best_box = (x, y + y1, w, h)
                detected = True

        self.pub_detected.publish(Bool(data=detected))
        self.pub_center_x.publish(Float32(data=norm_center_x))
        self.pub_area.publish(Float32(data=best_area))

        if detected:
            self.get_logger().info(
                f'Cone detected: center_x={norm_center_x:+.2f}, area={best_area:.0f}',
                throttle_duration_sec=0.5,
            )

        if self.get_parameter('debug_image').value:
            debug = frame.copy()
            if best_box is not None:
                x, y, w, h = best_box
                cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 140, 255), 2)
                cv2.putText(
                    debug,
                    f'cone x={norm_center_x:+.2f} area={best_area:.0f}',
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 140, 255),
                    2,
                )
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
