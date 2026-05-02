"""
mapping_node.py
===============
Accumulates lane line points published by lane_detection_node and builds a
persistent map of the lane layout in the /map frame for display in RViz.

Inputs
------
/lane/points   sensor_msgs/PointCloud2   sparse lane points in base_link frame,
                                         published each camera frame by
                                         lane_detection_node

Outputs
-------
/lane_map/points  sensor_msgs/PointCloud2  accumulated lane points in /map frame

How it works
------------
1. Each incoming /lane/points cloud is transformed from base_link -> map using
   TF2.  If the /map frame is not yet available (SLAM not running) the message
   is silently skipped and navigation continues normally.

2. Transformed points are inserted into a voxel dictionary keyed by
   (int(x/voxel_size), int(y/voxel_size)).  Only one point is kept per cell,
   so memory is naturally bounded and the cloud stays sparse.

3. A timer publishes the full accumulated cloud at publish_hz (default 2 Hz).
   This decouples the map publish rate from the camera frame rate.

Parameters
----------
base_frame   str    base_link   Source frame of incoming points
map_frame    str    map         Target accumulation frame (must be provided by SLAM)
voxel_size   float  0.05        Grid cell size in metres for downsampling
publish_hz   float  2.0         Rate at which the full map is re-published
max_points   int    50000       Hard cap on stored voxels; oldest evicted first
"""

import struct

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

import tf2_ros
from tf2_ros import TransformException


class MappingNode(Node):
    def __init__(self):
        super().__init__('mapping_node')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('publish_hz', 2.0)
        self.declare_parameter('max_points', 50000)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Voxel dict: (ix, iy) -> (x, y, z) in map frame.
        # Using an ordered-insertion list of keys so we can evict oldest first.
        self._voxel_map: dict = {}
        self._voxel_order: list = []

        self.create_subscription(
            PointCloud2,
            '/lane/points',
            self._lane_points_cb,
            10,
        )

        self.pub_map = self.create_publisher(PointCloud2, '/lane_map/points', 10)

        hz = self.get_parameter('publish_hz').value
        self.create_timer(1.0 / hz, self._publish_map)

        self.get_logger().info('Mapping node up — waiting for /lane/points and /map TF.')

    # ------------------------------------------------------------------ callback
    def _lane_points_cb(self, msg: PointCloud2):
        map_frame = self.get_parameter('map_frame').value
        base_frame = self.get_parameter('base_frame').value

        try:
            tf = self.tf_buffer.lookup_transform(
                map_frame,
                base_frame,
                rclpy.time.Time(),
            )
        except TransformException:
            self.get_logger().warn(
                f'TF not available: {base_frame} -> {map_frame}. '
                'Is SLAM running?',
                throttle_duration_sec=5.0,
            )
            return

        T = self._tf_to_matrix(tf)
        points_base = self._unpack_cloud(msg)
        if not points_base:
            return

        voxel = self.get_parameter('voxel_size').value
        max_pts = int(self.get_parameter('max_points').value)

        for x, y, z in points_base:
            p = np.array([x, y, z, 1.0])
            pm = T @ p
            xm, ym, zm = float(pm[0]), float(pm[1]), float(pm[2])

            key = (int(xm / voxel), int(ym / voxel))
            if key not in self._voxel_map:
                if len(self._voxel_order) >= max_pts:
                    # Evict the oldest voxel
                    old_key = self._voxel_order.pop(0)
                    self._voxel_map.pop(old_key, None)
                self._voxel_order.append(key)
            self._voxel_map[key] = (xm, ym, 0.0)

    # ------------------------------------------------------------------ timer
    def _publish_map(self):
        if not self._voxel_map:
            return

        pts = list(self._voxel_map.values())
        map_frame = self.get_parameter('map_frame').value

        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = map_frame
        msg.height = 1
        msg.width = len(pts)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * len(pts)
        msg.is_dense = True
        msg.data = np.array(pts, dtype=np.float32).tobytes()
        self.pub_map.publish(msg)

        self.get_logger().info(
            f'Lane map published: {len(pts)} voxels in {map_frame}',
            throttle_duration_sec=5.0,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _unpack_cloud(msg: PointCloud2):
        """Unpack a PointCloud2 message into a list of (x, y, z) tuples."""
        points = []
        point_step = msg.point_step
        data = msg.data
        n = msg.width * msg.height
        for i in range(n):
            offset = i * point_step
            if offset + 12 > len(data):
                break
            x, y, z = struct.unpack_from('fff', data, offset)
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue
            points.append((x, y, z))
        return points

    @staticmethod
    def _tf_to_matrix(tf):
        t = tf.transform.translation
        r = tf.transform.rotation
        qx, qy, qz, qw = r.x, r.y, r.z, r.w
        rot = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ])
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = [t.x, t.y, t.z]
        return T


def main(args=None):
    rclpy.init(args=args)
    node = MappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()