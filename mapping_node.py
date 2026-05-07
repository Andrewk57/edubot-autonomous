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
/lane_map/grid    nav_msgs/OccupancyGrid   2-D occupancy grid of lane lines (for RViz2 Map display)

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
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
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
        # deque gives O(1) pop from the front for oldest-first eviction.
        self._voxel_map: dict = {}
        self._voxel_order: deque = deque()

        self.create_subscription(
            PointCloud2,
            '/lane/points',
            self._lane_points_cb,
            10,
        )

        self.pub_map = self.create_publisher(PointCloud2, '/lane_map/points', 10)
        self.pub_grid = self.create_publisher(OccupancyGrid, '/lane_map/grid', 10)

        hz = self.get_parameter('publish_hz').value
        self.create_timer(1.0 / hz, self._publish_map)

        self.get_logger().info(
            'Mapping node up — waiting for /lane/points and /map TF.\n'
            'Add /lane_map/grid (nav_msgs/OccupancyGrid) in RViz2 to see the track map.'
        )

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
                    # Evict the oldest voxel (O(1) with deque)
                    old_key = self._voxel_order.popleft()
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
        self._publish_grid(pts, map_frame)

        self.get_logger().info(
            f'Lane map published: {len(pts)} voxels in {map_frame}',
            throttle_duration_sec=5.0,
        )

    # -------------------------------------------------------------- grid helper
    def _publish_grid(self, pts, map_frame: str):
        """Build and publish a nav_msgs/OccupancyGrid from accumulated voxels.

        Lane-line cells are marked 100 (occupied / black in RViz2).
        All other cells are -1 (unknown / grey).
        """
        voxel = self.get_parameter('voxel_size').value
        padding = 1.0  # extra metres around the bounding box

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        origin_x = min(xs) - padding
        origin_y = min(ys) - padding
        width  = int((max(xs) - origin_x + padding) / voxel) + 1
        height = int((max(ys) - origin_y + padding) / voxel) + 1

        # Guard against pathologically large grids
        if width * height > 4_000_000:
            self.get_logger().warn(
                'OccupancyGrid too large to publish; accumulate fewer points or increase voxel_size.',
                throttle_duration_sec=10.0,
            )
            return

        grid_data = np.full(width * height, -1, dtype=np.int8)

        for x, y, _ in pts:
            ix = int((x - origin_x) / voxel)
            iy = int((y - origin_y) / voxel)
            if 0 <= ix < width and 0 <= iy < height:
                grid_data[iy * width + ix] = 100

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = map_frame
        msg.info.resolution = voxel
        msg.info.width  = width
        msg.info.height = height
        msg.info.origin = Pose()
        msg.info.origin.position.x = origin_x
        msg.info.origin.position.y = origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid_data.tolist()
        self.pub_grid.publish(msg)

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