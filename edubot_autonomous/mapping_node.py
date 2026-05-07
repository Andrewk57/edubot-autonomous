"""
mapping_node.py
===============
Builds a persistent lane-line map in /map by accumulating points from
lane_detection_node. Publishes:

  /lane_map/points   PointCloud2     (combined, backward compat)
  /lane_map/markers  MarkerArray     colored cube list per class - this is the
                                     one that actually looks like a map in RViz
  /lane_map/grid     OccupancyGrid   top-down 2D occupancy of lane lines

Subscribes (any of the three; /lane/points alone still works):
  /lane/points          - combined (no class info, treated as 'white')
  /lane/points/white    - white solid line
  /lane/points/yellow   - yellow dashed line

FIXES applied:
  - TF lookup now uses msg.header.stamp instead of latest-available, which
    eliminates smearing during rotations.
  - Subscribes to /odom and rejects point clouds when |angular.z| is above
    a threshold (robot spinning -> projection is unreliable).
  - Subscribes to /lane/confidence and rejects clouds below a threshold
    so low-quality detections don't pollute the map.
"""

import math
import struct

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData

import tf2_ros
from tf2_ros import TransformException


class MappingNode(Node):
    def __init__(self):
        super().__init__('mapping_node')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('voxel_size', 0.08)
        self.declare_parameter('publish_hz', 2.0)
        self.declare_parameter('max_points_per_class', 30000)
        self.declare_parameter('grid_size_m', 12.0)
        self.declare_parameter('grid_resolution_m', 0.05)
        self.declare_parameter('marker_cube_size', 0.07)

        # --- NEW: spin rejection & confidence gating params ---
        self.declare_parameter('max_angular_for_mapping', 0.5)   # rad/s - skip clouds above this
        self.declare_parameter('min_confidence_for_mapping', 0.6) # skip clouds below this

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Two voxel maps, one per class. key=(ix,iy) -> (xm, ym).
        self._white_map: dict = {}
        self._white_order: list = []
        self._yellow_map: dict = {}
        self._yellow_order: list = []

        # --- NEW: track current angular velocity and confidence ---
        self._current_angular_z = 0.0
        self._current_confidence = 0.0

        self.create_subscription(Odometry, '/odom', self._odom_cb, 20)
        self.create_subscription(Float32, '/lane/confidence', self._conf_cb, 10)

        self.create_subscription(PointCloud2, '/lane/points',
                                 lambda m: self._lane_cb(m, 'white'), 10)
        self.create_subscription(PointCloud2, '/lane/points/white',
                                 lambda m: self._lane_cb(m, 'white'), 10)
        self.create_subscription(PointCloud2, '/lane/points/yellow',
                                 lambda m: self._lane_cb(m, 'yellow'), 10)

        self.pub_cloud   = self.create_publisher(PointCloud2,  '/lane_map/points',  10)
        self.pub_markers = self.create_publisher(MarkerArray,  '/lane_map/markers', 10)
        self.pub_grid    = self.create_publisher(OccupancyGrid,'/lane_map/grid',    10)

        hz = self.get_parameter('publish_hz').value
        self.create_timer(1.0 / hz, self._publish_all)

        self.get_logger().info(
            'Mapping node up - subscribing /lane/points{,/white,/yellow}, '
            'waiting on /map TF.'
        )

    # --- NEW callbacks for gating ---
    def _odom_cb(self, msg: Odometry):
        self._current_angular_z = msg.twist.twist.angular.z

    def _conf_cb(self, msg: Float32):
        self._current_confidence = float(msg.data)

    # ----------------------------------------------------------- ingest
    def _lane_cb(self, msg: PointCloud2, cls: str):
        # --- NEW: reject clouds when robot is spinning fast ---
        max_ang = float(self.get_parameter('max_angular_for_mapping').value)
        if abs(self._current_angular_z) > max_ang:
            self.get_logger().debug(
                f'Skipping cloud: angular_z={self._current_angular_z:.2f} > {max_ang}',
                throttle_duration_sec=2.0,
            )
            return

        # --- NEW: reject clouds when detection confidence is low ---
        min_conf = float(self.get_parameter('min_confidence_for_mapping').value)
        if self._current_confidence < min_conf:
            return

        map_frame  = self.get_parameter('map_frame').value
        base_frame = self.get_parameter('base_frame').value

        # --- FIX: use the message timestamp, not latest-available TF ---
        # This prevents smearing points around the robot's arc during turns.
        stamp = Time.from_msg(msg.header.stamp)
        try:
            tf = self.tf_buffer.lookup_transform(
                map_frame, base_frame, stamp,
                timeout=Duration(seconds=0.1),
            )
        except TransformException:
            self.get_logger().warn(
                f'TF not available at stamp for {base_frame} -> {map_frame}. SLAM running?',
                throttle_duration_sec=5.0,
            )
            return

        T = self._tf_to_matrix(tf)
        pts_base = self._unpack_cloud(msg)
        if not pts_base:
            return

        voxel = float(self.get_parameter('voxel_size').value)
        max_pts = int(self.get_parameter('max_points_per_class').value)

        if cls == 'yellow':
            vmap, vorder = self._yellow_map, self._yellow_order
        else:
            vmap, vorder = self._white_map, self._white_order

        for x, y, z in pts_base:
            p = np.array([x, y, z, 1.0])
            pm = T @ p
            xm, ym = float(pm[0]), float(pm[1])
            key = (int(xm / voxel), int(ym / voxel))
            if key not in vmap:
                if len(vorder) >= max_pts:
                    vmap.pop(vorder.pop(0), None)
                vorder.append(key)
            vmap[key] = (xm, ym)

    # ----------------------------------------------------------- publish
    def _publish_all(self):
        if not self._white_map and not self._yellow_map:
            return
        map_frame = self.get_parameter('map_frame').value
        stamp = self.get_clock().now().to_msg()

        self._publish_markers(map_frame, stamp)
        self._publish_combined_cloud(map_frame, stamp)
        self._publish_occupancy(map_frame, stamp)

        self.get_logger().info(
            f'lane_map: white={len(self._white_map)}  yellow={len(self._yellow_map)}',
            throttle_duration_sec=5.0,
        )

    def _publish_markers(self, frame_id, stamp):
        size = float(self.get_parameter('marker_cube_size').value)
        arr = MarkerArray()

        def make_marker(mid, pts, rgba):
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = stamp
            m.ns = 'lane_map'
            m.id = mid
            m.type = Marker.CUBE_LIST
            m.action = Marker.ADD
            m.scale.x = size
            m.scale.y = size
            m.scale.z = 0.02
            m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
            m.pose.orientation.w = 1.0
            for (x, y) in pts:
                p = Point()
                p.x, p.y, p.z = x, y, 0.01
                m.points.append(p)
            return m

        # White line: bright white, slightly transparent.
        arr.markers.append(make_marker(
            0, list(self._white_map.values()),  (1.0, 1.0, 1.0, 0.95)
        ))
        # Yellow line: high-vis yellow.
        arr.markers.append(make_marker(
            1, list(self._yellow_map.values()), (1.0, 0.9, 0.0, 0.95)
        ))
        self.pub_markers.publish(arr)

    def _publish_combined_cloud(self, frame_id, stamp):
        pts = [(x, y, 0.0) for (x, y) in self._white_map.values()]
        pts += [(x, y, 0.0) for (x, y) in self._yellow_map.values()]
        if not pts:
            return
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
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
        self.pub_cloud.publish(msg)

    def _publish_occupancy(self, frame_id, stamp):
        size_m = float(self.get_parameter('grid_size_m').value)
        res    = float(self.get_parameter('grid_resolution_m').value)
        n = int(size_m / res)

        # Center grid on the centroid of all stored points so it follows the map.
        all_pts = list(self._white_map.values()) + list(self._yellow_map.values())
        if not all_pts:
            return
        arr = np.array(all_pts, dtype=np.float32)
        cx, cy = float(arr[:, 0].mean()), float(arr[:, 1].mean())
        origin_x = cx - size_m / 2.0
        origin_y = cy - size_m / 2.0

        grid = np.full((n, n), -1, dtype=np.int8)  # unknown
        for (x, y) in all_pts:
            ix = int((x - origin_x) / res)
            iy = int((y - origin_y) / res)
            if 0 <= ix < n and 0 <= iy < n:
                grid[iy, ix] = 100  # occupied

        msg = OccupancyGrid()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp
        meta = MapMetaData()
        meta.resolution = res
        meta.width = n
        meta.height = n
        meta.origin.position.x = origin_x
        meta.origin.position.y = origin_y
        meta.origin.orientation.w = 1.0
        msg.info = meta
        msg.data = grid.flatten().tolist()
        self.pub_grid.publish(msg)

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _unpack_cloud(msg: PointCloud2):
        pts = []
        ps = msg.point_step
        data = msg.data
        n = msg.width * msg.height
        for i in range(n):
            off = i * ps
            if off + 12 > len(data):
                break
            x, y, z = struct.unpack_from('fff', data, off)
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue
            pts.append((x, y, z))
        return pts

    @staticmethod
    def _tf_to_matrix(tf):
        t = tf.transform.translation
        r = tf.transform.rotation
        qx, qy, qz, qw = r.x, r.y, r.z, r.w
        rot = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
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