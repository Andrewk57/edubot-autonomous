"""
mapping_node.py
===============
Builds only the white and yellow lane boundary lines from lane point clouds.

Inputs
------
  /lane/points/white   PointCloud2   white lane points in base_link
  /lane/points/yellow  PointCloud2   yellow lane points in base_link

Outputs
-------
  /lane_map/points   PointCloud2   densified line points in map frame
  /lane_map/markers  MarkerArray   white and yellow LINE_STRIP markers
  /lane_map/grid     OccupancyGrid line-only map, 100=line -1=unknown
"""

from collections import deque
import struct

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
from tf2_ros import TransformException

LANE_WHITE = '/lane/points/white'
LANE_YELLOW = '/lane/points/yellow'


def _densify_profile(profile: dict[int, int], max_gap_cells: int) -> dict[int, int]:
    if not profile:
        return {}

    xs = sorted(profile.keys())
    dense = {int(xs[0]): int(profile[xs[0]])}
    for idx in range(len(xs) - 1):
        x0 = int(xs[idx])
        y0 = int(profile[x0])
        x1 = int(xs[idx + 1])
        y1 = int(profile[x1])
        dense[x1] = y1
        gap = x1 - x0
        if gap <= 1 or gap > max_gap_cells:
            continue
        for step in range(1, gap):
            t = step / gap
            xi = x0 + step
            yi = int(round(y0 + t * (y1 - y0)))
            dense[xi] = yi
    return dense


def _smooth_profile(profile: dict[int, int], window_radius: int) -> dict[int, int]:
    if not profile or window_radius <= 0:
        return profile

    xs = sorted(profile.keys())
    smoothed: dict[int, int] = {}
    for x in xs:
        neighbors = [
            profile[xi]
            for xi in range(x - window_radius, x + window_radius + 1)
            if xi in profile
        ]
        smoothed[x] = int(round(float(np.median(neighbors)))) if neighbors else profile[x]
    return smoothed


class MappingNode(Node):
    def __init__(self):
        super().__init__('mapping_node')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('publish_hz', 3.0)
        self.declare_parameter('max_points_per_class', 12000)
        self.declare_parameter('grid_resolution_m', 0.05)
        self.declare_parameter('line_width_cells', 2)
        self.declare_parameter('max_interp_gap_m', 0.35)
        self.declare_parameter('padding_m', 0.5)
        self.declare_parameter('profile_smoothing_window', 2)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._white_map: dict[tuple[int, int], tuple[float, float]] = {}
        self._white_order: deque[tuple[int, int]] = deque()
        self._yellow_map: dict[tuple[int, int], tuple[float, float]] = {}
        self._yellow_order: deque[tuple[int, int]] = deque()

        self.create_subscription(PointCloud2, LANE_WHITE, self._white_cb, 10)
        self.create_subscription(PointCloud2, LANE_YELLOW, self._yellow_cb, 10)

        self.pub_cloud = self.create_publisher(PointCloud2, '/lane_map/points', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/lane_map/markers', 10)
        self.pub_grid = self.create_publisher(OccupancyGrid, '/lane_map/grid', 10)

        hz = float(self.get_parameter('publish_hz').value)
        self.create_timer(1.0 / hz, self._publish_all)

        self.get_logger().info('Mapping node up. Publishing only white/yellow lane lines.')

    def _white_cb(self, msg: PointCloud2):
        self._lane_cb(msg, self._white_map, self._white_order)

    def _yellow_cb(self, msg: PointCloud2):
        self._lane_cb(msg, self._yellow_map, self._yellow_order)

    def _lane_cb(self, msg: PointCloud2, lane_map, lane_order):
        map_frame = self.get_parameter('map_frame').value
        base_frame = self.get_parameter('base_frame').value
        try:
            tf = self.tf_buffer.lookup_transform(
                map_frame,
                base_frame,
                Time.from_msg(msg.header.stamp),
            )
        except TransformException:
            self.get_logger().warn(
                f'TF not available: {base_frame} -> {map_frame}.',
                throttle_duration_sec=5.0,
            )
            return

        points = self._unpack_cloud(msg)
        if not points:
            return

        transform = self._tf_to_matrix(tf)
        voxel = float(self.get_parameter('voxel_size').value)
        max_points = int(self.get_parameter('max_points_per_class').value)

        for x, y, z in points:
            point_map = transform @ np.array([x, y, z, 1.0], dtype=np.float64)
            xm = float(point_map[0])
            ym = float(point_map[1])
            key = (int(round(xm / voxel)), int(round(ym / voxel)))
            if key not in lane_map:
                if len(lane_order) >= max_points:
                    old_key = lane_order.popleft()
                    lane_map.pop(old_key, None)
                lane_order.append(key)
            lane_map[key] = (xm, ym)

    def _publish_all(self):
        white_profile = self._build_profile(list(self._white_map.values()))
        yellow_profile = self._build_profile(list(self._yellow_map.values()))
        if not white_profile and not yellow_profile:
            return

        stamp = self.get_clock().now().to_msg()
        frame_id = self.get_parameter('map_frame').value

        self._publish_markers(frame_id, stamp, white_profile, yellow_profile)
        self._publish_cloud(frame_id, stamp, white_profile, yellow_profile)
        self._publish_grid(frame_id, stamp, white_profile, yellow_profile)

    def _build_profile(self, points):
        if not points:
            return {}

        res = float(self.get_parameter('grid_resolution_m').value)
        max_gap_cells = max(1, int(round(float(self.get_parameter('max_interp_gap_m').value) / res)))
        smoothing_window = max(0, int(self.get_parameter('profile_smoothing_window').value))

        arr = np.array(points, dtype=np.float32)
        xy = arr[:, :2]
        center = xy.mean(axis=0)
        centered = xy - center

        if len(xy) >= 2:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            tangent = vh[0]
        else:
            tangent = np.array([1.0, 0.0], dtype=np.float32)

        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            tangent = np.array([1.0, 0.0], dtype=np.float32)
        else:
            tangent = tangent / tangent_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)

        longitudinal = centered @ tangent
        lateral = centered @ normal
        origin_s = float(longitudinal.min())
        gs = np.floor((longitudinal - origin_s) / res).astype(int)

        profile_local: dict[int, int] = {}
        for si in np.unique(gs):
            lateral_for_s = lateral[gs == si]
            if len(lateral_for_s) == 0:
                continue
            profile_local[int(si)] = int(round(float(np.median(lateral_for_s)) / res))

        dense_local = _densify_profile(profile_local, max_gap_cells)
        smooth_local = _smooth_profile(dense_local, smoothing_window)
        profile_world: dict[int, tuple[float, float]] = {}
        for si, ni in smooth_local.items():
            point = center + (origin_s + si * res) * tangent + (ni * res) * normal
            profile_world[int(si)] = (float(point[0]), float(point[1]))
        return profile_world

    def _publish_markers(self, frame_id, stamp, white_profile, yellow_profile):
        markers = MarkerArray()
        markers.markers.append(
            self._make_line_marker(frame_id, stamp, 0, white_profile, (1.0, 1.0, 1.0, 1.0))
        )
        markers.markers.append(
            self._make_line_marker(frame_id, stamp, 1, yellow_profile, (1.0, 0.9, 0.0, 1.0))
        )
        self.pub_markers.publish(markers)

    def _publish_cloud(self, frame_id, stamp, white_profile, yellow_profile):
        points = [(x, y, 0.0) for x, y in white_profile.values()]
        points += [(x, y, 0.0) for x, y in yellow_profile.values()]

        msg = PointCloud2()
        msg.header.stamp = stamp
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
        msg.data = np.array(points, dtype=np.float32).tobytes() if points else b''
        self.pub_cloud.publish(msg)

    def _publish_grid(self, frame_id, stamp, white_profile, yellow_profile):
        all_points = list(white_profile.values()) + list(yellow_profile.values())
        if not all_points:
            return

        res = float(self.get_parameter('grid_resolution_m').value)
        padding = float(self.get_parameter('padding_m').value)
        line_width = max(1, int(self.get_parameter('line_width_cells').value))

        arr = np.array(all_points, dtype=np.float32)
        origin_x = float(arr[:, 0].min()) - padding
        origin_y = float(arr[:, 1].min()) - padding
        width = int(np.ceil((float(arr[:, 0].max()) - origin_x + padding) / res)) + 1
        height = int(np.ceil((float(arr[:, 1].max()) - origin_y + padding) / res)) + 1

        grid = np.full((height, width), -1, dtype=np.int8)
        for profile in (white_profile, yellow_profile):
            for xw, yw in profile.values():
                gx = int(round((xw - origin_x) / res))
                gy = int(round((yw - origin_y) / res))
                for dy in range(-line_width + 1, line_width):
                    yy = gy + dy
                    if 0 <= gx < width and 0 <= yy < height:
                        grid[yy, gx] = 100

        msg = OccupancyGrid()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp
        meta = MapMetaData()
        meta.resolution = res
        meta.width = width
        meta.height = height
        meta.origin.position.x = origin_x
        meta.origin.position.y = origin_y
        meta.origin.orientation.w = 1.0
        msg.info = meta
        msg.data = grid.reshape(-1).tolist()
        self.pub_grid.publish(msg)

    def _make_line_marker(self, frame_id, stamp, marker_id, profile, rgba):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = 'lane_lines'
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.04
        marker.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
        marker.pose.orientation.w = 1.0
        for xi in sorted(profile.keys()):
            xw, yw = profile[xi]
            point = Point()
            point.x = float(xw)
            point.y = float(yw)
            point.z = 0.03
            marker.points.append(point)
        return marker

    @staticmethod
    def _unpack_cloud(msg: PointCloud2):
        points = []
        point_step = msg.point_step
        count = msg.width * msg.height
        for idx in range(count):
            offset = idx * point_step
            if offset + 12 > len(msg.data):
                break
            x, y, z = struct.unpack_from('fff', msg.data, offset)
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                points.append((x, y, z))
        return points

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
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = [t.x, t.y, t.z]
        return transform


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