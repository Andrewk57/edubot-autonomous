import math

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node

from geometry_msgs.msg import PolygonStamped, PoseStamped, TransformStamped
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from visualization_msgs.msg import Marker
import tf2_ros
from tf2_ros import TransformException


DRIVING = 'DRIVING'
STOPPED = 'STOPPED'


class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        self.declare_parameter('dry_run', True)
        self.declare_parameter('right_offset_m', 0.15)
        self.declare_parameter('lookahead_m', 0.50)
        self.declare_parameter('obstacle_stop_m', 0.40)
        self.declare_parameter('obstacle_cone_deg', 20.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        self.state = DRIVING
        self.center_line: PolygonStamped | None = None
        self.white_lines: PolygonStamped | None = None
        self.scan: LaserScan | None = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._pending_goal = False

        self.create_subscription(PolygonStamped, '/lane/center_line', self._center_cb, 10)
        self.create_subscription(PolygonStamped, '/lane/white_lines', self._white_cb, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        self.pub_obstacles = self.create_publisher(PointCloud2, '/lane_obstacles', 10)
        self.pub_wp_marker = self.create_publisher(Marker, '/nav/waypoint_marker', 10)

        self.create_timer(0.2, self._loop)

        dry = self.get_parameter('dry_run').value
        self.get_logger().info(f'Navigation node started  [dry_run={dry}]')

    def _center_cb(self, msg: PolygonStamped):
        self.center_line = msg

    def _white_cb(self, msg: PolygonStamped):
        self.white_lines = msg
        self._publish_obstacles(msg)

    def _scan_cb(self, msg: LaserScan):
        self.scan = msg

    def _loop(self):
        obstacle = self._obstacle_ahead()

        if self.state == DRIVING:
            if obstacle:
                self.state = STOPPED
                self.get_logger().info('Obstacle detected — stopping')
                return
            wp_base = self._compute_waypoint_base()
            if wp_base is None:
                self.get_logger().warn(
                    'No centre-line data — waiting', throttle_duration_sec=2.0
                )
                return
            wp_map = self._transform_to_map(wp_base)
            if wp_map is None:
                return
            self._publish_wp_marker(wp_map)
            self.get_logger().info(
                f'[{self.state}] waypoint → x={wp_map.pose.position.x:.2f} '
                f'y={wp_map.pose.position.y:.2f}',
                throttle_duration_sec=1.0,
            )
            if not self.get_parameter('dry_run').value:
                self._send_goal(wp_map)

        elif self.state == STOPPED:
            if not obstacle:
                self.get_logger().info('Path clear — resuming')
                self.state = DRIVING

    def _compute_waypoint_base(self) -> PoseStamped | None:
        if self.center_line is None or not self.center_line.polygon.points:
            return None

        pts = self.center_line.polygon.points
        lookahead = self.get_parameter('lookahead_m').value
        right_offset = self.get_parameter('right_offset_m').value

        best = max(pts, key=lambda p: p.x)

        target_x = min(float(best.x), lookahead)
        target_y = float(best.y) - right_offset

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.get_parameter('base_frame').value
        pose.pose.position.x = target_x
        pose.pose.position.y = target_y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    def _transform_to_map(self, pose_base: PoseStamped) -> PoseStamped | None:
        map_frame = self.get_parameter('map_frame').value
        base_frame = self.get_parameter('base_frame').value
        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                map_frame, base_frame, rclpy.time.Time()
            )
        except TransformException as e:
            self.get_logger().warn(f'TF base_link→map failed: {e}', throttle_duration_sec=2.0)
            return None

        t = tf.transform.translation
        r = tf.transform.rotation
        qx, qy, qz, qw = r.x, r.y, r.z, r.w
        rot = np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ])
        p_base = np.array([
            pose_base.pose.position.x,
            pose_base.pose.position.y,
            pose_base.pose.position.z,
        ])
        p_map = rot @ p_base + np.array([t.x, t.y, t.z])

        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = map_frame
        out.pose.position.x = float(p_map[0])
        out.pose.position.y = float(p_map[1])
        out.pose.position.z = float(p_map[2])
        out.pose.orientation = r
        return out

    def _obstacle_ahead(self) -> bool:
        if self.scan is None:
            return False
        threshold = self.get_parameter('obstacle_stop_m').value
        half_cone = math.radians(self.get_parameter('obstacle_cone_deg').value)
        a_min = self.scan.angle_min
        a_inc = self.scan.angle_increment
        for i, r in enumerate(self.scan.ranges):
            angle = a_min + i * a_inc
            if abs(angle) <= half_cone and 0.05 < r < threshold:
                return True
        return False

    def _publish_obstacles(self, msg: PolygonStamped):
        pts = msg.polygon.points
        if not pts:
            return
        arr = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)
        pc = PointCloud2()
        pc.header = msg.header
        pc.height = 1
        pc.width = len(pts)
        pc.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        pc.is_bigendian = False
        pc.point_step = 12
        pc.row_step = 12 * len(pts)
        pc.is_dense = True
        pc.data = arr.tobytes()
        self.pub_obstacles.publish(pc)

    def _send_goal(self, pose: PoseStamped):
        if self._pending_goal:
            return
        if not self._nav_client.wait_for_server(timeout_sec=0.05):
            self.get_logger().warn('Nav2 not available', throttle_duration_sec=5.0)
            return
        goal = NavigateToPose.Goal()
        goal.pose = pose
        future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_cb)
        self._pending_goal = True

    def _goal_response_cb(self, future):
        result = future.result()
        if result is None or not result.accepted:
            self.get_logger().warn('Nav2 rejected goal')
        self._pending_goal = False

    def _publish_wp_marker(self, pose: PoseStamped):
        m = Marker()
        m.header = pose.header
        m.ns = 'waypoint'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose = pose.pose
        m.scale.x = m.scale.y = m.scale.z = 0.12
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.2
        m.color.a = 1.0
        m.lifetime = Duration(seconds=0.5).to_msg()
        self.pub_wp_marker.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()