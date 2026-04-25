"""
navigation_node.py
==================
Closed-loop lane-following controller for EduBot.

Inputs
------
/lane/error          std_msgs/Float32   normalised lateral error from
                                        lane_detection_node ([-1.5, 1.5])
/lane/end_of_road    std_msgs/Bool      orange perpendicular line in view
/lane/white_detected std_msgs/Bool      sanity flag - we have a right boundary
/scan                sensor_msgs/LaserScan  forward LiDAR for obstacle stop
/odom                nav_msgs/Odometry  used to close the loop on the U-turn

Outputs
-------
/cmd_vel             geometry_msgs/Twist
/nav/state           std_msgs/String        current FSM state for debugging

State machine
-------------
DRIVING   - PD on the lateral error, scaled forward velocity
OBSTACLE  - LiDAR cone is occupied; hold zero velocity until clear
U_TURN    - end-of-road triggered; rotate ~180 then return to DRIVING
LOST      - no error signal at all in a while; creep forward slowly so we
            don't deadlock if a frame is dropped, but cap the duration

The PD gains, speeds, cone, and cooldown are all parameters so the demo can
be tuned trackside without rebuilding.
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String


DRIVING = 'DRIVING'
OBSTACLE = 'OBSTACLE'
U_TURN = 'U_TURN'
LOST = 'LOST'


class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # Safety
        self.declare_parameter('dry_run', True)

        # PD controller
        self.declare_parameter('kp', 1.1)
        self.declare_parameter('kd', 0.35)
        self.declare_parameter('max_angular', 1.4)

        # Linear speed schedule
        self.declare_parameter('max_linear', 0.16)
        self.declare_parameter('min_linear', 0.05)
        self.declare_parameter('linear_slowdown', 0.6)  # 0..1, how much |err| eats speed

        # Obstacle handling
        self.declare_parameter('obstacle_stop_m', 0.40)
        self.declare_parameter('obstacle_clear_m', 0.55)  # hysteresis
        self.declare_parameter('obstacle_cone_deg', 20.0)
        self.declare_parameter('obstacle_min_range', 0.05)

        # End-of-road / U-turn
        self.declare_parameter('u_turn_angular', 0.9)
        self.declare_parameter('u_turn_target_deg', 180.0)
        self.declare_parameter('u_turn_timeout_s', 8.0)
        self.declare_parameter('u_turn_cooldown_s', 6.0)

        # Lost / stale signal handling
        self.declare_parameter('error_timeout_s', 0.5)
        self.declare_parameter('lost_creep', 0.04)
        self.declare_parameter('lost_max_duration_s', 3.0)

        # Loop rate
        self.declare_parameter('control_hz', 20.0)

        self._error = 0.0
        self._error_stamp = 0.0
        self._last_error = 0.0
        self._last_error_time = None

        self._white_detected = False
        self._end_of_road = False
        self._scan = None
        self._yaw = None

        self._state = DRIVING
        self._u_turn_start_yaw = None
        self._u_turn_started_at = None
        self._u_turn_cooldown_until = 0.0
        self._lost_since = None

        self.create_subscription(Float32, '/lane/error', self._error_cb, 10)
        self.create_subscription(Bool, '/lane/end_of_road', self._eor_cb, 10)
        self.create_subscription(Bool, '/lane/white_detected', self._white_cb, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 20)

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_state = self.create_publisher(String, '/nav/state', 10)

        period = 1.0 / float(self.get_parameter('control_hz').value)
        self.create_timer(period, self._tick)

        dry = self.get_parameter('dry_run').value
        self.get_logger().info(f'Navigation node up [dry_run={dry}]')

    # ------------------------------------------------------------- inputs
    def _error_cb(self, msg: Float32):
        self._error = float(msg.data)
        self._error_stamp = self._now()

    def _eor_cb(self, msg: Bool):
        self._end_of_road = bool(msg.data)

    def _white_cb(self, msg: Bool):
        self._white_detected = bool(msg.data)

    def _scan_cb(self, msg: LaserScan):
        self._scan = msg

    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = math.atan2(siny, cosy)

    # ------------------------------------------------------------- helpers
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _error_is_fresh(self):
        return (self._now() - self._error_stamp) < self.get_parameter(
            'error_timeout_s'
        ).value

    def _obstacle_distance(self):
        if self._scan is None:
            return None
        cone = math.radians(self.get_parameter('obstacle_cone_deg').value)
        rmin = self.get_parameter('obstacle_min_range').value
        a = self._scan.angle_min
        inc = self._scan.angle_increment
        best = float('inf')
        for i, r in enumerate(self._scan.ranges):
            if not math.isfinite(r) or r < rmin:
                continue
            ang = a + i * inc
            while ang > math.pi:
                ang -= 2 * math.pi
            while ang < -math.pi:
                ang += 2 * math.pi
            if abs(ang) <= cone and r < best:
                best = r
        return None if best == float('inf') else best

    @staticmethod
    def _angle_diff(a, b):
        d = a - b
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d

    def _publish_cmd(self, lin, ang):
        if self.get_parameter('dry_run').value:
            lin = 0.0
            ang = 0.0
        cmd = Twist()
        cmd.linear.x = float(lin)
        cmd.angular.z = float(ang)
        self.pub_cmd.publish(cmd)

    def _publish_state(self):
        msg = String()
        msg.data = self._state
        self.pub_state.publish(msg)

    # ------------------------------------------------------------- control
    def _tick(self):
        # Obstacle override (highest priority except mid-U-turn).
        obs = self._obstacle_distance()
        stop_d = self.get_parameter('obstacle_stop_m').value
        clear_d = self.get_parameter('obstacle_clear_m').value

        if self._state != U_TURN:
            if obs is not None and obs < stop_d:
                if self._state != OBSTACLE:
                    self.get_logger().info(
                        f'Obstacle at {obs:.2f} m -> stopping'
                    )
                self._state = OBSTACLE
            elif self._state == OBSTACLE and (obs is None or obs > clear_d):
                self.get_logger().info('Path clear -> resuming')
                self._state = DRIVING

        # End-of-road -> U_TURN (with cooldown so we don't loop)
        if (
            self._state == DRIVING
            and self._end_of_road
            and self._now() > self._u_turn_cooldown_until
        ):
            self.get_logger().info('End of road -> U-turn')
            self._state = U_TURN
            self._u_turn_start_yaw = self._yaw
            self._u_turn_started_at = self._now()

        if self._state == OBSTACLE:
            self._publish_cmd(0.0, 0.0)
        elif self._state == U_TURN:
            self._run_u_turn()
        elif self._state == LOST:
            self._run_lost()
        else:  # DRIVING
            self._run_driving()

        self._publish_state()

    def _run_driving(self):
        if not self._error_is_fresh():
            self._state = LOST
            self._lost_since = self._now()
            self._publish_cmd(0.0, 0.0)
            return

        self._lost_since = None

        kp = self.get_parameter('kp').value
        kd = self.get_parameter('kd').value
        max_ang = self.get_parameter('max_angular').value

        now = self._now()
        if self._last_error_time is None:
            d_err = 0.0
        else:
            dt = max(1e-3, now - self._last_error_time)
            d_err = (self._error - self._last_error) / dt
        self._last_error = self._error
        self._last_error_time = now

        # err > 0 means white line is too far LEFT in image -> robot drifted
        # RIGHT toward the boundary -> steer LEFT (positive angular.z).
        ang = max(-max_ang, min(max_ang, kp * self._error + kd * d_err))

        max_lin = self.get_parameter('max_linear').value
        min_lin = self.get_parameter('min_linear').value
        slow = self.get_parameter('linear_slowdown').value
        lin = max(min_lin, max_lin * (1.0 - slow * min(1.0, abs(self._error))))

        self._publish_cmd(lin, ang)
        self.get_logger().info(
            f'[DRIVE] err={self._error:+.2f} d={d_err:+.2f} '
            f'-> lin={lin:.2f} ang={ang:+.2f}',
            throttle_duration_sec=0.5,
        )

    def _run_u_turn(self):
        target = math.radians(self.get_parameter('u_turn_target_deg').value)
        omega = self.get_parameter('u_turn_angular').value
        timeout = self.get_parameter('u_turn_timeout_s').value
        elapsed = self._now() - (self._u_turn_started_at or self._now())

        completed = False
        if self._yaw is not None and self._u_turn_start_yaw is not None:
            turned = abs(self._angle_diff(self._yaw, self._u_turn_start_yaw))
            if turned >= target - math.radians(8):
                completed = True
        else:
            if elapsed >= target / omega:
                completed = True

        if elapsed >= timeout:
            self.get_logger().warn('U-turn timeout, returning to DRIVING')
            completed = True

        if completed:
            self._state = DRIVING
            self._u_turn_cooldown_until = self._now() + self.get_parameter(
                'u_turn_cooldown_s'
            ).value
            self._u_turn_start_yaw = None
            self._u_turn_started_at = None
            self._publish_cmd(0.0, 0.0)
            self.get_logger().info('U-turn complete')
            return

        self._publish_cmd(0.0, omega)

    def _run_lost(self):
        if self._error_is_fresh():
            self._state = DRIVING
            self._lost_since = None
            return
        if self._lost_since is None:
            self._lost_since = self._now()
        if self._now() - self._lost_since > self.get_parameter(
            'lost_max_duration_s'
        ).value:
            self._publish_cmd(0.0, 0.0)
            return
        self._publish_cmd(self.get_parameter('lost_creep').value, 0.0)


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


if __name__ == '__main__':
    main()