"""
navigation_node.py
==================
Closed-loop lane-following controller for EduBot.

Inputs
------
/lane/error          std_msgs/Float32   normalised lateral error from
                                        lane_detection_node ([-1.5, 1.5])
/lane/heading        std_msgs/Float32   yellow-slope feed-forward heading
                                        in [-1, 1]; +ve = curve right ahead
/lane/confidence     std_msgs/Float32   detection confidence in [0,1]
/lane/end_of_road    std_msgs/Bool      orange perpendicular line in view
/lane/white_detected std_msgs/Bool      sanity flag - we have a right boundary
/scan                sensor_msgs/LaserScan  forward LiDAR for obstacle stop
/odom                nav_msgs/Odometry  used to close the loop on rotations

Outputs
-------
/cmd_vel             geometry_msgs/Twist
/nav/state           std_msgs/String        current FSM state for debugging

State machine
-------------
DRIVING       - PD on lateral error; angular and linear limits scale with
                detection confidence so a brief loss does not produce a
                max-rate spin
OBSTACLE      - LiDAR cone is occupied; hold zero velocity until clear
U_TURN        - end-of-road triggered; rotate ~180 then -> RECOVERY
INTERSECTION  - confidence collapsed while the robot was going straight ->
                90 deg right turn (assignment rule), then -> RECOVERY
RECOVERY      - slow forward creep with no PD until lane is reacquired or
                a short timeout elapses; protects against snapping to
                stale errors right after a turn
SPIN_HOLD     - the PD asked for sustained near-saturation angular while
                confidence was low; freeze velocity until confidence
                recovers (hard backstop against the saturated-spin loop)
LOST          - no error signal at all in a while; creep forward slowly so
                we don't deadlock if a frame is dropped, but cap the
                duration

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
INTERSECTION = 'INTERSECTION'
RECOVERY = 'RECOVERY'
SPIN_HOLD = 'SPIN_HOLD'
LOST = 'LOST'


class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # Safety
        self.declare_parameter('dry_run', True)

        # PD controller. Kp lowered + Kd raised vs. original to reduce the
        # snap-back oscillation seen at the track. kff is the feed-forward
        # gain on /lane/heading - it lets the robot start turning into a
        # curve based on yellow-line slope BEFORE the centroid has moved.
        # kff defaults to 0 because at the track a small perspective bias
        # in the slope fit was producing a sustained right-drift on
        # straightaways. Bring it up live with `ros2 param set kff 0.2`
        # while watching the curve behavior.
        # kff_sharp_amp adds a non-linear amplification so sharp inside
        # corners (heading near +/- 1.0) get a much stronger FF kick than
        # gentle outside curves: effective_term = kff * h * (1 + amp*|h|).
        # heading_slowdown drops linear speed in proportion to |heading|
        # so the robot doesn't carry too much momentum into a 90 deg corner.
        self.declare_parameter('kp', 0.7)
        self.declare_parameter('kd', 0.55)
        self.declare_parameter('kff', 0.1) 
        self.declare_parameter('kff_sharp_amp', 1.0) #for inside lane tight turns.
        self.declare_parameter('heading_slowdown', 0.0) #try 0 if stuff gets messed up.
        self.declare_parameter('max_angular', 1.4)

        # Linear speed schedule. Lower max + steeper slowdown to keep the
        # robot from carrying so much momentum into a curve that PD can't
        # correct in time.
        self.declare_parameter('max_linear', 0.10)
        self.declare_parameter('min_linear', 0.05)
        self.declare_parameter('linear_slowdown', 0.85)

        # Obstacle handling
        self.declare_parameter('obstacle_stop_m', 0.40)
        self.declare_parameter('obstacle_clear_m', 0.55)  # hysteresis
        self.declare_parameter('obstacle_cone_deg', 20.0)
        self.declare_parameter('obstacle_min_range', 0.05)

        # End-of-road / U-turn. Tighter angular + arrival window than before:
        # the old defaults overshot enough that we'd come out of the turn off
        # the lane.
        self.declare_parameter('u_turn_angular', 0.6)
        self.declare_parameter('u_turn_target_deg', 180.0)
        self.declare_parameter('u_turn_arrival_window_deg', 4.0)
        self.declare_parameter('u_turn_timeout_s', 8.0)
        self.declare_parameter('u_turn_cooldown_s', 6.0)

        # Intersection right-turn (assignment rule: turn right by default).
        # Negative target means clockwise; magnitude is 90 deg. Detection
        # is intentionally conservative: a brief lane gap on a straightaway
        # was firing the trigger and causing a phantom right turn, so we
        # require both a longer sustained loss AND deeper confidence drop.
        self.declare_parameter('intersection_target_deg', -90.0)
        self.declare_parameter('intersection_angular', 0.6)
        self.declare_parameter('intersection_timeout_s', 6.0)
        self.declare_parameter('intersection_cooldown_s', 8.0)
        self.declare_parameter('intersection_detect_secs', 1.5)
        self.declare_parameter('intersection_low_conf_threshold', 0.15)
        self.declare_parameter('intersection_steady_err_threshold', 0.25)

        # Confidence-aware control. conf_scale is clipped to
        # [low_conf_min_scale, 1.0] before scaling angular and linear limits.
        self.declare_parameter('low_conf_min_scale', 0.2)
        self.declare_parameter('steady_err_alpha', 0.2)

        # Spin-out guard. If the PD wants near-saturation angular for this
        # long while confidence is bad, freeze until detection recovers.
        self.declare_parameter('spin_out_secs', 1.2)
        self.declare_parameter('spin_out_low_conf', 0.4)
        self.declare_parameter('spin_out_recover_conf', 0.5)

        # Recovery (after U_TURN or INTERSECTION).
        self.declare_parameter('recovery_creep', 0.05)
        self.declare_parameter('recovery_max_secs', 4.0)
        self.declare_parameter('recovery_required_frames', 8)
        self.declare_parameter('recovery_conf_threshold', 0.6)

        # Lost / stale signal handling
        self.declare_parameter('error_timeout_s', 1.0)
        self.declare_parameter('lost_creep', 0.04)
        self.declare_parameter('lost_max_duration_s', 3.0)

        # Loop rate
        self.declare_parameter('control_hz', 20.0)

        self._error = 0.0
        self._error_stamp = 0.0
        self._last_error = 0.0
        self._last_error_time = None

        # Confidence defaults to 1.0 so a missing publisher (e.g. running
        # navigation against an old detection node) does not freeze the bot.
        self._confidence = 1.0
        self._heading = 0.0
        self._steady_err_ema = 0.0

        self._white_detected = False
        self._end_of_road = False
        self._scan = None
        self._yaw = None

        self._state = DRIVING
        self._u_turn_start_yaw = None
        self._u_turn_started_at = None
        self._u_turn_cooldown_until = 0.0
        self._lost_since = None

        # Intersection turn closes the loop on signed cumulative yaw so a
        # right-90 doesn't get confused at the +/- pi wrap.
        self._intersection_started_at = None
        self._intersection_last_yaw = None
        self._intersection_accum_yaw = 0.0
        self._intersection_cooldown_until = 0.0
        self._low_conf_since = None

        self._recovery_started_at = None
        self._recovery_white_streak = 0

        self._high_ang_dir = 0
        self._high_ang_since = None

        self.create_subscription(Float32, '/lane/error', self._error_cb, 10)
        self.create_subscription(Float32, '/lane/heading', self._heading_cb, 10)
        self.create_subscription(Float32, '/lane/confidence', self._conf_cb, 10)
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

    def _heading_cb(self, msg: Float32):
        h = float(msg.data)
        if h < -1.0:
            h = -1.0
        elif h > 1.0:
            h = 1.0
        self._heading = h

    def _conf_cb(self, msg: Float32):
        c = float(msg.data)
        if c < 0.0:
            c = 0.0
        elif c > 1.0:
            c = 1.0
        self._confidence = c

    def _eor_cb(self, msg: Bool):
        self._end_of_road = bool(msg.data)

    def _white_cb(self, msg: Bool):
        self._white_detected = bool(msg.data)

    def _scan_cb(self, msg: LaserScan):
        self._scan = msg

    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        # yaw from quaternion (z-axis rotation in the world frame)
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
            # normalise to [-pi, pi] so wrap-around doesn't lie about which way
            # the beam points.
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
        now = self._now()

        # 1. Obstacle override. Don't preempt an in-progress rotation; those
        # are time/yaw-bounded and aborting them halfway leaves the robot
        # mis-aligned.
        obs = self._obstacle_distance()
        stop_d = self.get_parameter('obstacle_stop_m').value
        clear_d = self.get_parameter('obstacle_clear_m').value

        rotating = self._state in (U_TURN, INTERSECTION)
        if not rotating:
            if obs is not None and obs < stop_d:
                if self._state != OBSTACLE:
                    self.get_logger().info(
                        f'Obstacle at {obs:.2f} m -> stopping'
                    )
                self._state = OBSTACLE
            elif self._state == OBSTACLE and (obs is None or obs > clear_d):
                self.get_logger().info('Path clear -> resuming')
                self._state = DRIVING

        # 2. End-of-road -> U_TURN (with cooldown so we don't loop)
        if (
            self._state == DRIVING
            and self._end_of_road
            and now > self._u_turn_cooldown_until
        ):
            self.get_logger().info('End of road -> U-turn')
            self._state = U_TURN
            self._u_turn_start_yaw = self._yaw
            self._u_turn_started_at = now

        # 3. Intersection detection. Only fire when we were going straight
        # (low steady-state error) and confidence has been near zero for
        # long enough that this is a structural lane loss, not a single
        # bad frame on a curve.
        if self._state == DRIVING:
            low_conf_thr = self.get_parameter(
                'intersection_low_conf_threshold'
            ).value
            if self._confidence < low_conf_thr:
                if self._low_conf_since is None:
                    self._low_conf_since = now
                detect_secs = self.get_parameter(
                    'intersection_detect_secs'
                ).value
                steady_thr = self.get_parameter(
                    'intersection_steady_err_threshold'
                ).value
                if (
                    now - self._low_conf_since >= detect_secs
                    and self._steady_err_ema < steady_thr
                    and now > self._intersection_cooldown_until
                    and now > self._u_turn_cooldown_until
                ):
                    self.get_logger().info(
                        f'Intersection (conf<{low_conf_thr:.2f} for '
                        f'{now - self._low_conf_since:.2f}s, steady_err='
                        f'{self._steady_err_ema:.2f}) -> right turn'
                    )
                    self._enter_intersection()
            else:
                self._low_conf_since = None

        # 4. Run the active state
        if self._state == OBSTACLE:
            self._publish_cmd(0.0, 0.0)
        elif self._state == U_TURN:
            self._run_u_turn()
        elif self._state == INTERSECTION:
            self._run_intersection()
        elif self._state == RECOVERY:
            self._run_recovery()
        elif self._state == SPIN_HOLD:
            self._run_spin_hold()
        elif self._state == LOST:
            self._run_lost()
        else:  # DRIVING
            self._run_driving()

        self._publish_state()

    def _enter_intersection(self):
        self._state = INTERSECTION
        self._intersection_started_at = self._now()
        self._intersection_last_yaw = self._yaw
        self._intersection_accum_yaw = 0.0
        # Reset transient PD/spin-out tracking so RECOVERY starts clean.
        self._high_ang_dir = 0
        self._high_ang_since = None
        self._low_conf_since = None

    def _run_driving(self):
        # No fresh error signal -> demote to LOST (briefly) instead of
        # publishing a stale command.
        if not self._error_is_fresh():
            self._state = LOST
            self._lost_since = self._now()
            self._publish_cmd(0.0, 0.0)
            return

        self._lost_since = None

        kp = self.get_parameter('kp').value
        kd = self.get_parameter('kd').value
        kff = self.get_parameter('kff').value
        sharp_amp = self.get_parameter('kff_sharp_amp').value
        max_ang = self.get_parameter('max_angular').value

        now = self._now()
        if self._last_error_time is None:
            d_err = 0.0
        else:
            dt = max(1e-3, now - self._last_error_time)
            d_err = (self._error - self._last_error) / dt
        self._last_error = self._error
        self._last_error_time = now

        # Sign convention from lane_detection_node:
        # err > 0 means white line is too far LEFT in image -> robot drifted
        # RIGHT toward the boundary -> steer LEFT (positive angular.z in REP-103).
        # heading > 0 means the lane curves RIGHT ahead -> steer RIGHT
        # (negative angular.z), so the FF term is subtracted. The amp factor
        # boosts sharp corners only - on a gentle curve (|h| ~ 0.2) it adds
        # ~40% with amp=2.0; on a sharp corner (|h| ~ 0.9) it adds ~180%.
        ff_term = kff * self._heading * (1.0 + sharp_amp * abs(self._heading))
        raw_ang = kp * self._error + kd * d_err - ff_term

        # Confidence-aware limits: low confidence -> slow gentle motion, not
        # max-rate rotation. This is the core fix for the back-and-forth
        # loop.
        min_scale = self.get_parameter('low_conf_min_scale').value
        conf_scale = max(min_scale, min(1.0, self._confidence))
        ang = max(-max_ang * conf_scale, min(max_ang * conf_scale, raw_ang))

        max_lin = self.get_parameter('max_linear').value
        min_lin = self.get_parameter('min_linear').value
        slow = self.get_parameter('linear_slowdown').value
        head_slow = self.get_parameter('heading_slowdown').value
        head_factor = max(0.2, 1.0 - head_slow * min(1.0, abs(self._heading)))
        lin = max(
            min_lin,
            max_lin
            * (1.0 - slow * min(1.0, abs(self._error)))
            * head_factor
            * conf_scale,
        )

        # Track steady-state error only during decent confidence so the EMA
        # reflects real driving conditions rather than the moments right
        # before a lane loss.
        if self._confidence >= 0.4:
            alpha = self.get_parameter('steady_err_alpha').value
            self._steady_err_ema = (
                (1.0 - alpha) * self._steady_err_ema + alpha * abs(self._error)
            )

        # Spin-out guard: the controller WANTS sustained near-saturation
        # angular while confidence is low -> freeze. Use the unclipped raw_ang
        # so this doesn't depend on the confidence-scaled limit clipping its
        # own indicator. Pre-existing as a backstop in case some HSV edge
        # case lets a stale error survive.
        spin_thresh = 0.8 * max_ang
        spin_secs = self.get_parameter('spin_out_secs').value
        spin_low_conf = self.get_parameter('spin_out_low_conf').value
        sign_ang = 0
        if raw_ang > spin_thresh:
            sign_ang = 1
        elif raw_ang < -spin_thresh:
            sign_ang = -1
        if sign_ang != 0:
            if self._high_ang_dir != sign_ang:
                self._high_ang_dir = sign_ang
                self._high_ang_since = now
            elif (
                self._high_ang_since is not None
                and now - self._high_ang_since > spin_secs
                and self._confidence < spin_low_conf
            ):
                self.get_logger().warn(
                    f'Spin-out guard: raw_ang={raw_ang:+.2f} sustained '
                    f'while conf={self._confidence:.2f} -> SPIN_HOLD'
                )
                self._state = SPIN_HOLD
                self._publish_cmd(0.0, 0.0)
                return
        else:
            self._high_ang_dir = 0
            self._high_ang_since = None

        self._publish_cmd(lin, ang)
        self.get_logger().info(
            f'[DRIVE] err={self._error:+.2f} d={d_err:+.2f} '
            f'hdg={self._heading:+.2f} conf={self._confidence:.2f} '
            f'-> lin={lin:.2f} ang={ang:+.2f}',
            throttle_duration_sec=0.5,
        )

    def _run_u_turn(self):
        target = math.radians(self.get_parameter('u_turn_target_deg').value)
        omega = self.get_parameter('u_turn_angular').value
        arrival = math.radians(
            self.get_parameter('u_turn_arrival_window_deg').value
        )
        timeout = self.get_parameter('u_turn_timeout_s').value
        now = self._now()
        elapsed = now - (self._u_turn_started_at or now)

        # Prefer odometry-based completion. Fall back to time-based so the
        # state machine still advances if /odom is absent at demo time. The
        # 180 case is symmetric so abs(angle_diff) is fine here.
        completed = False
        if self._yaw is not None and self._u_turn_start_yaw is not None:
            turned = abs(self._angle_diff(self._yaw, self._u_turn_start_yaw))
            if turned >= target - arrival:
                completed = True
        else:
            if elapsed >= target / omega:
                completed = True

        if elapsed >= timeout:
            self.get_logger().warn('U-turn timeout, entering RECOVERY')
            completed = True

        if completed:
            # Hand off to RECOVERY (slow forward creep until the lane is
            # reacquired) instead of snapping back into DRIVING with the
            # stale error from before the turn.
            self._u_turn_cooldown_until = now + self.get_parameter(
                'u_turn_cooldown_s'
            ).value
            self._u_turn_start_yaw = None
            self._u_turn_started_at = None
            self._enter_recovery('U-turn complete')
            return

        self._publish_cmd(0.0, omega)

    def _run_intersection(self):
        # 90 deg right turn (signed cumulative yaw so the +/- pi wrap doesn't
        # confuse arrival detection). target_deg is signed: -90 = right.
        target = math.radians(self.get_parameter('intersection_target_deg').value)
        omega_mag = self.get_parameter('intersection_angular').value
        direction = -1.0 if target < 0 else 1.0
        omega = omega_mag * direction
        arrival = math.radians(
            self.get_parameter('u_turn_arrival_window_deg').value
        )
        timeout = self.get_parameter('intersection_timeout_s').value
        now = self._now()
        elapsed = now - (self._intersection_started_at or now)

        if self._yaw is not None and self._intersection_last_yaw is not None:
            delta = self._angle_diff(self._yaw, self._intersection_last_yaw)
            self._intersection_accum_yaw += delta
            self._intersection_last_yaw = self._yaw
        elif self._yaw is not None:
            self._intersection_last_yaw = self._yaw

        completed = False
        if self._yaw is not None:
            if direction < 0:
                if self._intersection_accum_yaw <= target + arrival:
                    completed = True
            else:
                if self._intersection_accum_yaw >= target - arrival:
                    completed = True
        else:
            # Time fallback when odometry is unavailable.
            if elapsed >= abs(target / omega):
                completed = True

        if elapsed >= timeout:
            self.get_logger().warn('Intersection timeout, entering RECOVERY')
            completed = True

        if completed:
            self._intersection_cooldown_until = now + self.get_parameter(
                'intersection_cooldown_s'
            ).value
            self._intersection_started_at = None
            self._intersection_last_yaw = None
            self._intersection_accum_yaw = 0.0
            self._enter_recovery('Intersection turn complete')
            return

        self._publish_cmd(0.0, omega)

    def _enter_recovery(self, why):
        self._state = RECOVERY
        self._recovery_started_at = self._now()
        self._recovery_white_streak = 0
        # Reset PD derivative state so the first DRIVING tick after recovery
        # doesn't see a giant dt and produce a spike.
        self._last_error_time = None
        self._high_ang_dir = 0
        self._high_ang_since = None
        self._publish_cmd(0.0, 0.0)
        self.get_logger().info(f'{why} -> RECOVERY')

    def _run_recovery(self):
        timeout = self.get_parameter('recovery_max_secs').value
        threshold = self.get_parameter('recovery_conf_threshold').value
        required = int(self.get_parameter('recovery_required_frames').value)
        creep = self.get_parameter('recovery_creep').value

        now = self._now()
        elapsed = now - (self._recovery_started_at or now)

        if self._white_detected and self._confidence >= threshold:
            self._recovery_white_streak += 1
        else:
            self._recovery_white_streak = 0

        if self._recovery_white_streak >= required or elapsed >= timeout:
            reason = (
                'lane reacquired'
                if self._recovery_white_streak >= required
                else 'timeout'
            )
            self.get_logger().info(f'RECOVERY -> DRIVING ({reason})')
            self._state = DRIVING
            self._recovery_started_at = None
            self._recovery_white_streak = 0
            return

        self._publish_cmd(creep, 0.0)

    def _run_spin_hold(self):
        # Hard backstop against the saturated-spin failure mode. Stay parked
        # until the detector reports we have a real lane to drive on again.
        if self._confidence >= self.get_parameter('spin_out_recover_conf').value:
            self.get_logger().info('SPIN_HOLD -> DRIVING (confidence recovered)')
            self._state = DRIVING
            self._high_ang_dir = 0
            self._high_ang_since = None
            self._last_error_time = None
            return
        self._publish_cmd(0.0, 0.0)

    def _run_lost(self):
        # Try to recover: creep forward briefly. If the lane never reappears
        # we stop hard so we don't sail off the track.
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

# Somewhat working version
if __name__ == '__main__':
    main()