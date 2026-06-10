#!/usr/bin/env python3
#
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Wonho Yun, Jeonggeun Lim, Ryan Shim, Gilbert

# Python imports
import math
import os
import threading
import numpy
import onnxruntime as ort
import numpy as np
import random
# ROS2 imports
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PoseStamped, PointStamped
from rclpy.time import Time
from std_msgs.msg import Float32

# TF2 imports
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs.tf2_geometry_msgs as tf2_geometry_msgs
import tf2_ros

ros_distro = os.environ.get('ROS_DISTRO', 'humble').lower()
if ros_distro == 'humble':
    from geometry_msgs.msg import Twist as CmdVelMsg
else:
    from geometry_msgs.msg import Twist as CmdVelMsg

from experiments.load_yaml import load_yaml_config
from velocity_smoother import KobukiVelocitySmoother

class Turtlebot3AbsoluteMove(Node):

    def __init__(self):
        super().__init__('turtlebot3_absolute_move')

        self.get_logger().info("RL Agent node started")

        print('----------------------------------------------')

        self.goal_position = Point()
        self.goal_position.x = None
        self.goal_position.y = None
        self.goal_heading = 0.0
        self.position = Point()
        self.heading = 0.0
        self.position_error = Point()
        self.heading_error = 0.0

        self.angular_speed = 0.15
        self.linear_speed = 0.5

        qos = QoSProfile(depth=10)

        self.cmd_vel_pub = self.create_publisher(CmdVelMsg, '/commands/velocity', qos)
        self.cmd_vel = CmdVelMsg()
        self.goal_pub = self.create_publisher(Path, "/waypoints_path", qos)
        self.context_pub = self.create_publisher(Float32, "/energy_context", qos)

        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.get_odom,
            10)

        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 1)
        self.create_subscription(PoseStamped, '/clicked_point', self.clicked_point_cb, 1)
        

        # TF2 buffer & listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # State
        self.goal_robot = None
        self.goal_world = None
        self.odom = None
        self.world_frame = None

        self.get_logger().info('Ready to receive goal inputs.')

        #self.get_key()

        timer_period = 0.05
        # Timeouts in nanoseconds for easy compare
        """self.odom_timeout_ns = int(1.0 * 1e9)  # 1 second
        self.goal_timeout_ns = int(goal_timeout_value * 1e9)"""
        self.timer = self.create_timer(timer_period, self.timer_callback)
        # Delay publishing until ROS graph is ready
        self.timer_goal = self.create_timer(0.07, self.publish_path)
            
        onnx_path = "/home/GTL/fsangare/Isaaclab_kingfisher_sail/logs/rl_games/turtlebot3_maneuver/2026-05-21_14-21-23_good_amplitude/nn/turtlebot3_maneuver.onnx"
        #"/home/GTL/fsangare/Isaaclab_kingfisher_sail/logs/rl_games/turtlebot3_maneuver/2026-05-16_15-58-15/nn/turtlebot3_maneuver.onnx"
        #"/home/GTL/fsangare/Isaaclab_kingfisher_sail/logs/rl_games/turtlebot3_maneuver/2026-05-15_15-37-08/nn/turtlebot3_maneuver.onnx"
        #"/home/GTL/fsangare/Isaaclab_kingfisher_sail/logs/rl_games/turtlebot3_maneuver/2026-05-13_16-07-15/nn/turtlebot3_maneuver.onnx" # Good ACORD
        #"/home/GTL/fsangare/Isaaclab_kingfisher_sail/logs/rl_games/turtlebot3_maneuver/2026-05-10_14-33-13_work/nn/turtlebot3_maneuver.onnx" --> Good waypoint tracking
        #"/home/GTL/fsangare/Isaaclab_kingfisher_sail/logs/rl_games/turtlebot3_maneuver/2026-05-09_18-24-52/nn/turtlebot3_maneuver.onnx"
        #"/home/GTL/fsangare/Isaaclab_kingfisher_sail/logs/rl_games/turtlebot3_maneuver/2026-05-06_00-04-30/nn/turtlebot3_maneuver.onnx" 
        #"logs/rl_games/turtlebot3_maneuver/2026-04-27_19-04-15/nn/turtlebot3_maneuver.onnx" tilting checkpoint
        # Parameters (declare then read)
        self.declare_parameter('onnx_path', onnx_path)
        self.declare_parameter('dist_threshold', 0.3)
        self.declare_parameter('control_freq', 20.0)
        self.declare_parameter('goal_timeout', 5.0)
        

        model_path = self.get_parameter('onnx_path').value
        self.dist_threshold = self.get_parameter('dist_threshold').value
        self.control_freq = self.get_parameter('control_freq').value
        goal_timeout_value = self.get_parameter('goal_timeout').value

        config_file = 'config/grid.yaml'
        self.points = load_yaml_config(config_file)
        self.next_point_index = 0
        self.next_waypoint = None
        self.next_context_idx = 0

        self.velocity_smoother = KobukiVelocitySmoother(speed_lim_v=0.5, speed_lim_w=1.0, accel_lim_v=0.3, accel_lim_w=1.0)

        # Build model path. If you store models inside a package, adjust this as needed.
        # Here we attempt to find model path relative to this file if package not provided.
        # You might prefer to pass absolute path to onnx_path param.
        if os.path.isabs(model_path) and os.path.exists(model_path):
            self.rl_onnx_path = model_path
        else:
            # try to locate inside this package under models/
            package_share = os.path.dirname(os.path.realpath(__file__))  # fallback
            self.rl_onnx_path = os.path.join(package_share, 'models', model_path)

        self.ort_model = ort.InferenceSession(self.rl_onnx_path, providers=['CPUExecutionProvider'])


        # Locks
        self.odom_lock = threading.Lock()
        self.goal_lock = threading.Lock()

        self.model_name = os.path.basename(model_path).replace('.', '_')

        self.get_logger().info(f"Loading ONNX model from: {self.rl_onnx_path}")
        self.device = 'cpu'

        # Observation buffers
        self.actions = np.zeros((1,2), dtype=np.float32)
        self.lin_vel = np.zeros((1,2), dtype=np.float32)
        self.ang_vel = np.zeros((1,1), dtype=np.float32)
        self.goal_cos_sin = np.zeros((1,2), dtype=np.float32)
        self.head_cos_sin = np.zeros((1,2), dtype=np.float32)
        self.goal_distance = np.zeros((1,1), dtype=np.float32)
        self.initial_goal_distance = 0.0  # initial distance to goal, used for normalization
        self.energy_context = np.zeros((1,1), dtype=np.float32)  # energy context 
        self.energy = np.zeros((1,1), dtype=np.float32)  # energy level
        self.context_distance_combined = np.zeros((1,1), dtype=np.float32)  # combining context with distance
        self.evaluation_context_set = np.arange(0., 1.0, 0.1)  # example context values from 0 to 1

        self.max_angular_speed = 1.0
        self.max_linear_speed = self.velocity_smoother.speed_lim_v


    def publish_path(self):

        if self.world_frame is None:
            return
        else:
            self.timer_goal.cancel()

        path = Path()
        path.header.frame_id = self.world_frame
        path.header.stamp = self.get_clock().now().to_msg()

        for goal in self.points:
            
            x, y = goal[0], goal[1]

            pose = PoseStamped()
            pose.header.frame_id = self.world_frame
            pose.header.stamp = self.get_clock().now().to_msg()

            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0

            path.poses.append(pose)

        self.goal_pub.publish(path)
        self.get_logger().info(f"Published full waypoint path: {self.points}")


    # -------------------------
    # Observations & Actions
    # -------------------------
    def get_observations(self):
        
        with self.odom_lock:

            if self.odom is None:
                self.get_logger().warning('No odom available yet')
                return None
            
            if self.next_point_index==0:
                self.update_goal_to_nextwpt()

            if self.goal_position.x is None or self.goal_position.y is None or self.goal_robot is None:
                #self.get_logger().warning('No goal position available yet')
                return None
            
            transform_world2base = self.tf_buffer.lookup_transform('base_link',
                                                            self.goal_world.header.frame_id,
                                                            rclpy.time.Time())  # latest
            self.goal_robot = tf2_geometry_msgs.do_transform_pose_stamped(self.goal_world, transform_world2base)

            #print(f"goal_robot: {self.goal_robot.pose.position.x:.2f}, {self.goal_robot.pose.position.y:.2f}")
            self.goal_position.x = self.goal_world.pose.position.x
            self.goal_position.y = self.goal_world.pose.position.y


            robot_vel = [self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y]
            robot_ang_vel = [self.odom.twist.twist.angular.z]

            # goal_robot expected to be set by goal_reached() transformation earlier
            
            goal_pos = [self.goal_robot.pose.position.x, self.goal_robot.pose.position.y]

            goal_bearing = math.atan2(goal_pos[1], goal_pos[0])
            goal_distance = min(float(math.hypot(goal_pos[0], goal_pos[1])), 9.0)
            goal_cos_sin = [math.cos(goal_bearing), math.sin(goal_bearing)]
            head_cos_sin = [math.cos(self.heading), math.sin(self.heading)]

            # Fill buffers
            self.lin_vel[0] = robot_vel
            self.ang_vel[0] = robot_ang_vel
            self.goal_cos_sin[0] = goal_cos_sin
            self.goal_distance[0, 0] = goal_distance
            self.head_cos_sin[0] = head_cos_sin
            self.energy[0, 0] = np.square(self.actions[0, 0])

            if self.initial_goal_distance == 0.0:
                self.get_logger().info(f"Setting initial goal distance: {goal_distance:.2f}")
                return None  # skip first observation to allow initial distance to be set

            self.goal_distance[0, 0] = self.goal_distance[0, 0] / self.initial_goal_distance  # normalize distance to [0, 1]

            self.context_distance_combined[0, 0] = self.goal_distance[0, 0] * self.energy_context[0, 0]  # example of combining context with distance
            base_obs = np.concatenate(
                [
                    #self.actions,      # 2
                    (self.lin_vel/self.max_linear_speed).clip(-1.0, 1.0),      # 2
                    self.ang_vel,      # 1
                    self.goal_cos_sin, # 2
                    self.goal_distance,# 1
                    self.context_distance_combined, # 1
                    #self.energy,         # 1
                    self.energy_context,   # 1

                ],
                axis=1,
                dtype=np.float32
            )

            """current_config = np.array([[self.position.x, self.position.y]], dtype=np.float32)
            target_config = np.array([[self.goal_position.x, self.goal_position.y]], dtype=np.float32)"""

            observations = base_obs #np.concatenate([base_obs, current_config, target_config], axis=1)

            obs = {
                "obs": observations, 
                }

            print(f"Observations: {observations}")
            return obs

    def get_action(self, obs: dict):
        # ONNX runtime expects inputs keyed by input names or matching dict the model expects.
        # Original code passed {"obs": observations} so we do the same.
        try:
            output = self.ort_model.run(None, obs)
            action = np.clip(output[0].squeeze(0), -1.0, 1.0)
            self.actions[0] = action
        except Exception as e:
            self.get_logger().error(f"ONNX runtime failed: {e}")
            # fallback to zeros
            self.actions[0] = [0.0, 0.0]  # left, right, sail angle

        return self.actions

    
    def goal_cb(self, msg: PoseStamped):
        # store the goal transformed to the world frame
        with self.goal_lock:
            if self.world_frame is None:
                # we don't yet know the odom/world_frame
                self.get_logger().warning('World frame unknown - cannot transform goal yet')
                return
            
            if msg is None:
                self.get_logger().warning('Received empty goal message')
                self.goal_world = None
                return
            try:
                # convert msg.header.stamp (builtin_interfaces/Time) to rclpy Time for lookup
                
                print(f"[DEBUGGING] msg.header.frame_id: {msg.header.frame_id}")
                transform = self.tf_buffer.lookup_transform(self.world_frame,
                                                            msg.header.frame_id,
                                                            rclpy.time.Time())
                self.goal_world = tf2_geometry_msgs.do_transform_pose_stamped(msg, transform)

                transform_world2base = self.tf_buffer.lookup_transform('base_link',
                                                            self.goal_world.header.frame_id,
                                                            rclpy.time.Time())  # latest
                self.goal_robot = tf2_geometry_msgs.do_transform_pose_stamped(self.goal_world, transform_world2base)

                #print(f"goal_robot: {self.goal_robot.pose.position.x:.2f}, {self.goal_robot.pose.position.y:.2f}")
                self.goal_position.x = self.goal_world.pose.position.x
                self.goal_position.y = self.goal_world.pose.position.y
                print("frame_id of the goal message",msg.header.frame_id)
                self.initial_goal_distance = math.sqrt(pow(self.goal_robot.pose.position.x , 2) + pow(self.goal_robot.pose.position.y, 2))

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warning(f"Failed to find transform for goal: {e}")
                return
            
    def update_goal_to_nextwpt(self):
        
        self.next_point_index = (self.next_point_index + 1) % len(self.points)
        self.next_context_idx = (self.next_context_idx + 1) % len(self.evaluation_context_set)


        waypoint = PoseStamped()
        waypoint.header.frame_id = self.world_frame
        waypoint.pose.position.x = self.points[self.next_point_index][0]
        waypoint.pose.position.y = self.points[self.next_point_index][1]
        
        print(f"points: {self.points[self.next_point_index][0]}, {self.points[self.next_point_index][1]}")
        print(f"Updating goal to next waypoint: x={waypoint.pose.position.x:.2f}, y={waypoint.pose.position.y:.2f}, context={self.evaluation_context_set[self.next_context_idx]:.2f}")
        try:
            # convert msg.header.stamp (builtin_interfaces/Time) to rclpy Time for lookup
             
            self.goal_world = waypoint

            transform_world2base = self.tf_buffer.lookup_transform('base_link',
                                                        self.goal_world.header.frame_id,
                                                        rclpy.time.Time())  # latest
            self.goal_robot = tf2_geometry_msgs.do_transform_pose_stamped(self.goal_world, transform_world2base)
            
            self.goal_position.x = self.goal_world.pose.position.x
            self.goal_position.y = self.goal_world.pose.position.y
            
            self.initial_goal_distance = math.sqrt(pow(self.goal_robot.pose.position.x , 2) + pow(self.goal_robot.pose.position.y, 2))

            self.energy_context[0, 0] = self.evaluation_context_set[self.next_context_idx]  # pick a random context for evaluation

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warning(f"Failed to find transform for goal: {e}")

            
    def clicked_point_cb(self, msg: PointStamped):
        # store the clicked point transformed to the world frame as the new goal
        with self.goal_lock:
            if self.world_frame is None:
                self.get_logger().warning('World frame unknown - cannot transform clicked point yet')
                return
            
            if msg is None:
                self.get_logger().warning('Received empty clicked point message')
                return
            try:
                stamp_time = Time.from_msg(msg.header.stamp)
                transform = self.tf_buffer.lookup_transform(self.world_frame, msg.header.frame_id, stamp_time)
                clicked_point_world = tf2_geometry_msgs.do_transform_pose_stamped(msg, transform)
                self.goal_position.x = clicked_point_world.pose.position.x
                self.goal_position.y = clicked_point_world.pose.position.y

                transform_world2base = self.tf_buffer.lookup_transform('base_link',
                                                            clicked_point_world.header.frame_id,
                                                            rclpy.time.Time())  # latest
                self.goal_robot = tf2_geometry_msgs.do_transform_pose_stamped(clicked_point_world, transform_world2base)
                
                self.initial_goal_distance = math.sqrt(pow(self.goal_robot.pose.position.x , 2) + pow(self.goal_robot.pose.position.y, 2))
                
                self.energy_context[0, 0] = random.choice(self.evaluation_context_set)  # pick a random context for evaluation

                self.get_logger().info(f"Clicked point received, new goal set to x: {self.goal_position.x:.2f}, y: {self.goal_position.y:.2f}")
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warning(f"Failed to find transform for clicked point: {e}")
                return
            

    def timer_callback(self):
        
        self.context_pub.publish(Float32(data=float(self.energy_context[0, 0])))
        observation = self.get_observations()

        if self.goal_robot is None or self.goal_position.x is None or self.goal_position.y is None:
            # No goal set yet, just wait
            #self.get_logger().info('No goal position set yet, waiting for input...')
            return
        self.position_error.x = self.goal_robot.pose.position.x 
        self.position_error.y = self.goal_robot.pose.position.y

        distance = math.sqrt(pow(self.position_error.x, 2) + pow(self.position_error.y, 2))

        print(f"observation: {observation}")
        #print(f"position x {self.goal_position.x:.2f}, y {self.goal_position.y:.2f}, distance to goal: {distance:.2f}")
        if observation is None:
            return
        policy_actions = self.get_action(observation)
        smooth_actions = self.velocity_smoother.step(policy_actions)
        actions = smooth_actions.copy()
        v = abs(actions[0, 0])  # forward/backward
        w = actions[0, 1]  # left/right turn

        print(f"Action from ONNX model: v={actions[0, 0]:.2f} policy: {policy_actions[0, 0]:.2f}, w={actions[0, 1]:.2f} policy: {policy_actions[0, 1]:.2f}")

        self.cmd_vel.angular.z = float(w)
        self.cmd_vel.linear.x = float(v)
        
        self.get_logger().info(
            f'Moving to x: {self.goal_position.x:.2f}, y: {self.goal_position.y:.2f} '
            f'(current: {self.position.x:.2f}, {self.position.y:.2f})'
            f'index: {self.next_point_index-1}'
            f'context: {self.energy_context[0, 0]:.2f}'
        )

        #self.cmd_vel_pub.publish(self.cmd_vel)
        #print(f"3 Distance to goal: {distance:.2f}")

            #self.get_key()
        
        self.cmd_vel_pub.publish(self.cmd_vel)

        if distance <= 0.1:
            self.update_goal_to_nextwpt()

    def get_odom(self, msg:Odometry):
        self.position = msg.pose.pose.position
        self.world_frame = msg.header.frame_id
        self.odom = msg
        
        _, _, self.heading = self.transfrom_from_quaternion_to_eular(msg.pose.pose.orientation)

    def get_key(self):
        self.goal_position.x = float(input('goal x (absolute): '))
        self.goal_position.y = float(input('goal y (absolute): '))
        self.goal_heading = float(input('goal heading (absolute, degrees): '))

        self.initial_goal_distance = math.hypot(self.goal_position.x - self.position.x,
                                               self.goal_position.y - self.position.y)

        self.goal_heading = math.radians(self.goal_heading)
        if self.goal_heading > math.pi:
            self.goal_heading -= 2 * math.pi
        elif self.goal_heading < -math.pi:
            self.goal_heading += 2 * math.pi

        self.get_logger().info(
            f'New goal: x: {self.goal_position.x:.2f}, y: {self.goal_position.y:.2f}, '
            f'heading: {math.degrees(self.goal_heading):.2f}°'
        )

    def transfrom_from_quaternion_to_eular(self, q):
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init()
    node = Turtlebot3AbsoluteMove()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
