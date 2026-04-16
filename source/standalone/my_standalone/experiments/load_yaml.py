import yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from rclpy.qos import QoSProfile

def load_yaml_config(config_file):
    """Load the YAML configuration file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['experiments'][0]['points']

class PointPublisher(Node):
    def __init__(self, points):
        super().__init__('point_publisher')
        self.points = points
        # Publisher
        qos = QoSProfile(depth=10)
        self.pub = self.create_publisher(PoseArray, '/goals', qos)

        self.declare_parameter('pattern', 'grid')
        pattern = self.get_parameter('pattern').get_parameter_value().string_value
        self.get_logger().info(f'Publishing points with pattern: {pattern}')

        self.config_file = f"config/{pattern}.yaml"

        # Load YAML points
        self.points = load_yaml_config(self.config_file)
        # Timer to publish once when ready
        self.timer = self.create_timer(1.0, self.try_publish_points)

    def try_publish_points(self):

        self.publish_points()
        self.timer.cancel()  # Publish only once

    def publish_points(self):
        pose_array = PoseArray()
        #pose_array.header.frame_id = ""
        pose_array.header.stamp = self.get_clock().now().to_msg()

        # Fill PoseArray
        for point in self.points:
            pose = Pose()
            pose.position.x = float(point[0])
            pose.position.y = float(point[1])
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        #pose_array.header.frame_id = self.world_frame
        self.pub.publish(pose_array)
        self.get_logger().info(f"Published PoseArray with {len(pose_array.poses)} poses")


if __name__ == "__main__":
    config_file = "config/grid.yaml"  # Path to your YAML configuration file
    points = load_yaml_config(config_file)
    print("Loaded points from YAML configuration:")

    rclpy.init()
    node = PointPublisher(points)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()