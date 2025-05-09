import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from PyQt5.QtWidgets import QApplication, QSlider, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt

class RewardSlider(Node):
    def __init__(self, sliders_config):
        super().__init__('reward_slider_gui')
        self.slider_publishers = {}
        
        # PyQt setup
        
        self.app = QApplication([])
        self.window = QWidget()
        self.window.setWindowTitle("Reward Weight Sliders")
        self.layout = QVBoxLayout()
        
        for slider_name, config in sliders_config.items():
            min_val, max_val, initial = config['min'], config['max'], config['init']

            # ROS publisher
            topic_name = f"reward_{slider_name}_weight"
            self.slider_publishers[slider_name] = self.create_publisher(Float32, topic_name, 10)
            
            # UI Components
            slider_layout = QHBoxLayout()
            label = QLabel(f"{slider_name}:")
            value_label = QLabel(f"{initial:.2f}")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(min_val * 100))
            slider.setMaximum(int(max_val * 100))
            slider.setValue(int(initial * 100))
            
            # Connect signal to callback
            slider.valueChanged.connect(
                lambda value, name=slider_name, val_label=value_label: self.publish_value(name, value, val_label)
            )

            slider_layout.addWidget(label)
            slider_layout.addWidget(slider)
            slider_layout.addWidget(value_label)
            self.layout.addLayout(slider_layout)
            
        self.window.setLayout(self.layout)
        self.window.show()

    def publish_value(self, name, value, label):
        float_val = value / 100.0
        msg = Float32()
        msg.data = float_val
        self.slider_publishers[name].publish(msg)
        label.setText(f"{float_val:.2f}")

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            self.app.processEvents()

def main():
    rclpy.init()

    sliders_config = {
        'time':   {'min': 0., 'max': 2., 'init': 0.5},
        'energy': {'min': 0., 'max': 2., 'init': 0.5},
        'goal':   {'min': 0., 'max': 5., 'init': 1.0},
        'desired_speed':   {'min': 0.3, 'max': 1.5, 'init': 1.0},
        'wind_direct': {'min': -180., 'max': 180., 'init': -90.0},
        'wind_speed': {'min': 4., 'max': 8., 'init': 5}
        # Add more sliders here
    }

    node = RewardSlider(sliders_config)
    node.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
