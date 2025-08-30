# Create Publisher Node
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32
import torch

class RlAgentPublisher(rclpy.node.Node):
    def __init__(self, num_agents):
        super().__init__("rl_agent_publisher")
        self.num_agents = num_agents
        self.obs_publishers = [self.create_publisher(Float32MultiArray, f"rl_observations_{i}", 10) for i in range(num_agents)]
        self.act_publishers = [self.create_publisher(Float32MultiArray, f"rl_actions_{i}", 10) for i in range(num_agents)]
        self.rew_publishers = [self.create_publisher(Float32MultiArray, f"rl_rewards_{i}", 10) for i in range(num_agents)]
        self.aero_force_publisher =  [self.create_publisher(Float32MultiArray, f"rl_aero_force_{i}", 10) for i in range(num_agents)] 
        self.thruster_force_publisher =  [self.create_publisher(Float32MultiArray, f"rl_thruster_force_{i}", 10) for i in range(num_agents)] 
        self.speed_publisher =  [self.create_publisher(Float32MultiArray, f"rl_speed_{i}", 10) for i in range(num_agents)] 
        self.angle_of_attack_publisher =  [self.create_publisher(Float32MultiArray, f"rl_angle_of_attack_{i}", 10) for i in range(num_agents)]
        self.app_wind_angle_publisher =  [self.create_publisher(Float32MultiArray, f"rl_app_wind_angle_{i}", 10) for i in range(num_agents)]
        self.sail_angle_publisher =  [self.create_publisher(Float32MultiArray, f"rl_sail_angle_{i}", 10) for i in range(num_agents)]
        self.heading_publisher =  [self.create_publisher(Float32MultiArray, f"rl_sailboat_heading_w_{i}", 10) for i in range(num_agents)]
        self.head_wrt_wind_publisher =  [self.create_publisher(Float32MultiArray, f"rl_head_wrt_wind_{i}", 10) for i in range(num_agents)]
        self.lift_drag_ratio_publisher = [self.create_publisher(Float32MultiArray, f"rl_lift_drag_ratio_{i}", 10) for i in range(num_agents)]
        self.lift_publisher = [self.create_publisher(Float32MultiArray, f"rl_lift_{i}", 10) for i in range(num_agents)]
        self.drag_publisher = [self.create_publisher(Float32MultiArray, f"rl_drag_{i}", 10) for i in range(num_agents)]
        self.robot_pos_publisher = [self.create_publisher(Float32MultiArray, f"rl_robot_pos_w_{i}", 10) for i in range(num_agents)]
        self.goal_pos_publisher = [self.create_publisher(Float32MultiArray, f"rl_goal_pos_w_{i}", 10) for i in range(num_agents)]
        self.energy_publisher = [self.create_publisher(Float32MultiArray, f"rl_energy_{i}", 10) for i in range(num_agents)]
        self.episode_energy_publisher = [self.create_publisher(Float32MultiArray, f"rl_episode_energy_{i}", 10) for i in range(num_agents)]
        self.drag_coeff_publisher = [self.create_publisher(Float32MultiArray, f"rl_drag_coeff_{i}", 10) for i in range(num_agents)]
        self.lift_coeff_publisher = [self.create_publisher(Float32MultiArray, f"rl_lift_coeff_{i}", 10) for i in range(num_agents)]
        self.sum_angle_publisher =  [self.create_publisher(Float32MultiArray, f"rl_sum_angle_{i}", 10) for i in range(num_agents)]
        self.desired_pos_publisher =  [self.create_publisher(Float32MultiArray, f"rl_desired_pos_{i}", 10) for i in range(num_agents)]
        self.reward_progress_publisher =  [self.create_publisher(Float32MultiArray, f"rl_rew_progress_{i}", 10) for i in range(num_agents)]
        self.reward_energy_publisher =  [self.create_publisher(Float32MultiArray, f"rl_rew_energy_{i}", 10) for i in range(num_agents)]
        self.reward_bearing_publisher =  [self.create_publisher(Float32MultiArray, f"rl_rew_bearing_{i}", 10) for i in range(num_agents)]
        self.reward_backward_publisher =  [self.create_publisher(Float32MultiArray, f"rl_rew_backward_{i}", 10) for i in range(num_agents)]
        self.loss_disc_publisher =  [self.create_publisher(Float32MultiArray, f"rl_loss_disc_{i}", 10) for i in range(num_agents)]
        self.tack_wpts_publisher = [self.create_publisher(Float32MultiArray, f"rl_tack_wpts_{i}", 10) for i in range(num_agents)]
        self.energy_context_publisher = [self.create_publisher(Float32MultiArray, f"rl_energy_context_{i}", 10) for i in range(num_agents)]
        self.predicted_context_publisher =  [self.create_publisher(Float32MultiArray, f"rl_predicted_context_{i}", 10) for i in range(num_agents)]

    def publish(self, obs, act, rew, aero_force, thruster_force, speed, aoa, app_angle, sail_ang, head_w, head_wrt_wind, 
                ld_ratio, robot_pos, goal_pos, energy, episode_energy, lift, drag, lift_coeff, drag_coeff, sum_angle, desired_pos,
                rew_progress, rew_bearing, rew_energy, rew_backward, loss_disc, tack_wpts, energy_context, predicted_context):
        for i in range(self.num_agents):
            msg_obs = Float32MultiArray(data=obs[i].cpu().numpy().flatten().tolist())
            msg_act = Float32MultiArray(data=act[i].cpu().numpy().flatten().tolist())
            msg_rew = Float32MultiArray(data=rew[i].cpu().numpy().flatten().tolist())
            msg_aero_force = Float32MultiArray(data=aero_force[i].cpu().numpy().flatten().tolist())
            msg_thruster_force = Float32MultiArray(data=thruster_force[i].cpu().numpy().flatten().tolist())
            msg_speed = Float32MultiArray(data=speed[i].cpu().numpy().flatten().tolist())
            msg_angle_of_attack = Float32MultiArray(data=aoa[i].cpu().numpy().flatten().tolist())
            msg_app_wind_angle = Float32MultiArray(data=app_angle[i].cpu().numpy().flatten().tolist())
            msg_sail_angle = Float32MultiArray(data=sail_ang[i].cpu().numpy().flatten().tolist())
            msg_heading_w = Float32MultiArray(data=head_w[i].cpu().numpy().flatten().tolist())
            msg_head_wrt_wind = Float32MultiArray(data=head_wrt_wind[i].cpu().numpy().flatten().tolist())
            msg_lift_drag_ratio = Float32MultiArray(data=ld_ratio[i].cpu().numpy().flatten().tolist())
            msg_robot_pos = Float32MultiArray(data=robot_pos[i].cpu().numpy().flatten().tolist())
            msg_goal_pos = Float32MultiArray(data=goal_pos[i].cpu().numpy().flatten().tolist())
            msg_energy = Float32MultiArray(data=energy[i].cpu().numpy().flatten().tolist())
            msg_episode_energy = Float32MultiArray(data=episode_energy[i].cpu().numpy().flatten().tolist())
            msg_lift = Float32MultiArray(data=lift[i].cpu().numpy().flatten().tolist())
            msg_drag = Float32MultiArray(data=drag[i].cpu().numpy().flatten().tolist())
            msg_lift_coeff = Float32MultiArray(data=lift_coeff[i].cpu().numpy().flatten().tolist())
            msg_drag_coeff = Float32MultiArray(data=drag_coeff[i].cpu().numpy().flatten().tolist())
            msg_sum_angle = Float32MultiArray(data=sum_angle[i].cpu().numpy().flatten().tolist())
            msg_desired_pos = Float32MultiArray(data=desired_pos[i].cpu().numpy().flatten().tolist())
            msg_rew_progress = Float32MultiArray(data=rew_progress[i].cpu().numpy().flatten().tolist())
            msg_rew_backward = Float32MultiArray(data=rew_backward[i].cpu().numpy().flatten().tolist())
            msg_rew_energy = Float32MultiArray(data=rew_energy[i].cpu().numpy().flatten().tolist())
            msg_rew_bearing = Float32MultiArray(data=rew_bearing[i].cpu().numpy().flatten().tolist())
            msg_loss_disc = Float32MultiArray(data=loss_disc[i].cpu().numpy().flatten().tolist())
            msg_tack_wpts = Float32MultiArray(data=tack_wpts[i].cpu().numpy().flatten().tolist())
            msg_energy_context = Float32MultiArray(data=energy_context[i].cpu().numpy().flatten().tolist())
            msg_predicted_context = Float32MultiArray(data=predicted_context[i].cpu().numpy().flatten().tolist())

            self.obs_publishers[i].publish(msg_obs)
            self.act_publishers[i].publish(msg_act)
            self.rew_publishers[i].publish(msg_rew)
            self.aero_force_publisher[i].publish(msg_aero_force)
            self.thruster_force_publisher[i].publish(msg_thruster_force)
            self.speed_publisher[i].publish(msg_speed)
            self.angle_of_attack_publisher[i].publish(msg_angle_of_attack)
            self.app_wind_angle_publisher[i].publish(msg_app_wind_angle)
            self.sail_angle_publisher[i].publish(msg_sail_angle)
            self.heading_publisher[i].publish(msg_heading_w)
            self.head_wrt_wind_publisher[i].publish(msg_head_wrt_wind)
            self.lift_drag_ratio_publisher[i].publish(msg_lift_drag_ratio)
            self.robot_pos_publisher[i].publish(msg_robot_pos)
            self.energy_publisher[i].publish(msg_energy)
            self.episode_energy_publisher[i].publish(msg_episode_energy)
            self.goal_pos_publisher[i].publish(msg_goal_pos)
            self.lift_publisher[i].publish(msg_lift)
            self.drag_publisher[i].publish(msg_drag)
            self.drag_coeff_publisher[i].publish(msg_drag_coeff)
            self.lift_coeff_publisher[i].publish(msg_lift_coeff)
            self.sum_angle_publisher[i].publish(msg_sum_angle)
            self.desired_pos_publisher[i].publish(msg_desired_pos)
            self.reward_progress_publisher[i].publish(msg_rew_progress)
            self.reward_energy_publisher[i].publish(msg_rew_energy)
            self.reward_bearing_publisher[i].publish(msg_rew_bearing)
            self.reward_backward_publisher[i].publish(msg_rew_backward)
            self.loss_disc_publisher[i].publish(msg_loss_disc)
            self.tack_wpts_publisher[i].publish(msg_tack_wpts)
            self.energy_context_publisher[i].publish(msg_energy_context)
            self.predicted_context_publisher[i].publish(msg_predicted_context)


class RewardWeightSubscriber(Node):
    def __init__(self, slider_names):
        super().__init__('reward_weight_subscriber')

        self.reward_weights = {}
        self.slider_subscriptions = []

        for name in slider_names:
            topic_name = f"reward_{name}_weight"
            self.reward_weights[name] = 0.0  # Default value

            # Create a subscriber for each slider topic
            sub = self.create_subscription(
                Float32,
                topic_name,
                lambda msg, name=name: self.reward_callback(msg, name),
                10
            )
            self.slider_subscriptions.append(sub)

    def reward_callback(self, msg, name):
        self.reward_weights[name] = msg.data
        self.get_logger().info(f"Received {name} weight: {msg.data:.2f}")
