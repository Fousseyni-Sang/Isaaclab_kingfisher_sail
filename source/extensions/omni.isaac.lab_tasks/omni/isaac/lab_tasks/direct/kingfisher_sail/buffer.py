"""
This code mainly follows a Soft-Actor Critic YouTube tutorial found at:
https://www.youtube.com/watch?v=ioidsRlf79o&t=2649s
Channel name: Machine Learning with Phil
"""
import torch

class ReplayBuffer:
    def __init__(self, num_envs, max_size, input_shape, n_actions, device="cpu"):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.device = device

        self.state_memory = torch.zeros((num_envs, self.mem_size, input_shape), dtype=torch.float32, device=self.device)
        self.new_state_memory = torch.zeros((num_envs, self.mem_size, input_shape), dtype=torch.float32, device=self.device)
        self.action_memory = torch.zeros((num_envs, self.mem_size, n_actions), dtype=torch.float32, device=self.device)
        self.reward_memory = torch.zeros(num_envs, self.mem_size, dtype=torch.float32, device=self.device)
        self.terminal_memory = torch.zeros(num_envs, self.mem_size, dtype=torch.bool, device=self.device)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[:, index] = state
        self.new_state_memory[:, index] = new_state
        self.action_memory[:, index] = action
        self.reward_memory[:, index] = reward
        self.terminal_memory[:, index] = done

        self.mem_cntr += 1

    def sample_buffer(self, num_envs, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = torch.randint(0, max_mem, (num_envs, batch_size), device=self.device)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones

