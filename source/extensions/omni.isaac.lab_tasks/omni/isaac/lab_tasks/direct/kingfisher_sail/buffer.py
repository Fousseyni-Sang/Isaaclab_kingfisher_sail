"""
This code mainly follows a Soft-Actor Critic YouTube tutorial found at:
https://www.youtube.com/watch?v=ioidsRlf79o&t=2649s
Channel name: Machine Learning with Phil
"""
import torch

class ReplayBuffer:
    def __init__(self, num_envs, max_size, input_shape, device="cpu"):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.device = device

        # for the discriminator, it contains its state and context
        self.state_memory = torch.zeros((num_envs, self.mem_size, input_shape), dtype=torch.float32, device=self.device)
    
    def store_transition(self, state):
        index = self.mem_cntr % self.mem_size

        self.state_memory[:, index] = state
        
        self.mem_cntr += 1

    def sample_buffer(self, num_envs, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = torch.randint(0, max_mem*num_envs, (batch_size, ), device=self.device)
        # Create environment indices: [0, 1, ..., 99] repeated for each sample
        #env_ids = torch.arange(num_envs, device=self.device).unsqueeze(1).expand(-1, batch_size)  

        states = self.state_memory.reshape(num_envs*self.mem_size, -1)[batch]  

        #flat = states.reshape(-1, states.shape[-1])  # shape: (1024*1000, D)

        #idx = torch.randperm(flat.size(0))[:256]
        #state_context = flat[idx]

        #states = self.state_memory[batch]
        #print(f"batch: {batch.shape} state_buffer: {states.shape}")
        return states

    """def store_transition(self, state=None, action=None, reward=None, new_state=None, done=None, increment_mem_cntr=False):
        index = self.mem_cntr % self.mem_size

        if state is not None:
            self.state_memory[:, index] = state
        if new_state is not None:
            self.new_state_memory[:, index] = new_state
        if action is not None:
            self.action_memory[:, index] = action
        if reward is not None:
            self.reward_memory[:, index] = reward
        if done is not None:
            self.terminal_memory[:, index] = done

        if increment_mem_cntr:
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
        """

