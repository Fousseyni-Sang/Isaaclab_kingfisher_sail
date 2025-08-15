import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import os
import torch
import matplotlib.pyplot as plt
import time

class ReplayBuffer:
    def __init__(self, num_envs, max_size, state_shape, context_shape, device="cpu"):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.device = device

        # State and context memory
        self.state_memory = torch.zeros(
            (num_envs, self.mem_size, state_shape),
            dtype=torch.float32,
            device=self.device
        )
        self.context_memory = torch.zeros(
            (num_envs, self.mem_size, context_shape),
            dtype=torch.float32,
            device=self.device
        )

    def store_transition(self, state, context):
        index = self.mem_cntr % self.mem_size
        self.state_memory[:, index] = state
        self.context_memory[:, index] = context
        self.mem_cntr += 1

    def sample_buffer(self, num_envs, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = torch.randint(
            0, max_mem * num_envs, (batch_size,), device=self.device
        )

        flat_states = self.state_memory.reshape(num_envs * self.mem_size, -1)
        flat_contexts = self.context_memory.reshape(num_envs * self.mem_size, -1)

        return flat_states[batch], flat_contexts[batch]


class DiscriminatorNetwork(nn.Module):
    def __init__(self, lr, state_dim, context_dim,
                 fc1_dims=256, fc2_dims=256,
                 num_envs=1024, mem_size=1000,
                 name='discriminator', chkpt_dir='logs/acord', device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = lr
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.reparam_noise = 1e-6

        # Network layers
        self.fc1 = nn.Linear(self.state_dim, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.context_dim)
        self.sigma = nn.Linear(self.fc2_dims, self.context_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = device
        self.to(self.device)

        # Replay buffer for state-context pairs
        self.memory = ReplayBuffer(num_envs, mem_size, self.state_dim, self.context_dim, device)
        self.batch_size = 256
        self.num_envs = num_envs

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.tanh(self.fc2(x)) * 5
        mu = torch.sigmoid(self.mu(x))
        sigma = torch.sigmoid(self.sigma(x))
        sigma = torch.clamp(sigma, min=0.1, max=1)
        return mu, sigma

    def predict(self, state, reparameterize=True, requires_grad=True):
        if requires_grad:
            mu, sigma = self.forward(state)
            dist = Normal(mu, sigma)
            samples = dist.rsample() if reparameterize else dist.sample()
            samples = torch.clamp(samples, 0.0001, 0.9999)
            log_probs = dist.log_prob(samples)
            log_probs -= torch.log(1 - samples.pow(2) + self.reparam_noise)
            return samples, log_probs, dist
        else:
            with torch.no_grad():
                mu, sigma = self.forward(state)
                dist = Normal(mu, sigma)
                samples = dist.rsample() if reparameterize else dist.sample()
                samples = torch.clamp(samples, 0.0001, 0.9999)
                log_probs = dist.log_prob(samples)
                log_probs -= torch.log(1 - samples.pow(2) + self.reparam_noise)
                return samples, log_probs, dist

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return None, None

        states, contexts = self.memory.sample_buffer(self.num_envs, self.batch_size)
        predictions, log_probs, dist = self.predict(states)

        self.optimizer.zero_grad()
        loss = F.mse_loss(predictions, contexts)
        loss.backward()
        self.optimizer.step()

        return loss.item(), torch.mean(log_probs).item()


if __name__ == "__main__":
    num_envs = 1024
    mem_size = 1000
    discriminator1 = DiscriminatorNetwork(
        lr=0.0001, state_dim=1, context_dim=1,
        fc1_dims=256, fc2_dims=256,
        num_envs=num_envs, mem_size=mem_size
    )

    torch.manual_seed(0)

    # Fill buffer with simple relation: context = state * 0.5 + noise
    for _ in range(mem_size):
        state = torch.rand((num_envs, 1), device=discriminator1.device)
        context = torch.sin(state ** 0.5) + torch.tan(0.5 * torch.randn_like(state))
        discriminator1.memory.store_transition(state, context)

    losses = []
    for _ in range(1000):  # multiple training steps
        loss, _ = discriminator1.learn()
        if loss is not None:
            losses.append(loss)

    # Sample for evaluation
    state_batch, context_batch = discriminator1.memory.sample_buffer(num_envs, 100)
    predictions, _, _ = discriminator1.predict(state_batch, False, False)

    # Plot
    plt.figure(figsize=(8,6))
    plt.subplot(2, 1, 1)
    plt.plot(losses, label="loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(predictions.cpu().numpy(), label="predictions")
    plt.plot(context_batch.cpu().numpy(), label="context")
    plt.plot(torch.abs(predictions - context_batch).cpu().numpy(), label="pred_error")
    plt.legend()

    plt.savefig("loss_disc.png")
    print("Plot saved to loss_disc.png")

    
    """max_energy = 2
    max_speed = 2
    energy_context = torch.ones(num_envs)
    time_context = torch.ones(num_envs)

    energy = max_energy*torch.rand(num_envs)
    robot_speed = max_speed*torch.rand((num_envs, 2))

    energy_context = torch.zeros_like(energy_context).uniform_(0, 1)
    time_context = torch.zeros_like(time_context).uniform_(0, 1)
    
    discriminator1.memory.store_transition(torch.cat((energy.reshape(num_envs, -1), 
                                            energy_context.reshape(num_envs, -1)), dim=-1))
        
    discriminator1.memory.store_transition(torch.cat((torch.norm(robot_speed, 
                        dim=-1).reshape(num_envs, -1), time_context.reshape(num_envs, -1)), dim=-1))
    

    prediction1, log_prob1, distribution1 = discriminator1.predict(energy.reshape(num_envs, -1))
    prediction2, log_prob2, distribution2 = discriminator2.predict(torch.norm(robot_speed, dim=-1).reshape(num_envs, -1))

    predicted_energy_context, log_probs1, distrib1 = discriminator1.predict(energy.reshape(num_envs, -1))
    predic_error_energy = torch.abs(predicted_energy_context.reshape(-1) - energy_context.reshape(-1))
    predicted_time_context, log_probs2, distrib2= discriminator2.predict(energy.reshape(num_envs, -1))
    predic_error_time = torch.abs(predicted_time_context.reshape(-1) - time_context.reshape(-1))

    reward_acord = 0.5*(-torch.log(predic_error_energy) - torch.log(predic_error_time))
    #print(f"prediction error: {predic_error_energy.shape} \tprediction error time: {predic_error_time.shape}")
    print(f"prediction energy: {predicted_energy_context[:10]} \nreal: {energy_context[:10]}")
    print(f"prediction time: {predicted_time_context[:10]} \nreal: {time_context[:10]}")
    print(f"reward: {reward_acord[:10]}")

    plt.figure()
    plt.plot(predicted_energy_context.detach().numpy(), label="pr_ener")
    plt.plot(energy_context.detach().numpy(), label="rl_ener")
    plt.plot(predicted_time_context.detach().numpy(), label="pr_time")
    plt.plot(time_context.detach().numpy(), label="rl_time")
    plt.legend()
    plt.savefig("/tmp/acord_prediction.png")

    plt.figure()
    plt.plot(reward_acord.detach().numpy(), label="reward")
    plt.legend()
    plt.savefig("/tmp/acord_reward.png")"""
