import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import os
import torch
import matplotlib.pyplot as plt
#from .buffer import ReplayBuffer

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

class DiscriminatorNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims=256,
            fc2_dims=256, prediction_dims=1, memory_dim=2, num_envs=1024, mem_size=1000, name='discriminator', chkpt_dir='/tmp/sac', device='cpu'):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = lr
        self.prediction_dims = prediction_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.reparam_noise = 1e-6

        self.output_dims = self.prediction_dims

        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.output = nn.Linear(self.fc2_dims, self.prediction_dims)
        self.mu = nn.Linear(self.fc2_dims, self.prediction_dims)
        self.sigma = nn.Linear(self.fc2_dims, self.prediction_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = device #torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.memory = ReplayBuffer(num_envs, mem_size, memory_dim, device)
        self.batch_size = 256
        self.num_envs = num_envs

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)  # No activation, raw output for regression
        return x

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return None

        state_context = self.memory.sample_buffer(self.num_envs, self.batch_size)
        state = state_context[:, 0].reshape(-1, self.input_dims)
        context = state_context[:, -1].reshape(-1, self.output_dims)

        prediction = self.forward(state)
        loss = F.mse_loss(prediction, context)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    """def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.tanh(self.fc2(prob))*5

        mu = torch.sigmoid(self.mu(prob))
        sigma = F.softplus(self.sigma(prob)) + 1e-4 #torch.sigmoid(self.sigma(prob))

        return mu, sigma
    
    def predict(self, state, reparameterize=True, requires_grad=True):
        mu, sigma = self.forward(state)
        sigma = F.softplus(sigma) + 1e-2
        dist = Normal(mu, sigma)

        with torch.set_grad_enabled(requires_grad):
            predictions = dist.rsample() if reparameterize else dist.sample()
            prediction = torch.clamp(predictions, 0.0001, 0.9999)
            log_probs = dist.log_prob(predictions)
            log_probs -= torch.log(1 - prediction.pow(2) + self.reparam_noise)

        return prediction, log_probs, dist


    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
                return None, None
        
        state_context = self.memory.sample_buffer(self.num_envs, self.batch_size)
        print(f"state_context: {state_context[:10]}\n")
        state = state_context[:, 0].reshape(-1, self.input_dims)
        context = state_context[:, -1].reshape(-1, self.input_dims)
        print(f"state: {state[:10]} \ncontext: {context[:10]}\n")
        predictions, log_probs, dist1 = self.predict(state)
        print(f"prediction: {predictions[:10]}\n")
        #print(f"pred: {predictions.requires_grad}")
        #print(f"st_cont: {state_context.shape} state: {state.shape} context: {context.shape} predic: {predictions.shape}")
        self.optimizer.zero_grad()
        loss = (F.mse_loss(predictions, context)) #* 10 + (1 / torch.abs(torch.min(dist1.loc) - torch.max(dist1.loc)))) * 10
        print(f"loss: {loss}\n")
        #loss.requires_grad = True
        #print(f"required: {loss.requires_grad}")
        loss.backward()
        self.optimizer.step()

        return loss, torch.mean(log_probs).item()
        """


if __name__=="__main__":
    num_envs = 1024
    mem_size = 1000
    
    discriminator1 = DiscriminatorNetwork(lr=0.0001, input_dims=1, fc1_dims=256, fc2_dims=256, prediction_dims=1, 
                                                   num_envs=num_envs, mem_size=mem_size)
    discriminator2 = DiscriminatorNetwork(lr=0.0001, input_dims=1, fc1_dims=256, fc2_dims=256, prediction_dims=1, 
                                                num_envs=num_envs)
    
    # Instantiate and test
    torch.manual_seed(0)

    # Fill the buffer with synthetic data
    for _ in range(mem_size):
        s = torch.rand(num_envs, 2, device=discriminator1.device)  # state and context
        discriminator1.memory.store_transition(s)

    # Run learning steps
    losses = []
    logprobs = []
    for _ in range(1000):
        #loss, logp = discriminator1.learn()
        loss = discriminator1.learn()
        if loss is not None:
            losses.append(loss.detach().numpy())
            #logprobs.append(logp)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(losses, label="loss_determ")
    plt.legend()

    """plt.subplot(2, 1, 2)
    plt.plot(logprobs, label="log_prob")
    plt.legend()"""

    plt.savefig("/home/isaac_user/asv-sawasp-fousseyni/IsaacLab_kingfisher/loss.png")

    
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