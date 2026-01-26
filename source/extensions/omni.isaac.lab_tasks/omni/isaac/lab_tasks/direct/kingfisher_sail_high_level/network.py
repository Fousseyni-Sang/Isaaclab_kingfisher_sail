import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import os
import torch
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter
import glob


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


def find_latest_checkpoint_dir(base_root):
    """
    Finds the latest checkpoint file under logs/acord/discr/{time_dir}/nn/
    Returns the directory containing the latest checkpoint file.
    """
    candidate_files = []
    for time_dir in os.listdir(base_root):
        nn_dir = os.path.join(base_root, time_dir, "nn")
        if not os.path.isdir(nn_dir):
            continue
        for checkpoint_file in os.listdir(nn_dir):
            full_path = os.path.join(nn_dir, checkpoint_file)
            if os.path.isfile(full_path):
                candidate_files.append(full_path)

    if not candidate_files:
        raise FileNotFoundError(f"No checkpoint files found under {base_root}/**/nn/")

    # Find the latest checkpoint file by modification time
    latest_file = max(candidate_files, key=os.path.getmtime)
    # Return the directory containing the latest checkpoint file
    return os.path.dirname(latest_file)

class DiscriminatorNetwork(nn.Module):
    def __init__(self, lr, state_dim, context_dim, fc1_dims=256, fc2_dims=256, num_envs=1024, mem_size=1000000,
                name='discriminator', chkpt_dir='logs/acord/discr', device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = lr
        self.name = name
        checkpoint_dir = chkpt_dir
        #self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.reparam_noise = 1e-6

        # Network layers
        self.fc1 = nn.Linear(self.state_dim, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.context_dim)
        self.sigma = nn.Linear(self.fc2_dims, self.context_dim)

        self.dropout_fc1 = nn.Dropout(p=0.2)
        self.dropout_fc2 = nn.Dropout(p=0.2)
        self.dropout_sigma = nn.Dropout(p=0.2)
        self.dropoutmu = nn.Dropout(p=0.2)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = device
        self.to(self.device)
        

        # Replay buffer for state-context pairs
        self.memory = ReplayBuffer(num_envs, mem_size, self.state_dim, self.context_dim, device)
        self.batch_size = 256
        self.num_envs = num_envs

         # Logging
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        self.run_name_dir = f"{checkpoint_dir}/{time_str}"
        self.checkpoint_dir = f"{self.run_name_dir}/nn"
        self.summary_dir = f"{self.run_name_dir}/summaries"

        self.saving_frequency = 10

        self.writer = None

    def save(self, step_marker):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.state_dict(), f"{self.checkpoint_dir}/discr_{step_marker}.pt")
        return

    def init_writer(self):

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)

        self.writer = SummaryWriter(self.summary_dir)

    def load_checkpoint(self, path_checkpoint=None):
        """
        Loads the latest or specified discriminator checkpoint.
        If path_checkpoint is a file, loads from it.
        If path_checkpoint is a directory, loads the latest discr_*.pt in that directory.
        If path_checkpoint is None, finds the latest time_dir under logs/acord/discr/*/nn/
        and loads the latest discr_*.pt inside that.
        """
        if path_checkpoint is not None:
            if os.path.isfile(path_checkpoint):
                checkpoint_path = path_checkpoint
            elif os.path.isdir(path_checkpoint):
                candidates = glob.glob(os.path.join(path_checkpoint, "discr_*.pt"))
                if not candidates:
                    raise FileNotFoundError(f"No discr_*.pt found in {path_checkpoint}")
                checkpoint_path = max(candidates, key=os.path.getmtime)
            else:
                raise FileNotFoundError(f"{path_checkpoint} is not a valid file or directory")
        else:
            # Find latest time_dir under logs/acord/discr/*/nn/
            discr_root = os.path.join("logs", "acord", "kingfisher_direct")
            if not os.path.isdir(discr_root):
                raise FileNotFoundError(f"{discr_root} does not exist")
            # Find all logs/acord/discr/*/nn directories
            nn_dirs = []
            for d in os.listdir(discr_root):
                nn_dir = os.path.join(discr_root, d, "nn")
                if os.path.isdir(nn_dir):
                    nn_dirs.append(nn_dir)
            if not nn_dirs:
                raise FileNotFoundError(f"No nn directories found in {discr_root}")
            # Choose the most recent nn directory
            latest_nn_dir = max(nn_dirs, key=os.path.getmtime)
            candidates = glob.glob(os.path.join(latest_nn_dir, "discr_*.pt"))
            if not candidates:
                raise FileNotFoundError(f"No discr_*.pt found in {latest_nn_dir}")
            checkpoint_path = max(candidates, key=os.path.getmtime)

        print(f"[DISCRIMINATOR] Loading checkpoint: {checkpoint_path}")
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.eval()
        return

    def forward(self, state):
        x = self.dropout_fc1(state)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc2(x)
        x = F.tanh(self.fc2(x)) #* 5
        x = self.dropoutmu(x)
        mu = torch.sigmoid(self.mu(x))
        sigma = torch.sigmoid(self.sigma(x))
        sigma = torch.clamp(sigma, min=0.01, max=1)

        return mu, sigma

    def predict(self, state, reparameterize=True, requires_grad=True):
        if requires_grad:
            mu, sigma = self.forward(state)
            if mu.requires_grad==False: mu.requires_grad = True
            if sigma.requires_grad==False: sigma.requires_grad = True
            if state.requires_grad==False: state.requires_grad = True
            
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

    def learn(self, state_batch, context_batch):
        """if self.memory.mem_cntr < self.batch_size:
            return None, None"""

        #states, contexts = self.memory.sample_buffer(self.num_envs, self.batch_size)
        predictions, log_probs, dist = self.predict(state_batch)
        #if predictions.requires_grad == False: predictions.requires_grad = True

        self.optimizer.zero_grad()
        loss = loss = (F.mse_loss(predictions, context_batch) * 10 + 1 / torch.abs(torch.min(dist.loc) - torch.max(dist.loc))) * 10

        #loss = ((F.mse_loss(predictions, context_batch))*10 + (1 / torch.abs((min(dist.loc) - max(dist.loc))))) * 10
        
        #disc1_loss = (F.mse_loss(disc1_predictions, limit_factor1)*10+ (1/torch.abs((min(dist1.loc)-max(dist1.loc)))))*10
        #if loss.requires_grad == False: loss.requires_grad = True
        
        
        #print(f"loss: {loss.requires_grad} loss: {loss}")
        loss.backward()
        self.optimizer.step()

        self.loss = loss
        return loss.item(), torch.mean(log_probs).item()

class GeneratorNetwork(nn.Module):
    def __init__(self, lr, input_dim, output_dim, fc1_dims=256, fc2_dims=256, num_envs=1024, mem_size=1000000,
                name='generator', chkpt_dir='logs/acord/gener', device='cpu'):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, fc1_dims)
        self.fc2 = nn.Linear(fc2_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, output_dim)
        self.sigma = nn.Linear(fc2_dims, output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = device
        checkpoint_dir = chkpt_dir
        self.to(self.device)

        self.memory = ReplayBuffer(num_envs, mem_size, input_dim, output_dim, device)
        self.batch_size = 256
        self.num_envs = num_envs

         # Logging
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        self.run_name_dir = f"{checkpoint_dir}/{time_str}"
        self.checkpoint_dir = f"{self.run_name_dir}/nn"
        self.summary_dir = f"{self.run_name_dir}/summaries"

        self.saving_frequency = 10

        self.writer = None
        self.acord_reward_scale = 0.05

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = F.tanh(self.mu(x))
        sigma = F.sigmoid(self.sigma(x))

        return mu, sigma
    
    def get_energy(self, actions):
        return torch.sum(actions**2, dim=1, keepdim=True)
    
    def predict(self, state, reparameterize=True, requires_grad=True):
        if requires_grad:
            mu, sigma = self.forward(state)
            if mu.requires_grad==False: mu.requires_grad = True
            if sigma.requires_grad==False: sigma.requires_grad = True
            if state.requires_grad==False: state.requires_grad = True
            
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
            
            
    def learn(self, context_batch, predicted_context_batch):
        
        predic_error_energy = torch.clamp(torch.abs(predicted_context_batch-context_batch.reshape(-1))**2, min=0.000001, max=0.99999)
        reward_acord = self.acord_reward_scale * (- torch.log(predic_error_energy))

        self.optimizer.zero_grad()
        loss = -reward_acord
        loss.backward()
        self.optimizer.step()

        return loss.item()

if __name__ == "__main__":
    num_envs = 1024
    mem_size = 1000
    device = "cpu"
    discriminator = DiscriminatorNetwork(lr=0.0001, state_dim=1, context_dim=1, num_envs=num_envs, mem_size=mem_size, device=device)

    generator = GeneratorNetwork(lr=0.0001, input_dim=2, output_dim=2, num_envs=num_envs, mem_size=mem_size, device=device)
    torch.manual_seed(0)

    # Fill buffer with simple relation: context = state * 0.5 + noise
    actions = torch.zeros((num_envs, 2), device=device)
    energy = generator.get_energy(actions)
    
    for _ in range(mem_size):
        state = 2*torch.rand((num_envs, 1), device=discriminator.device)
        context = state * 0.5 + 0.05 * torch.randn_like(state)
        discriminator.memory.store_transition(state, context)

    losses = []
    state_batch, context_batch = discriminator.memory.sample_buffer(num_envs, discriminator.batch_size)
    """for _ in range(1000):  # multiple training steps
        loss, _ = discriminator.learn(state_batch, context_batch)
        if loss is not None:
            losses.append(loss)"""

    discriminator.load_checkpoint()

    # Sample for evaluation
    state_batch, context_batch = discriminator.memory.sample_buffer(num_envs, 100)
    predictions, _, _ = discriminator.predict(state_batch, False, False)

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
    
    discriminator.memory.store_transition(torch.cat((energy.reshape(num_envs, -1), 
                                            energy_context.reshape(num_envs, -1)), dim=-1))
        
    discriminator.memory.store_transition(torch.cat((torch.norm(robot_speed, 
                        dim=-1).reshape(num_envs, -1), time_context.reshape(num_envs, -1)), dim=-1))
    

    prediction1, log_prob1, distribution1 = discriminator.predict(energy.reshape(num_envs, -1))
    prediction2, log_prob2, distribution2 = discriminator2.predict(torch.norm(robot_speed, dim=-1).reshape(num_envs, -1))

    predicted_energy_context, log_probs1, distrib1 = discriminator.predict(energy.reshape(num_envs, -1))
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