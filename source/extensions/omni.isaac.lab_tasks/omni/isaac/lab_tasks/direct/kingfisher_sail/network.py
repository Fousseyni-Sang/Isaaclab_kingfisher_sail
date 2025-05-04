import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import os
import torch
from .buffer import ReplayBuffer

class DiscriminatorNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims=256,
            fc2_dims=256, prediction_dims=1, memory_dim=2, num_envs=1024, mem_size=1000, name='discriminator', chkpt_dir='/tmp/sac'):
        super(DiscriminatorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = lr
        self.prediction_dims = prediction_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.prediction_dims)
        self.sigma = nn.Linear(self.fc2_dims, self.prediction_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.memory = ReplayBuffer(num_envs, mem_size, memory_dim, 'cuda:0')
        self.batch_size = 256
        self.num_envs = num_envs

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.tanh(self.fc2(prob))*5

        mu = torch.sigmoid(self.mu(prob))
        sigma = torch.sigmoid(self.sigma(prob))

        sigma = torch.clamp(sigma, min=.1, max=1)

        return mu, sigma

    def predict(self, state, reparameterize=True, requires_grad=True):
        if requires_grad:
            mu, sigma = self.forward(state)
            probabilities = Normal(mu, sigma)
            if reparameterize:
                predictions = probabilities.rsample()
            else:
                predictions = probabilities.sample()

            prediction = predictions.to(self.device)
            prediction = torch.clamp(prediction, .0001, .9999)
            log_probs = probabilities.log_prob(predictions)
            log_probs -= torch.log(1-prediction.pow(2)+self.reparam_noise)

            return prediction, log_probs, probabilities
        else:
            with torch.no_grad():
                mu, sigma = self.forward(state)
                probabilities = Normal(mu, sigma)

                if reparameterize:
                    predictions = probabilities.rsample()
                else:
                    predictions = probabilities.sample()

                prediction = predictions.to(self.device)
                prediction = torch.clamp(prediction, .0001, .9999)
                log_probs = probabilities.log_prob(predictions)
                log_probs -= torch.log(1-prediction.pow(2)+self.reparam_noise)
                #log_probs = log_probs.sum(0, keepdim=True)

                return prediction, log_probs, probabilities

    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
                return None, None
        
        state_context = self.memory.sample_buffer(self.num_envs, self.batch_size)

        state = state_context[:, 0].reshape(-1, self.input_dims)
        context = state_context[:, -1].reshape(-1, self.input_dims)

        predictions, log_probs, dist1 = self.predict(state, requires_grad=True)
        #print(f"st_cont: {state_context.shape} state: {state.shape} context: {context.shape} predic: {predictions.shape}")
        self.optimizer.zero_grad()
        loss = (F.mse_loss(predictions, context) * 10 + (1 / torch.abs(torch.min(dist1.loc) - torch.max(dist1.loc)))) * 10
        loss.requires_grad = True
        #print(f"required: {loss.requires_grad}")
        loss.backward()
        self.optimizer.step()

        return loss, torch.mean(log_probs).item()
            

class Agent():
    def __init__(self, num_envs=1024, disc_lr=.0001, input_dims=[2], env=None, gamma=0.99, n_actions=2, tau=0.005,
             disc_layer1_size=256, disc_layer2_size=256, batch_size=256, reward_scale=2, 
                reparam_noise=1e-6, disc_input_dims=[1], predict_dims=1):
        self.gamma = gamma
        self.tau = tau
        
        self.disc_memory = ReplayBuffer(num_envs, 1000000, input_dims)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_action = 1
        self.env = env
        self.auto_entropy = True

        self.limit_factor_dist = torch.distributions.Uniform(low=torch.tensor(0.0).to('cuda:0' if torch.cuda.is_available() else 'cpu'),  high=torch.tensor(1.0).to('cuda:0' if torch.cuda.is_available() else 'cpu'))

        self.discriminator1 = DiscriminatorNetwork(lr=disc_lr, input_dims=disc_input_dims, fc1_dims=disc_layer1_size, fc2_dims=disc_layer2_size, prediction_dims=predict_dims)
        self.discriminator2 = DiscriminatorNetwork(lr=disc_lr, input_dims=disc_input_dims, fc1_dims=disc_layer1_size, fc2_dims=disc_layer2_size, prediction_dims=predict_dims)

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def remember(self, state, action, reward, new_state, done):
        self.disc_memory.store_transition(state, action, reward, new_state, done)
        

    def learn(self, update_params=True, update_disc=False):
        if(self.auto_entropy):
            actor_loss = None
            if self.memory.mem_cntr < self.batch_size:
                return None, None

            state, action, reward, new_state, done = \
                    self.disc_critic_1.sample_buffer(self.batch_size)

            reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
            done = torch.tensor(done).to(self.actor.device)
            state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
            state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
            action = torch.tensor(action, dtype=torch.float).to(self.actor.device)


            limit_factor = torch.clone(state)
            limit_factor = limit_factor[:, -2:]
            limit_factor1 = limit_factor[:,0]
            limit_factor2 = limit_factor[:,1]
            limit_factor1 = limit_factor1[:,None]
            limit_factor2 = limit_factor2[:,None]

            disc_state = torch.clone(state)
    

            #speed and hull angle:
            disc1_state =  disc_state[:, 0:1]
            disc2_state =  disc_state[:, 2:3]
            

            disc1_predictions, disc1_log_probs, dist1 = self.discriminator1.predict(disc1_state, requires_grad=False)
            disc1_log_probs.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            value_ = torch.clone(value_).detach().to('cuda:0' if torch.cuda.is_available() else 'cpu')
            
            # log probability of the limit factor
            log_prob_of_lf1 = self.limit_factor_dist.log_prob(disc1_predictions.detach())

            log_prob_of_lf1 = torch.clone(log_prob_of_lf1).to('cuda:0' if torch.cuda.is_available() else 'cpu')

            

            if torch.any(torch.isinf(disc1_log_probs)) or torch.any(torch.isnan(disc1_log_probs)) :
                print(disc1_log_probs, "disc_log_probs")
            if torch.any(torch.isinf(log_prob_of_lf1)) or torch.any(torch.isnan(log_prob_of_lf1)):
                log_prob_of_lf1 = torch.nan_to_num(log_prob_of_lf1, posinf=0, neginf=0)

            disc2_predictions, disc2_log_probs, dist2 = self.discriminator2.predict(disc2_state, requires_grad=False)
            disc2_log_probs.to('cuda:0' if torch.cuda.is_available() else 'cpu')

            log_prob_of_lf2 = self.limit_factor_dist.log_prob(disc2_predictions.detach())

            log_prob_of_lf2 = torch.clone(log_prob_of_lf2).to('cuda:0' if torch.cuda.is_available() else 'cpu')

            

            if torch.any(torch.isinf(disc2_log_probs)) or torch.any(torch.isnan(disc2_log_probs)) :
                print(disc2_log_probs, "disc_log_probs")
            if torch.any(torch.isinf(log_prob_of_lf2)) or torch.any(torch.isnan(log_prob_of_lf2)):
                log_prob_of_lf2 = torch.nan_to_num(log_prob_of_lf2, posinf=0, neginf=0)

            if torch.any(torch.isinf(value_)) or torch.any(torch.isnan(value_)):
                print(value_, "log_prob_of_lf")    

            rew=(-torch.log(torch.clamp(torch.abs(disc1_predictions-limit_factor1)**2, min=.000001, max=.99999)) + \
                -torch.log(torch.clamp(torch.abs(disc2_predictions-limit_factor2)**2, min=.000001, max=.99999)))/2

            if torch.any(torch.isinf(rew)) or torch.any(torch.isnan(rew)):
                print(rew, "rew")    
                print((dist1.cdf(disc1_predictions)-dist1.cdf(limit_factor))**2, "diff")
            

            rew = (rew[:,-1])*10
            rew = torch.where(reward<-29, reward*12, rew)

            

            

            disc_loss = None
            disc1_loss = None
            disc2_loss = None
            if update_disc:
                state, action, reward, new_state, done = \
                    self.disc_memory.sample_buffer(self.batch_size)

                reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
                done = torch.tensor(done).to(self.actor.device)
                state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
                state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
                action = torch.tensor(action, dtype=torch.float).to(self.actor.device)

                limit_factor = torch.clone(state)
                limit_factor = limit_factor[:, -2:]
                limit_factor1 = limit_factor[:,0]
                limit_factor2 = limit_factor[:,1]
                limit_factor1 = limit_factor1[:,None]
                limit_factor2 = limit_factor2[:,None]

                disc_state = torch.clone(state)

                ##speed and hull angle:
                disc1_state =  disc_state[:, 0:1]
                disc2_state =  disc_state[:, 2:3]
                
                disc1_predictions, disc1_log_probs, dist1 = self.discriminator1.predict(disc1_state, requires_grad=True)
                self.discriminator1.optimizer.zero_grad()
                disc1_loss = (F.mse_loss(disc1_predictions, limit_factor1)*10+ (1/torch.abs((min(dist1.loc)-max(dist1.loc)))))*10 
                disc1_loss.backward()
                self.discriminator1.optimizer.step()

                disc2_predictions, disc2_log_probs, dist2 = self.discriminator2.predict(disc2_state, requires_grad=True)
                self.discriminator2.optimizer.zero_grad()
                disc2_loss = (F.mse_loss(disc2_predictions, limit_factor2)*10+ (1/torch.abs((min(dist2.loc)-max(dist2.loc)))))*10 
                disc2_loss.backward()
                self.discriminator2.optimizer.step()

            
            if disc1_loss is not None and disc2_loss is not None:
                return disc1_loss, disc2_loss, torch.mean(disc1_log_probs).item(),\
                torch.mean(disc2_log_probs).item()
            else:
                return actor_loss, disc_loss