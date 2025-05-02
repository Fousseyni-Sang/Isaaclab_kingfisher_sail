import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import os
import torch

class DiscriminatorNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims=256,
            fc2_dims=256, prediction_dims=1, name='discriminator', chkpt_dir='/tmp/sac'):
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