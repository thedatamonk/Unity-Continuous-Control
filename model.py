# this class contains the definition of the Actor and the Critic Networks

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



# the implementation closely follows the DDPG paper
# Paper: https://arxiv.org/pdf/1509.02971.pdf

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model"""
    def __init__(self, state_size, action_size, seed, hidden1=400, hidden2=300):
        super(Actor, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)
        self.bn = nn.BatchNorm1d(state_size)
        
        self.tanh = nn.Tanh()
        # initialise the weights
        self.weight_initialiser()
        
        
    def weight_initialiser(self):
        """
        All layers but the final layer are initilaised from uniform
        distributions [-1/sqrt(f) , 1/sqrt(f)] where f is the fan-in of the layer
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        
        """
        The final layer is initialised from uniform distribution
        [-3*10^-3, 3*10^-3]
        """
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
        
    def forward(self, states):
        # Actor network maps states to action probabilities 
        if states.dim() == 1:
            states = torch.unsqueeze(states, 0)

        states = self.bn(states)
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        return self.tanh(self.fc3(x))
    
    
class Critic(nn.Module):
    """Critic (Value) Model"""
    def __init__(self, state_size, action_size, seed, hidden1=400, hidden2=300):
        super(Critic, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1 + action_size, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.bn = nn.BatchNorm1d(state_size)
        self.weight_initialiser()
    
    def weight_initialiser(self):
        """
        All layers but the final layer are initilaised from uniform
        distributions [-1/sqrt(f) , 1/sqrt(f)] where f is the fan-in of the layer
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        
        """
        The final layer is initialised from uniform distribution
        [-3*10^-3, 3*10^-3]
        """
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, states, actions):
        # Critic network maps (state, action) pairs to Q-values
        # For Batch Normalisation
        if states.dim() == 1:
            states = torch.unsqueeze(states, 0)
        
        states = self.bn(states)
        xs = F.relu(self.fc1(states))
        x = torch.cat([xs, actions], dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    