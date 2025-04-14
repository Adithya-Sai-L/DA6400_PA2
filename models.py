import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Policy network that maps states to action probabilities"""
    def __init__(self, state_dim, action_dim, layers, activation='ReLU', batch_norm=False, dropout=0.0):
        super(PolicyNetwork, self).__init__()
        all_layers = [state_dim] + layers + [action_dim]
        self.network = nn.Sequential()
        for i in range(len(all_layers) - 1):
            self.network.add_module(f"fc{i}", nn.Linear(all_layers[i], all_layers[i + 1]))            
            if i < len(all_layers) - 2:
                if batch_norm:
                    self.network.add_module(f"bn{i}", nn.BatchNorm1d(all_layers[i + 1]))
                if dropout > 0:
                    self.network.add_module(f"dropout{i}", nn.Dropout(dropout))
                self.network.add_module(f"activation{i}", getattr(nn, activation)())
        
    def forward(self, x):
        x = self.network(x)
        return F.softmax(x, dim=-1)
    
    def select_action(self, state):
        """Select action based on current policy and state"""
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob

class ValueNetwork(nn.Module):
    """Value network for baseline estimation"""
    def __init__(self, state_dim, layers, activation='ReLU', batch_norm=False, dropout=0.0):
        super(ValueNetwork, self).__init__()
        all_layers = [state_dim] + layers + [1]
        self.network = nn.Sequential()
        for i in range(len(all_layers) - 1):
            self.network.add_module(f"fc{i}", nn.Linear(all_layers[i], all_layers[i + 1]))
            if i < len(all_layers) - 2:
                if batch_norm:
                    self.network.add_module(f"bn{i}", nn.BatchNorm1d(all_layers[i + 1]))
                if dropout > 0:
                    self.network.add_module(f"dropout{i}", nn.Dropout(dropout))
                self.network.add_module(f"activation{i}", getattr(nn, activation)())
        
    def forward(self, x):
        x = self.network(x)
        return x
