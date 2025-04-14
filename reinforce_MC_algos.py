import torch
from torch import optim
from models import PolicyNetwork, ValueNetwork
import numpy as np
import torch.nn.functional as F

class REINFORCEWithoutBaseline:
    """REINFORCE algorithm without baseline (Type 1)"""
    def __init__(self, state_dim, action_dim, gamma=0.99, optimizer='Adam', lr=0.001, layers=[128, 64], activation='ReLU', batch_norm=False, dropout=0.0):
        self.gamma = gamma  # discount factor
        self.policy = PolicyNetwork(state_dim, action_dim, layers, activation, batch_norm, dropout)
        self.optimizer = getattr(optim,optimizer)(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []
        self.states = []
        
    def select_action(self, state):
        action, log_prob = self.policy.select_action(state)
        self.log_probs.append(log_prob)
        self.states.append(state)
        return action
    
    def store_reward(self, reward):
        self.rewards.append(reward)
        
    def calculate_returns(self):
        """Calculate returns Gₜ for each timestep"""
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        # Normalize returns for training stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
        return returns
    
    def update(self):
        """Update policy parameters using REINFORCE algorithm"""
        returns = self.calculate_returns()
        
        # Calculate policy loss: -log(π(Aₜ|Sₜ)) * Gₜ
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G) # since the question in the assignment does not dicount the returns based on time_step
        
        policy_loss = torch.cat(policy_loss).sum()
        
        # Update policy parameters
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear memory
        self.log_probs = []
        self.rewards = []
        self.states = []


class REINFORCEWithBaseline:
    """REINFORCE algorithm with baseline (Type 2)"""
    def __init__(self, state_dim, action_dim, gamma=0.99, optimizer='Adam', lr=0.001, layers=[128, 64], activation='ReLU', batch_norm=False, dropout=0.0):
        self.gamma = gamma  # discount factor
        self.policy = PolicyNetwork(state_dim, action_dim, layers, activation, batch_norm, dropout)
        self.value = ValueNetwork(state_dim, layers, activation, batch_norm, dropout)
        self.optimizer_policy = getattr(optim, optimizer)(self.policy.parameters(), lr=lr)
        self.optimizer_value = getattr(optim, optimizer)(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []
        self.states = []
        
    def select_action(self, state):
        action, log_prob = self.policy.select_action(state)
        self.log_probs.append(log_prob)
        self.states.append(state)
        return action
    
    def store_reward(self, reward):
        self.rewards.append(reward)
        
    def calculate_returns(self):
        """Calculate returns Gₜ for each timestep"""
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        return returns
    
    def update(self):
        """Update policy and value parameters using REINFORCE with baseline"""
        returns = self.calculate_returns()
        
        # Convert states to tensor
        states = torch.FloatTensor(np.array(self.states))
        
        # Calculate values for all states: V(s; Φ)
        values = self.value(states)
        values = values.squeeze()
        
        # Calculate advantages: Aₜ = Gₜ - V(Sₜ; Φ)
        advantages = returns - values.detach()
        
        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
        
        # Calculate value loss: MSE(V(Sₜ; Φ), Gₜ)
        value_loss = F.mse_loss(values, returns)
        
        # Calculate policy loss: -log(π(Aₜ|Sₜ)) * Aₜ
        policy_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        
        policy_loss = torch.cat(policy_loss).sum()
        
        # Update value network
        self.optimizer_value.zero_grad()
        value_loss.backward(retain_graph=True)
        self.optimizer_value.step()
        
        # Update policy network
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()
        
        # Clear memory
        self.log_probs = []
        self.rewards = []
        self.states = []
