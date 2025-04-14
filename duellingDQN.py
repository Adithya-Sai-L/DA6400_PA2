import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Type-1 Dueling DQN Network
class DuelingDQNType1(nn.Module):
    def __init__(self, state_size, action_size, layers, activation='ReLU', batch_norm=False, dropout=0.0):
        super(DuelingDQNType1, self).__init__()
        
        # Shared feature layer
        self.feature_layer = nn.Sequential()
        self.feature_layer.add_module("fc0", nn.Linear(state_size, layers[0]))
        if batch_norm:
            self.feature_layer.add_module("bn0", nn.BatchNorm1d(layers[0]))
        if dropout > 0:
            self.feature_layer.add_module("dropout0", nn.Dropout(dropout))
        self.feature_layer.add_module("activation0", getattr(nn, activation)())
        for i in range(1, len(layers) - 1):
            self.feature_layer.add_module(f"fc{i}", nn.Linear(layers[i-1], layers[i]))
            if batch_norm:
                self.feature_layer.add_module(f"bn{i}", nn.BatchNorm1d(layers[i]))
            if dropout > 0:
                self.feature_layer.add_module(f"dropout{i}", nn.Dropout(dropout))
            self.feature_layer.add_module(f"activation{i}", getattr(nn, activation)())
                
        # Value stream
        self.value_stream = nn.Sequential()
        self.value_stream.add_module("value_fc", nn.Linear(layers[-2], layers[-1]))
        if batch_norm:
            self.value_stream.add_module("value_nv", nn.BatchNorm1d(layers[-1]))
        if dropout > 0:
            self.value_stream.add_module("value_dropout", nn.Dropout(dropout))
        self.value_stream.add_module("value_activation", getattr(nn, activation)())
        self.value_stream.add_module("value_output", nn.Linear(layers[-1], 1))
        
        # Advantage stream
        self.advantage_stream = nn.Sequential()
        self.advantage_stream.add_module("advantage_fc", nn.Linear(layers[-2], layers[-1]))
        if batch_norm:
            self.advantage_stream.add_module("advantage_nv", nn.BatchNorm1d(layers[-1]))
        if dropout > 0:
            self.advantage_stream.add_module("advantage_dropout", nn.Dropout(dropout))
        self.advantage_stream.add_module("advantage_activation", getattr(nn, activation)())
        self.advantage_stream.add_module("advantage_output", nn.Linear(layers[-1], action_size))
        
    def forward(self, state):
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Type-1 combining: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # This is the average advantage aggregation
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

# Type-2 Dueling DQN Network
class DuelingDQNType2(nn.Module):
    def __init__(self, state_size, action_size, layers, activation='ReLU', batch_norm=False, dropout=0.0):
        super(DuelingDQNType2, self).__init__()
        
        # Shared feature layer
        self.feature_layer = nn.Sequential()
        self.feature_layer.add_module("fc0", nn.Linear(state_size, layers[0]))
        if batch_norm:
            self.feature_layer.add_module("bn0", nn.BatchNorm1d(layers[0]))
        if dropout > 0:
            self.feature_layer.add_module("dropout0", nn.Dropout(dropout))
        self.feature_layer.add_module("activation0", getattr(nn, activation)())
        for i in range(1, len(layers) - 1):
            self.feature_layer.add_module(f"fc{i}", nn.Linear(layers[i-1], layers[i]))
            if batch_norm:
                self.feature_layer.add_module(f"bn{i}", nn.BatchNorm1d(layers[i]))
            if dropout > 0:
                self.feature_layer.add_module(f"dropout{i}", nn.Dropout(dropout))
            self.feature_layer.add_module(f"activation{i}", getattr(nn, activation)())
                
        # Value stream
        self.value_stream = nn.Sequential()
        self.value_stream.add_module(nn.Linear(layers[-2], layers[-1]))
        if batch_norm:
            self.value_stream.add_module(nn.BatchNorm1d(layers[-1]))
        if dropout > 0:
            self.value_stream.add_module(nn.Dropout(dropout))
        self.value_stream.add_module("activation", getattr(nn, activation)())
        self.value_stream.add_module("value_output", nn.Linear(layers[-1], 1))
        
        # Advantage stream
        self.advantage_stream = nn.Sequential()
        self.advantage_stream.add_module(nn.Linear(layers[-2], layers[-1]))
        if batch_norm:
            self.advantage_stream.add_module(nn.BatchNorm1d(layers[-1]))
        if dropout > 0:
            self.advantage_stream.add_module(nn.Dropout(dropout))
        self.advantage_stream.add_module("activation", getattr(nn, activation)())
        self.advantage_stream.add_module("advantage_output", nn.Linear(layers[-1], action_size))
        
    def forward(self, state):
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Type-2 combining: Q(s,a) = V(s) + (A(s,a) - max(A(s,a')))
        # This is the max advantage aggregation
        q_values = value + (advantage - advantage.max(dim=1, keepdim=True)[0])
        # max() function returns a tuple of two tensors:
        # The first element `` contains the maximum values
        # The second element [1] contains the indices where those maximum values are located
        return q_values

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)).unsqueeze(1),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)

# Dueling DQN Agent
class DuelingDQNAgent:
    def __init__(self, state_size, action_size, layers, activation='ReLU', batch_norm=False, dropout=0.0,dueling_type=1, 
                 lr=0.001, gamma=0.99, batch_size=64,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, target_update=10, use_double_dqn=False):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.memory = ReplayBuffer(memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_double_dqn = use_double_dqn
        # Initialize networks
        if dueling_type == 1:
            self.policy_net = DuelingDQNType1(state_size, action_size, layers, activation, batch_norm, dropout).to(self.device)
            self.target_net = DuelingDQNType1(state_size, action_size, layers, activation, batch_norm, dropout).to(self.device)
        else:
            self.policy_net = DuelingDQNType2(state_size, action_size, layers, activation, batch_norm, dropout).to(self.device)
            self.target_net = DuelingDQNType2(state_size, action_size, layers, activation, batch_norm, dropout).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.steps_done = 0
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            print(state.shape)
            q_values = self.policy_net(state)
            print(q_values.shape)
            return q_values.max(1)[1].item()
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: get actions from policy_net, values from target_net
                next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                # Standard DQN: get max values from target_net
                next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            
            # Compute target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to avoid exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        self.steps_done += 1
        self.update_epsilon()
        self.update_target_network()

