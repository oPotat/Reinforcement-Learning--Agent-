import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Hyperparameters
ENV_NAME = "LunarLander-v3"
HIDDEN_SIZE = 64
BUFFER_SIZE = 10000
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LR = 0.001              # Learning Rate
BATCH_SIZE = 64
GAMMA = 0.99

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        """Initializes QNetwork.

        Args:
            state_dim (int): Dimension of state space.
            action_dim (int): Number of discrete actions the agent can do
            hidden (int): Number of Neuons.
        """
        super().__init__()
        self.net =nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self,x):
        return self.net(x)
    

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        # Add later
        """_summary_

        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            next_state (_type_): _description_
            done (function): _description_
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        # Add later
        """_summary_

        Args:
            batch_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return(
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)),
            )
    
    def size (self):
        return len(self.buffer)    
        
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        
        self.actioin_dim = action_dim
        self.epsilon = EPSILON_START
        
        self.q_net = QNetwork(state_dim, action_dim)
        self.target = QNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.q_net.state_dict())
        self.target.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()
     
     # Epsilon Greedy for actions.   
    def select_action (self, state):
        if random.random < self.epsilon: # Exploration
            return random.randrange(self.actioin_dim)
        with torch.no_grad(): # Exploration fail so use exploitation
            q = self.q_net(torch.FloatTensor(state).unsqueeze(0)) 
        return q.argmax().item()
    
    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        
        # Current Q Values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1).squeeze(1))
        
        # Target Q values (Bellman)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + GAMMA * max_next_q * (1-dones)
        
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    # Shrinks the epsilon while ensuring it doesn't go below 0.01 so that the agent can keep exploriong.
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        
    # Ensures stability by creating 2 networks.    
    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
    
def train():
    pass

if __name__ == "__main__":
    train()