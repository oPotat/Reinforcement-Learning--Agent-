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
            np.array(states), 
            np.array(actions), 
            np.array(rewards), 
            np.array(next_states), 
            np.array(dones)
            )
    
    def size (self):
        return len(self.buffer)    
        
class DQNAgent:
    def __init__(self):
        pass
    
def train():
    pass

if __name__ == "__main__":
    train()