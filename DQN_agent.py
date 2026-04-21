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


class QNetwork(nn.Module):
    def _init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.net =nn.Sequential(
            nn.Linear(state_dim, hidden), nn.RELU(),
            nn.Linear(hidden, hidden), nn.RELU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self,x):
        return self.net(x)
    
class ReplayBuffer:
    def __init__(self):
        pass