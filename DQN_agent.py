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
HIDDEN_SIZE = 128
BUFFER_SIZE = 50000
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LR = 0.0005              # Learning Rate
BATCH_SIZE = 64
GAMMA = 0.99            # Discount Factor
SEED = 42
EPISODES = 600
TARGET_UPDATE_C = 10    # hard-update target net every C episodes
MAX_STEPS = 1000


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden= HIDDEN_SIZE):
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
        """Stores transition in replay buffer.

        Args:
            state (np.darray): Current State.
            action (int): Action agent can do.
            reward (float): Given reward.
            next_state (np.darray): Result after action/
            done (bool): If episode ended after transition.
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Randomly samples a batch from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple: Containing states, actions, rewards, next states, and dones. as Torch Tensors.
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
    
    def __len__ (self):
        return len(self.buffer)    
        
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        
        self.buffer = ReplayBuffer()

        self.action_dim = action_dim
        self.epsilon = EPSILON_START
        
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()
     
     # Epsilon Greedy for actions.   
    def select_action (self, state):
        if random.random() <= self.epsilon: # Exploration
            return random.randrange(self.action_dim)   
        with torch.no_grad(): # Exploration fail so use exploitation
            q = self.q_net(torch.FloatTensor(state).unsqueeze(0)) 
        return q.argmax().item()
    
    def update(self):
        """
        Samples batch from memory and performs one step of gradient descent.
        
        Computes loss between current Q-values and the expected Q-values 
        then updates main network weights to minimize loss.
        """
        if len(self.buffer) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        
        # Current Q Values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
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
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
 
    env = gym.make(ENV_NAME)
    state_dim  = env.observation_space.shape[0]   # 8 for LunarLander
    action_dim = env.action_space.n               # 4 for LunarLander: 0: do nothing 1: fire left engine 2: fire main engine 3: fire right engine
 
    agent = DQNAgent(state_dim, action_dim)
    rewards_history = []
 
    for episode in range(1, EPISODES + 1):
        state, _ = env.reset(seed=SEED)
        total_reward = 0
 
        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
 
            agent.buffer.add(state, action, reward, next_state, float(done))
            agent.update()
 
            state = next_state
            total_reward += reward
            if done:
                break
 
        agent.decay_epsilon()
        if episode % TARGET_UPDATE_C == 0:
            agent.sync_target()
 
        rewards_history.append(total_reward)
 
        if episode % 50 == 0:
            avg = np.mean(rewards_history[-50:])
            print(f"[DQN] Episode {episode:4d} | Avg reward (last 50): {avg:7.2f} | ε={agent.epsilon:.3f}")
 
    env.close()
 
    # Save rewards for comparison
    np.save("dqn_rewards.npy", np.array(rewards_history))
    print("Saved dqn_rewards.npy")
 
    return rewards_history

if __name__ == "__main__":
    train()