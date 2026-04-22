import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Hyperparameters
ENV_NAME = "CartPole-v1"
EPISODES = 1000
GAMMA = 0.99
LR = 0.001
EPISODES = 500

# Might be subject to change.
env = gym.make(ENV_NAME, render_mode="rgb_array")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
print(f'Number of states: {n_states}, Number of actions: {n_actions}')

class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)


def reinforce(env, episodes=EPISODES, gamma=GAMMA, lr=LR):
    policy_net = PolicyNetwork(n_states, n_actions)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    
    reward_history = [] 

    for episode in range(episodes):
        log_probs = []
        rewards = []
        state = env.reset()[0]
        done = False

        while not done:
            state = torch.FloatTensor(state)
            action_probs = policy_net(state)
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[action])
            log_probs.append(log_prob)
            next_state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            state = next_state

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        loss = -sum(log_probs[i] * returns[i] for i in range(len(returns)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        total_reward = sum(rewards)
        reward_history.append(total_reward)

        if episode % 100 == 0: # Might tweak it to take average.
            print(f'Episode {episode}: Total Reward: {total_reward}')

    
    return policy_net, reward_history

# Run the training and capture the history
reinforce_agent, reward_history = reinforce(env, episodes=EPISODES)

np.save("reinforce_rewards.npy", np.array(reward_history)) # I think? Not sure.
print("Saved reinforce_rewards.npy")

plt.figure(figsize=(10, 6))
plt.plot(reward_history, color='blue', alpha=0.8)
# Test later.
window = 20
smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
plt.plot(range(window - 1, len(reward_history)), smoothed, label=f"{window}-ep average")
plt.title('REINFORCE Learning Curve on CartPole-v1')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.savefig("reinforce_rewards.png")
plt.show()