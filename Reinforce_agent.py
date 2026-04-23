import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
ENV_NAME = "LunarLander-v3"
EPISODES = 1500
GAMMA    = 0.99
LR       = 0.001


class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)


def reinforce(env, episodes=EPISODES, gamma=GAMMA, lr=LR):
    n_states  = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = PolicyNetwork(n_states, n_actions)
    optimizer  = optim.Adam(policy_net.parameters(), lr=lr)
    reward_history = []

    for episode in range(episodes):
        log_probs = []
        rewards   = []
        state, _  = env.reset()
        done      = False

        while not done:
            state_t      = torch.FloatTensor(state)
            action_probs = policy_net(state_t)
            action       = torch.multinomial(action_probs, 1).item()
            log_prob     = torch.log(action_probs[action])
            log_probs.append(log_prob)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            state = next_state

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -sum(log_probs[i] * returns[i] for i in range(len(returns)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reward_history.append(sum(rewards))

        if episode % 100 == 0:
            print(f"[REINFORCE] Episode {episode}: Total Reward = {sum(rewards):.2f}")

    return reward_history  


def train():               
    env = gym.make(ENV_NAME)
    rewards = reinforce(env)
    env.close()
    np.save("reinforce_rewards.npy", np.array(rewards))
    print("Saved reinforce_rewards.npy")
    return rewards


if __name__ == "__main__":
    train()