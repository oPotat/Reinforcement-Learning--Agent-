import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Actor(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)


# Critic network
class Critic(nn.Module):
    def __init__(self, n_states):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
def actor_critic(env, episodes=1000, gamma=0.99, lr=0.01):
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    actor = Actor(n_states, n_actions)
    critic = Critic(n_states)

    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

    rewards_history = []   
    
    for episode in range(episodes):
        state = env.reset()[0]
        done = False

        log_probs = []
        rewards = []
        values = []

        while not done:
            state_tensor = torch.FloatTensor(state)
            action_probs = actor(state_tensor)
            value = critic(state_tensor)

            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[action])

            next_state, reward, done, _, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            state = next_state

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        values_tensor = torch.cat(values)
        advantage = returns - values_tensor.detach()

        actor_loss = -sum(log_probs[i] * advantage[i] for i in range(len(advantage)))
        critic_loss = (returns - values_tensor).pow(2).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        episode_reward = sum(rewards)
        rewards_history.append(episode_reward)  

        if episode % 100 == 0:
            print(f"[A2C] Episode {episode}: Total Reward = {episode_reward:.2f}")

    return rewards_history  

#
def train():
    env = gym.make("LunarLander-v3")
    rewards = actor_critic(env, episodes=1000, gamma=0.99, lr=0.01)
    env.close()

    np.save("a2c_rewards.npy", np.array(rewards))
    print("Saved a2c_rewards.npy")

    return rewards


if __name__ == "__main__":
    train()