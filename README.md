# LunarLander-v3 — Reinforcement Learning Comparison

A comparison of three classic RL algorithms solving the [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment from Gymnasium.

## Algorithms

| Algorithm | File | Episodes | Key idea |
|---|---|---|---|
| DQN | `DQN_agent.py` | 600 | Off-policy Q-learning with experience replay and a target network |
| REINFORCE | `Reinforce_agent.py` | 1500 | On-policy Monte Carlo policy gradient with return normalisation |
| A2C | `A2C_agent.py` | 1500 | On-policy actor-critic; critic reduces variance of the policy gradient |

## Project Structure

```
.
├── DQN_agent.py        # Deep Q-Network agent
├── Reinforce_agent.py  # REINFORCE policy-gradient agent
├── A2C_agent.py        # Advantage Actor-Critic agent
└── compare.py          # Training runner + reward plotting
```

## Setup

```bash
pip install gymnasium[box2d] torch numpy matplotlib
```

> On some systems you may also need `swig` and `box2d-py` for the Box2D physics engine.

## Usage

**Train and compare all agents:**

```bash
python compare.py
```

Each agent saves its reward history to a `.npy` file (`dqn_rewards.npy`, `reinforce_rewards.npy`, `a2c_rewards.npy`). If a file already exists, `compare.py` loads it instead of retraining.

**Train a single agent:**

```bash
python DQN_agent.py
python Reinforce_agent.py
python A2C_agent.py
```

## Hyperparameters

### DQN
| Parameter | Value |
|---|---|
| Hidden size | 128 |
| Replay buffer | 50 000 |
| Batch size | 64 |
| Learning rate | 0.0005 |
| Discount (γ) | 0.99 |
| ε start / min / decay | 1.0 / 0.01 / 0.995 |
| Target net update | every 10 episodes |

### REINFORCE & A2C
| Parameter | Value |
|---|---|
| Hidden size | 128 |
| Learning rate | 0.001 |
| Discount (γ) | 0.99 |

## Notes

- DQN trains significantly faster (600 vs 1500 episodes) thanks to experience replay.
- REINFORCE is the simplest baseline but highest-variance due to full Monte Carlo returns.
- A2C adds a learned value baseline (critic) to reduce variance compared to REINFORCE.
