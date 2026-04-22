import numpy as np
import matplotlib.pyplot as plt
import os

# Hyperparameters
WINDOW_SIZE = 20

def load_train(name, train_fn):
    path = f"{name}_rewards.npy"
    if os.path.exists(path):
        print(f"Loading {path}...")
        return np.load(path)
    else:
        print(f"{path} not found. Go train.")
    
def smooth(rewards, window=WINDOW_SIZE):
    pass

if __name__ == "__main__":
    from DQN_agent import train as train_dqn
    from Reinforce_agent import reinforce as train_reinforce
    from A2C_agent import empty
    