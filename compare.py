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
        
def smooth(rewards, window=20):
    kernel   = np.ones(window) / window
    smoothed = np.convolve(rewards, kernel, mode='valid')
    # Pad the front so lengths match for plotting
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, smoothed])        
    
def plot_comparison(results: dict, window=20, save_path="comparison_rewards.png"):
    
    colors = {"DQN": "#2563EB", "REINFORCE": "#7C3AED", "A2C": "#D95F42"}
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    # Left: smoothed reward curves 
    ax = axes[0]
    for label, rewards in results.items():
        episodes = np.arange(1, len(rewards) + 1)
        ax.plot(episodes, rewards, alpha=0.18, color=colors[label])
        ax.plot(episodes, smooth(rewards, window),
                label=label, color=colors[label], linewidth=2)
 
    ax.axhline(200, color="gray", linestyle="--", linewidth=0.8, label="Solved (200)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Cumulative reward", fontsize=12)
    ax.set_title(f"Reward curves — LunarLander-v3\n({window}-episode rolling average)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2)
 
    # Right: final-50-episode bar chart 
    ax = axes[1]
    labels, means, stds = [], [], []
    for label, rewards in results.items():
        last50 = rewards[-50:]
        labels.append(label)
        means.append(np.mean(last50))
        stds.append(np.std(last50))
 
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6, width=0.5,
                  color=[colors[l] for l in labels], alpha=0.85)
 
    # Annotate bar tops
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05,
                f"{m:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
 
    ax.axhline(200, color="gray", linestyle="--", linewidth=0.8, label="Solved (200)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Mean reward (last 50 episodes)", fontsize=12)
    ax.set_title("Final performance comparison\n(mean ± std, last 50 episodes)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.2)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved {save_path}")
 
 
def print_summary(results: dict):
    print("\n")
    print(f"{'Algorithm':<12} {'Mean (last 50)':>16} {'Std':>10} {'Max ep':>10}")
    for label, rewards in results.items():
        last50 = rewards[-50:]
        print(f"{label:<12} {np.mean(last50):>16.2f} {np.std(last50):>10.2f} {np.max(rewards):>10.2f}")
    
    
if __name__ == "__main__":
    from DQN_agent import train as train_dqn
    from Reinforce_agent import train as train_reinforce # honestly, not sure if this is the right import or not.
    from A2C_agent import train as train_a2c
    
    results = {
        "DQN":       load_train("dqn",       train_dqn),
        "REINFORCE": load_train("reinforce",  train_reinforce),
        "A2C":       load_train("a2c",        train_a2c),
    }
 
    print_summary(results)
    plot_comparison(results, window=20)