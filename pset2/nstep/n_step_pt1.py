from collections import defaultdict
import gymnasium as gym
import numpy as np
from pathlib import Path
from typing import NamedTuple
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    savefig_folder: Path  # Root folder where plots are saved

params = Params(
    total_episodes=10,
    learning_rate=0.8,
    gamma=0.9,
    epsilon=0.1,
    map_size=5,
    seed=42, #12345
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    savefig_folder=Path("./plots/nstep_pt1/"),
)

params = params._replace(action_size=4)
params = params._replace(state_size=48)

class NstepTD:
    def __init__(self, n_step, env, params):
        self.n = n_step
        self.gamma = params.gamma
        self.epsilon = params.epsilon
        self.alpha = params.learning_rate
        self.num_episodes = params.total_episodes
        self.state_space = np.array(list(range(params.state_size)))
        self.action_space = np.array(list(range(params.action_size)))


        self.v_values = np.zeros(params.state_size)
        self.deltas = []
        self.env = env
        self.model = self.env.unwrapped.P
    def fixed_policy(self, state):
        # Check next_state for each action to determine valid movement
        action_right = 1
        action_down = 2
        action_left = 3
        action_up = 0
        
        # Check right: get next_state from transition
        transitions = self.model[state][action_right]
        # Assuming deterministic environment; take the first transition tuple
        _, next_state, _, _ = transitions[0]
        if next_state != state:
            return action_right
        
        # Check down
        transitions = self.model[state][action_down]
        _, next_state, _, _ = transitions[0]
        if next_state != state:
            return action_down
        
        # Check left
        transitions = self.model[state][action_left]
        _, next_state, _, _ = transitions[0]
        if next_state != state:
            return action_left
        
        # Check up
        transitions = self.model[state][action_up]
        _, next_state, _, _ = transitions[0]
        if next_state != state:
            return action_up
        
        # Default (shouldn't reach here if all actions are valid)
        return action_left
    def train_episode(self):
        T = float("inf")
        tau = -self.n - 1
        t = 0
        state, _ = self.env.reset()
        rollout = [(state, 0)]
        while tau != T - 1:
            if t < T:
                action = self.fixed_policy(state)
                next_state, reward, done, _, _ = self.env.step(action)
                rollout.append((next_state,reward))
                state = next_state
                if done:
                    T = t + 1
            tau = t - self.n - 1 
            if tau >= 0:
                G = 0
                for i in range(tau+1, min(tau+self.n+1, T)):
                    _, r_reward = rollout[i]
                    G += (self.gamma ** (i-tau-1))*r_reward
                if tau + self.n < T :
                    r_state, _ = rollout[tau + self.n]
                    G += (self.gamma**self.n)*self.v_values[r_state]
                # Update the value function
                r_state, _ = rollout[tau]
                self.v_values[r_state] += self.alpha*(G - self.v_values[r_state])                
            t += 1
    def train(self):
        for _ in range(self.num_episodes):
            prev_v_values = self.v_values.copy()
            self.train_episode()
            self.deltas.append(np.abs(prev_v_values - self.v_values))
    
    def get_v_values(self):
        return self.v_values

    def get_deltas(self):
        return self.deltas
    

def run_experiment(n_values):
    V_n_values = {}
    delta_n_values = {}
    env = gym.make('CliffWalking-v0')
    for n in n_values:
        td_agent = NstepTD(n, env, params)
        td_agent.train()
        V_n_values[n] = td_agent.get_v_values()
        delta_n_values[n] = td_agent.get_deltas()
    return V_n_values, delta_n_values

def plot_V_values(V_s_values, delta_values, n_values, savefig_folder):
    # Create the directory if it doesn't exist
    os.makedirs(savefig_folder, exist_ok=True)
    
    # 1. Plot individual V-values heatmaps as separate figures
    for n in n_values:
        fig, ax = plt.subplots(figsize=(10, 8))
        v_map = V_s_values[n].reshape(4, 12)
        im = sns.heatmap(v_map, ax=ax, cmap="coolwarm", annot=True, fmt=".2f", 
                      cbar=True, cbar_kws={'label': 'Value'})
        ax.set_title(f"V(s) for n={n}", fontsize=14)
        ax.set_xlabel("Column position")
        ax.set_ylabel("Row position")
        
        # Mark cliff positions with X
        for j in range(1, 11):
            ax.text(j + 0.5, 3.5, 'X', color='black', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Mark start and goal positions
        ax.text(0.5, 3.5, 'S', color='black', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(11.5, 3.5, 'G', color='black', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Save this individual heatmap
        heatmap_filename = f"V_values_n_{n}.png"
        heatmap_save_path = os.path.join(savefig_folder, heatmap_filename)
        plt.tight_layout()
        plt.savefig(heatmap_save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap for n={n} to {heatmap_save_path}")
        plt.close(fig)
    
    # 2. Plot convergence analysis as a separate figure
    fig, ax = plt.subplots(figsize=(12, 8))
    for n in n_values:
        # Calculate mean absolute error for each episode
        mean_errors = [np.mean(delta) for delta in delta_values[n]]
        ax.plot(range(1, len(mean_errors) + 1), mean_errors, label=f"n={n}", marker='o', markersize=4, linewidth=2)
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Mean Absolute Error", fontsize=12)
    ax.set_title("Convergence Analysis of N-step TD Learning", fontsize=14)
    ax.set_yscale('log')  # Log scale often shows convergence patterns better
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save convergence plot
    conv_filename = f"Convergence_Analysis_{'_'.join(map(str, n_values))}.png"
    conv_save_path = os.path.join(savefig_folder, conv_filename)
    plt.tight_layout()
    plt.savefig(conv_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved convergence analysis to {conv_save_path}")
    plt.close(fig)

    # 3. Create a difference visualization if we have multiple n values
    if len(n_values) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate the difference between largest and smallest n
        min_n = min(n_values)
        max_n = max(n_values)
        diff = V_s_values[max_n] - V_s_values[min_n]
        diff_map = diff.reshape(4, 12)
        
        im = sns.heatmap(diff_map, ax=ax, cmap="PiYG", center=0, annot=True, fmt=".2f", 
                      cbar=True, cbar_kws={'label': 'Value Difference'})
        ax.set_title(f"Value Difference: n={max_n} minus n={min_n}", fontsize=14)
        ax.set_xlabel("Column position")
        ax.set_ylabel("Row position")
        
        # Mark cliff positions with X
        for j in range(1, 11):
            ax.text(j + 0.5, 3.5, 'X', color='black', ha='center', va='center', fontsize=12)
            
        # Mark start and goal positions
        ax.text(0.5, 3.5, 'S', color='black', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(11.5, 3.5, 'G', color='black', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Save difference plot
        diff_filename = f"Value_Difference_{min_n}_vs_{max_n}.png"
        diff_save_path = os.path.join(savefig_folder, diff_filename)
        plt.tight_layout()
        plt.savefig(diff_save_path, dpi=300, bbox_inches='tight')
        print(f"Saved difference plot to {diff_save_path}")
        plt.close(fig)

# Example usage:
n_values = [1,3,5,7,11,13,17,19,50]  # Experiment with these n values

V_s_values, delta_s_values = run_experiment(n_values)
plot_V_values(V_s_values, delta_s_values, n_values, params.savefig_folder)