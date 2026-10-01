import os
import numpy as np
import cupy as cp
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import NamedTuple
import random

class Params(NamedTuple):
    """Parameters for the N-Step TD learning algorithm"""
    state_size: int = 48          # CliffWalking environment has 48 states
    action_size: int = 4          # CliffWalking has 4 actions (up, right, down, left)
    learning_rate: float = 0.1    # Alpha learning rate
    gamma: float = 0.95           # Discount factor
    epsilon: float = 0.1          # Exploration rate
    total_episodes: int = 500     # Total episodes for training
    savefig_folder: Path = Path("./figures")  # Folder to save figures

class SimpleNStepTD:
    def __init__(self, n_values, env_name, params):
        """
        Initialize the agent with the given n values.
        
        Args:
            n_values: List of n-step values to use (e.g. [1, 2, 4, 8])
            env_name: Name of the gymnasium environment
            params: Parameters for the agent
        """
        self.n_values = n_values
        self.env_name = env_name
        self.env = gym.make(env_name)
        
        # Set parameters
        self.alpha = params.learning_rate
        self.gamma = params.gamma
        self.epsilon = params.epsilon
        self.num_episodes = params.total_episodes
        self.state_size = params.state_size
        self.action_size = params.action_size
        self.params = params
        
        # Define action space
        self.action_space = list(range(self.action_size))
        
        # Initialize Q-values with optimistic initialization
        self.cu_q_values = {n: cp.zeros((self.state_size, self.action_size), dtype=cp.float32) + 0.1 
                           for n in self.n_values}
        
        # For dynamic n-step adaptation
        self.td_error_history = {n: cp.zeros((self.state_size, self.action_size, 10), dtype=cp.float32) 
                                for n in self.n_values}
        self.error_index = cp.zeros((self.state_size, self.action_size), dtype=cp.int32)
        self.active_n = cp.full((self.state_size, self.action_size), self.n_values[0], dtype=cp.int32)
        
        # Parameters for dynamic adaptation
        self.variance_threshold_low = 0.1
        self.variance_threshold_high = 0.5
        self.adaptation_period = 5
        
        # Create folder for saving figures
        os.makedirs(params.savefig_folder, exist_ok=True)
        
        # For tracking metrics
        self.rewards_history = []
        self.steps_history = []
        
        # Debug info
        print(f"Initialized SimpleNStepTD with n values: {n_values}")
        print(f"Environment: {env_name}, State size: {self.state_size}, Action size: {self.action_size}")
    
    def select_action(self, state):
        """Select an action using epsilon-greedy policy with dynamic n selection"""
        if random.random() < self.epsilon:
            # Exploration - random action
            return random.choice(self.action_space)
        else:
            # Exploitation - use active n for this state
            # Get the most common active n across actions for this state
            active_n_values = self.active_n[state].get()
            n = int(np.bincount(active_n_values).argmax())
            
            # Choose best action according to Q-values for this n
            return int(cp.argmax(self.cu_q_values[n][state]).get())
    
    def generate_trajectory(self):
        """Generate a single trajectory (episode) using current policy"""
        trajectory = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
        
        state, _ = self.env.reset()
        done = False
        max_steps = 1000  # Reasonable limit for environments like CliffWalking
        step_count = 0
        total_reward = 0
        
        while not done and step_count < max_steps:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['next_states'].append(next_state)
            trajectory['dones'].append(done)
            
            total_reward += reward
            state = next_state
            step_count += 1
        
        # Mark as done if max steps reached
        if step_count >= max_steps and not done:
            print(f"Hit max steps ({max_steps}) in trajectory generation")
            if trajectory['states']:
                trajectory['dones'][-1] = True
        
        # Store metrics
        self.rewards_history.append(total_reward)
        self.steps_history.append(step_count)
        
        return trajectory
    
    def calculate_n_step_returns(self, trajectory):
        """Calculate n-step returns for all n values in a trajectory"""
        n_step_returns = {}
        
        for n in self.n_values:
            returns_list = []
            states = trajectory['states']
            actions = trajectory['actions']
            rewards = trajectory['rewards']
            next_states = trajectory['next_states']
            dones = trajectory['dones']
            
            traj_length = len(states)
            if traj_length == 0:
                n_step_returns[n] = []
                continue
            
            # For each timestep in the trajectory
            for t in range(traj_length):
                # Calculate n-step return
                n_step_return = 0
                
                # Add rewards for up to n steps
                for i in range(min(n, traj_length - t)):
                    n_step_return += (self.gamma ** i) * rewards[t + i]
                    
                    # Stop if trajectory ended
                    if dones[t + i]:
                        break
                
                # Add bootstrapped value if needed
                bootstrap_idx = t + min(n, traj_length - t) - 1
                if bootstrap_idx < traj_length - 1 and not dones[bootstrap_idx]:
                    bootstrap_state = next_states[bootstrap_idx]
                    bootstrap_value = float(cp.max(self.cu_q_values[n][bootstrap_state]).get())
                    n_step_return += (self.gamma ** min(n, traj_length - t)) * bootstrap_value
                
                # Store the return
                returns_list.append((states[t], actions[t], n_step_return))
            
            n_step_returns[n] = returns_list
        
        return n_step_returns
    
    def update_active_n_values(self):
        """Update which n value is used for each state-action pair based on TD error variance"""
        print("Updating active n values based on TD error variance...")
        
        # Calculate variance for each state-action pair across all n values
        variances = {}
        for n in self.n_values:
            # Calculate variance of TD errors
            variances[n] = cp.var(self.td_error_history[n], axis=2)
        
        # Find best n for each state-action pair
        for s in range(self.state_size):
            for a in range(self.action_size):
                # Get variance for this state-action pair for each n
                var_values = [variances[n][s, a].get() for n in self.n_values]
                
                # Skip if not enough data
                if max(var_values) < 0.001:
                    continue
                
                # Target variance - neither too high nor too low
                target_var = (self.variance_threshold_low + self.variance_threshold_high) / 2
                
                # Find n value with variance closest to target
                best_n_idx = min(range(len(var_values)), 
                                key=lambda i: abs(var_values[i] - target_var))
                best_n = self.n_values[best_n_idx]
                
                # Update active n for this state-action pair
                self.active_n[s, a] = best_n
        
        # Visualize the distribution of active n values
        n_counts = {}
        active_n_cpu = self.active_n.get()
        for n in self.n_values:
            n_counts[n] = np.sum(active_n_cpu == n)
        
        print("Active n distribution:")
        for n, count in n_counts.items():
            print(f"  n={n}: {count} state-action pairs ({count/(self.state_size*self.action_size)*100:.1f}%)")
    
    def train(self):
        """Train the agent using n-step TD learning"""
        for episode in tqdm(range(self.num_episodes), desc="Training episodes"):
            # Generate single trajectory
            trajectory = self.generate_trajectory()
            
            # Calculate n-step returns
            n_step_returns = self.calculate_n_step_returns(trajectory)
            
            # Update Q-values and track TD errors
            for n, returns in n_step_returns.items():
                if returns:
                    states, actions, n_returns = zip(*returns)
                    states_gpu = cp.array(states, dtype=cp.int32)
                    actions_gpu = cp.array(actions, dtype=cp.int32)
                    n_returns_gpu = cp.array(n_returns, dtype=cp.float32)
                    
                    # Get current Q-values
                    current_q = self.cu_q_values[n][states_gpu, actions_gpu]
                    
                    # Calculate TD error
                    td_errors = n_returns_gpu - current_q
                    
                    # Update Q-values
                    self.cu_q_values[n][states_gpu, actions_gpu] += self.alpha * td_errors
                    
                    # Store TD errors in history for variance calculation
                    for i in range(len(states)):
                        s, a = states[i], actions[i]
                        idx = int(self.error_index[s, a].get())
                        self.td_error_history[n][s, a, idx] = td_errors[i]
                        self.error_index[s, a] = (idx + 1) % 10  # Circular buffer
            
            # Periodically update active n values
            if (episode + 1) % self.adaptation_period == 0:
                self.update_active_n_values()
            
            # Decay epsilon (optional)
            if self.epsilon > 0.01:
                self.epsilon *= 0.995
                
        return self.get_q_values()
    
    def get_q_values(self):
        """Get a CPU copy of Q-values for visualization or saving"""
        return {n: self.cu_q_values[n].get() for n in self.n_values}
    
    def get_active_n_distribution(self):
        """Get the distribution of active n values for visualization"""
        n_distribution = {}
        active_n_cpu = self.active_n.get()
        for n in self.n_values:
            n_distribution[n] = np.zeros(self.state_size)
            for s in range(self.state_size):
                n_count = np.sum(active_n_cpu[s] == n)
                n_distribution[n][s] = n_count / self.action_size
        
        return n_distribution
    
    def evaluate(self):
        """Evaluate the agent's performance"""
        results = {}
        
        for n in self.n_values:
            total_reward = 0
            num_eval_episodes = 10
            
            # Get CPU copy of Q-values for this n
            q_values = self.cu_q_values[n].get()
            
            for _ in range(num_eval_episodes):
                state, _ = self.env.reset()
                done = False
                episode_reward = 0
                steps = 0
                
                while not done and steps < 200:
                    steps += 1
                    action = np.argmax(q_values[state])
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    state = next_state
                
                total_reward += episode_reward
            
            results[n] = total_reward / num_eval_episodes
        
        return results
    
    def plot_training_progress(self):
        """Plot training progress metrics"""
        plt.figure(figsize=(12, 5))
        
        # Plot rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards_history, 'b-')
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        # Plot steps
        plt.subplot(1, 2, 2)
        plt.plot(self.steps_history, 'r-')
        plt.title('Episode Steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.params.savefig_folder / "training_progress.png")
        plt.close()
    
    def visualize_n_distribution(self):
        """Visualize the distribution of n values"""
        n_dist = self.get_active_n_distribution()
        
        plt.figure(figsize=(12, 5))
        for i, n in enumerate(self.n_values):
            plt.subplot(1, len(self.n_values), i+1)
            
            # For CliffWalking, reshape to 4x12 grid
            if self.state_size == 48:
                n_map = n_dist[n].reshape(4, 12)
            else:
                # For other environments, try square grid or simple bar chart
                grid_size = int(np.sqrt(self.state_size))
                if grid_size * grid_size == self.state_size:
                    n_map = n_dist[n].reshape(grid_size, grid_size)
                else:
                    # Fallback for non-square state spaces
                    plt.bar(range(self.state_size), n_dist[n])
                    plt.title(f"n={n} usage")
                    continue
            
            # Plot heatmap for grid environments
            sns.heatmap(n_map, cmap="viridis", vmin=0, vmax=1)
            plt.title(f"n={n} usage")
        
        plt.tight_layout()
        plt.savefig(self.params.savefig_folder / "n_distribution.png")
        plt.close()
        
    def visualize_policy(self):
        """Visualize the learned policy"""
        # Use the most common n value for each state
        active_n_cpu = self.active_n.get()
        
        plt.figure(figsize=(8, 6))
        
        if self.state_size == 48:  # CliffWalking
            # For each state, find most common n
            policy = np.zeros(self.state_size, dtype=int)
            for s in range(self.state_size):
                # Get most common n for this state
                n_counts = {}
                for n in self.n_values:
                    n_counts[n] = np.sum(active_n_cpu[s] == n)
                
                best_n = max(n_counts, key=n_counts.get)
                # Get best action using this n
                policy[s] = np.argmax(self.cu_q_values[best_n][s].get())
            
            # Reshape policy for visualization
            policy_grid = policy.reshape(4, 12)
            
            # Plot policy grid
            action_symbols = ['↑', '→', '↓', '←']  # [up, right, down, left]
            
            # Create a text-based visualization
            plt.figure(figsize=(12, 5))
            ax = plt.gca()
            ax.set_xlim(0, 12)
            ax.set_ylim(0, 4)
            ax.set_xticks(np.arange(12) + 0.5)
            ax.set_yticks(np.arange(4) + 0.5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            # Draw grid
            for i in range(13):
                plt.axvline(i, color='black', alpha=0.3)
            for i in range(5):
                plt.axhline(i, color='black', alpha=0.3)
            
            # Fill in policy and state information
            for row in range(4):
                for col in range(12):
                    state = row * 12 + col
                    
                    # Special cases
                    if row == 3 and 1 <= col <= 10:  # Cliff
                        plt.text(col + 0.5, 3.5, 'X', ha='center', va='center', color='red', fontsize=12)
                    elif row == 3 and col == 0:  # Start
                        plt.text(col + 0.5, 3.5, 'S', ha='center', va='center', color='green', fontsize=12)
                    elif row == 3 and col == 11:  # Goal
                        plt.text(col + 0.5, 3.5, 'G', ha='center', va='center', color='blue', fontsize=12)
                    else:
                        # Show policy
                        action = policy_grid[row, col]
                        plt.text(col + 0.5, 3.5 - row - 0.5, action_symbols[action], 
                                ha='center', va='center', fontsize=12)
            
            plt.title("Learned Policy")
            plt.savefig(self.params.savefig_folder / "policy_visualization.png")
            plt.close()
        
        # Visualize Q-values as heatmap
        for n in self.n_values:
            plt.figure(figsize=(8, 6))
            q_values = self.cu_q_values[n].get()
            state_values = np.max(q_values, axis=1)
            
            if self.state_size == 48:  # CliffWalking
                value_grid = state_values.reshape(4, 12)
                sns.heatmap(value_grid, cmap="viridis", annot=True, fmt=".1f")
                plt.title(f"State Values (n={n})")
            else:
                plt.bar(range(self.state_size), state_values)
                plt.title(f"State Values (n={n})")
            
            plt.savefig(self.params.savefig_folder / f"q_values_n{n}.png")
            plt.close()

# Example usage
if __name__ == "__main__":
    # Define parameters
    params = Params(
        state_size=48,           # CliffWalking has 48 states
        action_size=4,           # CliffWalking has 4 actions
        learning_rate=0.1,       # Alpha
        gamma=0.95,              # Discount factor
        epsilon=0.1,             # Exploration rate
        total_episodes=500,      # Total episodes
        savefig_folder=Path("./figures")  # Folder to save figures
    )
    
    # Create directories
    os.makedirs(params.savefig_folder, exist_ok=True)
    
    # Set range of n values to choose from
    n_values = [1, 2, 4, 8, 16]
    
    # Create and train agent
    env_name = "CliffWalking-v0"
    agent = SimpleNStepTD(n_values, env_name, params)
    q_values = agent.train()
    
    # Visualize results
    agent.plot_training_progress()
    agent.visualize_n_distribution()
    agent.visualize_policy()
    
    # Evaluate agent
    results = agent.evaluate()
    print("\nEvaluation Results:")
    for n, avg_reward in results.items():
        print(f"n={n}: Average reward = {avg_reward:.2f}")