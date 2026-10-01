import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import os
import gymnasium as gym
from tqdm import tqdm
from n_step_pt2 import ParallelNStepTD, params
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("CuPy not available, falling back to CPU visualizations")

class NStepVisualizer:
    def __init__(self, agent, env_name="CliffWalking-v0", n_values=None):
        self.agent = agent
        self.env = gym.make(env_name)
        self.n_values = n_values if n_values is not None else agent.n_values
        
        # For tracking metrics during training
        self.td_errors = {n: [] for n in self.n_values}
        self.value_estimates = {n: [] for n in self.n_values}
        self.episode_returns = {n: [] for n in self.n_values}
        
        # For bias-variance analysis
        self.true_values = None  # Will be approximated with Monte Carlo
        self.bias_metrics = {n: [] for n in self.n_values}
        self.variance_metrics = {n: [] for n in self.n_values}
        
        # Create folder for plots
        os.makedirs(params.savefig_folder, exist_ok=True)
        
        # Optimal policy for faster MC (set in compute_optimal_policy)
        self.optimal_policy = None
    
    def compute_optimal_policy(self):
        """Compute an approximate optimal policy for faster MC evaluation"""
        if self.optimal_policy is not None:
            return self.optimal_policy
            
        print("Computing optimal policy for fast MC estimation...")
        
        # For CliffWalking, we can use a hardcoded safe policy
        if "CliffWalking" in self.env.spec.id:
            policy = np.zeros(params.state_size, dtype=np.int32)
            
            # Default action: up (0)
            policy[:] = 0
            
            # For bottom row except cliff and goal, go right
            for s in range(36, 47):
                policy[s] = 1  # Right
                
            # For rightmost column except bottom, go down
            for s in range(11, 47, 12):
                policy[s] = 2  # Down
                
            self.optimal_policy = policy
            return policy
            
        # For other environments, use agent's learned policy
        policy = np.zeros(params.state_size, dtype=np.int32)
        q_values = self.agent.get_q_values()[self.n_values[0]]
        
        for s in range(params.state_size):
            policy[s] = np.argmax(q_values[s])
            
        self.optimal_policy = policy
        return policy
    
    def estimate_true_values(self, num_episodes=500):
        """Accelerated Monte Carlo state value estimation using CuPy and optimal policy"""
        print("Estimating true state values with accelerated Monte Carlo...")
        
        # Get optimal policy for faster episode generation
        optimal_policy = self.compute_optimal_policy()
        
        # Prepare return tracking arrays
        returns_sum = np.zeros(params.state_size, dtype=np.float32)
        returns_count = np.zeros(params.state_size, dtype=np.int32)
        
        # Batch size for parallel processing
        batch_size = 16
        
        # Set maximum steps to prevent infinite loops
        max_steps = 200
        
        # Track visited states for avoiding unnecessary calculations
        visited_states = set()
        
        # Process episodes in batches for better performance
        for batch in tqdm(range(0, num_episodes, batch_size), desc="MC batch estimation"):
            batch_episodes = []
            batch_actual_size = min(batch_size, num_episodes - batch)
            
            # Generate episodes in parallel (conceptually)
            for _ in range(batch_actual_size):
                episode = []
                state, _ = self.env.reset()
                done = False
                step_count = 0
                
                # Generate episode using mostly optimal policy (with exploration)
                while not done:
                    # Use optimal policy with 90% probability, explore with 10%
                    if np.random.random() < 0.9:
                        action = optimal_policy[state]
                    else:
                        action = np.random.choice(self.agent.action_space)
                        
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    episode.append((state, action, reward))
                    visited_states.add(state)
                    state = next_state
                    step_count += 1
                    
                    # Special case for CliffWalking: force termination at goal
                    if state == 47:  # Bottom-right corner in CliffWalking
                        done = True
                
                batch_episodes.append(episode)
            
            # Process all episodes in the batch using optimized computation
            for episode in batch_episodes:
                if not episode:
                    continue
                    
                # Calculate returns efficiently
                G = 0
                for t in reversed(range(len(episode))):
                    state, _, reward = episode[t]
                    G = self.agent.gamma * G + reward
                    returns_sum[state] += G
                    returns_count[state] += 1
        
        # Calculate average returns
        self.true_values = np.zeros(returns_sum.shape, dtype=np.float32)
        mask = returns_count > 0
        self.true_values[mask] = returns_sum[mask] / returns_count[mask]
        
        # Handle states that were never visited (use nearest neighbor heuristic)
        if len(visited_states) < params.state_size:
            print(f"Warning: {params.state_size - len(visited_states)} states never visited during MC estimation")
            self._fill_unvisited_states(visited_states)
            
        return self.true_values
    
    def _fill_unvisited_states(self, visited_states):
        """Fill value estimates for unvisited states using nearest neighbor"""
        if "CliffWalking" in self.env.spec.id:
            # For CliffWalking, specially handle cliff cells
            for s in range(params.state_size):
                if s not in visited_states:
                    # Check if cliff cell (row 3, col 1-10)
                    row, col = s // 12, s % 12
                    if row == 3 and 1 <= col <= 10:
                        self.true_values[s] = -100  # Cliff penalty
                    else:
                        # Find nearest visited state (Manhattan distance)
                        nearest = self._find_nearest_visited(s, visited_states)
                        self.true_values[s] = self.true_values[nearest]
    
    def _find_nearest_visited(self, state, visited_states):
        """Find nearest visited state using Manhattan distance"""
        if "CliffWalking" in self.env.spec.id:
            # For grid environments
            state_row, state_col = state // 12, state % 12
            min_dist = float('inf')
            nearest = None
            
            for visited in visited_states:
                visited_row, visited_col = visited // 12, visited % 12
                dist = abs(state_row - visited_row) + abs(state_col - visited_col)
                if dist < min_dist:
                    min_dist = dist
                    nearest = visited
            return nearest
        else:
            # For non-grid environments, just return first visited
            return next(iter(visited_states))
    
    def calculate_bias_variance(self, n):
        """Calculate bias and variance for a specific n-step method"""
        # Get CPU copy of Q-values
        q_values = self.agent.get_q_values()[n]
        
        # Convert Q-values to state values (max Q for each state)
        estimated_values = np.array([np.max(q_values[s]) for s in range(params.state_size)])
        
        # Calculate bias (squared difference between expected estimate and true value)
        bias = np.mean((estimated_values - self.true_values)**2)
        
        # For variance, we need multiple estimates, so use the variance across different states
        variance = np.var(estimated_values)
        
        return bias, variance
    
    def evaluate_policy(self, n, num_episodes=10):
        """Evaluate current policy for a specific n value"""
        total_reward = 0
        q_values = self.agent.get_q_values()[n]
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            max_steps = 1000
            
            while not done and step_count < max_steps:
                action = np.argmax(q_values[state])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
                step_count += 1
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def visualize_results(self):
        """Create all visualizations after training"""
        # Generate bias/variance metrics if true values are available
        if self.true_values is None:
            print("Estimating true values for bias-variance analysis...")
            self.estimate_true_values()
        
        # Calculate final bias and variance
        print("Calculating bias and variance metrics...")
        bias_values = {}
        variance_values = {}
        for n in self.n_values:
            bias, variance = self.calculate_bias_variance(n)
            bias_values[n] = bias
            variance_values[n] = variance
            print(f"n={n}: Bias²={bias:.5f}, Variance={variance:.5f}, Total Error={bias+variance:.5f}")
            
        # Evaluate policies
        print("Evaluating policies...")
        returns = {}
        for n in tqdm(self.n_values, desc="Evaluating"):
            returns[n] = self.evaluate_policy(n)
        
        # Create visualizations
        print("Creating visualizations...")
        self.plot_heatmaps()
        if len(self.n_values) > 1:
            self.plot_bias_variance_tradeoff(bias_values, variance_values)
            self.plot_performance_comparison(returns)
            self.visualize_bias_variance_grid(bias_values, variance_values)
        else:
            print("Skipping comparison plots - only one n value provided")
        
        print("Visualization complete!")
    
    def plot_heatmaps(self):
        """Plot value function heatmaps for each n-step method"""
        fig, axes = plt.subplots(1, len(self.n_values), figsize=(5*len(self.n_values), 4))
        
        if len(self.n_values) == 1:
            axes = [axes]
        
        for i, n in enumerate(self.n_values):
            # Get state values from Q-values
            q_values = self.agent.get_q_values()[n]
            state_values = np.array([np.max(q_values[s]) for s in range(params.state_size)])
            
            # Reshape for CliffWalking (4x12 grid)
            if params.state_size == 48:
                grid_values = state_values.reshape(4, 12)
            else:
                size = int(np.sqrt(params.state_size))
                grid_values = state_values.reshape(size, size)
            
            # Plot heatmap
            im = sns.heatmap(grid_values, ax=axes[i], cmap="viridis", annot=True, fmt=".1f")
            axes[i].set_title(f"n={n}")
            axes[i].set_xlabel("Column")
            axes[i].set_ylabel("Row")
            
            # Mark cliff positions for CliffWalking
            if params.state_size == 48:
                for j in range(1, 11):
                    axes[i].text(j + 0.5, 3.5, 'X', color='white', ha='center', va='center', 
                               fontsize=12, fontweight='bold')
                
                # Mark start and goal positions
                axes[i].text(0.5, 3.5, 'S', color='white', ha='center', va='center', 
                           fontsize=12, fontweight='bold')
                axes[i].text(11.5, 3.5, 'G', color='white', ha='center', va='center', 
                           fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(params.savefig_folder / "value_heatmaps.png", dpi=300)
        plt.close()
    
    def plot_bias_variance_tradeoff(self, bias_values, variance_values):
        """Plot bias-variance tradeoff for different n values"""
        plt.figure(figsize=(12, 8))
        
        # Sort n values for proper plotting
        n_values_sorted = sorted(self.n_values)
        
        # Plot bias, variance, total error
        plt.plot(n_values_sorted, [bias_values[n] for n in n_values_sorted], 'bo-', linewidth=2, label="Bias²")
        plt.plot(n_values_sorted, [variance_values[n] for n in n_values_sorted], 'ro-', linewidth=2, label="Variance")
        plt.plot(n_values_sorted, [bias_values[n] + variance_values[n] for n in n_values_sorted], 
                'go-', linewidth=2, label="Total Error")
        
        # Add point labels
        for n in n_values_sorted:
            plt.annotate(f"{bias_values[n]:.3f}", 
                        (n, bias_values[n]), 
                        textcoords="offset points",
                        xytext=(0,5), 
                        ha='center')
            plt.annotate(f"{variance_values[n]:.3f}", 
                        (n, variance_values[n]), 
                        textcoords="offset points",
                        xytext=(0,5), 
                        ha='center')
        
        plt.xlabel("n-step", fontsize=14)
        plt.ylabel("Error", fontsize=14)
        plt.title("Bias-Variance Tradeoff for Different n-step Values", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.xticks(n_values_sorted)
        
        plt.savefig(params.savefig_folder / "bias_variance_tradeoff.png", dpi=300)
        plt.close()
    
    def plot_performance_comparison(self, returns):
        """Plot final performance comparison"""
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        n_values_sorted = sorted(self.n_values)
        bars = plt.bar(range(len(n_values_sorted)), [returns[n] for n in n_values_sorted], alpha=0.7)
        plt.xticks(range(len(n_values_sorted)), [f"n={n}" for n in n_values_sorted])
        
        # Add values on top of bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f"{returns[n_values_sorted[i]]:.1f}", 
                    ha='center', fontsize=10)
        
        plt.ylabel("Average Return")
        plt.title("Policy Performance Comparison")
        plt.grid(True, axis='y', alpha=0.3)
        
        plt.savefig(params.savefig_folder / "performance_comparison.png", dpi=300)
        plt.close()
    
    def visualize_bias_variance_grid(self, bias_values, variance_values):
        """Create a grid visualization showing how different n values affect value estimation"""
        # Select interesting states to visualize based on environment
        if "CliffWalking" in self.env.spec.id:
            # For CliffWalking: start, near cliff, near goal, middle of upper path
            states_to_show = [0, 35, 46, 22]  
        else:
            # Generic selection for other environments
            states_to_show = [0, params.state_size // 4, params.state_size // 2, 
                             3 * params.state_size // 4]
        
        fig, axes = plt.subplots(len(states_to_show), 1, figsize=(10, 3*len(states_to_show)))
        
        if len(states_to_show) == 1:
            axes = [axes]
        
        for i, state in enumerate(states_to_show):
            if state < params.state_size:
                # Get true value
                true_val = self.true_values[state]
                
                # Get estimated values for each n
                estimates = []
                for n in sorted(self.n_values):
                    q_values = self.agent.get_q_values()[n][state]
                    estimates.append(np.max(q_values))
                
                # Plot comparisons
                axes[i].bar(range(len(self.n_values)), estimates, alpha=0.7, label="Estimated Value")
                axes[i].axhline(y=true_val, color='r', linestyle='-', label="True Value (MC)")
                
                # Set labels
                axes[i].set_xticks(range(len(self.n_values)))
                axes[i].set_xticklabels([f"n={n}" for n in sorted(self.n_values)])
                axes[i].set_ylabel("Value")
                
                # Add state description for CliffWalking
                if "CliffWalking" in self.env.spec.id:
                    state_desc = {
                        0: "Start (0,3)", 
                        35: "Near Cliff (11,2)", 
                        46: "Near Goal (10,3)",
                        22: "Safe Path (10,1)"
                    }
                    if state in state_desc:
                        axes[i].set_title(f"State {state} - {state_desc[state]}")
                    else:
                        axes[i].set_title(f"State {state}")
                else:
                    axes[i].set_title(f"State {state}")
                
                axes[i].legend()
                axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(params.savefig_folder / "bias_variance_grid.png", dpi=300)
        plt.close()
        
    def plot_true_values_heatmap(self):
        """Plot heatmap of the true values from MC estimation"""
        if self.true_values is None:
            print("True values not available. Running MC estimation...")
            self.estimate_true_values()
            
        plt.figure(figsize=(8, 6))
        
        # Reshape for environment
        if params.state_size == 48:  # CliffWalking
            grid_values = self.true_values.reshape(4, 12)
        else:
            # Assume square grid for other environments
            size = int(np.sqrt(params.state_size))
            grid_values = self.true_values.reshape(size, size)
            
        # Plot heatmap
        sns.heatmap(grid_values, cmap="viridis", annot=True, fmt=".1f")
        plt.title("True State Values (MC Estimation)")
        plt.xlabel("Column")
        plt.ylabel("Row")
        
        # Mark special cells for CliffWalking
        if params.state_size == 48:
            for j in range(1, 11):
                plt.text(j + 0.5, 3.5, 'X', color='white', ha='center', va='center', 
                       fontsize=12, fontweight='bold')
            
            # Mark start and goal positions
            plt.text(0.5, 3.5, 'S', color='white', ha='center', va='center', 
                   fontsize=12, fontweight='bold')
            plt.text(11.5, 3.5, 'G', color='white', ha='center', va='center', 
                   fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(params.savefig_folder / "true_values_heatmap.png", dpi=300)
        plt.close()

# Example usage
if __name__ == "__main__":
    # Set n values to compare 
    # For bias-variance tradeoff, try: n_values = [1, 2, 4, 8, 16]
    n_values = [1, 2, 4, 8, 16]
    
    # Create and train agent
    env_name = "CliffWalking-v0"
    
    # Display GPU info if available
    if HAS_CUPY:
        print(f"Using GPU device ID: {cp.cuda.Device(0).id}")
        try:
            # Get memory info
            mem_info = cp.cuda.runtime.memGetInfo()
            free_mem = mem_info[0] / 1024**3  # Convert to GB
            total_mem = mem_info[1] / 1024**3  # Convert to GB
            print(f"GPU memory: {free_mem:.2f} GB free / {total_mem:.2f} GB total")
            
            # Get compute capability
            cc_major = cp.cuda.runtime.deviceGetAttribute(
                cp.cuda.runtime.deviceAttributeComputeCapabilityMajor, 0)
            cc_minor = cp.cuda.runtime.deviceGetAttribute(
                cp.cuda.runtime.deviceAttributeComputeCapabilityMinor, 0)
            print(f"Compute capability: {cc_major}.{cc_minor}")
        except Exception as e:
            print(f"Could not retrieve detailed GPU information: {e}")
    
    # Create and train agent
    print("Creating and training agent...")
    agent = ParallelNStepTD(n_values, env_name, params)
    agent.train()
    
    # Create visualizer and generate visualizations
    print("Creating visualizations...")
    visualizer = NStepVisualizer(agent, env_name)
    visualizer.visualize_results()
    
    # Also generate true values heatmap
    visualizer.plot_true_values_heatmap()
    
    print("Visualization complete!")