import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from typing import NamedTuple
from pathlib import Path

import matplotlib.pyplot as plt


"ContinuousBanditEnv-v0"

class Params(NamedTuple):
    env_name: str = "MountainCarContinuous-v0"  # Gym environment name
    num_episodes: int = 1000                  # Total episodes
    gamma: float = 0.9                       # Discounting rate
    alpha: float = 0.001                        # Learning rate
    epsilon: float = 0.1                      # Exploration probability
    batch_size: int = 32                      # Batch size for mini-batch processing
    use_cuda: bool = False                    # Flag to use CUDA
    seed: int = 42                            # Define a seed for reproducible results
    feature_dim: int = None                   # Will be set based on environment state dimension
    eval_episodes: int = 100                  # Number of episodes for evaluation
    eval_frequency: int = 50                  # Frequency of evaluation
    log_interval: int = 100                   # Interval for logging
    savefig_folder: Path = Path("./pset2/plots/")  # Root folder where plots are saved

# Initialize parameters
params = Params()
random.seed(params.seed)

class LinearTD0():
    def __init__(self, params):
        self.env_name = params.env_name
        self.gamma = params.gamma
        self.alpha = params.alpha
        self.epsilon = params.epsilon
        self.num_episodes = params.num_episodes
        self.eval_episodes = params.eval_episodes
        self.eval_frequency = params.eval_frequency
        self.log_interval = params.log_interval
        self.seed = params.seed

        # Set up environment
        self.env = gym.make(self.env_name)
        self.env.reset(seed=self.seed)
        
        # Determine feature dimension from environment
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.weights = torch.randn(self.observation_space.shape[0], dtype=torch.float32) * 0.0
        
        # Set up device for computation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = self.weights.to(self.device)
    def fixed_policy(self,state):
        # """
        # A heuristic policy for MountainCarContinuous-v0.
        # Uses the car's velocity to decide the force:
        #   - if state[1] < 0, return a moderate force (0.5) to push right.
        #   - otherwise, return maximum force (1.0) to push right.
        # """
        # # Ensure action is a numpy array of the right shape.
        # if state[1] < 0:
        #     return np.array([0.5])
        # else:
        #     return np.array([1.0])
        """
        A more dynamic policy for MountainCarContinuous-v0.
        
        Uses both position and velocity:
          - If the car is far left (position < -0.6), use a gentle force 
            to gradually build momentum.
          - As the car moves to the center (position between -0.6 and 0.0),
            apply moderate force.
          - Near the goal (position >= 0.0), push with maximum force.
        
        Also considers velocity: if not moving much, push a bit harder to get moving.
        Adds small noise for exploration.
        """
        position, velocity = state
        # Default force determination based on position
        if position < -0.6:
            force = 0.4 if abs(velocity) > 0.01 else 0.5
        elif position < -0.2:
            force = 0.6
        elif position < 0.0:
            force = 0.8
        else:
            force = 1.0
        
        # Introduce slight randomness for exploration
        noise = np.random.uniform(-0.05, 0.05)
        # Ensure the action is within the environment’s valid range.
        action = np.clip(force + noise, self.env.action_space.low[0], self.env.action_space.high[0])
        return np.array([action])
    def train(self):
        """
        Train using Semi-gradient TD(0) with linear function approximation.
        The update rule is:
           w <- w + alpha [r + gamma*V(s') - V(s)] x
        where x is the feature vector (here, raw state).
        """
        episode_rewards = []
        for episode in tqdm(range(self.num_episodes), desc="Training episodes"):
            # Reset environment at the beginning of each episode
            state, _ = self.env.reset(seed=self.seed)
            total_reward = 0.0
            done = False
            
            # Run one episode
            while not done:
                # Use the fixed policy to select an action
                action = self.fixed_policy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                # Convert states to tensors and push them to the device
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                
                # Compute value estimates using the linear approximator
                V_state = torch.dot(self.weights, state_tensor)
                V_next = torch.dot(self.weights, next_state_tensor) if not done else 0.0
                
                # TD error: δ = r + γ V(next_state) - V(state)
                td_error = reward + self.gamma * (V_next - V_state)
                
                # Semi-gradient update: w <- w + α δ x
                with torch.no_grad():
                    self.weights += self.alpha * td_error * state_tensor
                
                # Move to next state
                state = next_state

            episode_rewards.append(total_reward)
            
            # Logging
            if (episode + 1) % self.log_interval == 0:
                print(f"Episode {episode+1}/{self.num_episodes}, Reward: {total_reward:.2f}")
        
        return episode_rewards

    def evaluate(self):
        """
        Evaluate the learned value function with a fixed policy over several episodes.
        """
        eval_rewards = []
        for _ in range(self.eval_episodes):
            state, _ = self.env.reset(seed=self.seed)
            total_reward = 0.0
            done = False
            while not done:
                action = self.fixed_policy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state
            eval_rewards.append(total_reward)
        avg_reward = np.mean(eval_rewards)
        print(f"Evaluation over {self.eval_episodes} episodes: Average Reward: {avg_reward:.2f}")
        return avg_reward

    def plot_rewards(self, rewards):
        """
        Plot the training rewards per episode.
        """
        os.makedirs(params.savefig_folder, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Rewards over Episodes")
        plt.grid(True)
        filename = f"{params.savefig_folder}/training_rewards.png"
        plt.savefig(filename)
        plt.close()

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 128)
        self.fc4 = nn.Linear(128, output_dim)
    
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x



class MLPTD0():
    def __init__(self, params):
        self.env_name = params.env_name
        self.gamma = params.gamma
        self.alpha = params.alpha
        self.epsilon = params.epsilon
        self.num_episodes = params.num_episodes
        self.eval_episodes = params.eval_episodes
        self.eval_frequency = params.eval_frequency
        self.log_interval = params.log_interval
        self.seed = params.seed

        # Set up environment
        self.env = gym.make(self.env_name)
        self.env.reset(seed=self.seed)
        
        # Determine feature dimension from environment
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.model = MLP(self.observation_space.shape[0], 32, self.action_space.shape[0])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
    
    def fixed_policy(self, state):
        """
        A heuristic policy for MountainCarContinuous-v0.
        Uses the car's velocity to decide the force:
          - if state[1] < 0, return a moderate force (0.5) to push right.
          - otherwise, return maximum force (1.0) to push right.
        """
        # # Ensure action is a numpy array of the right shape.
        # if state[1] < 0:
        #     return np.array([0.5])
        # else:
        #     return np.array([1.0])
        """
        A more dynamic policy for MountainCarContinuous-v0.
        
        Uses both position and velocity:
          - If the car is far left (position < -0.6), use a gentle force 
            to gradually build momentum.
          - As the car moves to the center (position between -0.6 and 0.0),
            apply moderate force.
          - Near the goal (position >= 0.0), push with maximum force.
        
        Also considers velocity: if not moving much, push a bit harder to get moving.
        Adds small noise for exploration.
        """
        position, velocity = state
        # Default force determination based on position
        if position < -0.6:
            force = 0.4 if abs(velocity) > 0.01 else 0.5
        elif position < -0.2:
            force = 0.6
        elif position < 0.0:
            force = 0.8
        else:
            force = 1.0
        
        # Introduce slight randomness for exploration
        noise = np.random.uniform(-0.05, 0.05)
        # Ensure the action is within the environment’s valid range.
        action = np.clip(force + noise, self.env.action_space.low[0], self.env.action_space.high[0])
        return np.array([action])
    def train(self):
        """
        Train using Semi-gradient TD(0) with linear function approximation.
        The update rule is:
           θ ← θ + α [r + γ V(s') - V(s)] ∇θV(s)
        where V(s) is computed by the network.
        """
        episode_rewards = []
        for episode in tqdm(range(self.num_episodes), desc="Training episodes"):
            # Reset environment at the beginning of each episode
            state, _ = self.env.reset(seed=self.seed)
            total_reward = 0.0
            done = False
            
            # Run one episode
            while not done:
                # Use the fixed policy to select an action
                action = self.fixed_policy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                # Convert states to tensors and push them to the device
                state_tensor = torch.tensor(state, dtype=torch.float32)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                
                # Compute value estimates using the linear approximator
                V_state = self.model(state_tensor)
                V_next = self.model(next_state_tensor) if not done else 0
                
                # TD error: δ = r + γ V(next_state) - V(state)
                td_error = reward + self.gamma * (V_next - V_state)
                
                loss = 0.5 * td_error**2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Move to next state
                state = next_state

            episode_rewards.append(total_reward)
            
            # Logging
            if (episode + 1) % self.log_interval == 0:
                print(f"Episode {episode+1}/{self.num_episodes}, Reward: {total_reward:.2f}")
        
        return episode_rewards
    def plot_rewards(self, rewards):
        os.makedirs(params.savefig_folder, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("MLP TD(0) Training Rewards over Episodes")
        plt.grid(True)
        filename = f"{params.savefig_folder}/mlp_training_rewards.png"
        plt.savefig(filename)
        plt.close()
    def visualize_value_function(self, resolution=30):
        """Visualize the learned value function across the state space"""
        x = np.linspace(-1.2, 0.6, resolution)  # Position range
        y = np.linspace(-0.07, 0.07, resolution)  # Velocity range
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(resolution):
            for j in range(resolution):
                state = np.array([X[i, j], Y[i, j]])
                state_tensor = torch.FloatTensor(state)
                Z[i, j] = self.model(state_tensor).item()
        
        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, Z, 20, cmap='viridis')
        plt.colorbar(label='Value Estimate')
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.title('Value Function Approximation')
        plt.savefig(f"{params.savefig_folder}/value_function.png")

    



# Example usage:
if __name__ == "__main__":
    # agent = LinearTD0(params)
    # rewards = agent.train()
    # agent.plot_rewards(rewards)
    # agent.evaluate()
    agent = MLPTD0(params)
    rewards = agent.train()
    agent.plot_rewards(rewards)
    agent.visualize_value_function()

