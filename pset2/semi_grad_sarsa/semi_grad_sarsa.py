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
import torch.nn.functional as F

import matplotlib.pyplot as plt

# Things to try: discritizing the action states

class Params(NamedTuple):
    env_name: str = "Acrobot-v1"  # Gym environment name
    num_episodes: int = 1000                  # Total episodes
    gamma: float = 0.99                       # Discounting rate
    alpha: float = 0.001                        # Learning rate
    epsilon: float = 0.1                      # Exploration probability
    beta: float = 0.01                          
    seed: int = 42                            # Define a seed for reproducible results
    feature_dim: int = None                   # Will be set based on environment state dimension
    eval_episodes: int = 100                  # Number of episodes for evaluation
    eval_frequency: int = 50                  # Frequency of evaluation
    log_interval: int = 100                   # Interval for logging
    savefig_folder: Path = Path("./pset2/semi_grad_sarsa/plots7/")  # Root folder where plots are saved

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc4 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.fc5 = nn.Linear(hidden_dim * 4 , hidden_dim * 2)
        self.fc6 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim , output_dim)
    
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = self.fc7(x)
        return x

class SemiGradSarsa():
    def __init__(self, params):
        self.env_name = params.env_name
        self.gamma = params.gamma
        self.alpha = params.alpha
        self.epsilon_start = 5.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 0.9995
        self.epsilon = self.epsilon_start
        self.beta = params.beta
        self.num_episodes = params.num_episodes
        self.eval_episodes = params.eval_episodes
        self.eval_frequency = params.eval_frequency
        self.log_interval = params.log_interval
        self.seed = params.seed
        self.savefig_folder = params.savefig_folder

        # Set up environment
        self.env = gym.make(self.env_name)
        self.env.reset(seed=self.seed)

        # Determine feature dimension from environment
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_values = MLP(self.observation_space.shape[0] + 1, 128 * 4, 1).to(self.device)
        self.optimizer = optim.Adam(self.q_values.parameters(), lr=self.alpha)  
        # self.avg_reward = 0
    def get_q_value(self, state, action):
        """Helper function to get Q value from network"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        action_tensor = torch.FloatTensor([action]).to(self.device)
        sa_pair = torch.cat([state_tensor, action_tensor]).unsqueeze(0)  # Add batch dimension
        return self.q_values(sa_pair).squeeze(0)  # Remove batch dimension from output
    def _policy(self, state):
        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                q_values = torch.zeros(self.action_space.n)
                for a in range(self.action_space.n):
                    q_values[a] = self.get_q_value(state, a)
                return q_values.argmax().item()

    def policy(self, state):
        with torch.no_grad():
            # Get Q-values for all actions
            q_values = torch.zeros(self.action_space.n, device=self.device)
            for a in range(self.action_space.n):
                q_values[a] = self.get_q_value(state, a)
            
            # Start with high temperature, gradually reduce
            # temperature = max(0.05, min(self.epsilon * 5, 2.0))  # Scale epsilon to reasonable temperature
            temperature = max(0.1, min(self.epsilon**0.8 * 10, 5.0))
            
            # Apply softmax with temperature
            probabilities = F.softmax(q_values / temperature, dim=0)
            
            # Sample action from probability distribution
            action = torch.multinomial(probabilities, 1).item()
            return action
    def _train_episode(self):
        state, _ = self.env.reset()
        action = self.policy(state)
        done = False
        rewards = []
        episode_reward = 0
        # torch.autograd.set_detect_anomaly(True)

        while not done:
            # Execute action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_reward += reward
            rewards.append(reward)

            # Get Next Action
            next_action = self.policy(next_state)

            current_q = self.get_q_value(state, action)

            with torch.no_grad():
                # EXPECTED SARSA: Calculate expected value across all actions
                expected_next_q = 0.0
                
                # Get Q-values for all actions in next state
                q_values = torch.zeros(self.action_space.n, device=self.device)
                for a in range(self.action_space.n):
                    q_values[a] = self.get_q_value(next_state, a)
                    
                # Calculate action probabilities under current policy (softmax)
                temperature = max(0.1, min(self.epsilon**0.8 * 10, 5.0))
                probabilities = F.softmax(q_values / temperature, dim=0)
                
                # Compute expected Q-value using probabilities
                for a in range(self.action_space.n):
                    expected_next_q += probabilities[a] * q_values[a]
                    
                # Compute target
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * expected_next_q
            # self.avg_reward += self.beta * td_error.item()

            # Calculate TD error using expected value
            td_error = target - current_q

            loss = 0.5 * td_error**2
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_values.parameters(), max_norm=1.0)
            self.optimizer.step()

            state = next_state
            action = next_action
        return rewards, episode_reward
    def _train(self):
        all_rewards = []
        episode_rewards = []  # Track per-episode totals
        
        for episode in tqdm(range(self.num_episodes), desc="Training Episodes: "):
            step_rewards, episode_reward = self.train_episode()
            all_rewards.extend(step_rewards)
            episode_rewards.append(episode_reward)
            
            # Periodically log average reward estimate
            if episode % self.log_interval == 0:
                print(f"Episode {episode}, Episode Reward: {episode_reward:.1f}")
            
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
        
        # Create and save plots
        self.plot_results(episode_rewards, all_rewards)
                
        return all_rewards, episode_rewards
    def train(self):
        all_rewards = []
        episode_rewards = []  # Track per-episode totals
        
        # Debugging statistics
        q_value_means = []     # Track mean Q-values
        q_value_maxes = []     # Track max Q-values
        td_errors = []         # Track TD errors
        gradient_norms = []    # Track gradient magnitudes
        action_counts = np.zeros(self.action_space.n)  # Track action distribution
        
        for episode in tqdm(range(self.num_episodes), desc="Training Episodes: "):
            # Track episode-specific stats
            episode_td_errors = []
            episode_q_values = []
            episode_actions = []
            
            # Train for one episode
            step_rewards, episode_reward = self.train_episode(
                track_stats=True, 
                stats_dict={
                    'td_errors': episode_td_errors,
                    'q_values': episode_q_values,
                    'actions': episode_actions
                }
            )
            
            # Collect rewards
            all_rewards.extend(step_rewards)
            episode_rewards.append(episode_reward)
            
            # Compute gradient norm
            total_norm = 0
            for p in self.q_values.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            gradient_norms.append(total_norm)
            
            # Compute statistics
            if episode_q_values:
                q_value_means.append(np.mean(episode_q_values))
                q_value_maxes.append(np.max(episode_q_values))
            if episode_td_errors:
                td_errors.append(np.mean(np.abs(episode_td_errors)))
            if episode_actions:
                for a in episode_actions:
                    action_counts[a] += 1
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Periodically log
            if episode % self.log_interval == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.1f}, Mean Q: {q_value_means[-1]:.4f}")
                
        # Create and save standard plots
        self.plot_results(episode_rewards, all_rewards)
        
        # Create and save debugging plots
        self.plot_debugging_graphs(
            episode_rewards, 
            q_value_means, 
            q_value_maxes, 
            td_errors, 
            gradient_norms, 
            action_counts
        )
        
        return all_rewards, episode_rewards

    def train_episode(self, track_stats=False, stats_dict=None):
        state, _ = self.env.reset()
        action = self.policy(state)
        done = False
        rewards = []
        episode_reward = 0
        
        while not done:
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_reward += reward
            rewards.append(reward)
            
            # Get Next Action
            next_action = self.policy(next_state)
            
            # Track actions if requested
            if track_stats and 'actions' in stats_dict:
                stats_dict['actions'].append(action)
            
            # Get Q-values
            current_q = self.get_q_value(state, action)
            
            # Track Q-values if requested
            if track_stats and 'q_values' in stats_dict:
                stats_dict['q_values'].append(current_q.item())
            
            # with torch.no_grad():
            #     next_q = self.get_q_value(next_state, next_action)
            #     if done:
            #         target = reward
            #     else:
            #         target = reward + self.gamma * next_q
            
            with torch.no_grad():
                # EXPECTED SARSA: Calculate expected value across all actions
                expected_next_q = 0.0
                
                # Get Q-values for all actions in next state
                q_values = torch.zeros(self.action_space.n, device=self.device)
                for a in range(self.action_space.n):
                    q_values[a] = self.get_q_value(next_state, a)
                    
                # Calculate action probabilities under current policy (softmax)
                temperature = max(0.1, min(self.epsilon**0.8 * 10, 5.0))
                probabilities = F.softmax(q_values / temperature, dim=0)
                
                # Compute expected Q-value using probabilities
                for a in range(self.action_space.n):
                    expected_next_q += probabilities[a] * q_values[a]
                    
                # Compute target
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * expected_next_q

            # Calculate TD error
            td_error = target - current_q
            
            # Track TD errors if requested
            if track_stats and 'td_errors' in stats_dict:
                stats_dict['td_errors'].append(td_error.item())
            
            # Update network
            loss = 0.5 * td_error**2
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_values.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update state and action
            state = next_state
            action = next_action
            
        return rewards, episode_reward

    def plot_debugging_graphs(self, episode_rewards, q_value_means, q_value_maxes, td_errors, gradient_norms, action_counts):
        """Create additional debugging plots"""
        os.makedirs(self.savefig_folder, exist_ok=True)
        
        # Plot Q-value statistics
        plt.figure(figsize=(10, 5))
        plt.plot(q_value_means, label='Mean Q-value')
        plt.plot(q_value_maxes, label='Max Q-value')
        plt.xlabel('Episode')
        plt.ylabel('Q-value')
        plt.title('Q-value Statistics During Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.savefig_folder, 'q_value_stats.png'))
        plt.close()
        
        # Plot TD errors
        plt.figure(figsize=(10, 5))
        plt.plot(td_errors)
        plt.xlabel('Episode')
        plt.ylabel('Mean Absolute TD Error')
        plt.title('TD Error During Training')
        plt.grid(True)
        plt.savefig(os.path.join(self.savefig_folder, 'td_errors.png'))
        plt.close()
        
        # Plot gradient norms
        plt.figure(figsize=(10, 5))
        plt.plot(gradient_norms)
        plt.xlabel('Episode')
        plt.ylabel('Gradient L2 Norm')
        plt.title('Gradient Magnitude During Training')
        plt.grid(True)
        plt.savefig(os.path.join(self.savefig_folder, 'gradient_norms.png'))
        plt.close()
        
        # Plot action distribution
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(action_counts)), action_counts)
        plt.xlabel('Action')
        plt.ylabel('Selection Count')
        plt.title('Action Selection Distribution')
        plt.grid(True)
        plt.savefig(os.path.join(self.savefig_folder, 'action_distribution.png'))
        plt.close()
        
        # Plot learning curves with different scales
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Regular scale
        ax1.plot(episode_rewards)
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Learning Curve (Regular Scale)')
        ax1.grid(True)
        
        # Log scale for better visibility of improvements
        ax2.plot(episode_rewards)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Reward')
        ax2.set_title('Learning Curve (Different Y Scale)')
        ax2.set_ylim([min(episode_rewards), min(max(episode_rewards), -300)])  # Adjusted y-limits
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.savefig_folder, 'learning_curves.png'))
        plt.close()
        
        # Plot network weight histograms (last layer)
        plt.figure(figsize=(10, 5))
        for name, param in self.q_values.named_parameters():
            if 'weight' in name and 'fc3' in name:  # Last layer weights
                weights = param.detach().cpu().numpy().flatten()
                plt.hist(weights, bins=50, alpha=0.7)
                plt.xlabel('Weight Value')
                plt.ylabel('Count')
                plt.title(f'Weight Distribution for {name}')
                plt.grid(True)
                plt.savefig(os.path.join(self.savefig_folder, f'weights_{name}.png'))
                plt.close()
    
    def plot_results(self, episode_rewards, all_rewards):
        """Create and save plots for the results"""
        # Create directory if it doesn't exist
        os.makedirs(self.savefig_folder, exist_ok=True)
        
        # Plot 1: Episode rewards over time
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Episode Reward')
        plt.title(f'Semi-Gradient SARSA Episode Rewards - {self.env_name}')
        plt.grid(True)
        plt.savefig(os.path.join(self.savefig_folder, 'episode_rewards.png'))
        plt.close()
        
        # Plot 2: Moving average of episode rewards
        plt.figure(figsize=(10, 5))
        window = min(100, len(episode_rewards))
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg)
        plt.xlabel('Episode')
        plt.ylabel(f'{window}-Episode Moving Average Reward')
        plt.title(f'Semi-Gradient SARSA Moving Average Reward - {self.env_name}')
        plt.grid(True)
        plt.savefig(os.path.join(self.savefig_folder, 'moving_avg_reward.png'))
        plt.close()
        
        # Plot 3: Average reward estimate over time
        plt.figure(figsize=(10, 5))
        # Collect avg_reward estimates at regular intervals
        steps = list(range(0, len(all_rewards), 1000))
        if steps and steps[-1] != len(all_rewards)-1:
            steps.append(len(all_rewards)-1)
        
        avg_reward_samples = [sum(all_rewards[:i])/i if i > 0 else 0 for i in steps]
        plt.plot(steps, avg_reward_samples)
        plt.xlabel('Environment Steps')
        plt.ylabel('True Average Reward')
        plt.title(f'Average Reward - {self.env_name}')
        plt.grid(True)
        plt.savefig(os.path.join(self.savefig_folder, 'avg_reward.png'))
        plt.close()


# Set random seed
params = Params()
random.seed(params.seed)
np.random.seed(params.seed)
torch.manual_seed(params.seed)

# Create and train the agent
agent = SemiGradSarsa(params)
all_rewards, episode_rewards = agent.train()

# print(f"Training complete. Final average reward estimate: {agent.avg_reward:.4f}")
print(f"Average episode reward over last 100 episodes: {np.mean(episode_rewards[-100:]):.1f}")

