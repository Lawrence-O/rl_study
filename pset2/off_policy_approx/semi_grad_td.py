import os
from pathlib import Path
from typing import NamedTuple
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import torch.nn.functional as F

import matplotlib.pyplot as plt

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=1, hidden_dem=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dem)
        self.fc2 = nn.Linear(hidden_dem, 4 * hidden_dem)
        self.fc3 = nn.Linear(4 * hidden_dem, hidden_dem)
        self.fc4 = nn.Linear(hidden_dem, 1)

        # Better weight initialization
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc4.weight, gain=np.sqrt(2))
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class BaseAgent:
    def __init__(self, env_name="Pendulum-v1", 
                 n_actions=15, 
                 lr=0.001, 
                 gamma=0.99, 
                 buffer_size=100000,
                 batch_size=64, 
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay=0.995,
                 target_update=10,
                 alg_name="base"):
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]

        # Discretize action space
        self.n_actions = n_actions
        self.action_min = float(self.env.action_space.low[0])
        self.action_max = float(self.env.action_space.high[0])
        self.discrete_actions = np.linspace(self.action_min, self.action_max, n_actions)

        #Initialize Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(self.state_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update

        # Tracking
        self.alg_name = alg_name
        self.training_rewards = []
        self.training_losses = []
        self.average_q_values = []
        self.target_q_values = []
        self.td_errors = []
    
    def select_action(self, state):
        if np.random.random < self.epsilon:
            return np.random.randint(self.n_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = torch.zeros(self.n_actions)
        with torch.no_grad():
            for a_idx in range(self.n_actions):
                action_tensor = torch.FloatTensor([self.discrete_actions[a_idx]]).to(self.device)
                q_values[a_idx] = self.q_network(state_tensor, action_tensor)
            return q_values.argmax.item()
    def get_continous_action(self, action_idx):
        return self.discrete_actions[action_idx]
    def train_step(self):
        return NotImplementedError()
    def train(self, n_episodes=1000, check_divergence_interval=10):
        self.training_rewards = []
        self.training_losses = []
        self.average_q_values = []

        for episode in tqdm(range(n_episodes), desc=f"Training {self.alg_name}"):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            episode_q_values = []
            done = False
            steps = 0

            while not done:
                action_idx = self.select_action(state)
                action = self.get_continous_action(action_idx)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                loss = self.train_step()
                episode_loss += loss

                # Get average Q-value for monitoring
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action_tensor = torch.FloatTensor([[action]]).to(self.device)
                    q_value = self.q_network(state_tensor, action_tensor).item()
                    episode_q_values.append(q_value)
                
                state = next_state
                episode_reward += reward
                steps += 1

                #Update Target Network
                if steps % self.target_update == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
                
            # Check for divergence periodically
            # if episode % check_divergence_interval == 0:
            #     self.diverged = self.detector.update(self.discrete_actions)
            #     if self.diverged:
            #         print(f"⚠️ Divergence detected at episode {episode}!")
            
            self.epsilon = max(self.epsilon_end, self.epsilon*self.epsilon_decay)

            # Track metrics
            self.training_rewards.append(episode_reward)
            self.training_losses.append(episode_loss / steps if steps > 0 else 0)
            self.average_q_values.append(np.mean(episode_q_values) if episode_q_values else 0)
            
            # Log progress periodically
            if episode % 100 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg Q: {self.average_q_values[-1]:.3f}, Epsilon: {self.epsilon:.2f}")

            return self.training_rewards