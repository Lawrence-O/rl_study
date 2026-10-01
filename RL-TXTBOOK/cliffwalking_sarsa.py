from collections import defaultdict
import gymnasium as gym
import numpy as np
from pathlib import Path
from typing import NamedTuple
import matplotlib.pyplot as plt
import random

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
    proba_slippery: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved

params = Params(
    total_episodes=500,
    learning_rate=0.8,
    gamma=0.9,
    epsilon=0.1,
    map_size=5,
    seed=42, #12345
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    proba_slippery=0.1,
    savefig_folder=Path("./plots/p4/"),
)


env = gym.make('CliffWalking-v0')
params = params._replace(action_size=4)
params = params._replace(state_size=48)


class SarsaAgent:
    def __init__(self, params):
        self.gamma = params.gamma
        self.epsilon = params.epsilon
        self.alpha = params.learning_rate
        self.num_episodes = params.total_episodes
        self.state_space = np.array(list(range(params.state_size)))
        self.action_space = np.array(list(range(params.action_size)))
        self.rewards = []
        self.steps = []


        self.q_values = defaultdict(float)
    
    def initialize(self):
        for state in self.state_space:
            for action in self.action_space:
                self.q_values[(state, action)] = np.random.uniform(0, 0.01)
    
    def choose_action(self, state):
        # Choose an action based on epsilon greedy
        # (1-epsilon) chance to pick greedy
        # epsilon chance to pick exploratory action

        if random.random() < self.epsilon:  # Get Random Action
            return random.choice(self.action_space)
        else: # Pick Greedy Action
            max_q = max(self.q_values[(state, a)] for a in self.action_space)
            best_actions = [a for a in self.action_space if self.q_values[(state, a)] == max_q]
        return random.choice(best_actions) # Random tie-break
    
    def train_episode(self):
        state, _ = env.reset()
        action = self.choose_action(state)
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_action = self.choose_action(next_state)

            td_error = reward + self.gamma * self.q_values[(next_state, next_action)] - self.q_values[(state, action)]

            self.q_values[(state,action)] += self.alpha*td_error

            total_reward += reward
            step_count += 1

            state = next_state
            action = next_action
        self.rewards.append(total_reward)
        self.steps.append(step_count)
    
    def train(self):
        self.initialize()
        for _ in range(self.num_episodes):
            self.train_episode()
    
    def policy(self, state):
        max_q = max(self.q_values[(state, a)] for a in self.action_space)
        best_actions = [a for a in self.action_space if self.q_values[(state, a)] == max_q]
        return random.choice(best_actions) # Random tie-break


# --- Training and Visualization ---

agent = SarsaAgent(params)
agent.train()

# Plot the sum of rewards vs. episodes (without smoothing)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(agent.rewards) + 1), agent.rewards, label="Total Reward per Episode")
plt.title("Total Reward vs. Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.show()

# Optionally, you can also plot the smoothed version if needed:
window_size = 20
smooth_rewards = np.convolve(agent.rewards, np.ones(window_size) / window_size, mode='valid')
plt.figure(figsize=(8, 5))
plt.plot(range(window_size, len(agent.rewards) + 1), smooth_rewards, color='orange', label="Smoothed Reward")
plt.title("Smoothed Total Reward vs. Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.show()

# # Final Policy Evaluation
test_rewards = []
test_steps = []
for _ in range(params.n_runs):
    print(f"Test Run: {_ + 1}")
    state, _ = env.reset()
    print("State :",state)
    done = False
    total_reward = 0
    step = 0
    while not done:
        action = agent.policy(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        step += 1
    test_steps.append(step)
    test_rewards.append(total_reward)
env.close()



plt.figure(figsize=(8, 5))
plt.plot(range(1, params.n_runs + 1), test_rewards, marker='o', linestyle='-', label='Test Reward')
plt.title("Total Reward per Test Run")
plt.xlabel("Test Run")
plt.ylabel("Total Reward")
plt.xticks(range(1, params.n_runs + 1))
plt.legend()
plt.grid(True)
plt.show()

print(f"\nFinal Policy Performance ({params.n_runs} test episodes):")
print(f"Average Reward: {np.mean(test_rewards):.2f}")
print(f"Average Step Length : {np.mean(test_steps):.2f}")





           

    