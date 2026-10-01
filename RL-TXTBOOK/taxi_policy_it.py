from collections import defaultdict
import gymnasium as gym
import numpy as np
from pathlib import Path
from typing import NamedTuple
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordEpisodeStatistics

class Params(NamedTuple):
    total_episodes: int  # Total episodes
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    theta: float # Delta Convergence
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    savefig_folder: Path  # Root folder where plots are saved

params = Params(
    total_episodes=2000,
    gamma=0.95,
    epsilon=0.1,
    theta=1e-3,
    map_size=5,
    seed=42, #12345
    n_runs=20,
    action_size=None,
    state_size=None,
    savefig_folder=Path("./plots/p2/"),
)

class PolicyIterationAgent:
    def __init__(self, env, params):
        self.theta = params.theta #Convergence Criteria
        self.gamma = params.gamma #Discount Reward
        self.state_space = np.array(list(range(params.state_size)))
        self.action_space = np.array(list(range(params.action_size)))
        self.model = env.unwrapped.P # accessed via [s][a][(prob, nextstate, reward, is_terminal), ...]
        
        self.v_values = defaultdict(float)
        self.policy = defaultdict(int)
    
    def initialize_values(self):
        for state in self.state_space:
            self.v_values[state] = 0
            self.policy[state] = np.random.choice(self.action_space)  # Random init

    def policy_evaluation(self):
        deltas = []
        while True:
            delta = 0
            for state in self.state_space:
                if not self.model[state]:
                    continue
                
                old_v = self.v_values[state]
                action = self.policy[state]

                #Sum Q value over all possible transition states by following policy
                q_val = 0.0
                for transition in self.model[state][action]:
                    prob, next_state, reward, _ =  transition
                    q_val += prob * (reward + self.gamma*self.v_values[next_state])

                self.v_values[state] = q_val
                delta = max(delta, abs(old_v - q_val))
            deltas.append(delta)
            if delta < self.theta:
                break
        return deltas
    
    def policy_improvement(self):
        policy_stable = True
        for state in self.state_space:
            old_action = self.policy[state]
            best_action = None
            best_v = float("-inf")
            for action in self.action_space:
                q_val = 0.0
                for transition in self.model[state][action]:
                    prob, next_state, reward, _ =  transition
                    q_val += prob * (reward + self.gamma*self.v_values[next_state])
                if q_val > best_v:
                    best_v = q_val
                    best_action = action
            if old_action != best_action:
                policy_stable = False
            self.policy[state] = best_action
        return policy_stable
    
    def train(self):
        self.initialize_values()
        policy_stable = False
        all_deltas = []  # Track deltas across all iterations
        policy_changes = []  # Track policy changes per iteration
        
        while not policy_stable:
            deltas = self.policy_evaluation()
            all_deltas.extend(deltas)
            
            # Track policy changes before improvement
            old_policy = dict(self.policy)
            policy_stable = self.policy_improvement()
            
            # Count changed actions
            changes = sum(1 for s in self.state_space if self.policy[s] != old_policy[s])
            policy_changes.append(changes)
            
        return all_deltas, policy_changes
    
    def get_next_action(self, state):
        return self.policy[state]

# Create the Env and wrap it with RecordEpisodeStatistics
env = gym.make(
    "Taxi-v3",
    render_mode="rgb_array" if params.savefig_folder else "human",
    max_episode_steps=200  # Prevent infinite episodes
)
env = RecordEpisodeStatistics(env, buffer_length=params.n_runs)  # Track last 100 episodes

# Update params with action and state sizes
params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)

# Initialize Agent
agent = PolicyIterationAgent(env, params)
deltas, policy_changes = agent.train()  # Get both metrics

# Run multiple episodes to collect statistics
num_episodes = 100
episode_rewards = []
episode_lengths = []

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        env.render()
        action = agent.get_next_action(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)

# Close environment
env.close()

# Create the figure folder if it doesn't exist
params.savefig_folder.mkdir(parents=True, exist_ok=True)

# Plot 1: Training Convergence
plt.figure(figsize=(15, 10))

# Plot 1a: Value Function Deltas
plt.subplot(2, 2, 1)
plt.plot(deltas)
plt.title("Policy Evaluation Convergence")
plt.xlabel("Evaluation Iterations")
plt.ylabel("Max Delta")
plt.yscale('log')

# Plot 1b: Policy Changes per Iteration
plt.subplot(2, 2, 2)
plt.plot(policy_changes, marker='o')
plt.title("Policy Changes per Iteration")
plt.xlabel("Policy Iteration")
plt.ylabel("Number of Changed Actions")

# Plot 2: Performance Metrics
plt.subplot(2, 2, 3)
plt.plot(episode_rewards, label='Episode Reward')
plt.plot(np.convolve(episode_rewards, np.ones(10)/10, mode='valid'), 
         label='10-Episode Moving Avg', color='red')
plt.title("Reward Progression")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()

# Plot 2b: Episode Lengths
plt.subplot(2, 2, 4)
plt.plot(episode_lengths, label='Episode Length')
plt.plot(np.convolve(episode_lengths, np.ones(10)/10, mode='valid'), 
         label='10-Episode Moving Avg', color='red')
plt.title("Episode Length Progression")
plt.xlabel("Episode")
plt.ylabel("Steps Taken")
plt.legend()

plt.tight_layout()
plt.savefig(params.savefig_folder / "training_performance.png")
plt.close()

# Plot 3: Final Value Function Distribution
plt.figure(figsize=(10, 6))
all_values = [agent.v_values[s] for s in agent.state_space]
plt.hist(all_values, bins=20, edgecolor='black')
plt.title("State Value Distribution")
plt.xlabel("Value Estimate")
plt.ylabel("Number of States")
plt.savefig(params.savefig_folder / "value_distribution.png")
plt.close()

# Plot 4: Action Distribution
plt.figure(figsize=(10, 6))
action_counts = np.bincount(list(agent.policy.values()))
plt.bar(range(params.action_size), action_counts, edgecolor='black')
plt.title("Action Distribution in Learned Policy")
plt.xlabel("Action")
plt.ylabel("Count")
plt.xticks(range(params.action_size), ['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'])
plt.savefig(params.savefig_folder / "action_distribution.png")
plt.close()

# Plot 5: Success Rate
successes = [1 if r > 0 else 0 for r in episode_rewards]  # Taxi gives +20 for success
plt.figure(figsize=(10, 6))
plt.bar(["Success Rate"], np.cumsum(successes)/np.arange(1, len(successes)+1))
plt.title("Cumulative Success Rate")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.ylim(0, 1)
plt.savefig(params.savefig_folder / "success_rate.png")
plt.close()

print(f"Success rate: {sum(episode_rewards)/len(episode_rewards):.2%}")