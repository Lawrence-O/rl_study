from collections import defaultdict
import gymnasium as gym
import numpy as np
from pathlib import Path
from typing import NamedTuple
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt

class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    theta: float
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved

params = Params(
    total_episodes=2000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    theta=1e-3,
    map_size=5,
    seed=42, #12345
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    proba_frozen=0.1,
    savefig_folder=Path("./plots/p1/"),
)

# Value Iteration
class ValueIterationAgent:
    def __init__(self, env, params):
        self.theta = params.theta #Convergence Criteria
        self.gamma = params.gamma #Discount Reward
        self.state_space = np.array(list(range(params.state_size)))
        self.action_space = np.array(list(range(params.action_size)))
        self.model = env.unwrapped.P # accessed via [s][a][(prob, nextstate, reward, is_terminal), ...]


        self.v_values = defaultdict(float) #V Values
        self.policy = defaultdict(int)

    def value_iteraiton(self):
        deltas = []  # Track delta for convergence plot
        while True:
            delta = 0
            for state in self.state_space:
                if not self.model[state]: # No Transition States
                    continue

                old_v = self.v_values[state]
                best_v = float("-inf")
                for action in self.action_space:
                    # Sum over all possible transitions for this state-action pair
                    q_val = 0.0
                    for transition in self.model[state][action]:
                        prob, next_state, reward, _ =  transition
                        q_val += prob * (reward + self.gamma*self.v_values[next_state])
                    
                    if q_val > best_v:
                        best_v = q_val
                
                self.v_values[state] = best_v
                delta = max(delta, abs(old_v - best_v))
                deltas.append(delta)
            if delta < self.theta:
                break
        return deltas
        
    def update_policy(self):
        for state in self.state_space:
            if not self.model[state]: # No Transition States
                    continue
            
            best_action = None
            best_v = float("-inf")
            for action in self.action_space:
                # Sum over all possible transitions for this state-action pair
                q_val = 0.0
                for transition in self.model[state][action]:
                    prob, next_state, reward, _ =  transition
                    q_val += prob * (reward + self.gamma*self.v_values[next_state])
                if q_val > best_v:
                    best_v = q_val
                    best_action = action
            self.policy[state] = best_action
    
    def get_next_action(self, state):
        return self.policy[state]

# Initialize Env and Agent
# Set the seed
rng = np.random.default_rng(params.seed)

# Create the figure folder if it doesn't exist
params.savefig_folder.mkdir(parents=True, exist_ok=True)

#Create the Env
env = gym.make(
    "FrozenLake-v1",
    is_slippery=params.is_slippery,
    render_mode="human",
    desc=generate_random_map(
        size=params.map_size, p=params.proba_frozen, seed=params.seed
    ),
)

params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)
print(f"Action size: {params.action_size}")
print(f"State size: {params.state_size}")     
desc = [row for row in env.unwrapped.desc]  # Decode bytes to strings    

#Create Agent
agent = ValueIterationAgent(env, params)

#Run Value Iteration
deltas = agent.value_iteraiton()
agent.update_policy()

# Test the learned policy
state, _ = env.reset()
done = False
total_reward = 0

print("Running Policy:")
while not done:
    env.render()
    action = agent.get_next_action(state)
    state, reward, done, _, _ = env.step(action)
    total_reward += reward

print(f"Total reward: {total_reward}")
env.close()

#Plots

plt.figure()
plt.plot(deltas)
plt.xlabel("Iterations")
plt.ylabel("Delta")
plt.title("Value Iteration Convergence")
plt.yscale("log")
plt.savefig(params.savefig_folder / "convergence.png")
plt.close()

# Plot 2: Value Function Heatmap
v_values = np.array([agent.v_values[state] for state in range(params.state_size)])
v_grid = v_values.reshape(params.map_size, params.map_size)

plt.figure()
plt.imshow(v_grid, cmap="coolwarm")
plt.colorbar()
for row in range(params.map_size):
    for col in range(params.map_size):
        tile = desc[row][col]
        value = v_grid[row, col]
        plt.text(col, row, f"{tile}\n{value:.2f}", ha="center", va="center", color="black")
plt.title("State Value Heatmap")
plt.axis("off")
plt.savefig(params.savefig_folder / "heatmap.png")
plt.close()

# Plot 3: Policy Visualization
policy_actions = np.array([agent.policy[state] for state in range(params.state_size)])
policy_grid = policy_actions.reshape(params.map_size, params.map_size)

action_to_arrow = {3: "↑", 2: "→", 1: "↓", 0: "←"}  # Map actions to arrows
plt.figure()
plt.imshow(v_grid, cmap="coolwarm")
for row in range(params.map_size):
    for col in range(params.map_size):
        tile = desc[row][col]
        action = policy_grid[row, col]
        plt.text(col, row, f"{tile}\n{action_to_arrow[action]}", ha="center", va="center", color="black")
plt.title("Optimal Policy")
plt.axis("off")
plt.savefig(params.savefig_folder / "policy.png")
plt.close()

# Plot 4: Success Rate
test_env = gym.make(
    "FrozenLake-v1",
    is_slippery=params.is_slippery,
    render_mode="rgb_array",
    desc=generate_random_map(size=params.map_size, p=params.proba_frozen, seed=params.seed),
)

n_test_episodes = 1000
successes = 0
for _ in range(n_test_episodes):
    state, _ = test_env.reset()
    done = False
    while not done:
        action = agent.policy[state]
        state, reward, done, _, _ = test_env.step(action)
    successes += (reward == 1)

plt.figure()
plt.bar(["Success Rate"], [successes / n_test_episodes], color="skyblue")
plt.ylim(0, 1)
plt.ylabel("Rate")
plt.title(f"Success Rate (Avg. over {n_test_episodes} Episodes)")
plt.savefig(params.savefig_folder / "success_rate.png")
plt.close()
test_env.close()



