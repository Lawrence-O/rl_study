from collections import defaultdict
import gymnasium as gym
import numpy as np
from pathlib import Path
from typing import NamedTuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

class Params(NamedTuple):
    total_episodes: int  # Total episodes
    gamma: float  # Discounting rate
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    savefig_folder: Path  # Root folder where plots are saved

params = Params(
    total_episodes=500000,
    gamma=0.95,
    map_size=5,
    seed=42, #12345
    n_runs=20,
    action_size=None,
    state_size=None,
    savefig_folder=Path("./plots/p3/"),
)

env = gym.make("Blackjack-v1", sab=True)  # Simplified Blackjack rules
params = params._replace(action_size=2)

class FirstVisitMCAgent:
    def __init__(self):

        self.v_value = defaultdict(float)
        self.returns = defaultdict(list)
        self.params = params  

    def policy(self, state):
        # Consider the policy that sticks if the player’s sum is 20 or 21, and otherwise hits\
        player_sum, dealer_value, usable_ace = state
        return 1 # 0 is Stick; 1 is Hit

    def generate_episode(self):
        episode = []
        state, _ = env.reset()
        done = False
        while not done:
            action = self.policy(state)
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state,action,reward))
            state = next_state
        return episode
    
    def reset(self):
        self.v_value = defaultdict(float)
        self.returns =  defaultdict(list)
    
    def first_visit_mc(self):
        for _ in range(params.total_episodes):
            episode = self.generate_episode()
            states, _, rewards = zip(*episode) #Unpacks episode and encodes all states into states etc
            first_visit = set()
            G = 0
            for t in range(len(states)-1, -1, -1):
                state = states[t]
                G = params.gamma*G + rewards[t]
                # Check if this is the first time we've seen the state in this episode
                if states[t] not in first_visit:
                    first_visit.add(state)
                    self.returns[state].append(G)
                    self.v_value[state] = np.mean(self.returns[state])


# Instantiate and run the agent
agent = FirstVisitMCAgent()
agent.first_visit_mc()

# Create the folder for saving plots if it doesn't exist
params.savefig_folder.mkdir(parents=True, exist_ok=True)

# Function to plot V(s) for a given value of usable_ace (True/False)
def plot_value_function(agent, usable_ace, params, title="Value Function"):
    # Define the range of player's sum and dealer's card we want to plot.
    player_sums = np.arange(12, 22)   # Player sums 12 through 21
    dealer_cards = np.arange(1, 11)    # Dealer cards 1 through 10

    # Prepare an array to store the value for each (player_sum, dealer_card)
    value_grid = np.zeros((len(player_sums), len(dealer_cards)))
    for i, player_sum in enumerate(player_sums):
        for j, dealer_card in enumerate(dealer_cards):
            # The state is defined as (player_sum, dealer_card, usable_ace)
            state = (player_sum, dealer_card, usable_ace)
            # Use the estimated value if it exists, otherwise default to 0.
            value_grid[i, j] = agent.v_value.get(state, 0)
    
    # Create a meshgrid for plotting
    X, Y = np.meshgrid(dealer_cards, player_sums)
    
    # Plot using a 3D surface plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, value_grid, cmap='viridis')
    ax.set_xlabel("Dealer Showing")
    ax.set_ylabel("Player Sum")
    ax.set_zlabel("V(s)")
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Save the figure
    plot_path = params.savefig_folder / f"value_function_usable_{usable_ace}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to: {plot_path}")

# Generate plots for both cases: usable ace and no usable ace
plot_value_function(agent, usable_ace=True, params=params, title="Value Function (Usable Ace)")
plot_value_function(agent, usable_ace=False, params=params, title="Value Function (No Usable Ace)")

            
