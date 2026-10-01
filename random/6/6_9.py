from collections import defaultdict
import numpy as np
import random as random
import matplotlib.pyplot as plt

class WindyGridworld:
    def __init__(self, king_moves=False, stochastic_wind=False, no_move_action=False):
        # Grid configuration (rows x columns)
        self.rows = 7
        self.cols = 10
        
        # Start and goal positions (row, column)
        self.start = (3, 0)
        self.goal = (3, 7)
        self.pos = None
        
        # Wind configuration (strength per column)
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        
        # Action space configuration
        self.king_moves = king_moves
        self.no_move_action = no_move_action
        self.actions = self._get_action_space()
        
        # Stochastic wind settings
        self.stochastic_wind = stochastic_wind
        
        # Initialize state
        self.reset()

    def _get_action_space(self):
        """Define action space based on configuration"""
        actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
        }
        
        if self.king_moves:
            actions.update({
                4: (-1, -1),  # Up-Left
                5: (-1, 1),   # Up-Right
                6: (1, -1),   # Down-Left
                7: (1, 1),    # Down-Right
            })
            
        if self.no_move_action:
            actions[len(actions)] = (0, 0)  # No movement
            
        return actions

    def reset(self):
        """Reset to start position"""
        self.pos = self.start
        return self.pos

    def step(self, action):
        """Take a step in the environment"""
        # Calculate movement
        dr, dc = self.actions[action]
        new_r = self.pos[0] + dr
        new_c = self.pos[1] + dc
        
        # Apply boundaries
        new_r = np.clip(new_r, 0, self.rows-1)
        new_c = np.clip(new_c, 0, self.cols-1)
        
        # Apply wind
        wind_strength = self.wind[new_c]
        
        if self.stochastic_wind and wind_strength != 0:
            wind_strength += np.random.choice([-1, 0, 1])
            # wind_strength = max(0, wind_strength)
            
        new_r += wind_strength
        new_r = np.clip(new_r, 0, self.rows-1)
        
        # Update position
        self.pos = (new_r, new_c)
        
        # Calculate reward and done
        done = (self.pos == self.goal)
        reward = -1 if not done else 0  # Standard reward scheme

        # if done:
            # print(f"Step - Action: {action}, Prev Pos: {self.start if not hasattr(self, 'pos') else self.pos}, Move: {(dr, dc)}, Wind: {wind_strength}, New Pos: {self.pos}, Goal?: {done}") # DEBUG PRINT
        
        return self.pos, reward, done, {}

    def get_valid_actions(self):
        """Get list of valid actions for current position"""
        return list(self.actions.keys())

    def is_terminal_state(self, state):
        return state == self.goal

class SarsaAgent:
    def __init__(self, env, state_space_size, action_space_size, alpha=0.1, gamma=1, epsilon=0.1):
        """
        Initialize SARSA agent with:
        - Q-table: Dictionary/array storing state-action values
        - alpha: Learning rate (step size)
        - gamma: Discount factor
        - epsilon: Exploration rate for ε-greedy policy
        """
        self.q_values = defaultdict(float)
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon


    def choose_action(self, current_state, epsilon=None, valid_actions=None):
        """
        Select action using ε-greedy policy:
        1. With probability (1-ε): choose action with highest Q-value
        2. With probability ε: choose random valid action
        Returns selected action
        """
        if not epsilon:
            epsilon = self.epsilon
        possible_actions = self.env.actions
        if random.random() < epsilon:
            return random.choice(list(possible_actions.keys()))
        else:
            max_q_val = float('-inf')
            best_action = None
            for action_key in possible_actions.keys():
                # dx,dy = possible_actions[action_key]
                # new_state = current_state[0] + dx, current_state[1] + dy
                new_state_val = self.q_values[(current_state, action_key)]
                if new_state_val > max_q_val:
                    max_q_val = new_state_val
                    best_action = action_key
            return best_action

    def update_q_values(self, current_state, action, reward, next_state, next_action):
        """
        Update Q-table using SARSA update rule:
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        """
        self.q_values[(current_state,action)] += self.alpha * (reward + self.gamma*self.q_values[(next_state,next_action)] 
                                       - self.q_values[(current_state, action)])

    def decay_epsilon(self, decay_rate):
        """
        Reduce exploration rate ε over time
        (Optional method for annealing exploration)
        """
        self.epsilon *= decay_rate

    def get_q_value(self, state, action):
        """
        Retrieve Q-value for specific state-action pair
        Returns float value
        """
        return self.q_values.get((state,action), 0)

    def reset(self):
        """
        Reset agent while maintaining learned Q-values
        (Optional method for episodic resets)
        """
        pass

    def train_episode(self, env):
        """
        Run a single SARSA episode in the Windy Gridworld.
        Returns: Total reward and number of steps for the episode
        """
        state = env.reset()
        action = self.choose_action(state)
        total_reward = 0
        steps = 0
        done = False
        while not done:
            # Take action, get next state and reward
            next_state, reward, reached_goal, _ = env.step(action)

            # Compute our next action from Q
            next_action = self.choose_action(next_state)

            #Update Q Values
            self.update_q_values(state, action, reward, next_state, next_action)

            #Update Values
            state, action = next_state, next_action

            # Update metrics
            total_reward += reward
            steps += 1

            #Terminal
            done = reached_goal

            # if steps > 10000:
            #     # print("We Broke Due to Going Over Steps")
            #     break
            
        print(f"Episode Finished - Steps: {steps}, Reward: {total_reward}, Goal Reached?: {done}") # DEBUG PRINT
        return total_reward, steps


    def train(self, env, num_episodes=1000):
        """
        Main training loop for SARSA.
        
        Pseudocode logic:
        1. For each episode:
            a. Reset environment to start state
            b. Choose initial action using ε-greedy policy
            c. While episode not terminated:
                i. Take action, observe reward and next state
                ii. Choose next action using ε-greedy policy
                iii. Update Q-values using SARSA rule
                iv. Transition to next state/action
        2. Track performance metrics
        """
        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            rewards, length = self.train_episode(env)

            episode_rewards.append(rewards)
            episode_lengths.append(length)

            # Print progress
            if (episode+1) % 100 == 0:
                avg_steps = np.mean(episode_lengths[-100:])
                print(f"Episode {episode+1} | Avg Steps: {avg_steps:.1f}")

        return episode_rewards, episode_lengths
    def get_trajectory(self, env):
        """
        Get the trajectory of the agent following the learned policy (greedy).
        Returns: A list of state tuples representing the trajectory.
        """
        state = env.reset()
        trajectory = [state]
        done = False
        rewards = []
        steps = 0
        while not done:
            action_key = self.choose_action(state, epsilon=0)  # Choose action greedily (epsilon=0)
            next_state, reward, done, _ = env.step(action_key)
            trajectory.append(next_state)
            state = next_state
            rewards.append([reward])
            steps += 1
        trajectory.append(state)
        return trajectory, rewards, steps
    
    def render_trajectory(self, env, trajectory):
        """
        Render the trajectory on the Windy Gridworld.
        """
        grid = [['-' for _ in range(env.cols)] for _ in range(env.rows)] # Initialize grid with '-'

        start_pos = env.start
        goal_pos = env.goal

        grid[start_pos[0]][start_pos[1]] = 'S' # Mark start
        grid[goal_pos[0]][goal_pos[1]] = 'G' # Mark goal

        for pos in trajectory[1:-1]: # Mark path, excluding start and goal (already marked)
            grid[pos[0]][pos[1]] = 'P' # 'P' for path

        print("Trajectory:")
        for row in grid:
            print(''.join(row))
    def render_trajectory_mpl(self, env, trajectory):
        """
        Render the trajectory on the Windy Gridworld using matplotlib.
        """
        grid_np = np.zeros((env.rows, env.cols)) # Initialize grid with 0s
        start_pos = env.start
        goal_pos = env.goal

        # Mark start, goal and path with different values
        for pos in trajectory[1:-1]: # Mark path, excluding start and goal
            grid_np[pos[0], pos[1]] = 3     # Path (e.g., 3)
        
        grid_np[start_pos[0], start_pos[1]] = 1  # Start (e.g., 1)
        grid_np[goal_pos[0], goal_pos[1]] = 2   # Goal (e.g., 2)

        # Colors for visualization
        cmap = plt.cm.get_cmap('viridis', 4) # Use 'viridis' colormap with 4 colors

        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(grid_np, cmap=cmap, origin='upper', vmin=0, vmax=3) # Display grid

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, ticks=[0.5, 1.5, 2.5, 3.5])
        cbar.ax.set_yticklabels(['Empty', 'Start', 'Goal', 'Path'])  # Label colorbar

        # Annotate Start and Goal
        ax.text(start_pos[1], start_pos[0], 'S', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        ax.text(goal_pos[1], goal_pos[0], 'G', ha='center', va='center', color='white', fontsize=12, fontweight='bold')

        # Plot path as points
        path_x = [pos[1] for pos in trajectory[1:-1]] # Columns (x-coords)
        path_y = [pos[0] for pos in trajectory[1:-1]] # Rows (y-coords)
        ax.plot(path_x, path_y, marker='o', linestyle='-', color='red', markersize=5, linewidth=1, label='Path')

        # Set grid lines
        ax.set_xticks(np.arange(env.cols)-0.5, minor=True)
        ax.set_yticks(np.arange(env.rows)-0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", size=0) # Hide minor ticks

        # Remove ticks and labels
        ax.set_xticks(np.arange(env.cols))
        ax.set_yticks(np.arange(env.rows))
        ax.set_xticklabels(np.arange(env.cols))
        ax.set_yticklabels(np.arange(env.rows)) # You can remove these if you don't want axis labels
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title("Windy Gridworld Trajectory (SARSA Agent)")
        ax.invert_yaxis() # To have origin at top-left as in gridworld convention

        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Initialize environment and agent
    env = WindyGridworld(king_moves=True,no_move_action=True, stochastic_wind=True)
    agent = SarsaAgent(
        env,
        state_space_size=(7, 10),  # Grid dimensions
        action_space_size=len(env.actions),
        alpha=0.5,  # Learning rate from example
        gamma=0.2,
        epsilon=0.1  # ε from problem description
    )

    # Train the agent
    rewards, steps = agent.train(env, num_episodes=8000)
    print(f"Training Average # of Steps: {np.sum(steps) / len(steps)} \n Average # of Rewards: {np.sum(rewards) / len(rewards)}")

    
    #Trajectory
    trajectory,rewards,steps = agent.get_trajectory(env)
    print(f"Policy # of Steps: {steps} \n  # of Rewards: {np.sum(rewards) }")
    agent.render_trajectory_mpl(env, trajectory)

