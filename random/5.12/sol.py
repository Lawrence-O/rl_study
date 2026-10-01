import numpy as np
import random
import matplotlib.pyplot as plt

class RacetrackEnv:
    def __init__(self, racetrack_map):
        """
        Initializes the Racetrack environment.

        Args:
            racetrack_map: A 2D list or numpy array representing the racetrack.
                           Different values can represent:
                           - 0: Track
                           - 1: Boundary
                           - 2: Start Line
                           - 3: Finish Line
        """
        self.racetrack = np.array(racetrack_map) # Store map as numpy array
        print(self.racetrack)
        self.start_line_positions = [] # List of (x, y) tuples for start line
        self.finish_line_positions = [] # List of (x, y) tuples for finish line
        self._extract_start_finish_lines()
        self.current_state = None # (position (x,y), velocity (vx, vy))
        print("Start Line Positions:", self.start_line_positions) # Print start positions
        print("Finish Line Positions:", self.finish_line_positions) # Print finish positions

    def _extract_start_finish_lines(self):
        """
        Identifies start and finish line positions from the racetrack map.
        Populates self.start_line_positions and self.finish_line_positions.
        """
        # TODO: Implement logic to find start and finish line cells in the map
        # and populate self.start_line_positions and self.finish_line_positions.
        # Start line is marked with 2, Finish line with 3.
        self.start_line_positions = []
        self.finish_line_positions = []

        # Find start line positions (value 2)
        rows, cols = np.where(self.racetrack == 2)
        self.start_line_positions = list(zip(cols, rows))  # (x,y) = (column, row)

        # Find finish line positions (value 3)
        rows, cols = np.where(self.racetrack == 3)
        self.finish_line_positions = list(zip(cols, rows))  # (x,y) = (column, row)

    def reset(self):
        """
        Resets the environment to the start of a new episode.
        - Randomly chooses a start position.
        - Sets velocity to (0, 0).
        - Returns the initial state.

        Returns:
            state: Initial state (position (x,y), velocity (vx, vy)).
        """
        # TODO: Implement reset logic:
        # 1. Choose a random start position from self.start_line_positions.
        # 2. Set initial velocity to (0, 0).
        # 3. Set self.current_state.
        # 4. Return initial state.
        startx, starty = random.choice(self.start_line_positions)
        self.current_state = [(startx,starty), (0,0)]
        return [(startx,starty), (0,0)]

    def get_possible_actions(self):
        """
        Returns a list of possible actions from the current state.
        Actions are velocity increments (dvx, dvy) in the range [-1, 0, 1] for each component.

        Returns:
            list of tuples: List of possible actions [(dvx1, dvy1), (dvx2, dvy2), ...]
        """
        possible_actions = []
        for dvx in [-1, 0, 1]:
            for dvy in [-1, 0, 1]:
                possible_actions.append((dvx, dvy))
        return possible_actions

    # def _is_valid_velocity(self, vx, vy):
    #     """
    #     Checks if a given velocity (vx, vy) is valid according to the constraints.
    #     """
    #     return 0 <= vx < 5 and 0 <= vy < 5 and not (vx == 0 and vy == 0 and self.current_state[0] not in self.start_line_positions) # Added condition for start line
    def _is_valid_velocity(self, vx, vy, position):
        if vx == 0 and vy == 0:
            return position in self.start_line_positions
        return 0 <= vx < 5 and 0 <= vy < 5
    def _bresenham_line(self, start, end):
        # Bresenham's line algorithm to get all cells along the path
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        cells = []
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return cells

    def _check_path_collision(self, start_pos, end_pos):
        """
        Checks path collision with detailed debugging prints.
        """
        cells = self._bresenham_line(start_pos, end_pos)
        finished_crossed = False
        for cell_index, (x, y) in enumerate(cells):
            # print(f"  Checking cell {cell_index}: ({x}, {y})", end=" - ") # Print cell coordinates
            if y < 0 or x < 0 or y >= self.racetrack.shape[0] or x >= self.racetrack.shape[1]:
                # print("Out of bounds - Boundary Collision")
                return 'boundary', False
            cell_value = self.racetrack[y,x]
            # print(f"Value: {cell_value}", end=" - ") # Print cell value
            if cell_value == 1:
                # print("Boundary Collision")
                return "boundary", False
            if cell_value == 3:
                # print("Finish Line Crossed")
                finished_crossed = True
            # else:
                # print("Track/Start/Finish") # Indicate track, start, or finish
        return None, finished_crossed

    
    def step(self, action):
        """
        Executes one time step in the environment.

        Args:
            action: A tuple (dvx, dvy) representing the velocity increments.

        Returns:
            tuple: (next_state, reward, done, info)
                   next_state: The next state (position (x,y), velocity (vx, vy)).
                   reward: Reward obtained in this step (always -1).
                   done: True if the episode is finished (finish line crossed or off-track), False otherwise.
                   info: Additional information (can be an empty dictionary for now).
        """
        dvx, dvy = action

        # Stochastic velocity increment failure
        if random.random() < 0.1:
            dvx, dvy = 0, 0

        current_pos, current_vel = self.current_state
        vx, vy = current_vel

        # Proposed new velocity
        new_vx = vx + dvx
        new_vy = vy + dvy

        # Velocity Constraints check
        if not self._is_valid_velocity(new_vx, new_vy, current_pos):
            new_vx = max(0, min(new_vx, 4)) # Clip velocity to valid range, but this might not be the intended behavior, consider invalid action handling differently if needed.
            new_vy = max(0, min(new_vy, 4)) #  For now, clipping to keep things running, but think if action should be invalid instead.


        # Proposed new position
        start_pos = current_pos
        end_pos = (current_pos[0] + new_vx, current_pos[1] + new_vy)


        collision_type, finish_line_crossed = self._check_path_collision(start_pos, end_pos)

        if collision_type == 'boundary':
            # Off-track: reset to start line
            next_pos = random.choice(self.start_line_positions)
            next_vel = (0, 0)
            self.current_state = (next_pos, next_vel)
            return self.current_state, -1, False, {"collision": "boundary"} # Episode continues, not done
        elif finish_line_crossed:
            # Finish line crossed
            next_pos = end_pos # or keep the last position? define based on problem spec if needed
            next_vel = (new_vx, new_vy) # velocity at finish line
            self.current_state = (next_pos, next_vel)
            return self.current_state, 1, True, {"finish": True} # Episode done, success
        else:
            # Valid move within track or on finish line (but path check determined no boundary collision)
            next_pos = end_pos
            next_vel = (new_vx, new_vy)
            self.current_state = (next_pos, next_vel)
            return self.current_state, -1, False, {} # Episode continues, no collision


    def render(self, state):
        """
        Prints a text-based representation of the racetrack with the car's current position.
        """
        grid = [[' ' for _ in range(self.racetrack.shape[1])] for _ in range(self.racetrack.shape[0])]
        for y in range(self.racetrack.shape[0]):
            for x in range(self.racetrack.shape[1]):
                if self.racetrack[y, x] == 1:
                    grid[y][x] = '#'
                elif self.racetrack[y, x] == 2:
                    grid[y][x] = 'S'
                elif self.racetrack[y, x] == 3:
                    grid[y][x] = 'F'
                else:
                    grid[y][x] = '.' # Track
        
        current_pos, current_vel = state
        if 0 <= current_pos[1] < self.racetrack.shape[0] and 0 <= current_pos[0] < self.racetrack.shape[1]: #position in (x,y) format
            grid[current_pos[1]][current_pos[0]] = 'C' # Car at current position

        print("-" * (self.racetrack.shape[1] + 2)) # Border
        for row in grid:
            print("|" + "".join(row) + "|")
        print("-" * (self.racetrack.shape[1] + 2)) # Border
    def render_matplotlib(self, trajectory=None):
        """
        Renders the racetrack and the car's trajectory using matplotlib.

        Args:
            trajectory: An optional list of states (positions) representing the car's trajectory.
                        If provided, the trajectory will be plotted on the racetrack.
        """
        plt.figure(figsize=(6, 6)) # Adjust figure size as needed
        racetrack_grid = self.racetrack.copy() # Create a copy to avoid modifying the original map

        # Define colors for different cell types
        colors = {
            0: 'lightgray',  # Track
            1: 'black',      # Boundary
            2: 'lightgreen', # Start Line
            3: 'lightcoral'  # Finish Line
        }
        cmap = plt.cm.colors.ListedColormap([colors[i] for i in range(4)])
        bounds = [0, 0.5, 1.5, 2.5, 3.5] # Define boundaries for the colormap
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)


        # Plot the racetrack
        plt.imshow(racetrack_grid, cmap=cmap, norm=norm, origin='lower') # origin='lower' to match grid indexing

        # Plot trajectory if provided
        if trajectory:
            trajectory_positions = [state[0] for state in trajectory] # Extract positions from states
            x_coords = [pos[1] for pos in trajectory_positions] # Column index is x-coordinate
            y_coords = [pos[0] for pos in trajectory_positions] # Row index is y-coordinate
            plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue', markersize=5) # Plot trajectory line

            # Mark start and end points of trajectory
            start_pos = trajectory_positions[0]
            end_pos = trajectory_positions[-1]
            plt.plot(start_pos[1], start_pos[0], marker='o', markersize=8, color='green', label='Start') # Start point
            plt.plot(end_pos[1], end_pos[0], marker='x', markersize=8, color='red', label='End')     # End point
            plt.legend() # Show legend for start and end points

        # Get current car position and plot it
        current_pos, _ = self.current_state
        if 0 <= current_pos[1] < self.racetrack.shape[0] and 0 <= current_pos[0] < self.racetrack.shape[1]:
            plt.plot(current_pos[0], current_pos[1], marker='*', markersize=10, color='yellow', label='Current Car') # Mark current car
            plt.legend()


        plt.xticks([]) # Remove x ticks
        plt.yticks([]) # Remove y ticks
        plt.grid(False) # Disable grid lines
        plt.title("Racetrack with Trajectory")
        plt.gca().invert_yaxis() # Invert y-axis to match array indexing (optional, for top-left origin)
        plt.show()


class MCAgent:
    def __init__(self, possible_actions):
        """
        Initializes the Monte Carlo Agent.

        Args:
            possible_actions: List of possible actions (velocity increments).
        """
        self.possible_actions = possible_actions
        self.q_values = {} # Q(s, a) dictionary: {(state, action): value}
        self.policy = {}    # Target policy: {state: action} - greedy policy derived from Q
        self.behavior_policy = {} # Behavior policy for exploration (e.g., epsilon-greedy)


    def _get_state_str(self, state):
        """
        Helper function to convert a state tuple (position, velocity) to a string for dictionary keys.
        """
        pos, vel = state
        return f"pos_{pos[0]}_{pos[1]}_vel_{vel[0]}_{vel[1]}"


    def _get_action_str(self, action):
        """
        Helper function to convert an action tuple (dvx, dvy) to a string for dictionary keys.
        """
        return f"dvx_{action[0]}_dvy_{action[1]}"


    # def initialize_q_values(self, env):
    #     """
    #     Initializes Q-values for all possible state-action pairs encountered in the environment.
    #     You might need to explore the state space to pre-initialize or initialize on demand.
    #     For simplicity, we can initialize Q-values to 0 for all state-action pairs encountered during learning.
    #     """
    #     states = env.racetrack
    #     for row in range(states.shape[0]):
    #         for col in range(states.shape[1]):
    #             for action in self.possible_actions:
    #                 state_str = self._get_state_str((row,col))
    #                 action_str = self._get_action_str(action)
    #                 if not self.q_values[(state_str,action_str)]:
    #                     self.q_values[(state_str,action_str)] = 0
    #     pass # Q-values can be initialized on demand when a state-action pair is first encountered.


    def choose_action(self, state, policy_type='behavior', epsilon=0.1):
        """
        Chooses an action based on the specified policy type.

        Args:
            state: Current state.
            policy_type: 'behavior' or 'target'.
            epsilon: Epsilon value for epsilon-greedy behavior policy.

        Returns:
            action: Chosen action (velocity increment tuple).
        """
        state_str = self._get_state_str(state)

        if policy_type == 'behavior':
            # Epsilon-greedy behavior policy
            if random.random() < epsilon:
                return random.choice(self.possible_actions) # Explore
            else:
                if state_str not in self.policy or self.policy[state_str] is None: # If no target policy yet, act randomly
                    return random.choice(self.possible_actions)
                else:
                    return self.policy[state_str] # Exploit based on target policy
        elif policy_type == 'target':
            # Greedy target policy
            if state_str not in self.policy or self.policy[state_str] is None: # If no target policy yet, act randomly
                    return random.choice(self.possible_actions)
            return self.policy[state_str]
        else:
            raise ValueError("Invalid policy_type. Choose 'behavior' or 'target'.")


    def off_policy_mc_control(self, env, num_episodes, epsilon=0.1, discount_reward = 1):
        """
        Implements off-policy Monte Carlo control using importance sampling.

        Args:
            env: Racetrack environment object.
            num_episodes: Number of episodes to run for learning.
            epsilon: Epsilon for the behavior policy.
        """
        C = {} # Cumulative importance sampling ratios for weighted average: C(s, a)
        N = {} # Counts of state-action pairs for simple average (optional, for comparison or debugging)

        for episode_count in range(num_episodes):
            episode = self.run_episode(env, policy_type='behavior', epsilon=epsilon) # Generate episode using behavior policy
            G = 0 # Initialize return
            W = 1 # Initialize importance sampling ratio

            print(f"\n--- Episode {episode_count + 1} ---") # Episode start marker

            for t in reversed(range(len(episode))): # Loop backwards through the episode
                state_t, action_t, reward_t, behavior_policy_prob = episode[t]
                state_str_t = self._get_state_str(state_t)
                action_str_t = self._get_action_str(action_t)
                state_action_pair = (state_str_t, action_str_t)
                # 1. Compute G
                G = discount_reward*G + reward_t

                # behavior_policy_prob = 0.5 #Debug

                # 2. Compute importance sampling ratio (rho_t)
                target_policy_action = self.choose_action(state_t, policy_type='target')
                # Target policy is greedy, prob of chosen action is 1 if it's the greedy action
                pi_prob = 1.0 if target_policy_action == action_t else 0.0

                # # Behavior policy probability (assuming uniform random over 9 actions for simplicity in IS)
                # behavior_policy_prob = 1.0 / len(self.possible_actions) # Uniform random behavior policy

                rho_t = pi_prob / behavior_policy_prob if behavior_policy_prob > 0 else 0 # Avoid division by zero
                W = W * rho_t # Cumulative importance ratio


                # Update Q-value using weighted importance sampling
                if state_action_pair not in C:
                    C[state_action_pair] = 1
                if state_action_pair not in self.q_values:
                    self.q_values[state_action_pair] = 0

                # Update Cumulative W
                C[state_action_pair] += W

                update_amount = (W / C[state_action_pair]) * (G - self.q_values[state_action_pair]) # Calculate update amount
                self.q_values[state_action_pair] += update_amount
            
                # Update Q-value using weighted importance sampling
                # self.q_values[state_action_pair] =  self.q_values[state_action_pair] + (W / C[state_action_pair])*(G -  self.q_values[state_action_pair])

                print(f"\n  Time step t={t}:") # Time step marker
                print(f"    State: {state_t}, Action: {action_t}, Reward: {reward_t}")
                print(f"    Target Action: {target_policy_action}, pi_prob: {pi_prob}, behavior_policy_prob: {behavior_policy_prob:.2f}, rho_t: {rho_t:.2f}, W: {W:.2f}")
                print(f"    Return G: {G:.2f}, Old Q(s,a): {self.q_values[state_action_pair]:.2f}, C(s,a): {C[state_action_pair]:.2f}")
                print(f"    Update Amount: {update_amount:.4f}, New Q(s,a): {self.q_values[state_action_pair]:.2f}")

                best_action = None
                best_q_value = -float('inf')
                for action in self.possible_actions:
                    state_action_test_pair = (self._get_state_str(state_t), self._get_action_str(action))
                    q_val = self.q_values.get(state_action_test_pair, 0) # Default to 0 if not seen yet
                    if q_val > best_q_value:
                        best_q_value = q_val
                        best_action = action
    
                self.policy[self._get_state_str(state_t)] = best_action # Update target policy
                
                if pi_prob == 0: # If target policy probability was 0 for the taken action, then break (as per importance sampling logic, further updates will be 0)
                    break
        print("Off-policy MC learning finished.")


    def run_episode(self, env, policy_type='behavior', epsilon=0.1, disable_stochasticity_for_optimal=False):
        """
        Runs one episode of the Racetrack game.

        Args:
            env: Racetrack environment.
            policy_type: 'behavior' or 'target' policy to use.
            epsilon: Epsilon for behavior policy if used.
            disable_stochasticity_for_optimal: If True, disables the 0.1 stochasticity in the environment (for optimal policy demonstration).

        Returns:
            episode: List of (state, action, reward) tuples for the episode.
        """
        episode = []
        state = env.reset()
        # print("Initial State (Episode Start):", state) # Print initial state
        done = False

        while not done:
            if disable_stochasticity_for_optimal:
                original_stochasticity_prob = 0.1 # Assuming 0.1 is the default in env.
                env.stochasticity_prob = 0.0 # Temporarily disable stochasticity if needed for optimal policy run (if env has such an attribute, or modify step function directly)

            action = self.choose_action(state, policy_type=policy_type, epsilon=epsilon)
            next_state, reward, done, _ = env.step(action) # Take action in env

            # Compute behavior policy probability
            greedy_action = self.policy.get(self._get_state_str(state), None)
            if action == greedy_action:
                behavior_prob = (1 - epsilon) + epsilon/len(self.possible_actions)
            else:
                behavior_prob = epsilon/len(self.possible_actions)
            episode.append((state, action, reward, behavior_prob))  # Store probability
            # episode.append((state, action, reward))
            state = next_state

            if disable_stochasticity_for_optimal:
                env.stochasticity_prob = original_stochasticity_prob # Restore stochasticity if modified

            if done:
                break
        return episode


    def get_optimal_trajectory(self, env, start_state=None, num_steps=100):
        """
        Runs an episode using the learned optimal policy (target policy) without stochasticity to visualize the trajectory.

        Args:
            env: Racetrack environment.
            start_state: Optional starting state (position and velocity). If None, environment is reset to a random start.
            num_steps: Maximum number of steps to run the trajectory for.

        Returns:
            trajectory: List of states visited in the trajectory.
        """
        trajectory = []
        if start_state:
            env.current_state = start_state # Set specific start state if provided
        else:
            env.reset() # Reset to random start if no start_state given
        current_state = env.current_state
        print("Initial State (Trajectory Run):", current_state)
        trajectory.append(current_state)
        done = False
        steps = 0

        while not done and steps < num_steps:
            action = self.choose_action(current_state, policy_type='target') # Choose action from target policy (greedy)
            # print("Chosen Action (Trajectory):", action) # ADDED: Print chosen action
            next_state, reward, done, _ = env.step(action) # Take step in env (stochasticity still active unless you modify env.step to disable it for demonstration)
            trajectory.append(next_state)
            current_state = next_state
            steps += 1
            if done:
                break
        return trajectory



def load_racetrack_map(map_file):
    """
    Loads the racetrack map from a file.
    Assumes the file is a text file where each character represents a cell:
    ' ': Track (0)
    '#': Boundary (1)
    'S': Start Line (2)
    'F': Finish Line (3)
    Args:
        map_file: Path to the map file.

    Returns:
        A 2D list representing the racetrack map.
    """
    racetrack_map_list = []
    with open(map_file, 'r') as f:
        for line in f:
            row = []
            for char in line.strip():
                if char == ' ':
                    row.append(0) # Track
                elif char == '#':
                    row.append(1) # Boundary
                elif char == 'S':
                    row.append(2) # Start Line
                elif char == 'F':
                    row.append(3) # Finish Line
                else:
                    raise ValueError(f"Invalid character in map file: {char}")
            racetrack_map_list.append(row)
    return racetrack_map_list


if __name__ == "__main__":
    # Load racetrack map
    map_file = "./racetrack.txt" # Replace with your map file path
    racetrack_map_data = load_racetrack_map(map_file)

    # Create environment
    env = RacetrackEnv(racetrack_map_data)

    # Create agent
    possible_actions = env.get_possible_actions()
    agent = MCAgent(possible_actions)

    # --- DEBUGGING STEP FUNCTION ---
    debug_env = env # Create a separate env for debugging to not interfere with training.
    debug_env.current_state = [(5, 1), (0, 0)] # Set the state where it's stuck
    debug_action = (1, 1) # Try a positive velocity increment action
    print("\n--- STEP FUNCTION DEBUGGING ---")
    print("Current State (Debug):", debug_env.current_state)
    print("Action (Debug):", debug_action)
    next_state, reward, done, info = debug_env.step(debug_action)
    print("Next State (Debug):", next_state)
    print("Reward (Debug):", reward)
    print("Done (Debug):", done)
    print("Info (Debug):", info)
    print("--- DEBUGGING END ---")

    # Train the agent using Off-Policy Monte Carlo Control
    num_episodes = 300 # Adjust as needed
    agent.off_policy_mc_control(env, num_episodes, epsilon=0.1)

    # Get optimal trajectory and visualize (without stochasticity for visualization)
    print("Running optimal policy trajectory (no stochasticity):")
    optimal_trajectory = agent.get_optimal_trajectory(env, num_steps=500) # You might need to modify get_optimal_trajectory to disable stochasticity in env.step for demonstration
    
    print("\n--- Learned Policy (Complete) ---")
    for state_str, action in agent.policy.items():
        print(f"State: {state_str}, Action: {action}")
    print("--- Policy End ---\n")
    print("\n--- Q-values for Start States ---")
    for start_pos in env.start_line_positions:
        start_state_str = agent._get_state_str((start_pos, (0, 0)))
        print(f"State: {start_state_str}")
        for action in agent.possible_actions:
            state_action_pair = (start_state_str, agent._get_action_str(action))
            q_value = agent.q_values.get(state_action_pair, 0) # Get Q-value, default to 0 if not seen
            print(f"  Action: {action}, Q-value: {q_value}")
    print("--- Q-values End ---")
    for state in optimal_trajectory:
        env.render(state) # If render function is implemented.
    env.render_matplotlib(optimal_trajectory) # Render trajectory using matplotlib
    print("Optimal policy trajectory finished.")