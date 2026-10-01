import numpy as np
import matplotlib.pyplot as plt
import random  # Import the random module

# Define Gridworld (example 4x4 grid as described)
GRID_SIZE = 4
# TERMINAL_STATES = [(0, 0), (3, 2)] # Coordinates of terminal states
TERMINAL_STATES = [(0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)]
REWARD_NON_TERMINAL = -1
REWARD_TERMINAL = 0
DISCOUNT_FACTOR = 0.9
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
POLICY_PROB = 0.25 # Probability of each action in the uniform random policy
P_SUCCESS_REWARD = 1  # Probability of getting +1 reward in non-terminal state

def get_state_index(row, col):
    """Convert row and column to a single state index."""
    # TODO: Implement the logic to convert row and column to a single state index.
    # Example: row * GRID_SIZE + col
    return row * GRID_SIZE + col

def get_coords_from_index(index):
    """Convert state index back to row and column coordinates."""
    # TODO: Implement the logic to convert a state index back to row and column coordinates.
    # Example: index // GRID_SIZE, index % GRID_SIZE
    return index // GRID_SIZE, index % GRID_SIZE

def is_terminal_state(row, col):
    """Check if a given coordinate is a terminal state."""
    # TODO: Implement the logic to check if the given (row, col) is in TERMINAL_STATES.
    return (row, col) in TERMINAL_STATES

def deterministic_get_reward(row, col):
    """Get the reward for being in a given state (coordinate)."""
    # TODO: Implement the reward logic.
    # If it's a terminal state, return REWARD_TERMINAL, otherwise return REWARD_NON_TERMINAL.
    return REWARD_TERMINAL if is_terminal_state(row ,col ) else REWARD_NON_TERMINAL
def get_reward(row, col):
    """Get the reward for being in a given state (coordinate), now stochastic."""
    if is_terminal_state(row, col):
        return REWARD_TERMINAL  # Terminal states still have deterministic reward
    else:
        # Sample reward from Bernoulli distribution
        if random.random() < P_SUCCESS_REWARD:
            return REWARD_NON_TERMINAL # +1 reward with probability P_SUCCESS_REWARD
        else:
            return 0 # 0 reward with probability (1 - P_SUCCESS_REWARD)

def get_next_state(row, col, action):
    """
    Determine the next state (coordinates) given current state and action.
    Handles boundary conditions (staying in the same state if action is off-grid).
    """
    # new_row, new_col = row, col # Default: stay in the same state

    # TODO: Implement the logic to determine the next state based on the action.
    # Consider actions: "UP", "DOWN", "LEFT", "RIGHT".
    # Ensure you handle boundary conditions: if an action takes the agent off-grid, it stays in the same state.
    action_dict = {
        "UP" : (-1,0),
        "DOWN" : (1,0),
        "LEFT" : (0,-1),
        "RIGHT": (0,1)
    }

    new_row, new_col = action_dict[action]

    if not (0 <= new_row < GRID_SIZE ) or not(0 <= new_col < GRID_SIZE):
        return row , col 
    
    return new_row, new_col

def policy_evaluation(convergence_threshold=1e-1, visualize_iterations=False):
    """
    Implement Iterative Policy Evaluation to estimate v_pi.
    Args:
        convergence_threshold: Threshold for convergence check.
        visualize_iterations: If True, print grid values at each iteration.
    Returns:
        V: Estimated state-value function.
        deltas: List of delta values for each iteration (for convergence plot).
    """
    V = np.zeros(GRID_SIZE * GRID_SIZE) # Initialize V(s) to 0 for all states (indexed 0 to 15)
    delta_history = []
    delta = float('inf')
    iteration = 0

    if visualize_iterations:
        print("Initial Value Function (V):")
        print_value_grid(V)

    while delta > convergence_threshold and iteration < 1e5:
        delta = 0
        iteration += 1
        if visualize_iterations:
            print(f"\nIteration: {iteration}")

        for state_index in range(GRID_SIZE * GRID_SIZE): # Iterate through all states (0 to 15)
            row, col = get_coords_from_index(state_index)

            if is_terminal_state(row, col): # Skip terminal states
                continue

            v_old = V[state_index]
            v_new = 0

            # TODO: Implement the Bellman Expectation Equation here.
            # For each action in ACTIONS:
            #   1. Get the next state (next_row, next_col) using get_next_state(row, col, action).
            #   2. Convert next state coordinates to state index: next_state_index = get_state_index(next_row, next_col).
            #   3. Get the reward for the next state: reward = get_reward(next_row, next_col).
            #   4. Update v_new using the Bellman Expectation Equation:
            #      v_new += POLICY_PROB * (reward + DISCOUNT_FACTOR * V[next_state_index])

            for action in ACTIONS:
                next_row, next_col = get_next_state(row,col,action)
                next_state_index = get_state_index(next_row,next_col)
                reward = get_reward(next_row, next_col)
                v_new += POLICY_PROB * (reward + DISCOUNT_FACTOR * V[next_state_index])

            V[state_index] = v_new
            delta = max(delta, abs(v_new - v_old))

        delta_history.append(delta)
        if visualize_iterations:
            print_value_grid(V)

    return V, delta_history

def print_value_grid(value_function):
    """Prints the value function in a grid format."""
    for row in range(GRID_SIZE):
        row_values = []
        for col in range(GRID_SIZE):
            state_index = get_state_index(row, col)
            row_values.append(f"{value_function[state_index]:.2f}") # Format to 2 decimal places
        print(row_values)

def plot_convergence(deltas):
    """Plots the convergence of Policy Evaluation."""
    plt.figure(figsize=(8, 6))
    plt.plot(deltas)
    plt.xlabel("Iteration")
    plt.ylabel("Delta (Maximum Value Change)")
    plt.title("Convergence of Iterative Policy Evaluation")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    estimated_v_pi, deltas = policy_evaluation(visualize_iterations=True) # Set visualize_iterations=True to see grid updates

    print("\nFinal Estimated State-Value Function (v_pi) for the Random Policy:")
    print_value_grid(estimated_v_pi)

    plot_convergence(deltas)