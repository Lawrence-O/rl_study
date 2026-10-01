import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # For drawing arrows in policy grid
import random

# Define Gridworld (example 4x4 grid as described)
GRID_SIZE = 5
TERMINAL_STATES = [(0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)]
# TERMINAL_STATES = [(GRID_SIZE // 2 , GRID_SIZE // 2 )]
REWARD_NON_TERMINAL = -1
REWARD_TERMINAL = 10
DISCOUNT_FACTOR = 0.1
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
POLICY_PROB = 0.25
P_SUCCESS_REWARD = 1 # For stochastic rewards example (optional - can set to 1.0 for deterministic)

def get_state_index(row, col):
    return row * GRID_SIZE + col

def get_coords_from_index(index):
    return index // GRID_SIZE, index % GRID_SIZE

def is_terminal_state(row, col):
    return (row, col) in TERMINAL_STATES

def is_out_of_bounds(row, col):
    return not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE)

def get_reward(row, col):
    if is_terminal_state(row, col):
        return REWARD_TERMINAL
    elif is_out_of_bounds(row, col):
        return -10
    else:
        return REWARD_NON_TERMINAL
        # if random.random() < P_SUCCESS_REWARD: # Stochastic reward example
        #     return REWARD_NON_TERMINAL
        # else:
        #     return 0
        # return REWARD_NON_TERMINAL # Deterministic reward (original)

def get_next_state(row, col, action):
    action_dict = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1)
    }
    row_change, col_change = action_dict[action]
    new_row, new_col = row + row_change, col + col_change

    if not (0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE):
        return row, col
    return new_row, new_col

def policy_evaluation(convergence_threshold=1e-6, visualize_iterations=False, save_visualizations=False):
    """
    Implement Iterative Policy Evaluation with visualizations.
    """
    V = np.zeros(GRID_SIZE * GRID_SIZE)
    delta_history = []
    delta = float('inf')
    iteration = 0
    value_history = [] # To store value function at each iteration

    if visualize_iterations:
        print("Initial Value Function (V):")
        print_value_grid(V)
        value_history.append(V.copy()) # Store initial V

    while delta > convergence_threshold and iteration < 1e5:
        delta = 0
        iteration += 1
        if visualize_iterations:
            print(f"\nIteration: {iteration}")

        V_iteration = np.copy(V) # Store value function for this iteration for visualization
        for state_index in range(GRID_SIZE * GRID_SIZE):
            row, col = get_coords_from_index(state_index)
            if is_terminal_state(row, col):
                continue

            v_old = V[state_index]
            v_new = 0

            for action in ACTIONS:
                next_row, next_col = get_next_state(row, col, action)
                next_state_index = get_state_index(next_row, next_col)
                reward = get_reward(next_row, next_col)
                v_new += POLICY_PROB * (reward + DISCOUNT_FACTOR * V[next_state_index])

            V[state_index] = v_new
            delta = max(delta, abs(v_new - v_old))

        delta_history.append(delta)
        value_history.append(V.copy()) # Store V after iteration
        if visualize_iterations:
            print_value_grid(V)
        if save_visualizations and iteration % 5 == 0: # Save visualizations every 5 iterations
            visualize_value_function_heatmap(V, iteration, save=True)
            visualize_policy_arrows(V, iteration, save=True)


    if save_visualizations: # Final visualizations after convergence
        visualize_value_function_heatmap(V, iteration, save=True, final=True)
        visualize_policy_arrows(V, iteration, save=True, final=True)
    else:
        visualize_value_function_heatmap(V, iteration, save=False, final=True)
        visualize_policy_arrows(V, iteration, save=False, final=True)


    return V, delta_history, value_history

def print_value_grid(value_function):
    for row in range(GRID_SIZE):
        row_values = []
        for col in range(GRID_SIZE):
            state_index = get_state_index(row, col)
            row_values.append(f"{value_function[state_index]:.2f}")
        print(row_values)

def plot_convergence(deltas):
    plt.figure(figsize=(8, 6))
    plt.plot(deltas)
    plt.xlabel("Iteration")
    plt.ylabel("Delta (Maximum Value Change)")
    plt.title("Convergence of Iterative Policy Evaluation")
    plt.grid(True)
    plt.show()

def visualize_value_function_heatmap(value_function, iteration, save=False, final=False):
    """Visualizes the value function as a heatmap."""
    value_grid = value_function.reshape((GRID_SIZE, GRID_SIZE))
    plt.figure(figsize=(6, 5))
    plt.imshow(value_grid, cmap='viridis', interpolation='nearest') # Use 'viridis' or other cmap
    plt.colorbar(label='State Value (V)')
    plt.title(f'Value Function - Iteration {iteration if not final else "Final"}')

    # Annotate cells with value
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            plt.text(j, i, f'{value_grid[i, j]:.2f}', ha="center", va="center", color="black")

    plt.xticks([]) # Remove x ticks
    plt.yticks([]) # Remove y ticks
    plt.gca().invert_yaxis() # Invert y axis to match gridworld orientation

    if save:
        filename = f"value_heatmap_iter_{iteration:03d}.png" if not final else "value_heatmap_final.png"
        plt.savefig(filename)
        plt.close() # Close figure to prevent display in notebook if saving
    else:
        plt.show()

def visualize_policy_arrows(value_function, iteration, save=False, final=False):
    """Visualizes the greedy policy based on value function using arrows."""
    policy_grid = np.empty((GRID_SIZE, GRID_SIZE), dtype=object) # To store actions as arrows
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    ax.set_xlim(-0.5, GRID_SIZE - 0.5) # Set limits for grid lines
    ax.set_ylim(GRID_SIZE - 0.5, -0.5) # Inverted y axis
    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1)) # Grid line positions
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1))
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False) # Remove ticks/labels
    ax.grid(True, linewidth=1, color='black') # Display grid lines

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if is_terminal_state(row, col):
                policy_grid[row, col] = "T" # Mark terminal state
                continue

            state_index = get_state_index(row, col)
            q_values = {}
            for action in ACTIONS:
                next_row, next_col = get_next_state(row, col, action)
                next_state_index = get_state_index(next_row, next_col)
                reward = get_reward(next_row, next_col) # Reward not actually used for policy, but included for consistency
                q_values[action] = reward + DISCOUNT_FACTOR * value_function[next_state_index] # Q-value based on V

            best_action = max(q_values, key=q_values.get) # Greedy action
            policy_grid[row, col] = best_action

            # Draw arrows for policy visualization
            arrow_dict = {"UP": (0, -0.3), "DOWN": (0, 0.3), "LEFT": (-0.3, 0), "RIGHT": (0.3, 0)} # Arrow offsets
            arrow_offset = arrow_dict[best_action]
            ax.add_patch(mpatches.Arrow(col, GRID_SIZE - 1 - row, arrow_offset[0], arrow_offset[1], width=0.15, color='blue')) # Draw arrow

    plt.title(f'Greedy Policy - Iteration {iteration if not final else "Final"}')
    if save:
        filename = f"policy_arrows_iter_{iteration:03d}.png" if not final else "policy_arrows_final.png"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    estimated_v_pi, deltas, value_history = policy_evaluation(visualize_iterations=True, save_visualizations=False) # visualize_iterations=False, save_visualizations=True
    print("\nFinal Estimated State-Value Function (v_pi) for the Random Policy:")
    print_value_grid(estimated_v_pi)
    plot_convergence(deltas)