import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

# Define Gridworld (example 5x5 grid)
GRID_SIZE = 5
# TERMINAL_STATES = [(GRID_SIZE // 2 , GRID_SIZE // 2 )] # Center cell as terminal
TERMINAL_STATES = [(0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)]
REWARD_NON_TERMINAL = -1
REWARD_TERMINAL = 1
DISCOUNT_FACTOR = 0.6
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
POLICY_PROB = 0.25
P_SUCCESS_REWARD = 1  # Deterministic reward for simplicity

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
        return REWARD_NON_TERMINAL
    else:
        return REWARD_NON_TERMINAL
        # if random.random() < P_SUCCESS_REWARD: # Stochastic reward example (currently deterministic as P_SUCCESS_REWARD = 1.0)
        #     return REWARD_NON_TERMINAL
        # else:
        #     return 0

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

def policy_evaluation_step(policy, V, convergence_threshold=1e-6):
    """
    One step of Policy Evaluation used within Policy Iteration.
    """
    delta = float('inf')
    iter1 = 0
    while delta > convergence_threshold: # Continue evaluation until convergence
        delta = 0
        iter1 += 1
        for state_index in range(GRID_SIZE * GRID_SIZE):
            row, col = get_coords_from_index(state_index)
            if is_terminal_state(row, col):
                continue

            # v_old = V[state_index]
            # v_new = 0
            # action = policy[row, col] # Get the action from the current policy
            # next_row, next_col = get_next_state(row, col, action)
            # next_state_index = get_state_index(next_row, next_col)
            # reward = get_reward(next_row, next_col)
            # v_new = reward + DISCOUNT_FACTOR * V[next_state_index] # Policy evaluation update for the chosen action

            # V[state_index] = v_new
            # delta = max(delta, abs(v_new - v_old))
            v_new = 0
            # Average over all actions (for stochastic policies)
            for action in ACTIONS:
                next_row, next_col = get_next_state(row, col, action)
                next_state_index = get_state_index(next_row, next_col)
                reward = get_reward(next_row, next_col)
                # Weight by POLICY_PROB (e.g., 0.25 for uniform random)
                v_new += POLICY_PROB * (reward + DISCOUNT_FACTOR * V[next_state_index])

            delta = max(delta, abs(V[state_index] - v_new))
            V[state_index] = v_new
    return V


def policy_iteration(convergence_threshold=1e-6, visualize_iterations=True, save_visualizations=False):
    """
    Implement Policy Iteration with visualizations.
    """
    V = np.zeros(GRID_SIZE * GRID_SIZE)
    V[get_state_index(0, 0)] = REWARD_TERMINAL  # Terminal (0,0)
    V[get_state_index(4, 4)] = REWARD_TERMINAL  # Terminal (4,4)
    policy = np.array([random.choice(ACTIONS) for _ in range(GRID_SIZE * GRID_SIZE)]) # Initialize with a random policy
    policy = policy.reshape((GRID_SIZE, GRID_SIZE))
    policy_stable = False
    iteration = 0

    while not policy_stable and iteration < 1e4:
        iteration += 1
        policy_stable = False
        if visualize_iterations:
            print(f"\nPolicy Iteration - Iteration: {iteration}")

        # **1. Policy Evaluation:**
        V = policy_evaluation_step(policy, V, convergence_threshold) # Use a single step for faster iteration in Policy Iteration

        # **2. Policy Improvement:**
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if is_terminal_state(row, col):
                    continue

                old_action = policy[row, col]
                q_values = {}
                for action in ACTIONS:
                    next_row, next_col = get_next_state(row, col, action)
                    next_state_index = get_state_index(next_row, next_col)
                    reward = get_reward(next_row, next_col)
                    q_values[action] = reward + DISCOUNT_FACTOR * V[next_state_index]

                best_action = max(q_values, key=q_values.get)
                policy[row, col] = best_action
                if old_action != best_action:
                    policy_stable = False # Policy changed, need another iteration
                # print("Done {} {}".format(row, col))

        # if visualize_iterations:
        #     visualize_value_function_heatmap(V, iteration, save=False)
        #     visualize_policy_arrows_policy_iteration(policy, iteration, save=False) # Modified policy arrow visualization for policy iteration
        # if save_visualizations and iteration % 5 == 0:
        #     visualize_value_function_heatmap(V, iteration, save=True)
        #     visualize_policy_arrows_policy_iteration(policy, iteration, save=True)

    if save_visualizations:
        visualize_value_function_heatmap(V, iteration, save=True, final=True)
        visualize_policy_arrows_policy_iteration(policy, iteration, save=True, final=True)
    else:
        visualize_value_function_heatmap(V, iteration, save=False, final=True)
        visualize_policy_arrows_policy_iteration(policy, iteration, save=False, final=True)

    return policy, V

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
    plt.title("Convergence of Iterative Policy Evaluation") # Title might be slightly misleading now, consider changing if only used for Policy Iteration's evaluation step
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


def visualize_policy_arrows_policy_iteration(policy, iteration, save=False, final=False):
    """Visualizes the policy from Policy Iteration using arrows."""
    policy_grid = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1))
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1))
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    ax.grid(True, linewidth=1, color='black')

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if is_terminal_state(row, col):
                policy_grid[row, col] = "T"
                continue

            best_action = policy[row, col] # Get action directly from policy
            policy_grid[row, col] = best_action

            arrow_dict = {"UP": (0, -0.3), "DOWN": (0, 0.3), "LEFT": (-0.3, 0), "RIGHT": (0.3, 0)}
            arrow_offset = arrow_dict[best_action]
            ax.add_patch(mpatches.Arrow(col, GRID_SIZE - 1 - row, arrow_offset[0], arrow_offset[1], width=0.15, color='blue'))

    plt.title(f'Policy Iteration Policy - Iteration {iteration if not final else "Final"}')
    if save:
        filename = f"policy_arrows_policy_iter_{iteration:03d}.png" if not final else "policy_arrows_policy_final.png"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def visualize_policy_arrows(value_function, iteration, save=False, final=False): # Original policy arrow visualization - not directly used in main execution for Policy Iteration
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

    plt.title(f'Greedy Policy - Iteration {iteration if not final else "Final"}') # Title might be misleading now, consider changing if not used for Policy Evaluation
    if save:
        filename = f"policy_arrows_iter_{iteration:03d}.png" if not final else "policy_arrows_final.png"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    optimal_policy, optimal_v = policy_iteration(visualize_iterations=True, save_visualizations=False) # Policy Iteration
    print("\nOptimal Policy (from Policy Iteration):")
    for row in range(GRID_SIZE):
        print(optimal_policy[row, :])
    print("\nOptimal Value Function (v*) (from Policy Iteration):")
    print_value_grid(optimal_v)