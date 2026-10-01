import numpy as np
import matplotlib.pyplot as plt

'''
Exercise 4.9 (programming) Implement value iteration for the gambler’s problem and
solve it for ph = 0.25 and ph = 0.55. In programming, you may find it convenient to
introduce two dummy states corresponding to termination with capital of 0 and 100,
giving them values of 0 and 1 respec
'''

def _gamblers_problem(ph, theta=1e-9):
    # Initialize value function (0 to 100 inclusive)
    V = np.zeros(101)
    discount = 1

    while True:
        delta = 0
        V_prev = np.copy(V)
        for s in range(1, 100): # (1,99)
            current_max = -np.inf
            # Determine valid actions for state s
            max_stake = min(s, 100 - s)
            for a in range(0, max_stake + 1):
                # Compute next states and their values
                if a == 0:
                    current_val = 0.0
                else:
                    win_state = s + a # win the gamble and earns a from his state
                    lose_state = s - a # loses the gamble and removes a from his state

                    # reward calculation:
                    win_reward = 1 if win_state == 100 else 0
                    lose_reward = 0.0

                    # calculate expectation
                    current_val = ph * (win_reward + discount*V_prev[win_state]) + (1 - ph) * (lose_reward + discount*V_prev[lose_state])
                
                # Update current_max if expected value is higher
                current_max = max(current_max, current_val)
            # Update delta and V_new[s]
            V[s] = current_max
            delta = max(delta, abs(V[s] - V_prev[s]))
        # Break condition and update V
        if delta < theta:
            break
    return V, extract_policy(V, ph)
def gamblers_problem(ph, theta=1e-9):
    V = np.zeros(101)  # States 0-100 (terminal states at 0)
    discount = 1
    sweep_history = []  # Store value functions after each sweep
    
    while True:
        delta = 0
        V_prev = np.copy(V)
        sweep_history.append(V_prev.copy())  # Capture before update
        
        for s in range(1, 100):
            max_stake = min(s, 100 - s)
            current_max = -np.inf
            
            for a in range(0, max_stake + 1):
                if a == 0:
                    current_val = 0.0
                else:
                    win_state = min(s + a, 100)
                    lose_state = max(s - a, 0)
                    
                    win_reward = 1.0 if win_state == 100 else 0.0
                    current_val = ph * (win_reward + V_prev[win_state]) + \
                                (1 - ph) * V_prev[lose_state]
                
                current_max = max(current_max, current_val)
            
            delta = max(delta, abs(current_max - V_prev[s]))
            V[s] = current_max
        
        if delta < theta:
            sweep_history.append(V.copy())  # Capture final update
            break
    
    return V, extract_policy(V, ph), sweep_history
def plot_sweeps(sweep_history, ph):
    plt.figure(figsize=(10, 6))
    
    # Select key sweeps: first 3 and either 32nd or final
    target_sweeps = [1, 2, 3, 32] if ph == 0.4 else [1, 2, 3, -1]
    
    for sweep in target_sweeps:
        if sweep < len(sweep_history):
            label = f'Sweep {sweep}' if sweep != -1 else f'Sweep {len(sweep_history)-1}'
            plt.plot(sweep_history[sweep][1:100], label=label)
    
    plt.title(f'Value Function Evolution (ph = {ph})')
    plt.xlabel('Capital')
    plt.ylabel('Probability of Winning')
    plt.legend()
    plt.ylim(0, 1)
    plt.show()

def extract_policy(V, ph, discount=1):
    policy = np.zeros(100)  # States 1-99 (0 and 100 are terminal)
    for s in range(1, 100):
        max_val = -1
        best_action = 0
        max_stake = min(s, 100 - s)
        for a in range(0, max_stake + 1):
            # Compute expected value for each action
            win_state = s + a
            lose_state = s - a
            # reward calculation:
            win_reward = 1 if win_state == 100 else 0
            lose_reward = 0.0
            expected_val = ph * (win_reward + discount*V[win_state]) + (1-ph)*(lose_reward + discount*V[lose_state])
            # TODO: Find the best action(s)
            if expected_val > max_val or (expected_val == max_val and a < best_action):
                max_val = expected_val
                best_action = a
        policy[s-1] = best_action  # Adjust index for 0-99
    return policy

def plot_results(V, policy, ph):
    plt.figure(figsize=(10, 6))
    
    # Value Function Plot
    plt.subplot(2, 1, 1)
    plt.plot(V[1:100], 'b-')
    plt.title(f'Value Function (ph = {ph})')
    plt.xlabel('Capital')
    plt.ylabel('Probability of Winning')
    plt.ylim(0, 1)
    
    # Policy Plot
    plt.subplot(2, 1, 2)
    plt.stem(policy, linefmt='C0-', markerfmt='C0o', basefmt='C0-')
    plt.title(f'Optimal Policy (ph = {ph})')
    plt.xlabel('Capital')
    plt.ylabel('Stake')
    plt.ylim(0, 50 if ph < 0.5 else 100)
    
    plt.tight_layout()
    plt.show()

# Run for ph=0.25 and ph=0.55
if __name__ == "__main__":
    # for ph in [0.25, 0.4, 0.55]:
    #     # V, policy = gamblers_problem(ph)
    #     # plot_results(V, policy, ph)
    V, policy, history = gamblers_problem(ph=0.4, theta=1e-9)
    plot_sweeps(history, ph=0.4)
    
    # For exercise solutions (ph=0.25 and 0.55)
    for ph in [0.25, 0.4, 0.55]:
        V, policy, history = gamblers_problem(ph)
        plot_sweeps(history, ph)
        plot_results(V, policy, ph)