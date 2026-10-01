import numpy as np
import matplotlib.pyplot as plt

# Parameters
k = 10
n_runs = 200
n_steps = 100
epsilon = 0.1
alpha = 0.1
step_size_walk = 0.01

# Track performance metrics
rewards_sample = np.zeros(n_steps)
rewards_alpha = np.zeros(n_steps)
optimal_sample = np.zeros(n_steps)
optimal_alpha = np.zeros(n_steps)

for run in range(n_runs):
    print(f'Run: {run+1}/{n_runs}', end='\r')
    # Initialize true action values (q*) and estimates (Q) for both methods
    q_true = np.zeros(k)
    Q_sample = np.zeros(k)
    Q_alpha = np.zeros(k)
    N_sample = np.zeros(k)  # Action counts for sample-average
    
    for step in range(n_steps):
        print(f'Run: {run+1}/{n_runs}, Step: {step+1}/{n_steps}', end='\r')
        # Add random walk to q* (nonstationarity)
        q_true += np.random.normal(0, step_size_walk, k)
        current_optimal = np.argmax(q_true)
        
        # Epsilon-greedy action selection for both methods
        # --- Sample-Average Method ---
        if np.random.rand() < epsilon:
            action_sample = np.random.randint(k)
        else:
            action_sample = np.argmax(Q_sample)
        
        # Get reward and update Q for sample-average
        reward_sample_val = np.random.normal(q_true[action_sample], 1)
        N_sample[action_sample] += 1
        Q_sample[action_sample] += (reward_sample_val - Q_sample[action_sample]) / N_sample[action_sample]
        
        # Track performance
        rewards_sample[step] += reward_sample_val
        optimal_sample[step] += (action_sample == current_optimal)
        
        # --- Constant Step-Size (α=0.1) Method ---
        if np.random.rand() < epsilon:
            action_alpha = np.random.randint(k)
        else:
            action_alpha = np.argmax(Q_alpha)
        
        # Get reward and update Q for constant α
        reward_alpha_val = np.random.normal(q_true[action_alpha], 1)
        Q_alpha[action_alpha] += alpha * (reward_alpha_val - Q_alpha[action_alpha])
        
        # Track performance
        rewards_alpha[step] += reward_alpha_val
        optimal_alpha[step] += (action_alpha == current_optimal)

# Average across runs
rewards_sample /= n_runs
rewards_alpha /= n_runs
optimal_sample = (optimal_sample / n_runs) * 100  # Convert to percentage
optimal_alpha = (optimal_alpha / n_runs) * 100

# Plot results
plt.figure(figsize=(12, 5))

# Average Reward
plt.subplot(1, 2, 1)
plt.plot(rewards_sample, label='Sample Average')
plt.plot(rewards_alpha, label=r'Constant $\alpha=0.1$')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)

# Optimal Action %
plt.subplot(1, 2, 2)
plt.plot(optimal_sample, label='Sample Average')
plt.plot(optimal_alpha, label=r'Constant $\alpha=0.1$')
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()