from collections import defaultdict
import pickle
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
from pathlib import Path
from typing import NamedTuple
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cupy as cp
import random
from tqdm import tqdm

class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    savefig_folder: Path  # Root folder where plots are saved

params = Params(
    total_episodes=500,
    learning_rate=0.8,
    gamma=0.9,
    epsilon=0.1,
    map_size=5,
    seed=42, #12345
    n_runs=20,
    action_size=None,
    state_size=None,
    savefig_folder=Path("./plots/nstep_pt2/"),
)

params = params._replace(action_size=4)
params = params._replace(state_size=48)

class ParallelNStepTD:
    def __init__(self, n_values, env_name, params, num_trajectories=1):
        self.n_values = n_values
        self.params = params
        self.gamma = params.gamma
        self.epsilon = params.epsilon
        self.alpha = params.learning_rate
        self.num_episodes = params.total_episodes
        self.state_space = np.array(list(range(params.state_size)))
        self.action_space = np.array(list(range(params.action_size)))
        self.env_name = env_name
        self.num_trajectories = num_trajectories
        np.random.seed(params.seed)
        random.seed(params.seed)

        #Initial Q Values
        self.cu_q_values = {n: cp.random.uniform(low=0, high=0.01, size=(params.state_size, params.action_size)) for n in n_values}
        self.vec_env = AsyncVectorEnv([lambda: gym.make(env_name) for _ in range(min(32, num_trajectories))])
        self.env = gym.make(env_name)
    def select_action_single(self, state):
        """Select action for a single state (used for MC estimation)"""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            # Consider all n-values and take the maximum Q-value across them
            q_stack = cp.stack([self.cu_q_values[n][state] for n in self.n_values])
            max_q_values = cp.max(q_stack, axis=0).get()  # Transfer to CPU only at the end
            return np.argmax(max_q_values)
        
    def old_select_actions(self, states):
        batch_size = len(states)
        
        # Move states to GPU
        cu_states = cp.array(states, dtype=cp.int32)
        
        # Generate random values for epsilon-greedy
        random_values = np.random.random(batch_size)
        explore_mask = random_values < self.epsilon
        exploit_mask = ~explore_mask
        
        # Initialize actions array
        actions = np.zeros(batch_size, dtype=int)
        
        # For exploration: choose random actions
        if np.any(explore_mask):
            actions[explore_mask] = np.random.choice(self.action_space, size=np.sum(explore_mask))
        
        # For exploitation: choose greedy actions
        if np.any(exploit_mask):
            # Create a stack of Q-values for all n
            q_max = None
            exploit_states = cp.array(np.array(states)[exploit_mask], dtype=cp.int32)
            
            for n in self.n_values:
                q_values = self.cu_q_values[n][exploit_states]
                if q_max is None:
                    q_max = q_values
                else:
                    q_max = cp.maximum(q_max, q_values)
            
            # Get best actions
            best_actions = cp.argmax(q_max, axis=1).get()
            actions[exploit_mask] = best_actions
        if random.random() < 0.001:  # 0.1% of the time
            print(f"Action distribution: {np.bincount(actions, minlength=4)}")
        
        return actions
    def select_actions(self, states):
        states_gpu = cp.asarray(states, dtype=cp.int32)
        batch_size = len(states)
        
        # Epsilon-greedy masks
        explore_mask = np.random.rand(batch_size) < self.epsilon
        exploit_mask = ~explore_mask

        actions = np.empty(batch_size, dtype=int)
        
        # Exploration
        if np.any(explore_mask):
            actions[explore_mask] = np.random.choice(self.action_space, 
                                           size=np.sum(explore_mask))
        
        # Exploitation
        if np.any(exploit_mask):
            exploit_states = states_gpu[exploit_mask]
            
            # Stack Q-values for all n
            all_q = cp.stack([self.cu_q_values[n][exploit_states] 
                            for n in self.n_values])
            q_max = cp.max(all_q, axis=0)
            best_actions = cp.argmax(q_max, axis=1).get()
            
            actions[exploit_mask] = best_actions
        return actions
        
    def generate_trajectories(self):
        n_envs = self.vec_env.num_envs
        trajectories = []
        for batch_start in range(0, self.num_trajectories, n_envs):
            batch_size = min(n_envs, self.num_trajectories - batch_start)

            #Initialize batch trajectories
            batch_trajectories = [{'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []} 
                             for _ in range(batch_size)]
            
            states, _ = self.vec_env.reset()
            states = states[:batch_size]
            trajectory_done = np.zeros(batch_size, dtype=bool)
            # Add maximum step limit to prevent infinite loops
            max_steps = 1000
            steps = 0
            while not np.all(trajectory_done):
                actions = self.select_actions(states)
                next_states, rewards, terminated, truncated, _ = self.vec_env.step(actions)

                new_dones = np.logical_or(terminated, truncated)

                # Store transitions
                for i in range(batch_size):
                    if not trajectory_done[i]:
                        batch_trajectories[i]['states'].append(states[i])
                        batch_trajectories[i]['actions'].append(actions[i])
                        batch_trajectories[i]['rewards'].append(rewards[i])
                        batch_trajectories[i]['next_states'].append(next_states[i])
                        batch_trajectories[i]['dones'].append(new_dones[i])
                    # Mark trajectory as done if episode ended
                    if new_dones[i]:
                        trajectory_done[i] = True

                states = next_states
                steps += 1
                
                # Break if taking too long
                if steps >= max_steps:
                    print(f"Hit max steps ({max_steps}) in trajectory generation")
                    # Mark remaining trajectories as done
                    for i in range(batch_size):
                        if not trajectory_done[i] and batch_trajectories[i]['states']:
                            batch_trajectories[i]['dones'][-1] = True
                            trajectory_done[i] = True  # Add this line
                    
                    break
            trajectories.extend(batch_trajectories)
            # Check if we have enough trajectories
            if len(trajectories) >= self.num_trajectories:
                break
        return trajectories[:self.num_trajectories]
    def calculate_n_step_returns(self, trajectories):

        batch_size = len(trajectories)
        max_trajectory_length = max(len(traj['states']) for traj in trajectories)

        # Create Arrays
        states = cp.array([traj['states'] + [0] * (max_trajectory_length - len(traj['states'])) for traj in trajectories], dtype=cp.int32)
        actions = cp.array([traj['actions'] + [0] * (max_trajectory_length - len(traj['actions'])) for traj in trajectories], dtype=cp.int32)
        rewards = cp.array([traj['rewards'] + [0] * (max_trajectory_length - len(traj['rewards'])) for traj in trajectories], dtype=cp.float32)
        next_states = cp.array([traj['next_states'] + [0] * (max_trajectory_length - len(traj['next_states'])) for traj in trajectories], dtype=cp.int32)
        dones = cp.array([traj['dones'] + [True] * (max_trajectory_length - len(traj['dones'])) for traj in trajectories], dtype=cp.bool_)
        traj_lengths = cp.array([len(traj['states']) for traj in trajectories], dtype=cp.int32)

        # N Step Return Arrays
        n_step_returns = {}

        # Calculate N Step Returns
        for n in self.n_values:
            # Pre-allocate arrays for results
            all_states = []
            all_actions = []
            all_returns = []

            # Process in Batches
            for batch_idx in range(0, batch_size, 32):
                batch_end = min(batch_idx + 32, batch_size)
                batch_size_actual = batch_end - batch_idx

                #Extract the batch's data
                batch_states = states[batch_idx:batch_end]
                batch_actions = actions[batch_idx:batch_end]
                batch_rewards = rewards[batch_idx:batch_end]
                batch_next_states = next_states[batch_idx:batch_end]
                batch_dones = dones[batch_idx:batch_end]
                batch_lengths = traj_lengths[batch_idx:batch_end]

                # Vectorized N Step Returns
                for t_offset in range(max_trajectory_length):
                    # Broadcast 1D array to 2D array and then compare
                    # Create an array of valid steps for each time step, each trajectory in the batch
                    valid_steps = cp.arange(batch_size_actual)[:, None] < batch_lengths[:, None]
                    # Create a mask to identify which time steps are within valid range for each trajectory
                    valid_steps = valid_steps & (cp.arange(max_trajectory_length)[None, :] + t_offset < batch_lengths[:, None])
                    valid_n_steps = valid_steps & (cp.arange(max_trajectory_length)[None, :] + t_offset + n <= batch_lengths[:, None])

                    # Check if there any valid trajectories to process
                    if not cp.any(valid_n_steps):
                        continue

                    # Extract states and actions at t
                    t_states = batch_states[valid_steps & (cp.arange(max_trajectory_length)[None, :] == t_offset)]
                    t_actions = batch_actions[valid_steps & (cp.arange(max_trajectory_length)[None, :] == t_offset)]

                    if len(t_states) == 0:
                        continue

                    # Compute Returns
                    returns = cp.zeros(len(t_states), dtype=cp.float32)

                    #Add Discount Rewards
                    for j in range(n):
                        step_idx = t_offset + j
                        if step_idx >= max_trajectory_length:
                            break
                        # Get Rewards at t + j
                        step_mask = valid_steps & (cp.arange(max_trajectory_length)[None, :] == step_idx)
                        step_rewards = batch_rewards[step_mask]

                        # Check if we have rewards for all states
                        if len(step_rewards) != len(t_states):
                            break

                        returns += (self.gamma ** j) * step_rewards
                    
                    # Add Bootstrapped values
                    bootstrap_idx = t_offset + n - 1
                    if bootstrap_idx < max_trajectory_length:
                        # Get Next States and Dones
                        bootstrap_mask = valid_steps & (cp.arange(max_trajectory_length)[None, :] == bootstrap_idx)
                        bootstrap_next_states = batch_next_states[bootstrap_mask]
                        bootstrap_dones = batch_dones[bootstrap_mask]

                        if len(bootstrap_next_states) == len(t_states):
                            # Only bootstrap non terminal states
                            bootstrap_values = cp.zeros(len(bootstrap_next_states), dtype=cp.float32)
                            non_terminal = ~bootstrap_dones

                            if cp.any(non_terminal):
                                # Get Max Q values for non temrinal next states
                                next_state_values = cp.max(self.cu_q_values[n][bootstrap_next_states], axis=1)
                                bootstrap_values[non_terminal] = next_state_values
                            # Add bootstrapped values
                            returns += (self.gamma ** n) * bootstrap_values
                # Store results
                all_states.append(t_states)
                all_actions.append(t_actions)
                all_returns.append(returns)
            
            # Combine results from all batches
            if all_states:
                combined_states = cp.concatenate(all_states).get()
                combined_actions = cp.concatenate(all_actions).get()
                combined_returns = cp.concatenate(all_returns).get()
                
                n_step_returns[n] = list(zip(combined_states, combined_actions, combined_returns))
            else:
                n_step_returns[n] = []
        
        return n_step_returns
    
    def train(self):
        for episode in tqdm(range(self.num_episodes), desc="Training episodes"):
            #Generate Trajectories
            trajectories = self.generate_trajectories()

            #Calculate N Step Returns
            n_step_returns = self.calculate_n_step_returns(trajectories)
            
            for n, returns in n_step_returns.items():
                if returns:
                    states, actions, n_returns  = zip(*returns)
                    states = cp.array(states, dtype=cp.int32)
                    actions = cp.array(actions, dtype=cp.int32)
                    n_returns = cp.array(n_returns, dtype=cp.float32)

                    # Current Q-values
                    current_q = self.cu_q_values[n][states, actions]

                    #calculate TD error
                    td_errors = n_returns - current_q

                    #Update Q
                    self.cu_q_values[n][states, actions] += self.alpha * td_errors
        return self.get_q_values()
    def get_q_values(self):
        """Get a CPU copy of Q-values for visualization or saving"""
        return {n: self.cu_q_values[n].get() for n in self.n_values}
    def evaluate(self):
        results = {}
        
        for n in self.n_values:
            total_reward = 0
            num_eval_episodes = 10
            
            # Get CPU copy of Q-values for this n
            q_values = self.cu_q_values[n].get()

            
            for _ in range(num_eval_episodes):
                state, _ = self.env.reset()
                done = False
                episode_reward = 0
                steps = 0

                
                while not done and steps < 1000:
                    steps += 1
                    action = np.argmax(q_values[state])
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    state = next_state
                
                total_reward += episode_reward
            
            results[n] = total_reward / num_eval_episodes
        
        return results
    def save_trajectories(self, trajectories, episode):
        """Save trajectories to disk"""
        save_dir = params.savefig_folder / "trajectories"
        os.makedirs(save_dir, exist_ok=True)
        trajectory_file = save_dir / f"{self.env_name}_ep{episode}_n{len(self.n_values)}_traj{self.num_trajectories}.pkl"
        
        # Convert any CuPy arrays to NumPy before saving
        with open(trajectory_file, 'wb') as f:
            pickle.dump(trajectories, f)
        
        print(f"Saved trajectories to {trajectory_file}")
        return trajectory_file

    def load_trajectories(self, episode):
        """Load trajectories from disk if they exist"""
        save_dir = params.savefig_folder / "trajectories"
        trajectory_file = save_dir / f"{self.env_name}_ep{episode}_n{len(self.n_values)}_traj{self.num_trajectories}.pkl"
        
        if os.path.exists(trajectory_file):
            with open(trajectory_file, 'rb') as f:
                trajectories = pickle.load(f)
            print(f"Loaded trajectories from {trajectory_file}")
            return trajectories
        
        return None

if __name__ == "__main__":
    # Set n values to compare
    n_values = [1]
    
    # Create agent
    env_name = "CliffWalking-v0"  
    agent = ParallelNStepTD(n_values, env_name, params)
    q_values = agent.train()
    # Evaluate the trained agent
    print("Evaluating agent performance...")
    results = agent.evaluate()
    print("\nEvaluation Results:")
    for n, avg_reward in results.items():
        print(f"n={n}: Average reward = {avg_reward:.2f}")
    
    # Simple visualization of the policy
    for n in n_values:
        policy = np.argmax(q_values[n], axis=1)
        grid_policy = policy.reshape(4, 12)
        
        print(f"\nLearned Policy (n={n}):")
        action_symbols = ['↑', '→', '↓', '←']  # Up, Right, Down, Left
        for row in range(4):
            row_str = ""
            for col in range(12):
                state = row * 12 + col
                # Special cases for the map
                if row == 3 and 1 <= col <= 10:  # Cliff
                    row_str += " X "
                elif row == 3 and col == 11:  # Goal
                    row_str += " G "
                else:
                    action = policy[state]
                    row_str += f" {action_symbols[action]} "
            print(row_str)



