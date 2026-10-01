import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sns

class MLP(nn.Module):
    """Q-value network"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, output_dim)
        
        # Better initialization
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SemiGradSarsaGA:
    def __init__(self, params):
        self.env_name = params.env_name
        self.env = gym.make(params.env_name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.action_dim = self.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-network setup
        self.q_net = MLP(
            input_dim=self.observation_space.shape[0] + 1,  # state + action
            hidden_dim=64,
            output_dim=1
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=params.alpha)
        self.avg_reward = torch.tensor(0.0, device=self.device)

        # Genetic Algorithm setup
        self.ga_pop_size = 10
        self.mutation_rate = 0.2  # Increased for more exploration
        self.mutation_scale = 0.2
        self.policy_population = self._initialize_population()
        self.current_policy = self.policy_population[0]
        self.fitness_scores = []
        self.best_policy = None
        self.best_fitness = -float('inf')

        # Exploration parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start

        # Training parameters
        self.num_episodes = params.num_episodes
        self.gamma = params.gamma
        self.beta = params.beta
        
        # Debugging parameters
        self.savefig_folder = Path(params.savefig_folder)
        self.savefig_folder.mkdir(parents=True, exist_ok=True)
        
        # Stats tracking
        self.q_values_history = []
        self.td_errors = []
        self.gradient_norms = []
        self.action_counts = np.zeros(self.action_dim)
        self.ga_vs_q_selections = [0, 0]  # [GA, Q-network]
        self.generation_best_fitness = []
        self.population_diversity = []

    def _initialize_population(self):
        """Initialize population of action probability distributions"""
        population = []
        for _ in range(self.ga_pop_size):
            # Initialize with some bias toward successful policies for Acrobot
            # 0: left torque, 1: no torque, 2: right torque
            probs = np.random.dirichlet(np.ones(self.action_dim) * 2)
            population.append(probs)
        return population

    def _calculate_diversity(self, population):
        """Calculate diversity of the population using average pairwise distance"""
        if len(population) <= 1:
            return 0.0
            
        total_dist = 0.0
        count = 0
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                total_dist += np.sum(np.abs(population[i] - population[j]))
                count += 1
        return total_dist / count if count > 0 else 0.0

    def policy(self, state):
        """Select action using ε-decay: GA policy vs Q-greedy"""
        # Record if we're using GA for this selection
        using_ga = np.random.rand() < self.epsilon
        
        if using_ga:
            # Update selection tracking
            self.ga_vs_q_selections[0] += 1
            # Use GA exploration policy
            action = np.random.choice(self.action_dim, p=self.current_policy)
            return action, True
        else:
            # Update selection tracking
            self.ga_vs_q_selections[1] += 1
            # Use Q-network greedy policy
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = torch.zeros(self.action_dim, device=self.device)
            
            for a in range(self.action_dim):
                sa = torch.cat([state_tensor, torch.tensor([a], dtype=torch.float32, device=self.device)])
                q_values[a] = self.q_net(sa.unsqueeze(0)).squeeze()
                
            # Save Q-values for debugging
            self.q_values_history.append(q_values.detach().cpu().numpy())
            
            action = q_values.argmax().item()
            # Update action counts
            self.action_counts[action] += 1
            return action, False

    def train_episode(self):
        """Run one training episode, return total reward"""
        state, _ = self.env.reset()
        action, used_ga = self.policy(state)
        done = False
        rewards = []
        episode_reward = 0
        ga_used_in_episode = used_ga
        steps = 0
        episode_td_errors = []
        
        while not done:
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_reward += reward
            rewards.append(reward)
            steps += 1
            
            next_action, next_used_ga = self.policy(next_state)
            
            if next_used_ga:
                ga_used_in_episode = True

            # Convert to tensors
            state_tensor = torch.FloatTensor(state).to(self.device)
            action_tensor = torch.tensor([float(action)], device=self.device)
            sa_pair = torch.cat([state_tensor, action_tensor]).unsqueeze(0)
            current_q = self.q_net(sa_pair).squeeze()

            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            next_action_tensor = torch.tensor([float(next_action)], device=self.device)
            next_sa_pair = torch.cat([next_state_tensor, next_action_tensor]).unsqueeze(0)
            
            with torch.no_grad():
                next_q = self.q_net(next_sa_pair).squeeze()

            # SARSA update with average reward
            reward_tensor = torch.tensor(reward, device=self.device)
            td_error = reward_tensor - self.avg_reward + self.gamma * next_q - current_q
            episode_td_errors.append(td_error.item())
            self.avg_reward += self.beta * td_error.detach()

            # Update Q-network
            loss = 0.5 * td_error.pow(2)
            self.optimizer.zero_grad()
            loss.backward()
            
            # Track gradient norms
            total_norm = 0
            for p in self.q_net.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.gradient_norms.append(total_norm)
            
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            state = next_state
            action = next_action
            used_ga = next_used_ga

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Store TD errors for this episode
        self.td_errors.append(np.mean(np.abs(episode_td_errors)))
        
        # Calculate fitness - convert rewards to fitness score
        if terminated:  # Successfully reached goal
            fitness = 500 - steps  # Reward quick solutions
        else:
            fitness = episode_reward  # Use episode reward for unsuccessful runs
            
        if ga_used_in_episode:
            self.fitness_scores.append(fitness)
            
            # Track best policy
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_policy = np.copy(self.current_policy)

        return episode_reward

    def _evolve_policies(self):
        """Evolve GA population using fitness-proportional selection"""
        if not self.fitness_scores:
            return
            
        # Convert fitness to selection probability
        fitness = np.array(self.fitness_scores)
        min_fit = np.min(fitness)
        if min_fit < 0:
            fitness = fitness - min_fit + 1.0
            
        # Track best fitness for this generation
        self.generation_best_fitness.append(np.max(fitness))
        
        # Calculate diversity before evolution
        diversity = self._calculate_diversity(self.policy_population)
        self.population_diversity.append(diversity)
        
        # Tournament selection
        selected = []
        for _ in range(self.ga_pop_size):
            if len(fitness) > 2:
                candidates = np.random.choice(len(fitness), 3)
                winner = candidates[np.argmax(fitness[candidates])]
                selected.append(np.copy(self.policy_population[winner % self.ga_pop_size]))
            else:
                selected.append(np.copy(self.policy_population[0]))
        
        # Always include best policy seen so far
        if self.best_policy is not None:
            selected[0] = np.copy(self.best_policy)

        # Crossover and mutation to create new population
        new_population = []
        while len(new_population) < self.ga_pop_size:
            # Select two parents
            parent1, parent2 = np.random.choice(len(selected), 2, replace=False)
            p1, p2 = selected[parent1], selected[parent2]
            
            # Two-point crossover
            crossover_points = sorted(np.random.choice(self.action_dim, 2, replace=False))
            child = np.copy(p1)
            child[crossover_points[0]:crossover_points[1]] = p2[crossover_points[0]:crossover_points[1]]
            
            # Mutation
            if np.random.rand() < self.mutation_rate:
                mutation_idx = np.random.randint(0, self.action_dim)
                mutation = np.random.normal(0, self.mutation_scale)
                child[mutation_idx] += mutation
                
            # Ensure valid probability distribution
            child = np.clip(child, 0.01, None)
            child /= child.sum()
            
            new_population.append(child)
        
        self.policy_population = new_population
        self.fitness_scores = []

    def plot_debug_graphs(self):
        """Create and save all debug plots"""
        # 1. Q-values distribution
        plt.figure(figsize=(10, 6))
        if self.q_values_history:
            q_values = np.concatenate(self.q_values_history[-1000:])  # Last 1000 samples
            plt.hist(q_values, bins=50, alpha=0.7)
            plt.axvline(np.mean(q_values), color='r', linestyle='dashed', linewidth=1)
            plt.title(f'Q-value Distribution (mean={np.mean(q_values):.3f})')
            plt.xlabel('Q-value')
            plt.ylabel('Frequency')
            plt.grid(alpha=0.3)
            plt.savefig(self.savefig_folder / 'q_value_distribution.png')
            plt.close()
        
        # 2. TD Error over time
        plt.figure(figsize=(10, 6))
        plt.plot(self.td_errors)
        plt.title('Average TD Error Magnitude Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Mean Absolute TD Error')
        plt.grid(alpha=0.3)
        plt.savefig(self.savefig_folder / 'td_errors.png')
        plt.close()
        
        # 3. Gradient norms
        plt.figure(figsize=(10, 6))
        plt.plot(self.gradient_norms)
        plt.title('Gradient L2 Norm During Training')
        plt.xlabel('Update Step')
        plt.ylabel('Gradient Norm')
        plt.yscale('log')  # Log scale for better visibility
        plt.grid(alpha=0.3)
        plt.savefig(self.savefig_folder / 'gradient_norms.png')
        plt.close()
        
        # 4. Action distribution
        plt.figure(figsize=(10, 6))
        plt.bar(range(self.action_dim), self.action_counts)
        plt.title('Action Selection Distribution')
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
        plt.savefig(self.savefig_folder / 'action_distribution.png')
        plt.close()
        
        # 5. GA vs Q-network selection
        plt.figure(figsize=(10, 6))
        plt.bar(['GA Policy', 'Q-network'], self.ga_vs_q_selections)
        plt.title('GA vs Q-network Action Selection')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
        plt.savefig(self.savefig_folder / 'ga_vs_q_selection.png')
        plt.close()
        
        # 6. Best policy visualization
        if self.best_policy is not None:
            plt.figure(figsize=(10, 6))
            plt.bar(range(self.action_dim), self.best_policy)
            plt.title(f'Best GA Policy (Fitness: {self.best_fitness:.1f})')
            plt.xlabel('Action')
            plt.ylabel('Probability')
            plt.grid(alpha=0.3)
            plt.savefig(self.savefig_folder / 'best_policy.png')
            plt.close()
        
        # 7. Generation best fitness
        if self.generation_best_fitness:
            plt.figure(figsize=(10, 6))
            plt.plot(self.generation_best_fitness)
            plt.title('Best Fitness Per Generation')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.grid(alpha=0.3)
            plt.savefig(self.savefig_folder / 'generation_fitness.png')
            plt.close()
            
        # 8. Population diversity
        if self.population_diversity:
            plt.figure(figsize=(10, 6))
            plt.plot(self.population_diversity)
            plt.title('Population Diversity Over Generations')
            plt.xlabel('Generation')
            plt.ylabel('Diversity (Avg Pairwise Distance)')
            plt.grid(alpha=0.3)
            plt.savefig(self.savefig_folder / 'population_diversity.png')
            plt.close()
            
        # 9. Epsilon decay
        plt.figure(figsize=(10, 6))
        episodes = range(self.num_episodes)
        epsilon_values = [self.epsilon_start * (self.epsilon_decay ** e) 
                          for e in episodes]
        epsilon_values = [max(self.epsilon_end, e) for e in epsilon_values]
        plt.plot(episodes, epsilon_values)
        plt.title('Epsilon Decay Schedule')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(alpha=0.3)
        plt.savefig(self.savefig_folder / 'epsilon_decay.png')
        plt.close()

    def evaluate(self, num_episodes=50):
        """Evaluate the trained agent"""
        eval_rewards = []
        success_count = 0
        
        # Create evaluation environment
        eval_env = gym.make(self.env_name)
        
        for _ in tqdm(range(num_episodes), desc="Evaluating"):
            state, _ = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Use the best policy or Q-network for evaluation
                if self.best_policy is not None and np.random.rand() < 0.5:
                    action = np.random.choice(self.action_dim, p=self.best_policy)
                else:
                    # Use Q-network greedy policy
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    q_values = torch.zeros(self.action_dim, device=self.device)
                    
                    for a in range(self.action_dim):
                        sa = torch.cat([state_tensor, torch.tensor([a], dtype=torch.float32, device=self.device)])
                        q_values[a] = self.q_net(sa.unsqueeze(0)).squeeze()
                    
                    action = q_values.argmax().item()
                
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
                
                if terminated:
                    success_count += 1
            
            eval_rewards.append(episode_reward)
        
        success_rate = success_count / num_episodes * 100
        avg_reward = np.mean(eval_rewards)
        
        print(f"Evaluation Results:")
        print(f"  Average reward: {avg_reward:.1f}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        # Plot evaluation results
        plt.figure(figsize=(10, 6))
        plt.hist(eval_rewards, bins=20, alpha=0.7)
        plt.axvline(avg_reward, color='r', linestyle='dashed', linewidth=2)
        plt.title(f'Evaluation Rewards (Success Rate: {success_rate:.1f}%)')
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.savefig(self.savefig_folder / 'evaluation_results.png')
        plt.close()
        
        return avg_reward, success_rate

    def train(self):
        """Main training loop"""
        all_rewards = []
        avg_rewards = []  # Track moving average
        
        for episode in tqdm(range(self.num_episodes), desc="Training"):
            # Cycle through GA population
            self.current_policy = self.policy_population[episode % self.ga_pop_size]
            
            # Train episode
            episode_reward = self.train_episode()
            all_rewards.append(episode_reward)
            
            # Calculate moving average
            window = min(100, len(all_rewards))
            avg_reward = sum(all_rewards[-window:]) / window
            avg_rewards.append(avg_reward)
            
            # Evolve population after each generation
            if (episode + 1) % self.ga_pop_size == 0:
                self._evolve_policies()
                
            # Logging
            if episode % 100 == 0 or episode == self.num_episodes - 1:
                print(f"Episode: {episode}, Reward: {episode_reward:.1f}, " 
                      f"Avg(100): {avg_reward:.1f}, Epsilon: {self.epsilon:.3f}")
                
        # Plot training progress
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(all_rewards, alpha=0.3, label='Episode Rewards')
        plt.plot(avg_rewards, linewidth=2, label='100-Episode Average')
        plt.title('Training Progress')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 1, 2)
        successful = [r > -100 for r in all_rewards]
        window_size = 50
        success_rate = [sum(successful[max(0, i-window_size):i+1]) / 
                        min(i+1, window_size) * 100 
                        for i in range(len(successful))]
        plt.plot(success_rate)
        plt.title(f'Success Rate (50-episode window)')
        plt.xlabel('Episode')
        plt.ylabel('Success %')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.savefig_folder / 'training_progress.png')
        plt.close()
        
        # Create and save debug plots
        self.plot_debug_graphs()
        
        # Evaluate the final agent
        self.evaluate()
        
        return all_rewards, avg_rewards

# Hyperparameters and Usage
class Params:
    env_name = "Acrobot-v1"
    alpha = 0.001
    beta = 0.01
    gamma = 0.99
    num_episodes = 1000
    savefig_folder = "./plots8/"

if __name__ == "__main__":
    params = Params()
    agent = SemiGradSarsaGA(params)
    rewards, avg_rewards = agent.train()
    
    print(f"Training complete!")
    print(f"Final 100-episode average reward: {avg_rewards[-1]:.1f}")
    
    if agent.best_policy is not None:
        print(f"Best GA policy found with fitness: {agent.best_fitness:.1f}")
        print(f"Best policy distribution: {agent.best_policy}")