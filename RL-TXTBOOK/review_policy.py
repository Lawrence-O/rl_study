import numpy as np
from collections import defaultdict
import random

# ============================================================
# ENVIRONMENTS FOR POLICY GRADIENT METHODS
# ============================================================

class SimpleContinuousEnv:
    """
    Simple 1D continuous control task.
    
    State: position x in [-1, 1]
    Action: force in [-1, 1] (continuous!)
    Goal: reach x > 0.9
    
    Dynamics: x' = clip(x + 0.1*action + noise, -1, 1)
    Reward: -1 per step, +10 at goal
    
    This is simpler than Mountain Car but captures the key challenge:
    learning a continuous policy.
    """
    
    def __init__(self, noise=0.05):
        self.noise = noise
        self.state = None
        
    def reset(self):
        self.state = np.random.uniform(-0.5, 0.0)  # Start left of center
        return self.state
    
    def step(self, action):
        # Clip action to valid range
        action = np.clip(action, -1, 1)
        
        # Simple dynamics with noise
        self.state = self.state + 0.1 * action + np.random.normal(0, self.noise)
        self.state = np.clip(self.state, -1, 1)
        
        # Check if goal reached
        done = self.state > 0.9
        reward = 10.0 if done else -0.1
        
        return self.state, reward, done, {}


class DiscreteCartPoleSimple:
    """
    Simplified CartPole-like environment for discrete policy gradients.
    
    State: [position, velocity, angle, angular_velocity]
    Actions: 0 = push left, 1 = push right
    
    Goal: Keep pole balanced (angle near 0) as long as possible.
    Episode ends if |angle| > 0.3 or |position| > 2.0 or 200 steps reached.
    """
    
    def __init__(self):
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.pole_length = 0.5
        self.force_mag = 10.0
        self.dt = 0.02
        self.max_steps = 200
        
        self.state = None
        self.steps = 0
        
    def reset(self):
        # Small random initial state
        self.state = np.random.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()
    
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        
        force = self.force_mag if action == 1 else -self.force_mag
        
        # Physics (simplified)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        total_mass = self.cart_mass + self.pole_mass
        pole_mass_length = self.pole_mass * self.pole_length
        
        # Angular acceleration
        temp = (force + pole_mass_length * theta_dot**2 * sin_theta) / total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / \
                    (self.pole_length * (4/3 - self.pole_mass * cos_theta**2 / total_mass))
        
        # Linear acceleration
        x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass
        
        # Euler integration
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_acc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        
        # Check termination
        done = (abs(theta) > 0.3 or abs(x) > 2.0 or self.steps >= self.max_steps)
        reward = 1.0 if not done else 0.0
        
        return self.state.copy(), reward, done, {}


class MultiArmedBandit:
    """
    Simple multi-armed bandit for understanding REINFORCE basics.
    
    No state transitions - just pick an action and get a reward.
    Perfect for understanding the policy gradient without complexity.
    
    Actions: 0, 1, 2, 3, 4 (5 arms)
    Rewards: Each arm has a different mean reward (unknown to agent)
    """
    
    def __init__(self):
        # True reward means (agent doesn't know these)
        self.true_means = [0.2, 0.5, 0.1, 0.8, 0.3]
        self.n_actions = 5
        
    def reset(self):
        return 0  # Dummy state (bandits are stateless)
    
    def step(self, action):
        # Reward is Bernoulli with probability = true_mean[action]
        reward = 1.0 if random.random() < self.true_means[action] else 0.0
        return 0, reward, True, {}  # Always terminates (1-step episodes)
    
    def optimal_action(self):
        return np.argmax(self.true_means)


# ============================================================
# POLICY REPRESENTATIONS
# ============================================================

class SoftmaxPolicy:
    """
    Softmax policy for discrete actions.
    
    π(a|s) = exp(θ[s,a]) / Σ exp(θ[s,a'])
    
    For simplicity, we use a tabular representation.
    For the bandit (stateless), we just have θ[a].
    """
    
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        # Initialize preferences to 0 (uniform policy)
        self.theta = np.zeros((n_states, n_actions))
    
    def get_probabilities(self, state):
        """Return action probabilities for given state"""
        # Softmax with numerical stability
        preferences = self.theta[state]
        exp_prefs = np.exp(preferences - np.max(preferences))
        return exp_prefs / np.sum(exp_prefs)
    
    def sample_action(self, state):
        """Sample action from policy"""
        probs = self.get_probabilities(state)
        return np.random.choice(self.n_actions, p=probs)
    
    def log_probability(self, state, action):
        """Return log π(a|s) - needed for policy gradient"""
        probs = self.get_probabilities(state)
        return np.log(probs[action] + 1e-10)


class GaussianPolicy:
    """
    Gaussian policy for continuous actions.
    
    π(a|s) = N(μ(s), σ²)
    
    For simplicity:
    - μ(s) = θ_μ · s (linear in state)
    - σ = exp(θ_σ) (learnable but state-independent)
    """
    
    def __init__(self):
        # Simple linear policy: μ = θ_μ * state
        self.theta_mu = 0.0  # Weight for mean
        self.theta_sigma = 0.0  # Log of std dev
    
    @property
    def sigma(self):
        return np.exp(self.theta_sigma)
    
    def get_mean(self, state):
        return self.theta_mu * state
    
    def sample_action(self, state):
        """Sample action from Gaussian"""
        mean = self.get_mean(state)
        return np.random.normal(mean, self.sigma)
    
    def log_probability(self, state, action):
        """Return log π(a|s) for Gaussian"""
        mean = self.get_mean(state)
        sigma = self.sigma
        log_prob = -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((action - mean) / sigma) ** 2
        return log_prob


# ============================================================
# YOUR IMPLEMENTATIONS GO BELOW
# ============================================================

# --- CHALLENGE 1: REINFORCE for Bandit ---
def reinforce_bandit(env, n_episodes=1000, alpha=0.1):
    """
    Implement REINFORCE for the multi-armed bandit.
    
    This is the simplest case:
    - No states (or single dummy state)
    - 1-step episodes
    - Reward = return (no discounting needed)
    
    The policy gradient for softmax is:
    ∇θ log π(a|s) = e_a - π(s)
    
    where e_a is one-hot vector for action a.
    
    Update: θ += α * G * ∇θ log π(a|s)
    
    Args:
        env: MultiArmedBandit environment
        n_episodes: Number of episodes
        alpha: Learning rate
    
    Returns:
        policy: Trained SoftmaxPolicy
        rewards: List of rewards per episode
        action_probs_history: List of action probabilities over time
    
    TODO: Implement the algorithm
    """
    policy = SoftmaxPolicy(n_states=1, n_actions=env.n_actions)
    rewards = []
    action_probs_history = []
    
    # YOUR CODE HERE
    pass
    
    return policy, rewards, action_probs_history


# --- CHALLENGE 2: REINFORCE with Baseline ---
def reinforce_with_baseline(env, n_episodes=2000, alpha_policy=0.01, alpha_baseline=0.1, gamma=0.99):
    """
    Implement REINFORCE with baseline for CartPole.
    
    The baseline reduces variance without adding bias:
    ∇θ J(θ) ≈ Σ_t (G_t - b(s_t)) * ∇θ log π(a_t|s_t)
    
    Common baseline: b(s) = V(s) (learned value function)
    
    For function approximation, we'll use simple linear features.
    
    Args:
        env: DiscreteCartPoleSimple environment
        n_episodes: Number of episodes
        alpha_policy: Learning rate for policy
        alpha_baseline: Learning rate for baseline
        gamma: Discount factor
    
    Returns:
        policy_weights: Learned policy parameters
        baseline_weights: Learned baseline parameters
        episode_lengths: List of episode lengths (measure of performance)
    
    TODO: Implement the algorithm
    
    Hints:
    - Use linear function approximation: π(a=1|s) = sigmoid(θ · φ(s))
    - Baseline: V(s) ≈ w · φ(s)
    - Features φ(s) can just be the state vector (or add polynomial features)
    """
    # Simple linear policy: logit = θ · s, then sigmoid
    policy_weights = np.zeros(4)  # 4 state features
    baseline_weights = np.zeros(4)
    episode_lengths = []
    
    # YOUR CODE HERE
    pass
    
    return policy_weights, baseline_weights, episode_lengths


# --- CHALLENGE 3: REINFORCE for Continuous Control ---
def reinforce_continuous(env, n_episodes=1000, alpha_mu=0.01, alpha_sigma=0.001, gamma=0.99):
    """
    Implement REINFORCE with Gaussian policy for continuous actions.
    
    Policy: π(a|s) = N(μ(s), σ²)
    - μ(s) = θ_μ * s (linear)
    - σ = exp(θ_σ) (learned)
    
    Policy gradient for Gaussian:
    ∇θ_μ log π(a|s) = (a - μ) * s / σ²
    ∇θ_σ log π(a|s) = ((a - μ)² / σ² - 1)
    
    Args:
        env: SimpleContinuousEnv
        n_episodes: Number of episodes
        alpha_mu: Learning rate for mean parameter
        alpha_sigma: Learning rate for std parameter
        gamma: Discount factor
    
    Returns:
        policy: Trained GaussianPolicy
        episode_rewards: List of total rewards per episode
    
    TODO: Implement the algorithm
    """
    policy = GaussianPolicy()
    episode_rewards = []
    
    # YOUR CODE HERE
    pass
    
    return policy, episode_rewards


# --- CHALLENGE 4: Actor-Critic (TD) ---
def actor_critic(env, n_episodes=2000, alpha_actor=0.01, alpha_critic=0.1, gamma=0.99):
    """
    Implement one-step Actor-Critic for CartPole.
    
    Unlike REINFORCE which uses full returns (Monte Carlo),
    Actor-Critic uses TD error for lower variance updates:
    
    TD error: δ = r + γV(s') - V(s)
    
    Actor update: θ += α_actor * δ * ∇θ log π(a|s)
    Critic update: w += α_critic * δ * ∇w V(s)
    
    Args:
        env: DiscreteCartPoleSimple environment
        n_episodes: Number of episodes
        alpha_actor: Learning rate for policy (actor)
        alpha_critic: Learning rate for value function (critic)
        gamma: Discount factor
    
    Returns:
        actor_weights: Learned policy parameters
        critic_weights: Learned value function parameters
        episode_lengths: List of episode lengths
    
    TODO: Implement the algorithm
    """
    actor_weights = np.zeros(4)
    critic_weights = np.zeros(4)
    episode_lengths = []
    
    # YOUR CODE HERE
    pass
    
    return actor_weights, critic_weights, episode_lengths


# --- CHALLENGE 5: A2C (Advantage Actor-Critic) ---
def a2c(env, n_episodes=2000, alpha_actor=0.01, alpha_critic=0.1, gamma=0.99, n_steps=5):
    """
    Implement n-step Advantage Actor-Critic.
    
    Uses n-step returns for the advantage estimate:
    A(s,a) = G_t^{(n)} - V(s_t)
    
    where G_t^{(n)} = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})
    
    Args:
        env: DiscreteCartPoleSimple environment
        n_episodes: Number of episodes
        alpha_actor: Learning rate for actor
        alpha_critic: Learning rate for critic
        gamma: Discount factor
        n_steps: Number of steps for n-step returns
    
    Returns:
        actor_weights: Learned policy parameters
        critic_weights: Learned value function parameters
        episode_lengths: List of episode lengths
    
    TODO: Implement the algorithm
    """
    actor_weights = np.zeros(4)
    critic_weights = np.zeros(4)
    episode_lengths = []
    
    # YOUR CODE HERE
    pass
    
    return actor_weights, critic_weights, episode_lengths


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sigmoid(x):
    """Numerically stable sigmoid"""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))


def softmax(x):
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def compute_returns(rewards, gamma):
    """Compute discounted returns for each timestep"""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


# ============================================================
# CONCEPTUAL QUESTIONS
# ============================================================

"""
POLICY GRADIENT THEORY QUESTIONS

Q1: Why do we use log π(a|s) in the policy gradient instead of just π(a|s)?
    
    YOUR ANSWER:


Q2: The REINFORCE estimator has high variance. Why? What causes this variance?
    
    YOUR ANSWER:


Q3: How does the baseline reduce variance without adding bias?
    Mathematically, why is E[b(s) * ∇ log π(a|s)] = 0?
    
    YOUR ANSWER:


Q4: What's the key difference between REINFORCE and Actor-Critic?
    When would you prefer one over the other?
    
    YOUR ANSWER:


Q5: In the Gaussian policy, we learn both μ and σ. Why might we want to learn σ
    instead of keeping it fixed? What happens to σ during training?
    
    YOUR ANSWER:


Q6: Actor-Critic uses TD error (bootstrapping) while REINFORCE uses full returns.
    Connect this to what you learned about TD vs MC in value-based methods.
    What are the tradeoffs?
    
    YOUR ANSWER:

"""


# ============================================================
# TEST YOUR IMPLEMENTATIONS
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("CHALLENGE 1: REINFORCE on Multi-Armed Bandit")
    print("="*60)
    
    bandit = MultiArmedBandit()
    print(f"Optimal action: {bandit.optimal_action()} (true mean: {max(bandit.true_means)})")
    
    policy, rewards, probs_history = reinforce_bandit(bandit, n_episodes=1000)
    
    if policy is not None and len(probs_history) > 0:
        final_probs = policy.get_probabilities(0)
        print(f"Learned action probabilities: {final_probs}")
        print(f"Most likely action: {np.argmax(final_probs)}")
        print(f"Average reward (last 100): {np.mean(rewards[-100:]):.3f}")
    else:
        print("Not implemented yet!")
    
    print("\n" + "="*60)
    print("CHALLENGE 2: REINFORCE with Baseline on CartPole")
    print("="*60)
    
    cartpole = DiscreteCartPoleSimple()
    policy_w, baseline_w, lengths = reinforce_with_baseline(cartpole, n_episodes=1000)
    
    if len(lengths) > 0:
        print(f"Average episode length (last 100): {np.mean(lengths[-100:]):.1f}")
        print(f"Max episode length: {max(lengths)}")
    else:
        print("Not implemented yet!")
    
    print("\n" + "="*60)
    print("CHALLENGE 3: REINFORCE for Continuous Control")
    print("="*60)
    
    cont_env = SimpleContinuousEnv()
    policy, rewards = reinforce_continuous(cont_env, n_episodes=500)
    
    if policy is not None and len(rewards) > 0:
        print(f"Learned μ weight: {policy.theta_mu:.3f}")
        print(f"Learned σ: {policy.sigma:.3f}")
        print(f"Average reward (last 100): {np.mean(rewards[-100:]):.2f}")
    else:
        print("Not implemented yet!")
    
    print("\n" + "="*60)
    print("CHALLENGE 4: Actor-Critic on CartPole")
    print("="*60)
    
    cartpole = DiscreteCartPoleSimple()
    actor_w, critic_w, lengths = actor_critic(cartpole, n_episodes=1000)
    
    if len(lengths) > 0:
        print(f"Average episode length (last 100): {np.mean(lengths[-100:]):.1f}")
        print(f"Max episode length: {max(lengths)}")
    else:
        print("Not implemented yet!")
    
    print("\n" + "="*60)
    print("CHALLENGE 5: A2C on CartPole")
    print("="*60)
    
    cartpole = DiscreteCartPoleSimple()
    actor_w, critic_w, lengths = a2c(cartpole, n_episodes=1000, n_steps=5)
    
    if len(lengths) > 0:
        print(f"Average episode length (last 100): {np.mean(lengths[-100:]):.1f}")
        print(f"Max episode length: {max(lengths)}")
    else:
        print("Not implemented yet!")
    
    print("\n" + "="*60)
    print("Summary: Policy Gradient Methods")
    print("="*60)
    print("""
    Method              | Update Target           | Variance | Bias
    --------------------|-------------------------|----------|------
    REINFORCE           | Full return G_t         | High     | None
    REINFORCE + Baseline| G_t - b(s_t)            | Medium   | None
    Actor-Critic (TD)   | r + γV(s') - V(s)       | Low      | Some
    A2C (n-step)        | G_t^(n) - V(s_t)        | Medium   | Some
    """)