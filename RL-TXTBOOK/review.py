import copy
import numpy as np
from collections import defaultdict, deque
import random

# ============================================================
# PRE-BUILT SIMULATOR - DO NOT MODIFY THIS SECTION
# ============================================================

class GridWorld:
    """
    Simple 5x5 GridWorld with deterministic transitions.
    
    Layout:
    +---+---+---+---+---+
    | S |   |   |   | G |  (G = Goal, +10 reward)
    +---+---+---+---+---+
    |   | X |   | X |   |  (X = Pit, -5 reward, episode ends)
    +---+---+---+---+---+
    |   |   |   |   |   |
    +---+---+---+---+---+
    |   | X |   |   |   |
    +---+---+---+---+---+
    |   |   |   | X |   |
    +---+---+---+---+---+
    
    States: 0-24 (5x5 grid, row-major)
    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    """
    
    def __init__(self, slip_prob=0.0):
        self.size = 5
        self.n_states = 25
        self.n_actions = 4
        self.slip_prob = slip_prob  # Probability of slipping to random adjacent state
        
        self.start_state = 0
        self.goal_state = 4
        self.pits = {6, 8, 16, 23}
        
        # Build transition model P[s][a] = [(prob, next_state, reward, done), ...]
        self.P = self._build_model()
        
    def _build_model(self):
        P = {s: {a: [] for a in range(self.n_actions)} for s in range(self.n_states)}
        
        for s in range(self.n_states):
            if s == self.goal_state or s in self.pits:
                # Terminal states - all actions lead to self with 0 reward
                for a in range(self.n_actions):
                    P[s][a] = [(1.0, s, 0, True)]
                continue
                
            row, col = s // self.size, s % self.size
            
            for a in range(self.n_actions):
                transitions = []
                
                # Intended direction
                next_states_probs = self._get_next_state_probs(row, col, a)
                
                for next_s, prob in next_states_probs:
                    reward = self._get_reward(next_s)
                    done = (next_s == self.goal_state or next_s in self.pits)
                    transitions.append((prob, next_s, reward, done))
                
                P[s][a] = transitions
        
        return P
    
    def _get_next_state_probs(self, row, col, action):
        """Returns list of (next_state, probability) tuples"""
        # Action effects: 0=Up(-1,0), 1=Right(0,+1), 2=Down(+1,0), 3=Left(0,-1)
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        def apply_action(r, c, a):
            dr, dc = deltas[a]
            nr, nc = r + dr, c + dc
            # Boundary check
            if 0 <= nr < self.size and 0 <= nc < self.size:
                return nr * self.size + nc
            return r * self.size + c  # Stay in place if hitting wall
        
        intended_next = apply_action(row, col, action)
        
        if self.slip_prob == 0:
            return [(intended_next, 1.0)]
        
        # With slipping: intended action with (1-slip_prob), uniform random with slip_prob
        results = defaultdict(float)
        results[intended_next] += (1 - self.slip_prob)
        
        for a in range(self.n_actions):
            slip_next = apply_action(row, col, a)
            results[slip_next] += self.slip_prob / self.n_actions
        
        return list(results.items())
    
    def _get_reward(self, state):
        if state == self.goal_state:
            return 10
        elif state in self.pits:
            return -5
        return -0.1  # Small penalty to encourage efficiency
    
    def reset(self):
        self.current_state = self.start_state
        return self.current_state
    
    def step(self, action):
        transitions = self.P[self.current_state][action]
        probs = [t[0] for t in transitions]
        idx = np.random.choice(len(transitions), p=probs)
        prob, next_state, reward, done = transitions[idx]
        self.current_state = next_state
        return next_state, reward, done, {}
    
    def render_policy(self, policy):
        """Visualize a policy on the grid"""
        arrows = ['↑', '→', '↓', '←']
        print("\nPolicy Visualization:")
        print("+" + "---+" * self.size)
        for row in range(self.size):
            line = "|"
            for col in range(self.size):
                s = row * self.size + col
                if s == self.goal_state:
                    line += " G |"
                elif s in self.pits:
                    line += " X |"
                else:
                    a = policy.get(s, 0)
                    line += f" {arrows[a]} |"
            print(line)
            print("+" + "---+" * self.size)
    
    def render_values(self, V):
        """Visualize state values on the grid"""
        print("\nValue Function:")
        print("+" + "------+" * self.size)
        for row in range(self.size):
            line = "|"
            for col in range(self.size):
                s = row * self.size + col
                v = V.get(s, 0)
                line += f"{v:6.2f}|"
            print(line)
            print("+" + "------+" * self.size)


# ============================================================
# HELPER FUNCTION FOR TESTING
# ============================================================

def evaluate_policy(env, policy, n_episodes=100, max_steps=100):
    """Evaluate a policy by running episodes and computing average return"""
    returns = []
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            action = policy.get(state, 0)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break
        returns.append(total_reward)
    return np.mean(returns), np.std(returns)


# ============================================================
# YOUR IMPLEMENTATIONS GO BELOW
# ============================================================

# --- CHALLENGE 1: Value Iteration ---
def value_iteration(env, gamma=0.99, theta=1e-6):
    """
    Implement Value Iteration algorithm.
    
    Args:
        env: GridWorld environment with env.P[s][a] = [(prob, next_s, reward, done), ...]
        gamma: Discount factor
        theta: Convergence threshold
    
    Returns:
        V: Dictionary mapping state -> value
        policy: Dictionary mapping state -> best action
        deltas: List of max delta per iteration (for plotting convergence)
    
    """
    V = {s: 0 for s in range(env.n_states)}
    policy = {s: 0 for s in range(env.n_states)}
    deltas = []
    curr_delta = theta * 10
    while curr_delta > theta:
        for s in range(env.n_states):
            old_v = V[s]
            max_v = float("-inf")
            for a in range(env.n_actions):
                q = sum(p * (r + gamma * V[ns]) for p, ns, r, _ in env.P[s][a])
                max_v = max(q, max_v)
            V[s] = max_v
            curr_delta = abs(old_v - V[s])
            deltas.append(curr_delta)

    for s in range(env.n_states):
        best_action = None
        max_v = float("-inf")
        for a in range(env.n_actions):
            q = sum(p * (r + gamma * V[ns]) for p, ns, r, _ in env.P[s][a])
            if max_v < q:
                best_action = a
                max_v = q
        policy[s] = best_action
    return V, policy, deltas


# --- CHALLENGE 2: Policy Iteration ---
def policy_iteration(env, gamma=0.99, theta=1e-6):
    """
    Implement Policy Iteration algorithm.
    
    Args:
        env: GridWorld environment
        gamma: Discount factor
        theta: Convergence threshold for policy evaluation
    
    Returns:
        V: Dictionary mapping state -> value
        policy: Dictionary mapping state -> best action
        n_iterations: Number of policy improvement iterations until convergence
    
    """
    V = {s: 0 for s in range(env.n_states)}
    policy = {s: random.randint(0, env.n_actions - 1) for s in range(env.n_states)}
    n_iterations = 0

    # Policy iteration works in two loops:
    # Policy evaluation, policy improvement

    def policy_evaluation():
        deltas = []
        while True:
            delta = 0.0
            for s in range(env.n_states):
                a = policy[s]
                q = sum(p * (r + gamma * V[ns]) for p, ns, r, _ in env.P[s][a])
                delta = max(abs(V[s] - q), delta)
                V[s] = q
                deltas.append(delta)
            if delta < theta:
                break
        return deltas
    
    def policy_improvement():
        for s in range(env.n_states):
            best_action = None
            max_v = float("-inf")
            for a in range(env.n_actions):
                q = sum(p * (r + gamma * V[ns]) for p, ns, r, _ in env.P[s][a])
                if q > max_v:
                    max_v = q
                    best_action = a
            policy[s] = best_action
    
    old_policy = copy.deepcopy(policy)
    policy_stable = False
    all_deltas = []
    while not policy_stable:
        deltas = policy_evaluation()
        all_deltas.extend(deltas)
        policy_improvement()
        if policy == old_policy:
            policy_stable = True
        old_policy = copy.deepcopy(policy)
        n_iterations += 1

    
    return V, policy, n_iterations


# --- CHALLENGE 3: SARSA ---
def sarsa(env, n_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Implement SARSA (On-policy TD Control).
    
    Args:
        env: GridWorld environment
        n_episodes: Number of episodes to train
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate for epsilon-greedy
    
    Returns:
        Q: Dictionary mapping (state, action) -> value
        policy: Dictionary mapping state -> best action
        episode_rewards: List of total rewards per episode
    
    """
    Q = defaultdict(float)
    policy = {}
    episode_rewards = []

    def get_action(s, training=False):
        if random.random() < epsilon and training:
            return random.randint(0, env.n_actions - 1)
        else:
            best_action = None
            best_q = float("-inf")
            for a in range(env.n_actions):
                q = Q[(s, a)]
                if q > best_q:
                    best_q = q
                    best_action = a
            return best_action

    def train_one_step():
        episode_reward = 0.0
        state = env.reset()
        while True:
            action = get_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            next_action = get_action(next_state, training=True)

            td_error = (reward + gamma*Q[(next_state, next_action)]) - Q[(state,action)]
            Q[(state,action)] += alpha*td_error

            state = next_state
            action = next_action
            episode_reward += reward

            if done:
                return episode_reward


    for i in range(n_episodes):
        episode_reward = train_one_step()
        episode_rewards.append(episode_reward)

    for s in range(env.n_states):
        best_action = None
        best_q = float("-inf")
        for a in range(env.n_actions):
            if Q[(s, a)] > best_q:
                best_q = Q[(s, a)]
                best_action = a
        policy[s] = best_action
    
    return Q, policy, episode_rewards


# --- CHALLENGE 4: Q-Learning ---
def q_learning(env, n_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Implement Q-Learning (Off-policy TD Control).
    
    Args:
        env: GridWorld environment
        n_episodes: Number of episodes to train
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate for epsilon-greedy
    
    Returns:
        Q: Dictionary mapping (state, action) -> value
        policy: Dictionary mapping state -> best action
        episode_rewards: List of total rewards per episode
    
    """
    Q = defaultdict(float)
    policy = {}
    episode_rewards = []
    
    def get_action(state, training=False):
        if random.random() < epsilon and training:
            return random.randint(0, env.n_actions - 1)
        else:
            best_q, best_action = float("-inf"), None
            for a in range(env.n_actions):
                q = Q[(state, a)]
                if q > best_q:
                    best_q = q
                    best_action = a
            return best_action
        
    def train_one_step():
        state = env.reset()
        episode_reward = 0.0
        while True:
            action = get_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            max_q_next = max(Q[(next_state, a)] for a in range(env.n_actions))
            td_error = (reward + gamma * max_q_next) - Q[(state, action)]
            Q[(state, action)] += alpha * td_error
            episode_reward += reward
            state = next_state

            if done:
                return episode_reward
            
    for i in range(n_episodes):
        episode_reward = train_one_step()
        episode_rewards.append(episode_reward)

    for s in range(env.n_states):
        best_action = None
        best_q = float("-inf")
        for a in range(env.n_actions):
            if Q[(s, a)] > best_q:
                best_q = Q[(s, a)]
                best_action = a
        policy[s] = best_action
    
    return Q, policy, episode_rewards


# --- CHALLENGE 5: TD(0) Prediction ---
def td0_prediction(env, policy, n_episodes=1000, alpha=0.1, gamma=0.99):
    """
    Implement TD(0) for policy evaluation.
    
    Args:
        env: GridWorld environment
        policy: Dictionary mapping state -> action (fixed policy to evaluate)
        n_episodes: Number of episodes
        alpha: Learning rate
        gamma: Discount factor
    
    Returns:
        V: Dictionary mapping state -> value estimate
        td_errors: List of TD errors for analysis
    """
    V = defaultdict(float)
    td_errors = []
    
    for i in range(n_episodes):
        state = env.reset()
        while True:
            action = policy[state]
            next_state, reward, done, _ = env.step(action)
            td_error = (reward + gamma * V[next_state]) - V[state]
            V[state] += alpha*td_error

            state = next_state
            td_errors.append(td_error)

            if done:
                break
    
    return V, td_errors


# --- CHALLENGE 6: n-step TD Prediction ---
def n_step_td_prediction(env, policy, n_episodes=1000, alpha=0.1, gamma=0.99, n=3):
    """
    Implement n-step TD for policy evaluation.
    
    Args:
        env: GridWorld environment
        policy: Dictionary mapping state -> action (fixed policy to evaluate)
        n_episodes: Number of episodes
        alpha: Learning rate
        gamma: Discount factor
        n: Number of steps to look ahead before bootstrapping
    
    Returns:
        V: Dictionary mapping state -> value estimate
    
    Think about:
        - TD(0): n=1, bootstrap after 1 step
        - n-step: wait n steps, then bootstrap with V[S_{t+n}]
        - MC: n=infinity, never bootstrap (use full return)
    """
    V = defaultdict(float)

    for _ in range(n_episodes):
        state = env.reset()
        queue = deque(maxlen=n)
        while True:
            action = policy[state]
            next_state, reward, done, _ = env.step(action)

            if len(queue) >= n:
                (pop_state, _, pop_reward, _, pop_done) = queue.popleft()
                if not pop_done:
                    G = pop_reward
                    for i, (s, a, r, ns, d) in enumerate(queue):
                        G += r * gamma**(i + 1)
                    G += gamma**(n) * V[next_state]
                    td_error = G - V[pop_state]
                    V[pop_state] += alpha * td_error
                    
            queue.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                break
        # Finish off queue

        while queue:
            (pop_state, _, pop_reward, _, pop_done) = queue.popleft()
            G = pop_reward
            for i, (s, a, r, ns, d) in enumerate(queue):
                G += r * gamma**(i + 1)
            td_error = G - V[pop_state]
            V[pop_state] += alpha * td_error
    
    return V


# --- CHALLENGE 7: TD(λ) Prediction with Eligibility Traces ---
def td_lambda_prediction(env, policy, n_episodes=1000, alpha=0.1, gamma=0.99, lambda_=0.8):
    """
    Implement TD(λ) for policy evaluation using eligibility traces.
    
    Args:
        env: GridWorld environment
        policy: Dictionary mapping state -> action (fixed policy to evaluate)
        n_episodes: Number of episodes
        alpha: Learning rate
        gamma: Discount factor
        lambda_: Trace decay parameter (0 to 1)
    
    Returns:
        V: Dictionary mapping state -> value estimate
    """
    V = defaultdict(float)
    
    for _ in range(n_episodes):
        state = env.reset()
        E = defaultdict(float)

        while True:
            action = policy[state]
            next_state, reward, done, _ = env.step(action)
            td_error = (reward + gamma * V[next_state]) - V[state]
            E[state] += 1
            for s in E:
                V[s] += alpha * td_error * E[s]
            for s in E:
                E[s] *= gamma * lambda_
            
            state = next_state
            if done:
                break
    
    return V


# ============================================================
# CONCEPTUAL CHALLENGES (Answer in comments or separate file)
# ============================================================

"""
CHALLENGE 7: Theoretical Questions

Q1: In your SARSA implementation, you use the actual next action (A') chosen by the 
    epsilon-greedy policy. In Q-Learning, you use max_a Q(S', a). 
    
    Explain with a concrete example from the GridWorld why SARSA might learn a 
    "safer" but suboptimal path compared to Q-Learning near the pits.
    
    YOUR ANSWER:
    

Q2: Policy Iteration is guaranteed to converge in a finite number of iterations.
    Why? What property of the policy improvement step ensures this?
    
    YOUR ANSWER:
    

Q3: TD(0) updates V(s) after every step, while MC waits until episode end.
    What are the tradeoffs? When would you prefer one over the other?
    (Hint: Think about variance, bias, and learning speed)
    
    YOUR ANSWER:
    

Q4: In TD(λ), what happens when λ=0 vs λ=1? Why is λ=0.8 or 0.9 often used
    in practice instead of the extremes?
    
    YOUR ANSWER:
    

Q5: You implemented both model-based (Value/Policy Iteration) and model-free 
    (SARSA, Q-Learning, TD) methods. If you had access to env.P (the transition 
    model), why would you ever use model-free methods?
    
    YOUR ANSWER:
    

Q6: Look at the GridWorld. State 5 is directly below the start (state 0) and 
    adjacent to a pit (state 6). 
    
    After training with Q-Learning (epsilon=0.1), what action would you expect 
    the optimal policy to take in state 5? Why?
    
    YOUR ANSWER:


Q7: Eligibility traces in TD(λ) allow credit assignment to states visited earlier
    in the episode. Consider this sequence: S0 -> S1 -> S2 -> S3 (reward=10).
    
    With λ=0.9 and γ=0.99, which state gets the most credit for the reward?
    Which gets the least? Why?
    
    YOUR ANSWER:
    
"""


# ============================================================
# TEST YOUR IMPLEMENTATIONS
# ============================================================

if __name__ == "__main__":
    # Create environments
    env_deterministic = GridWorld(slip_prob=0.0)
    env_stochastic = GridWorld(slip_prob=0.2)
    
    print("="*60)
    print("CHALLENGE 1: Value Iteration (Deterministic)")
    print("="*60)
    V, policy, deltas = value_iteration(env_deterministic)
    if V and policy:
        env_deterministic.render_values(V)
        env_deterministic.render_policy(policy)
        avg_return, std_return = evaluate_policy(env_deterministic, policy)
        print(f"Average Return: {avg_return:.2f} ± {std_return:.2f}")
        print(f"Converged in {len(deltas)} iterations")
    else:
        print("Not implemented yet!")
    
    print("\n" + "="*60)
    print("CHALLENGE 2: Policy Iteration (Deterministic)")
    print("="*60)
    V, policy, n_iter = policy_iteration(env_deterministic)
    if V and policy:
        env_deterministic.render_values(V)
        env_deterministic.render_policy(policy)
        avg_return, std_return = evaluate_policy(env_deterministic, policy)
        print(f"Average Return: {avg_return:.2f} ± {std_return:.2f}")
        print(f"Policy converged in {n_iter} improvement steps")
    else:
        print("Not implemented yet!")
    
    print("\n" + "="*60)
    print("CHALLENGE 3: SARSA (Stochastic Environment)")
    print("="*60)
    Q, policy, rewards = sarsa(env_stochastic, n_episodes=1000)
    if Q and policy:
        env_stochastic.render_policy(policy)
        avg_return, std_return = evaluate_policy(env_stochastic, policy)
        print(f"Average Return: {avg_return:.2f} ± {std_return:.2f}")
        print(f"Final 100-episode avg reward: {np.mean(rewards[-100:]):.2f}")
    else:
        print("Not implemented yet!")
    
    print("\n" + "="*60)
    print("CHALLENGE 4: Q-Learning (Stochastic Environment)")
    print("="*60)
    Q, policy, rewards = q_learning(env_stochastic, n_episodes=1000)
    if Q and policy:
        env_stochastic.render_policy(policy)
        avg_return, std_return = evaluate_policy(env_stochastic, policy)
        print(f"Average Return: {avg_return:.2f} ± {std_return:.2f}")
        print(f"Final 100-episode avg reward: {np.mean(rewards[-100:]):.2f}")
    else:
        print("Not implemented yet!")
    
    print("\n" + "="*60)
    print("CHALLENGE 5: TD(0) Prediction")
    print("="*60)
    # First get a policy from Q-learning to evaluate
    _, eval_policy, _ = q_learning(env_deterministic, n_episodes=500)
    if eval_policy:
        V, td_errors = td0_prediction(env_deterministic, eval_policy, n_episodes=2000)
        if V:
            env_deterministic.render_values(dict(V))
            print(f"Total TD errors recorded: {len(td_errors)}")
        else:
            print("Not implemented yet!")
    else:
        print("Need Q-Learning first!")
    
    print("\n" + "="*60)
    print("CHALLENGE 6: TD(λ) Prediction")
    print("="*60)
    if eval_policy:
        V = td_lambda_prediction(env_deterministic, eval_policy, n_episodes=2000, lambda_=0.8)
        if V:
            env_deterministic.render_values(dict(V))
        else:
            print("Not implemented yet!")
    else:
        print("Need Q-Learning first!")
    
    print("\n" + "="*60)
    print("BONUS: Compare SARSA vs Q-Learning near pits")
    print("="*60)
    print("Run both on stochastic env and compare policies at state 5 (near pit)")
    # Add your comparison code here