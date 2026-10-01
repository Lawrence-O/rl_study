import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# ---------------------
# Helper: Bresenham’s algorithm to determine cells touched by a line.
def bresenham_line(x0, y0, x1, y1):
    """Return list of (x,y) positions on the grid that a line from (x0,y0) to (x1,y1) passes through."""
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            cells.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            cells.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    cells.append((x1, y1))
    return cells

# ---------------------
# Racetrack Environment

class RaceTrack:
    def __init__(self, track, start_states, finish_states, max_speed=4, noise_prob=0.1):
        """
        track: 2D numpy array of shape (rows, cols). 1 if in-track, 0 if off-track.
        start_states: list of (row, col) for starting positions.
        finish_states: set of (row, col) for finish cells.
        max_speed: maximum speed (each component is 0...max_speed).
        noise_prob: probability that the acceleration is replaced by (0,0).
        """
        self.track = track
        self.rows, self.cols = track.shape
        self.start_states = start_states
        self.finish_states = set(finish_states)
        self.max_speed = max_speed
        self.noise_prob = noise_prob
        self.reset(noisy=True)

    def reset(self, noisy=True):
        # choose a random starting cell; velocity must be zero.
        self.position = random.choice(self.start_states)
        self.velocity = (0, 0)
        self.noisy = noisy  # if False, then no noise is applied (for demonstration trajectories)
        return (self.position, self.velocity)

    def in_track(self, pos):
        row, col = pos
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        return self.track[row, col] == 1

    def step(self, acceleration):
        """
        acceleration: tuple (a_row, a_col) chosen from {-1,0,1}^2.
        Returns: new_state, reward, done.
        """
        # With probability noise_prob, override acceleration with (0,0)
        if self.noisy and random.random() < self.noise_prob:
            acceleration = (0, 0)
            
        # update velocity
        v_row, v_col = self.velocity
        a_row, a_col = acceleration
        new_v_row = v_row + a_row
        new_v_col = v_col + a_col
        # enforce constraints: components between 0 and max_speed.
        new_v_row = min(max(new_v_row, 0), self.max_speed)
        new_v_col = min(max(new_v_col, 0), self.max_speed)
        
        # Do not allow both velocity components to be 0 except if on start line.
        if (new_v_row, new_v_col) == (0,0) and self.position not in self.start_states:
            # force a minimum speed in one direction; here we choose to keep previous velocity if possible.
            if v_row + v_col > 0:
                new_v_row, new_v_col = self.velocity
            else:
                # if velocity was zero, allow a small nudge.
                new_v_row = 1

        old_row, old_col = self.position
        # compute new intended position
        new_row = old_row - new_v_row   # here we assume “up” is decreasing row index
        new_col = old_col + new_v_col
        new_position = (new_row, new_col)
        
        # get the cells that the car passes through (including start and finish)
        path = bresenham_line(old_row, old_col, new_row, new_col)
        
        crashed = False
        finished = False
        for cell in path:
            # if cell is off track
            if not self.in_track(cell):
                # Check: if cell is a finish cell then that is allowed.
                if cell in self.finish_states:
                    finished = True
                    break
                else:
                    crashed = True
                    break
            # Also, if cell is finish, we finish the episode.
            if cell in self.finish_states:
                finished = True
                break
        
        reward = -1  # reward is 1 per time step
        if finished:
            self.position = new_position  # update to finish position
            self.velocity = (new_v_row, new_v_col)
            done = True
            return ((self.position, self.velocity), 0, done)
        elif crashed:
            # Reset to a random starting position with zero velocity.
            self.position = random.choice(self.start_states)
            self.velocity = (0,0)
            done = False
            return (((self.position, self.velocity), reward, done))
        else:
            # Normal update.
            self.position = new_position
            self.velocity = (new_v_row, new_v_col)
            done = False
            return ((self.position, self.velocity), reward, done)

# ---------------------
# Example racetrack specification

def make_racetrack():
    """
    We create a simple racetrack.
    For example, we use a grid of size 10 x 30.
      - The track is defined as the set of cells with value 1.
      - The track has a bend.
      - The starting line is defined as the leftmost column between rows 6 and 8.
      - The finish line is defined as the top row between columns 25 and 29.
    """
    rows, cols = 10, 30
    track = np.zeros((rows, cols), dtype=int)
    # Fill in a simple track: a “road” that winds upward.
    # For simplicity, assume that from row 3 to 9, columns 0-29 are valid,
    # but with a “wall” below row 3 in columns 25+.
    for r in range(3, rows):
        for c in range(cols):
            track[r, c] = 1
    # Also, allow a horizontal segment at the top (rows 0-3) in columns 25 to 29.
    for r in range(0, 4):
        for c in range(25, cols):
            track[r, c] = 1

    # Starting line: bottom row of the track (for example, row 9, columns 0-4)
    start_states = [(9, c) for c in range(0, 5) if track[9, c] == 1]

    # Finish line: top row in columns 25 to 29 (row 0, columns 25-29)
    finish_states = [(r, c) for r in range(0,4) for c in range(25, cols) if track[r, c]==1]

    return track, start_states, finish_states

# ---------------------
# Off-Policy MC Control with Weighted Importance Sampling

def off_policy_mc_control(env, num_episodes=5000, gamma=1.0):
    """
    env: instance of RaceTrack.
    num_episodes: number of episodes to generate.
    gamma: discount factor.
    Returns: Q (dict mapping state-action pairs to values) and target policy pi.
    A state is ((row, col), (v_row, v_col)) and action is (a_row, a_col).
    """
    # Initialize Q(s,a) arbitrarily and C(s,a)=0 for every state-action pair.
    Q = defaultdict(float)
    C = defaultdict(float)
    # For simplicity, we define the set of possible actions.
    actions = [(a_row, a_col) for a_row in [-1, 0, 1] for a_col in [-1, 0, 1]]
    
    # Initialize the target policy to be greedy w.r.t. Q.
    # We represent the target policy as a dict mapping state to action.
    pi = {}
    
    # Behavior policy: we use an ε-soft (here uniform random) policy that assigns nonzero probability to every action.
    b_prob = 1.0 / len(actions)
    
    for episode in range(num_episodes):
        # Generate an episode following the behavior policy.
        episode_data = []
        state = env.reset(noisy=True)
        done = False
        while not done:
            # Choose action uniformly at random (the behavior policy b)
            action = random.choice(actions)
            # Take step
            next_state, reward, done = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state

        # Compute the return G for each time step (no discounting here, but gamma=1)
        G = 0
        W = 1.0
        # Process episode in reverse order.
        for t in reversed(range(len(episode_data))):
            state_t, action_t, reward_t = episode_data[t]
            G = gamma * G + reward_t
            key = (state_t, action_t)
            C[key] += W
            # Incremental update to Q using weighted importance sampling.
            Q[key] += (W / C[key]) * (G - Q[key])
            
            # Update the target policy for state_t: choose the action with the highest Q.
            # Note: we assume all actions have been visited at least once eventually.
            q_vals = {}
            for a in actions:
                q_vals[a] = Q[(state_t, a)]
            best_action = max(q_vals, key=q_vals.get)
            pi[state_t] = best_action

            # If the action taken is not what the target policy would do,
            # then the importance sampling ratio becomes zero.
            if action_t != pi[state_t]:
                break
            # Since the behavior policy is uniform, the importance sampling ratio is:
            W = W / b_prob

    return Q, pi

# ---------------------
# Generate and display several trajectories following the optimal (target) policy (noise turned off)

def simulate_optimal_trajectory(env, pi, max_steps=100):
    """
    Simulate one episode following the target policy pi.
    Turn off noise for the demonstration.
    Returns: list of positions visited.
    """
    trajectory = []
    state = env.reset(noisy=False)
    trajectory.append(state[0])
    done = False
    steps = 0
    while not done and steps < max_steps:
        # Get the greedy action if defined, else random.
        if state in pi:
            action = pi[state]
        else:
            action = random.choice([(a_row, a_col) for a_row in [-1,0,1] for a_col in [-1,0,1]])
        state, reward, done = env.step(action)
        trajectory.append(state[0])
        steps += 1
    return trajectory

# ---------------------
# Main execution

if __name__ == '__main__':
    # Create the racetrack.
    track, start_states, finish_states = make_racetrack()
    print(track)
    env = RaceTrack(track, start_states, finish_states, max_speed=4, noise_prob=0.1)
    
    # Run off-policy MC control to learn Q and the target policy.
    print("Learning policy ...")
    Q, pi = off_policy_mc_control(env, num_episodes=10000)
    print("Learning complete.")

    # Plot several trajectories following the optimal policy (with noise turned off).
    num_trajs = 5
    plt.figure(figsize=(10,5))
    plt.imshow(track, cmap='gray_r')
    for i in range(num_trajs):
        traj = simulate_optimal_trajectory(env, pi, max_steps=200)
        traj = np.array(traj)
        plt.plot(traj[:,1], traj[:,0], marker='o', label=f'Traj {i+1}')
    plt.gca().invert_yaxis()
    plt.title("Trajectories following the optimal policy")
    plt.legend()
    plt.show()
