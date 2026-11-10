# Solving a Markov Decision Process using Policy Iteration

## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, by repeatedly performing policy evaluation and policy improvement to obtain the optimal value function and optimal policy.

---

## Problem Statement

In this experiment, the `FrozenLake-v1` environment is solved using the **Policy Iteration** algorithm.

The agent starts from the start state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process. Policy Iteration is used to repeatedly evaluate the current policy and improve it until the policy becomes stable.

The objective is to find:

1. The optimal state-value function $V^*(s)$
2. The optimal policy $pi^*(s)$

---

## Software Requirements

```bash
pip install gymnasium numpy
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---

## Theory

Policy Iteration is a Dynamic Programming method used to find the optimal policy of a Markov Decision Process.

It consists of two major steps:

1. **Policy Evaluation**
2. **Policy Improvement**

These two steps are repeated until the policy becomes stable.

---

## Policy Evaluation

Policy evaluation estimates the value function for the current policy.

The Bellman expectation equation is:

$$
V^\pi(s) =
\sum_a \pi(a \mid s)
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $pi(a \mid s)$ | Probability of taking action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $pi$ |

---

## Policy Improvement

Policy improvement updates the policy greedily with respect to the current value function.

The improved policy is obtained as:

$$
\pi'(s) =
\arg\max_a
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

If the improved policy is the same as the old policy, the policy is considered stable.

---

## Algorithm

1. Create the Gymnasium `FrozenLake-v1` environment.
2. Initialize a random policy.
3. Repeat until the policy becomes stable:
   - Evaluate the current policy using iterative policy evaluation.
   - Improve the policy greedily using the current value function.
   - Compare the old policy and the new policy.
4. Stop when the policy does not change.
5. Display the optimal value function and optimal policy.

---

## Python Program

```python
import gymnasium as gym
import numpy as np
# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
env = env.unwrapped

n_states = env.observation_space.n
n_actions = env.action_space.n

gamma = 0.99
theta = 1e-8
# -------------------------------------------------
# Policy Evaluation
# -------------------------------------------------
def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    """
    Performs iterative policy evaluation using the Bellman expectation equation.

    Parameters:
        env    : Gymnasium FrozenLake environment
        policy : Fixed policy to be evaluated
        gamma  : Discount factor
        theta  : Convergence threshold

    Returns:
        V         : Estimated state-value function
        iteration : Number of iterations used for convergence
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    iteration = 0

    while True:
        delta = 0
        iteration += 1
        for state in range(n_states):
            old_value = V[state]
            new_value = 0
            for action in range(n_actions):
                action_probability = policy[state][action]
                for probability, next_state, reward, terminated in env.P[state][action]:
                    new_value += action_probability * probability * (reward + gamma * V[next_state])
            V[state] = new_value
            delta = max(delta, abs(old_value - V[state]))

        if delta < theta:
            break

    return V, iteration


# -------------------------------------------------
# Policy Improvement
# -------------------------------------------------
def policy_improvement(env, V, gamma=0.99):
    """
    Improves the policy by making it greedy with respect to the value function V.

    Parameters:
        env   : Gymnasium FrozenLake environment
        V     : State-value function from policy evaluation
        gamma : Discount factor

    Returns:
        new_policy : An updated, deterministic policy
        policy_stable: Boolean indicating if the policy stopped changing
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize a new deterministic policy
    new_policy = np.zeros((n_states, n_actions))

    for state in range(n_states):
        # Calculate Q-values for all actions from this state
        Q = np.zeros(n_actions)
        for action in range(n_actions):
            for probability, next_state, reward, terminated in env.P[state][action]:
                Q[action] += probability * (reward + gamma * V[next_state])

        # Find the action with the maximum Q-value
        best_action = np.argmax(Q)

        # Set the probability of the best action to 1.0
        new_policy[state][best_action] = 1.0

    return new_policy
#-------------------------------------------------
# Policy Iteration
# -------------------------------------------------
def policy_iteration(env, gamma=0.99, theta=1e-8):
    """
    Performs full Policy Iteration to find the optimal policy and value function.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # 1. Start with a random uniform policy
    policy = np.ones((n_states, n_actions)) / n_actions

    iteration = 0
    while True:
        iteration += 1

        # 2. Evaluate current policy
        V, eval_iters = policy_evaluation(env, policy, gamma, theta)

        # 3. Improve policy based on new V
        new_policy = policy_improvement(env, V, gamma)

        # 4. Check for convergence (if the policy didn't change)
        if np.array_equal(new_policy, policy):
            print(f"Policy converged after {iteration} iterations.")
            break

        policy = new_policy

    return policy, V

# Run the algorithm
optimal_policy, optimal_V = policy_iteration(env)

# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(V):
    print("\nOptimal State-Value Function:")
    print(np.round(V.reshape(4, 4), 4))


def print_policy(policy):
    action_symbols = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑"
    }

    best_actions = np.argmax(policy, axis=1)
    policy_grid = np.array(
        [action_symbols[action] for action in best_actions]
    ).reshape(4, 4)


    print("\nOptimal Policy:")
    print(policy_grid)
# -------------------------------------------------
# Run Policy Iteration
# -------------------------------------------------

optimal_policy, optimal_value_function = policy_iteration(
    env,
    gamma=gamma,
    theta=theta
)

print("Name:   PYNAM VINODH    ")
print("Register Number:   212223240131    ")
print_value_function(optimal_value_function)
print_policy(optimal_policy)

env.close()

```

## Output

```text

Total policy iterations: 

Optimal State-Value Function:


Optimal Policy:

```
<img width="403" height="265" alt="image" src="https://github.com/user-attachments/assets/c69f3e69-ecf1-4d33-b3e3-4a5eb4fe1ce0" />



---

## Result

```text

Policy Iteration was successfully implemented for the FrozenLake-v1 environment. The algorithm repeatedly performed policy evaluation and policy improvement until the policy became stable. The final state-value function represents the optimal expected discounted return for each state, while the final policy specifies the best action to take in each state.
```
---

## Inference
```text
Policy Iteration efficiently solves a finite Markov Decision Process by alternating between evaluating the current policy and improving it greedily. During policy evaluation, the Bellman expectation equation estimates the value of each state. During policy improvement, the action with the highest expected value is selected. The process terminates when further policy improvement produces no changes, indicating that the optimal policy has been reached. In FrozenLake, stochastic transitions are handled using the transition probabilities provided by env.P[s][a].

```
---

