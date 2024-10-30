# Reinforcement Learning

## Problem Statement
The interaction between an agent and its environment is formalized as Markov Decision Process.

The agent takes an action $a_i$ at time i, and this action causes the state changing from $s_i$ to $s_{i+1}$. 

A policy $\theta$ is about which action to take under that state: $\pi_{\theta}(a_{i} | s_i)$ denotes the probablity of taking action $a_i$ under $s_i$.

From a starting state $s_0$ to a termination state $s_N$, the trajectory of the state, action is like:

$\tau = s_0, a_0, s_1, a_1, \dots, a_{N-1}, s_N$

The reward of this trajectory is defined as

$R(\tau) = \sum_{i=0}^{N} \gamma^i r_i $

where $r_i = r(s_i, a_i)$ is the reward function at the time i.

If starting from time $t$, this is the action-value function $Q(s_i, a_i) = \sum_{i=t}^{N} \gamma^i r_i$. 

It's also called "discounted" reward because a future rewards x steps away get "discounted" by a factor of $\gamma^x << 1$.

The reinforcement learning aims to find a best policy that maximizes the expected summed "discounted" reward:

$max_{\theta} \mathbb{E}[ R(\tau) ]$

## Discrete Actions (Deep Q-learning)

Assume the action space consists of K different discrete choices.

Q-learning estimates the "value" taking each action under the current state $s_i$ at time $t$:

$Q(s_t, a_t) = \sum_{i=t}^{N} \gamma^{i - t} r_i$

The best policy $\theta^{\*}$ will pick an action $a_i^{\*}$ that maximizes $Q(s_i, a_i) $. Thus,

$Q^{\*}(s_t, a_t) = max_{\theta} Q (s_t, a_t) = max_{\theta} \sum_{i=t}^{N} \gamma^{i-t} r_i = max_{\theta} \left( r(s_t, a_t) + \gamma Q^{\*}(s_{t+1}, a_{t+1}) \right) $

This recursion says: given the current state $s_t$, the best policy will pick an action that maximizes the sum of the **instant** reward $r_t(s_t, a_t)$ and the discounted **future** "value".

DQN (Deep Q-Network) builds a deep neutral network which:
- takes the current state $s$ as input
- outputs K head, each head being the action value $Q^{\*}(s, a)$ for each action $a$.

If we have such a model, under any state, the agent can pick the best action corresponding to max output head (ie. the best policy).

How to train such a model? 

We want to **iteratively** update this model's outputs as the agent iteract with the environment.

Define a loss function as:

$SmoothL1Loss \left( Q(s_i, a_i) - max_{\theta} (r_i + \gamma Q(s_{i+1}, a_{i+1})) \right) = SmoothL1Loss \left( Q(s_i, a_i) - r_i  - \gamma max_{\theta} ( Q(s_{i+1}, a_{i+1})) \right) $

where $r_i$ is the instant reward at step $i$, and $Q (s_{i+1}, a_{i+1}) $ is the action value at the next state $s_{i+1}$ taking next an action $a_{i+1}$.

The intuition is -- the current estimation $Q(s,a)$ will gradually get closer to $Q^{\*}(s,a)$; the agent is correcting itself by learning from the instant rewards.

So we can let the agent play a game and store the trajectory of the state, action and reward.

Then we can compute a loss based on a batch of the tuples of $(s_i, a_i, s_{i+1}, r_{i})$ and then update the network weights.

Notice that we will need to balance the exploitation and exploration of the agent. 

So by certain chance, we let the agent pick a random action, instead of picking the best action that maximizes the current $Q(s, a)$.

## Continous Actions (PPO)

The expectation of the "discounted" reward is:

$\mathbb{E}_{\tau} [ R(\tau)] $

