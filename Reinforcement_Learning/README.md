# Reinforcement Learning

## Problem Statement
The interaction between an agent and its environment is formalized as Markov Decision Process.

The agent takes an action $a_i$ at time i, and this action causes the state changing from $s_i$ to $s_{i+1}$. 

A policy is about which action to take under that state: $\theta = P(a_{i+1} | s_i)$

From a starting state $s_0$ to a termination state $s_N$, the trajectory of the state, action is like:

$\tau = s_0, a_0, s_1, a_1, \dots, a_{N-1}, s_N$

The reward of this trajectory is defined as

$R(\tau) = \sum_{i=0}^{N} \gamma^i r_i $

where $r_i(s_i, a_i)$ is the reward at the time i (when the state was $s_i$ and the agent took action $a_i$).

This is called action-value function. Also called "discounted" reward because a future rewards x steps away get "discounted" by a factor of $\gamma^x << 1$.

The reinforcement learning aims to find a best policy that maximizes the "discounted" reward:

$max_{\theta} R(\tau)$

## Discrete Actions (Deep Q-learning)

Assume the action space consists of K different discrete choices.

Q-learning estimates the "value" taking each action under the current state $s_i$ at time $t$:

$Q(s_i, a_i) = \sum_{i=t}^{N} \gamma^{i - t} r_t$

The best policy $\theta^{\*}$ will pick an action $a_i^{\*}$ that maximizes $Q(s_i, a_i) $. Thus,

$Q^{\*}(s_i, a_i) = max_{\theta} Q (s_i, a_i) = max_{\theta} \sum_{i=t}^{N} \gamma^{i-t} r_i = max_{\theta} \left( r(s_i, a_i) + \gamma Q^{\*}(s_{i+1}, a_{i+1}) \right) $

This recursion says: given the current state $s_i$, the best policy will pick an action that maximizes the sum of instant reward $r(s_i, a_i)$ and the discounted future "value".

DQN (Deep Q-Network) builds a deep neutral network which:
- takes the current state $s$ as input
- outputs K head, each head being the action value $Q^{\*}(s, a)$ for each action $a$.

If we have such a model, under any state, the agent can pick the best action corresponding to max output head (ie. we obtained the best policy).

How to train such a model? We want to iteratively update this model's outputs as the agent iteract with the environment (aka online learning).

We define a loss function as:

$SmoothL1Loss \left( Q(s_i, a_i) - max (r_i + \gamma Q(s_{i+1}, a_{i+1})) \right) $

where $r_i$ is the instant reward at step $i$, and $Q (s_{i+1}, a_{i+1}) $ is the action value at the next state $s_{i+1}$ taking next an action $a_{i+1}$.

The intuition is -- the current estimation $Q(s,a)$ will gradually get closer to $Q^{\*}(s,a)$.

So we can let the agent play a game and store the trajectory of the state, action and reward. 
Then we can compute a loss based on the tuple of $(s_i, a_i, s_{i+1}, r_{i})$.

## Continous Actions (PPO)
