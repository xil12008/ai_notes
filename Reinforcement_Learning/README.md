# Reinforcement Learning

## Problem Statement
The interaction between an agent and its environment is formalized as Markov Decision Process.

The agent takes an action $a_t$ at time t, and this action causes the state changing from $s_t$ to $s_{t+1}$. 

A policy is about which action to take under that state: $\theta = P(a_{t+1} | s_t)$

From a starting state to a termination state, the trajectory of the state, action is like:

$\tau = s_0, a_0, s_1, a_1, \dots, a_{N-1}, s_N$

The reward of this trajectory is defined as

$R(\tau) = \sum_{t=0}^{N} \gamma^t r_t $

where $r_t(s_t, a_t)$ is the reward at the time t (when the state was $s_t$ and the agent took action $a_t$).

The reinforcement learning aims to maximize the reward:

$max_{\theta} R(\tau)$

## Discrete Actions (Deep Q-learning)

Assume the action space consists of K different discrete choices.

Q-learning estimates the "value" taking each action under the current state $s_i$:

$ Q(s_i, a_i) = \sum_{i=t}^{N} \gamma^(i -t) r_t $

This is called action-value function. Also called "discounted" reward because a future rewards x steps away get "discounted" by a factor of $\gamma^x << 1$.

The best policy $\theta^{\*}$ will pick an action $a_i^{\*}$ that maximizes $ Q(s_i, a_i) $. Thus,

$Q^{\*}(s_i, a_i) = max_{\theta} Q (s_i, a_i) = max_{\theta} \sum_{i=t}^{N} \gamma^(i-t) r_i = max_{\theta} \left r(s_i, a_i) + \gamma Q^{\*}(s_{i+1}, a_{i+1}) \right $

This recursion says: given the current state $s_i$, the best policy will pick an action that maximizes the sum of instant reward $r(s_i, a_i)$ and the discounted future "value".

DQN (Deep Q-Network) builds a deep neutral network which:
- takes the current state $s$ as input
- outputs K head, each head being the action value $Q^{\*}(s, a) \for each action a$.

If we have such a model, under any state, the agent can pick the best action corresponding to max output head (ie. we obtained the best policy).

How to train such a model? We want to iteratively update this model's outputs as the agent iteract with the environment (aka online learning).

We define a loss function as:

$SmoothL1Loss \left Q(s_i, a_i) - max (r_i + \gamma Q(s_{i+1}, a_{i+1})) \right $

where $r_i$ is the instant reward at step $i$, and $Q (s_{i+1}, a_{i+1}) $ is the action value at the next state $s_{i+1}$ taking next an action $a_{i+1}$.

The intuition is -- the current estimation $Q(s,a)$ will gradually get closer to $Q^{\*}(s,a)$.

So we can let the agent play a game and store the trajectory of the state, action and reward. 
Then we can compute a loss based on the tuple of $(s_i, a_i, s_{i+1}, r_{i})$.

## Continous Actions (PPO)
