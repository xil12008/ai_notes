# Reinforcement Learning

## Problem Statement
The interaction between an agent and its environment is formalized as Markov Decision Process (MDP).

The policy defines the probablity of an agent taking action $a_i$ under $s_i$, which is:

$$
\pi_{\theta}(a_i | s_i)
$$

Here the policy is parameterized by $\theta$.

This action $a_i$ causes the state changing from $s_i$ to $s_{i+1}$ with probability:

$$
P(s_{i+1} | s_i, a_i)
$$ 

From a starting state $s_0$ to a termination state $s_N$, as the agent interacts with the environment, the trajectory of the state, action is:

$$
\tau = s_0, a_0, s_1, a_1, \dots, a_{N-1}, s_N
$$

The reward of this trajectory is defined as

$$
R(\tau) = \sum_{i=0}^{N} \gamma^i r_i
$$

where $r_i = r(s_i, a_i)$ is the reward at the time i. Notice the reward can also be probablistic.

If starting from time $t$, the action-value function is defined as

$$
Q_{\theta}(s_t, a_t) = \sum_{i=t}^{N} \gamma^{i-t} r_i
$$

It's also called the "discounted" reward because a future rewards get "discounted" by a factor of $\gamma^{i-t} < 1$.

The state-value function (aka. value function) is the expected action-value:

$$
V_{\theta}(s_t) = \sum_i \pi_{\theta}(a_i | s_t) Q(s_t, a_i)
$$

Notice that the value for the same state can be different for different policies, eg. a state can be very useful for a strategy, but not useful for another strategy.

For this reason, both state-value $V_{\theta}$ and action-value $Q_{\theta}$ functions are subscript-ed by $\theta$.

The reinforcement learning aims to find a best policy that maximizes the expected summed "discounted" reward:

$$
\theta^{*} = argmax_{\theta} \mathbb{E}[ R(\tau) ]
$$

If we know the starting state is going to be some particular $s_0$, then a simplier expression of this objective is,

$$
\theta^{*} = argmax_{\theta} \left[ V(s_0) \right] = argmax_{\theta} \left[ \sum_i \pi_{\theta}(a_i | s_0) Q(s_0, a_i) \right]
$$

Looking at the expression above, there are mainly two terms: $\pi_{\theta}(a_i | s_0)$ and $Q(s_0, a_i)$

Correspondingly, there are generally two stylies of reinforcement learning:

- approximate the action-value $Q(s, a)$ for each action $a$ under each state $s$, so the derived policy can choose the best action.
- approximate the policy $\pi_{\theta}(a | s)$ directly; the derived policy should maximize the total rewards.

Q-learning, as the name indicates, follows the first schema. 

Policy Gradient Approximation follows the second schema.

## Policy Gradient Approximation

Since we are only interested in the $\pi_{\theta}(a | s)$ that produces the maximum reward rather than what exactly the reward is,

we just need its gradient over the policy parameters $\theta$, which is

$$
\nabla_{\theta} V(s_0) = \nabla_{\theta} \left[ \sum_i \pi_{\theta}(a_i | s_0) Q(s_0, a_i) \right]
$$

Notice the enviroment is independent of the policy $\pi$, so 

$$
\nabla_{\theta} P(s_{i+1} | s_i, a_i) = 0
$$ 

Thus,

$$
\nabla_{\theta} Q(s_0, a_i) = \nabla_{\theta} \sum_s P(s, r | s_0, a_i) [ r(s_0, a_i) + V(s) ] = \sum_s \nabla_{\theta} P(s | s_0, a_i) V(s) = \sum_s P(s | s_0, a_i) \nabla_{\theta} V(s)
$$

That way, we derived a recurisive definition from $s_0$ to any state $s$ as follow,

$$
\nabla_{\theta} V(s_0) 
= \sum_i \nabla_{\theta} \pi_{\theta}(a_i | s_0) Q(s_0, a_i) + \pi_{\theta}(a_i | s_0) \nabla_{\theta} Q(s_0, a_i)
= \sum_i \left[ \nabla_{\theta} \pi_{\theta}(a_i | s_0) Q(s_0, a_i) + \pi_{\theta}(a_i | s_0) \sum_s P(s | s_0, a_i) \nabla_{\theta} V(s) \right]
$$

Unrolling the equation, for any state $s_k$, the term $\nabla_{\theta} \pi_{\theta}(a_{i_k} | s_k) Q(s_k, a_{i_k})$ will appears repeatedly with multipliers:

$$
\sum_{s_0, a_0, s_1, a_1, ..., s_{k-1}, a_{k-1}, s_k} \pi_{\theta}(a_{i_0} | s_0) P(s_1 | s_0, a_{i_0})  ... \pi_{\theta}(a_{i_{k-1}} | s_{k-1}) P(s_k | s_{k-1}, a_{i_{k-1}})
$$

Therefore,

$$
\nabla_{\theta} V(s_0) \propto E_{\pi} \[ \sum_a Q(s, a) \nabla_{\theta} \pi_{\theta}(a | s) \] 
$$

Here the expectation $E_{\pi}\[\]$ over a policy $\pi$ means we can sample this term $\sum_a Q(s, a) \nabla_{\theta} \pi_{\theta}(a | s)$ by Monto Carlo method as this policy visits each state.

The summation can be inconvenient. Thus,

$$
\nabla_{\theta} V(s_0) \propto E_{\pi} \[ \sum_a \pi_{\theta}(a | s) Q(s, a) \frac{\nabla_{\theta} \pi_{\theta}(a | s)}{\pi_{\theta}(a | s)} \] = E_{\pi} \[ Q(s, a) \frac{\nabla_{\theta} \pi_{\theta}(a | s)}{\pi_{\theta}(a | s)} \] 
$$

Then the sampling is also applied over the action this policy takes.

It turns out for any term $b(s)$ that is irrelevant to the actions, the following expectation is zero:

$$
E_{\pi} \left[ b(s) \nabla \log \left( \pi_\theta(a | s) \right) \right] = 0
$$

Therefore, the gradient is also:

$$
E_{\pi} \left[ A(s, a) \nabla \log \left( \pi_\theta(a | s) \right) \right]
$$
where the advantage function $A(s, a) = Q(s, a) - V(s)$ reflects the advantage of selecting some action $a$ compared to selecting the average action.

To maximize $E_{\tau}[R(\tau)]$, adding the gradient means:
- if $A(s, a) > 0$, increasing $\pi_\theta(a|s)$;
- if $A(s, a) < 0$, decreasing $\pi_\theta(a|s)$.

The loss function that has the same gradient is (the negative sign is needed because we minimize loss):

$$
E_{\pi} \left[ A(s, a) \log \left( \pi_\theta(a | s) \right) \right]
$$

Notice we also need to estimate the $A(s_i, a_i) = Q(s_i, a_i) - V(s_i)$. 

Actor-critic methods consist of two models, which may optionally share parameters:

- Critic: updates the estimation of $A(s_i, a_i)$.
- Actor: use the estimated $A(s_i, a_i)$ to update the policy $\theta(a | s)$.

Off-policy learning:

The behavior policy $\pi_{\text{old}}$ used for sampling can be different than the target policy the agent $\pi$ wants to learn.

## Discrete Actions (Deep Q-learning)

Assume the action space consists of K different discrete choices.

Q-learning estimates the "value" taking each action under the current state $s_i$ at time $t$, ie.

$Q(s_t, a_t) = \sum_{i=t}^{N} \gamma^{i - t} r_i$

The best policy $\theta^{\*}$ will pick an action $a_t^{\*}$ that maximizes $Q(s_t, a_t) $. Thus,

$Q^{\*}(s_t, a_t) = max_{\theta} Q (s_t, a_t) = max_{\theta} \sum_{i=t}^{N} \gamma^{i-t} r_i = max_{\theta} \left( r(s_t, a_t) + \gamma Q^{\*}(s_{t+1}, a_{t+1}) \right) $

This recursion says: given the current state $s_t$, the best policy will pick an action that maximizes the sum of the **instant** reward $r_t(s_t, a_t)$ and the discounted **future** "value".

If we have such a model, under any state $s_t$, this model can output all $Q^*(s_t, a_t)$ for each valid action $a_t$. 

Then one can pick the best action corresponding to the max action value, ie. we derived the best policy.

DQN (Deep Q-Network) builds a deep neutral network which:
- takes the current state $s$ as input
- outputs K head, each head being the action value $Q^{\*}(s, a)$ for each action $a$.

How to train such a model?

The "memoryless" Markov property says the future evolution of the process is independent of its history -- to maximize $Q(s_t, a_t)$, one must maximize $Q(s_{t+1}, a_{t+1})$.

We want to **iteratively** update this model's outputs as the agent iteract with the environment.

The intuition is -- the current estimation $Q(s,a)$ will gradually get closer to $Q^{\*}(s,a)$; the agent is correcting itself by learning from the instant rewards.

Let's define a loss function (temporal difference of one step $TD(0)$, looking at the error by comparing current estimate and the future estimate, aka. boostraping -- making a guess out of a guess) as:

$SmoothL1Loss \left( Q(s_i, a_i) - max_{\theta} (r_i + \gamma Q(s_{i+1}, a_{i+1})) \right) = SmoothL1Loss \left( Q(s_i, a_i) - r_i  - \gamma max_{\theta} ( Q(s_{i+1}, a_{i+1})) \right) $

where $r_i$ is the instant reward at step $i$, and $Q(s_{i+1}, a_{i+1})$ is the action value at the next state $s_{i+1}$ taking next an action $a_{i+1}$.

So we can let the agent play a game and store the trajectory of the state, action and reward.

Then we can compute a loss based on a batch of the tuples of $(s_i, a_i, s_{i+1}, r_{i})$ and then update the network weights.

Notice that we will need to balance the **exploitation and exploration** of the agent. 

So by certain chance, we let the agent pick a random action, instead of picking the best action that maximizes the current $Q(s, a)$.

Notice this way the policy generating the trajectories/replays is not the final policy that DQN converges to. So the DQN is off-policy RL.

Off-policy is usually preferred than on-policy because it can also leverage the histories/replays generated from other policies (random/prior policies, instead of the one Q-learning aims to learn) .

The **maximization bias** can cause q-learning to converge slowly, this is because: the max in the loss function can have some bias simply due to noise, then it sticks to that action max a future Q.

To be more robust, double Q-learning use two networks -- one to pick the action, the other to estimate the action-value, in order to reduce the **maximization bias**.
