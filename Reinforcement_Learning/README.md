The interaction between an agent and its environment is formalized as Markov Decision Process.

The agent takes an action $a_t$ at time t, and this action causes the state changing from $s_t$ to $s_(t+1)$. 

From a starting state to a termination state, the trajectory of the state, action is like:

$\tau = s_1, a_1, s_2, a_2, \dots, a_{N-1}, s_N$

The reward of this trajectory is defined as

$R(\tau) = \sum_{t=0}^{N} \gamma^t r_t $

where $r_t(s_t, a_t)$ is the reward at the time t (when the state was $s_t$ and the agent took action $a_t$).
