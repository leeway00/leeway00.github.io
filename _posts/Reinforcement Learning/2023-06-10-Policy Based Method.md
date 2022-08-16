---
title: "Policy Based Method"
category: Reinforcement Learning
tags:
  - [RL]
permalink: /rl/policy/
excerpt: 
last_modified_at: Now

layout: single_v2
katex: true
---


# Policy Based Method
What about larger state space?
If State varaible is continuous, it's hard to make it. In this case, use Neural Network to calculate the probability of each action.

## Policy-Based Approximation
Neural network approximates!!

In the case of discrete action spaces, the neural network has one node for each possible action.\
For continuous action spaces, the neural network has one node for each action entry (or index).

## Hill Climbing
Agent can gradually improve expected return $J$. 
$\theta$ encodes the policy and influecnes rewrad and affect$J$
$J(\theta) = \sum_{s,a} P(s,a) R(s,a)$\
### Gradient Ascent?
Gradient descent steps in the direction opposite the gradient, since it wants to minimize a function.
Gradient ascent is otherwise identical, except we step in the direction of the gradient, to reach the maximum.
### Hill Climbing
Hill Climbing is similar but different.  It is an iterative algorithm that starts with an arbitrary solution to a problem, then attempts to find a better solution by making an incremental change to the solution. At each iteration, hill climbing will adjust a single element in $\mathbf{x}$  and determine whether the change improves the value of $f(\mathbf{x})$. This differs from gradient descent methods, which adjust all of the values in {\displaystyle \mathbf {x} }\mathbf {x}  at each iteration according to the gradient of the hill.

### Pseudocode

Ititialize $\theta$ to a policy arbitrarily
Collect an eplisode with $\theta$ and record the return $G$
Repeat until environment solved:
1. Add a little bit of random noise. $\theta_{new} \leftarrow \theta_{best}+\epsilon$
2. Collect an episode with $new \theta$ and record the return with $G_{new}$.
3. If $G_{new} > G_{best}$:
   1. $\theta_{best} \leftarrow \theta_{new}$
   2. $G_{best} \leftarrow G_{new}$

cf) the return that the agent collects in a single episode $G$ and the expected return $J$

### Additional topic
- Steepest hill climbing: pick the candidate policy that looks best in the direction of the gradient, and then iterate. Help mitigate the case to be finished as suboptimal solution.
- Simulated Annealing: uses a pre-defined schedule to control how the policy space is explored, and gradually reduces the search radius as we get closer to the optimal solution.
- Adaptive Noise Scaling: decreases the search radius with each iteration when a new best policy is found, and otherwise increases the search radius.


- Entropy method: What if we select top 10% and take average?
- Evolution Strategies: better get higher weights

![Simple RL for CartPole](https://kvfrans.com/simple-algoritms-for-solving-cartpole/)
![Evolution Strategy](https://github.com/alirezamika/evostra)

## Why Policy based methods
1. Simplicty
   1. Value based method needs value function and policy function. But Policy based does not need it. Rather, it uses stohchastic notation to define its policy. 
2. Stochastic Policy
   1. Aliased States: Two states are considered same optimal points. Either go right or left is fine but not both. In this case get into trench which object oscillates. 
   2. Rock Sissor Paper: uniform distribution is best
3. Continuous action spaces: Policy-based methods are well-suited for continuous action spaces.


