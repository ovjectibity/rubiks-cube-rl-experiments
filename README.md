## The Plan

Inspired by OpenAI's [2019 paper](https://arxiv.org/abs/1910.07113) on solving the Rubik's Cube with a robot hand, I'm trying to solve the standard 3x3x3 Rubik's in 4 grades of difficulty using deep RL:
1. Grade-1: Purely representational I/O
2. Grade-2: Representiational input + Assisted articulation
3. Grade-3: Representational input + Embodied articulation
4. Grade-4: Fully embodied perception + articulation

I'm still on Grade-1. Down the line, I also want to build policies that are able to solve more difficult variations as well such as 4x4x4, 5x5x5 & so on, including other spatial puzzles.

The idea is for this problem domain - a geometric puzzle - to be a test bed of experiments with various on-policy & off-policy RL algorithms (especially interested in taking a [crack at this](https://seohong.me/blog/q-learning-is-not-yet-scalable/)), algorithm hyperparameters, training & regularization techniques & derive a domain-specific scaling law.
