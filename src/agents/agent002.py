#! /usr/bin/env
"""
Second agent - learn from agent 001 to exceed performance.

Initial reading:
- [0], https://huggingface.co/learn/deep-rl-course/unit2/mc-vs-td (important chapter on Q-learning) <- Very important chapter in entirety
- [1], https://stats.stackexchange.com/questions/336974/when-are-monte-carlo-methods-preferred-over-temporal-difference-ones
- [2], https://stats.stackexchange.com/questions/355820/why-do-temporal-difference-td-methods-have-lower-variance-than-monte-carlo-metp
- [3], https://huggingface.co/learn/deep-rl-course/unit3/from-q-to-dqn <- This is what we are actually doing...

- A book: http://incompleteideas.net/book/RLbook2020.pdf

Initial thoughts:
- Do it as Monte-Carlo, to reward winning the game (I think). See https://huggingface.co/learn/deep-rl-course/unit2/mc-vs-td
- Formulate as a state-value problem (value of actions between states)
- Do some math on the potential state-size... need a bigger NN?
- Work with the reward-system to learn to win the game
   - Be quick about it
   - Put pieces next to each other
   - Block the other player!
- Train on the mirror of the state also - data augmentation
- From [3]:
    - Use experience replay (still) - avoid catastrophic forgetting
    - Use convolutional network to learn spatial information
    - Fixed Q-target to stabilize training (update after every C steps), don't chase a moving target
    - Double DQN? Maybe not... we are using Monte-Carlo (still) to generate Q-values of states


"""
