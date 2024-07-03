#!/usr/bin/env python3
"""
The first hacked together agent. Can we make it go for 4 in a row?

Do not worry about structure or niceness - just make it work and do
something. For now it just plays against itself...

Monkey-see, monkey-do - from:
https://towardsdatascience.com/develop-your-first-ai-agent-deep-q-learning-375876ee2472
"""

import argparse
import collections
import logging
import pdb
import random
import traceback

import numpy as np
import torch
from fiar.board import FiarBoard

_LOG = logging.getLogger(__name__)


# ######################### Entry #########################
def main():
    """Entry of program."""
    parser = argparse.ArgumentParser(description=__doc__)

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--train", action="store_true", help="Retrain the agent")
    grp.add_argument("--play", action="store_true", help="Face off against the agent")

    parser.add_argument("--debug", action="store_true", help="Help with debugging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)4s: %(message)s",
        encoding="utf-8",
        handlers=[logging.StreamHandler()],
    )
    if args.debug:
        _LOG.setLevel(logging.DEBUG)

    # ########## Select what to do ##########
    if args.train:
        train_agent_001()
    if args.play:
        play_agent_001()


# ######################### Reusables #########################


# ######################### Training #########################
def train_agent_001():
    """(Re)-Train the agent."""

    board = FiarBoard()

    agent = Agent001(board.cols * board.rows, board.cols).to("cpu")
    _LOG.info("Training agent:\n%s", agent)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

    player = 1  # players number
    episodes = 1000  # how many games to train across
    max_steps = board.cols * board.rows * 5  # illegal steps can be taken

    # NN-related
    batch_size = 4  # 32
    gamma = 0.98  # discount on future rewards (dampening)
    epsilon_cur = 1  # Current chance of exploration
    epsilon_dec = 0.998  # Decay over episodes
    epsilon_end = 0.010  # Minimum chance of exploration

    # Saving for experience replay
    Experience = collections.namedtuple(
        "Experience", ["state", "action", "reward", "next_state", "done"]
    )
    memory = collections.deque(maxlen=batch_size * 200)

    # ##### Playing games #####
    for i_ep in range(episodes):
        board.clear_state()
        # Random starting places ?
        # board.add_coin(player + 1, np.random.randint(0, board.cols))

        for i_step in range(max_steps):
            action = None
            x_ten = torch.Tensor(board.get_flat_state())

            if np.random.rand() < epsilon_cur:  # Explore \o/
                action = np.random.randint(0, board.cols)
            else:  # ##### Present the state to the agent and get an action #####
                q_vals = agent.forward(x_ten)
                action = torch.argmax(q_vals)
                # _LOG.debug("Ep: %4d, step: %4d, q: %s -> %d", i_ep, i_step, q_vals, action)

            if epsilon_cur > epsilon_end:
                epsilon_cur *= epsilon_dec

            # ##### Take a step, save the experience (including reward) #####
            result = board.add_coin(player, action)
            if result is None:  # illegal move
                continue

            # How did that go?
            n = board.get_span(1, result[0], result[1])
            done = n >= 4

            # Set the reward
            reward = -1
            if done:  # won the game?
                reward = 100
            elif n > 1:  # placing coins next to each other?
                reward += n * 10

            # Save it all
            exp = Experience(
                state=x_ten,
                action=action,
                reward=reward,
                next_state=torch.Tensor(board.get_flat_state()),
                done=done,
            )
            memory.append(exp)

            # ##### Train the neural network in a batch #####
            if len(memory) > batch_size * 4:  # have some entropy to pick from
                batch = random.sample(memory, batch_size)
                states = torch.stack([exp.state for exp in batch])
                actions = [exp.action for exp in batch]
                rewards = [exp.reward for exp in batch]
                n_states = torch.stack([exp.next_state for exp in batch])
                dones = [exp.done for exp in batch]

                agent.train(True)
                cur_q_vals = agent(states)
                nex_q_vals = agent(n_states)
                trg_q_vals = cur_q_vals.clone().detach()

                # Setup all rewards
                for i in range(batch_size):
                    if dones[i]:  # Reinforce winning the game is good
                        trg_q_vals[i, actions[i]] = rewards[i]
                        # Realization: We only change the q-vals (nn output) for the action taken.
                        # No back-propagation changing the other outputs.
                    else:
                        trg_q_vals[i, actions[i]] = rewards[i] + gamma * torch.max(
                            nex_q_vals
                        )

                # Do the training-step
                pred = agent(states)
                loss = loss_fn(pred, trg_q_vals)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                agent.eval()
                # _LOG.debug("Loss: %s, action: %d, reward: %f", loss, action, reward)

            # ##### Check for done #####
            if done:
                break

        if i_ep % 100 == 0:
            _LOG.debug(
                "Episode: %d/%d (%d steps), result:\n", i_ep, episodes, i_step + 1
            )
            board.print_state()


class Agent001(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.n_inputs = num_inputs
        self.n_outputs = num_outputs

        self.the_stack = torch.nn.Sequential(
            # Hidden layer
            torch.nn.Linear(self.n_inputs, 128),  # input to internals
            torch.nn.ReLU(),
            # Hidden layer
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            # Output layer (pick an action - drop a coin in a column)
            torch.nn.Linear(64, self.n_outputs),
        )

    def forward(self, board_state):
        q_values = self.the_stack(board_state)
        return q_values


# ######################### Playing #########################
def play_agent_001():
    """Play against the agent."""


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_stack()
        traceback.print_exception(exc)
        pdb.post_mortem()
