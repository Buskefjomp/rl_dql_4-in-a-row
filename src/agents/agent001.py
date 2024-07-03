#!/usr/bin/env python3
"""
The first hacked together agent. Can we make it go for 4 in a row?

Do not worry about structure or niceness - just make it work and do
something. For now it just plays against itself...

Monkey-see, monkey-do - from:
https://towardsdatascience.com/develop-your-first-ai-agent-deep-q-learning-375876ee2472
"""

import argparse
import logging
import pdb
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
    episodes = 100  # how many games to train across
    max_steps = board.cols * board.rows * 5  # illegal steps can be taken

    for i_ep in range(episodes):
        board.clear_state()
        # Random starting places
        board.add_coin(player + 1, np.random.randint(0, board.cols))

        for i_step in range(max_steps):
            # ##### Present the state to the agent and get an action #####
            x_ten = torch.Tensor(board._state)
            q_vals = agent.forward(x_ten).detach().numpy()
            action = np.argmax(q_vals)
            # _LOG.debug("Ep: %4d, step: %4d, q: %s -> %d", i_ep, i_step, q_vals, action)

            # TODO: Add epsilon-greedy functionality (exploration)

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

            # ##### Train the neural network #####
            # TODO: Add batches and add experience replay
            target_q_vals = q_vals.copy()
            if done:
                target_q_vals[action] = reward  # reinforce winning the game is good
                # Realization: We only change the q-vals (nn output) for the action taken.
                # No back-propagation changing the other outputs.
            else:
                # Extend with Q-values from what is achieved
                x_new = torch.Tensor(board._state)
                next_q_vals = agent.forward(x_new).detach().numpy()
                target_q_vals[action] = reward + 0.99 * np.max(next_q_vals)

            agent.train(True)
            y_ten = torch.Tensor(target_q_vals)
            pred = agent(x_ten)
            loss = loss_fn(pred, y_ten)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            agent.eval()
            # _LOG.debug("Loss: %s, action: %d, reward: %f", loss, action, reward)

            # ##### Check for done #####
            if done:
                break

        _LOG.debug("Episode: %d (%d steps), result:\n", i_ep, i_step + 1)
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
        x = torch.flatten(board_state)
        q_values = self.the_stack(x)
        return q_values


# ######################### Playing #########################
def play_agent_001():
    """Play against the agent."""


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_stack()
        pdb.post_mortem()
