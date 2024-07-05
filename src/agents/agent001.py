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
import pathlib
import pdb
import random
import time
import traceback

import numpy as np
import torch
from fiar.board import FiarBoard

_LOG = logging.getLogger(__name__)
_MODEL_SAVE_PATH = pathlib.Path(__file__).absolute().parent / "agent001.torch_save"


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
    episodes = 35000  # how many games to train across
    max_steps = board.cols * board.rows * 5  # illegal steps can be taken

    # NN-related
    batch_size = 32
    gamma = 0.99  # discount on future rewards (dampening)
    epsilon_cur = 1  # Current chance of exploration
    epsilon_dec = 0.9996  # Decay over episodes
    epsilon_end = 0.010  # Minimum chance of exploration
    _LOG.info(
        "Epsilon decay: %f, will take %d/%d episodes to reach min: %f",
        epsilon_dec,
        int(np.log(epsilon_end) / np.log(epsilon_dec)),  # end = dec^k
        episodes,
        epsilon_end,
    )
    latest_loss = None
    latest_action = None
    t_last = time.time()

    # Saving for experience replay
    Experience = collections.namedtuple(
        "Experience", ["state", "action", "reward", "next_state", "done"]
    )
    memory = collections.deque(maxlen=batch_size * 150)

    # ##### Playing games #####
    _LOG.info("Training %d episodes. Memory is: %d", episodes, memory.maxlen)
    for i_ep in range(episodes):
        board.clear_state()
        # Random starting places ?
        # board.add_coin(player + 1, np.random.randint(0, board.cols))

        for i_step in range(max_steps):
            # Pick player
            if player == 1:
                player = 2
            else:
                player = 1

            # Start stepping
            action = None
            x_ten = torch.Tensor(board.get_flat_state(player))
            # TODO: this should really report '1' for the active player and '-1' for all others

            if np.random.rand() < epsilon_cur:  # Explore \o/
                action = np.random.randint(0, board.cols)
            else:  # ##### Present the state to the agent and get an action #####
                q_vals = agent.forward(x_ten)
                action = torch.argmax(q_vals)
                # _LOG.debug("Ep: %4d, step: %4d, q: %s -> %d", i_ep, i_step, q_vals, action)
            latest_action = action

            # ##### Take a step, save the experience (including reward) #####
            result = board.add_coin(player, action)

            # How did that go?
            n = 0
            if result is not None:  # Can do an illegal move here
                n = board.get_span(player, result[0], result[1])
            done = n >= 4

            # Set the reward
            reward = -1
            if done:  # won the game?
                reward = 100
            elif result is None:
                reward = -100  # please don't do illegal moves
            # elif n > 1:  # placing coins next to each other?
            #     reward += n * 10
            # ... try very sparse

            # Save it all
            exp = Experience(
                state=x_ten,
                action=action,
                reward=reward,
                next_state=torch.Tensor(board.get_flat_state(player)),
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

                latest_loss = loss
                # _LOG.debug("Loss: %s, action: %d, reward: %f", loss, action, reward)

            # ##### Check for done #####
            if done:
                break

        if epsilon_cur > epsilon_end:
            epsilon_cur *= epsilon_dec

        if (i_ep + 1) % 100 == 0:
            t_elap = time.time() - t_last
            t_ep = t_elap / 100
            t_left = (episodes - i_ep) * t_ep / 60
            _LOG.info(
                "Episode: %d/%d (%d steps), epsilon: %1.3f, loss: %s,\n\tt_ep: %1.6fs, t_left: %2.1fm, win_action: %d,\nresult:\n",
                i_ep,
                episodes,
                i_step + 1,
                epsilon_cur,
                latest_loss,
                t_ep,
                t_left,
                latest_action,
            )
            board.print_state()
            t_last = time.time()

    _LOG.info("Did %d episodes, saving model to: %s", episodes, _MODEL_SAVE_PATH)
    torch.save(agent, _MODEL_SAVE_PATH)


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
    board = FiarBoard()
    agent = torch.load(_MODEL_SAVE_PATH)

    board.clear_state()
    player_agent = 1
    player_user = 2

    _LOG.info("Player actions: s: skip, q: quit, <0, .. n> add coin to column n")
    _LOG.info("Agent is player %d", player_agent)
    _LOG.info("User is player %d", player_user)

    for i in range(1000):
        _LOG.info("Move %4d", i)

        # ########## Agent move ##########
        _LOG.info("Agent moves")
        x_ten = torch.Tensor(board.get_flat_state(player_agent))
        q_vals = agent.forward(x_ten)
        action = torch.argmax(q_vals)
        _LOG.debug("\tQ-vals: %s", q_vals)
        _LOG.debug("\tAction: %d", action)
        result = board.add_coin(player_agent, action)
        board.print_state()
        if result is None:
            _LOG.warning("\tIllegal move from agent")
        else:
            n_conn = board.get_span(player_agent, result[0], result[1])
            _LOG.debug("\tAgent connected %d coins", n_conn)
            if n_conn >= 4:
                _LOG.debug("\tAgent wins \o/ !!!")
                break

        # ########## Player action ##########
        stri = input("Enter action:")
        if stri == "s":
            _LOG.debug("\tSkipping action")
        elif stri == "q":
            _LOG.debug("\tQuitting")
            break
        else:
            try:
                action = int(stri)
            except ValueError:
                _LOG.debug("Whut number?")
                continue
            result = board.add_coin(player_user, action)
            board.print_state()
            if result is None:
                _LOG.warning("\tIllegal move from player")
            else:
                n_conn = board.get_span(player_user, result[0], result[1])
                _LOG.debug("\tUser connected %d coins", n_conn)
                if n_conn >= 4:
                    _LOG.debug("\tUser wins (blast!!) !!!")
                    break
    _LOG.info("End of game after %d turns", i + 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_stack()
        traceback.print_exception(exc)
        pdb.post_mortem()
        pdb.post_mortem()
