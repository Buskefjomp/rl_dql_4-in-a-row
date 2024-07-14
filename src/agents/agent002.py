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
- (done) Do it as Monte-Carlo, to reward winning the game (I think). See https://huggingface.co/learn/deep-rl-course/unit2/mc-vs-td
- (done) Formulate as a state-value problem (value of actions between states)
- (done) Do some math on the potential state-size... need a bigger NN?
- (done) Work with the reward-system to learn to win the game
   - (done) Be quick about it
   - (done) Put pieces next to each other
   - (done) Block the other player!
- Train on the mirror of the state also - data augmentation
- From [3]:
    - (done) Use experience replay (still) - avoid catastrophic forgetting
    - (later) Use convolutional network to learn spatial information -> agent003
    - (maybe? doing MC and fixed across batch...) Fixed Q-target to stabilize training (update after every C steps), don't chase a moving target
    - Double DQN? Maybe not... we are using Monte-Carlo (still) to generate Q-values of states
- (done) Have one player with a different set of epsilon-greedy parameters - to keep exploring bad moves also. Does the other one win more?

- Use a hyper-parameter-tuner to optimize stuff

Postponed / later:
    - (done) Look at the loss-curve to see how we are doing
    - (done) Investigate Huber-loss - doing better than MSE?
    - Learning-rate and other hyper-parameters
        o https://developers.google.com/machine-learning/testing-debugging/metrics/interpretic
        -> Too high learning rate with oscillating curve...
    - After the above (to exhaust performance) -> Use a convolutional network to better get spatial information (3x3 cells?)

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
from torch.utils.tensorboard import SummaryWriter

_LOG = logging.getLogger(__name__)
_MODEL_SAVE_PATH = pathlib.Path(__file__).absolute().parent / "agent002.torch_save"


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
        train_agent_002()
    if args.play:
        play_agent_002()


# ######################### Reusables #########################


# ######################### Training #########################
def train_agent_002():
    """(Re)-Train the agent."""

    # TensorBoard - see https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
    writer = SummaryWriter(log_dir="_agent002-logs")

    board = FiarBoard()

    agent = Agent002(board.cols * board.rows, board.cols).to("cpu")
    _LOG.info("Training agent:\n%s", agent)
    loss_fn = torch.nn.HuberLoss(delta=2)  # Alternative to MSELoss()
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4, weight_decay=1e-5)

    episodes = 100000  # how many games to train across
    max_steps = board.cols * board.rows * 1 + 10  # few illegal steps can be taken

    # NN-related
    batch_size = 32  # We want to train on a few games every time we add ~one (and forget another), # TODO: Statistics on number of moves in a game?
    n_batches = 1
    memory = 25000  # How many moves to store
    gamma = 1.0  # Discount across steps, can be 1.0 for full reward
    epsilon_cur = [1, 1]  # Starting/Current chance of exploration, for each player
    epsilon_dec = [0.99995, 0.9993]  # Decay over episodes
    epsilon_end = [
        0.010,
        0.60,
    ]  # Minimum chance of exploration, worse for the latter player to train making bad moves
    _LOG.info(
        "Epsilon decay: %f, will take %d/%d episodes to reach min: %f (for player 0). Epsilon-min for other: %f",
        epsilon_dec[0],
        int(np.log(epsilon_end[0]) / np.log(epsilon_dec[0])),  # end = dec^k
        episodes,
        epsilon_end[0],
        epsilon_end[1],
    )
    t_last = time.time()

    # Saving for experience replay - as we are doing Monte-Carlo this is much more limited
    Experience = collections.namedtuple("Experience", ["state", "action", "reward"])
    memory = collections.deque(maxlen=memory)

    # Saving data through an episode, to be expanded later with reward once episode is done
    OneStep = collections.namedtuple("OneStep", ["state", "action", "reward"])

    # ########## Playing games ##########
    _LOG.info(
        "Doing %d episodes against itself. Memory is: %d moves.",
        episodes,
        memory.maxlen,
    )

    i_player = 0  # players number

    # Runtime reporting
    t_ep = None
    t_last = time.time()
    n_for_report = 100  # How many episodes before something is reported
    latest_loss = None
    who_wins = np.zeros((3,), dtype=int)
    t_start = time.time()

    for i_ep in range(episodes):
        # ##### Play a game #####
        # Is now Monte-Carlo like - play an entire game to its end.
        board.clear_state()

        player_steps = [[], []]  # steps of OneStep for players
        win_player = None
        for i_step in range(max_steps):
            i_player = (i_player + 1) % 2
            act_player = i_player + 1  # board cannot handle player = 0
            not_player = ((i_player + 1) % 2) + 1

            # Current state
            x_ten = torch.Tensor(board.get_flat_state(act_player))

            # Explore or exploit?
            action = None
            if np.random.rand() < epsilon_cur[i_player]:  # Explore \o/
                action = np.random.randint(0, board.cols)
            else:  # ##### Present the state to the agent and get an action #####
                q_vals = agent.forward(x_ten)
                action = int(torch.argmax(q_vals))

            # Take the action
            result = board.add_coin(act_player, action)

            # ##### Calculate reward #####
            n_lined = 0  # how many connected tiles?
            if result is not None:  # Can do an illegal move here
                n_lined = board.get_span(act_player, result[0], result[1])
            done = n_lined >= 4

            reward = 0
            if done:
                reward = 50
            elif result is None:
                reward -= 50  # Do not do illegal moves
            else:
                reward += -0.1  # Making a move is costly, make sure to win fast
                if n_lined >= 2:
                    reward += 0.2 * n_lined  # Placing adjacent is good after all
                n_block = board.get_span(
                    not_player, result[0], result[1], allow_not_owned=True
                )
                if n_block >= 2:
                    reward += 0.1 * (n_block**2)  # Blocking other player is good also

            # Save the move
            player_steps[i_player].append(OneStep(x_ten, action, reward))

            if done:
                win_player = i_player
                break
        if win_player is None:
            who_wins[2] += 1
        else:
            who_wins[win_player] += 1

        # Maybe show what is happening
        if (i_ep + 1) % n_for_report == 0:
            the_win = win_player
            if the_win is not None:
                the_win += 1  # map to board-state
            _LOG.info(
                "Played episode %d, num moves: %d, win_player: %s, end-state:",
                i_ep,
                i_step,
                the_win,
            )
            board.print_state()
            # _LOG.info(
            #     "\tPlayer 0 moves: %s",
            #     " | ".join(
            #         [
            #             f"({step.action} -> { step.reward:2.1f}"
            #             for step in player_steps[0]
            #         ]
            #     ),
            # )
            # _LOG.info(
            #     "\tPlayer 1 moves: %s",
            #     " | ".join(
            #         [
            #             f"({step.action} -> { step.reward:2.1f})"
            #             for step in player_steps[1]
            #         ]
            #     ),
            # )
            _LOG.info("Win-rates: %s (0, 1, draw)", who_wins / who_wins.sum())

        # ##### Post-process the game #####
        reward_loss = -10  # Clearly your last move was a bad move
        g_t = 0  # Back-summed reward
        # Both are loosers?
        if win_player is None:
            win_player = 0
            g_t = reward_loss  # inject a loss penalty
            # not writable: player_steps[win_player][-1].reward = reward_loss
        loss_player = (win_player + 1) % 2
        # Back-calculate the reward for the winning player, see [0]
        for i_step in range(len(player_steps[win_player]) - 1, -1, -1):
            the_step = player_steps[win_player][i_step]
            g_t = g_t + gamma * the_step.reward

            # Save this as an experience
            exp = Experience(state=the_step.state, action=the_step.action, reward=g_t)
            memory.append(exp)

        # Back-calculate the reward for the loosing player
        g_t = reward_loss
        for i_step in range(len(player_steps[loss_player]) - 1, -1, -1):
            the_step = player_steps[loss_player][i_step]
            g_t = g_t + gamma * the_step.reward

            # Also save this as an experience - we learn from losing
            exp = Experience(state=the_step.state, action=the_step.action, reward=g_t)
            memory.append(exp)

        # TODO: Train the mirrored state to enforce symmetry

        # ##### Train model #####
        if len(memory) > batch_size * 5:  # Have some entropy to pick from
            for i_batch in range(n_batches):
                batch = random.sample(memory, batch_size)
                states = torch.stack([exp.state for exp in batch])
                actions = [exp.action for exp in batch]
                rewards = [exp.reward for exp in batch]

                agent.train(True)
                cur_q_vals = agent(states)
                trg_q_vals = cur_q_vals.clone().detach()
                for i in range(batch_size):
                    trg_q_vals[i, actions[i]] = rewards[i]

                loss = loss_fn(cur_q_vals, trg_q_vals)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                agent.eval()

                latest_loss = loss

            writer.add_scalar("Loss/train", latest_loss, i_ep)

        if ((i_ep + 1) % n_for_report == 0) and (latest_loss is not None):
            t_elap = time.time() - t_last
            if t_ep is None:
                t_ep = t_elap / n_for_report
            else:
                t_ep = t_ep * 0.8 + 0.2 * t_elap / n_for_report
            t_left = (episodes - i_ep) * t_ep / 60
            _LOG.info(
                "Episode: %d/%d (%d steps), epsilon: %1.3f, %1.3f, loss: %s,\n\tt_ep: %1.6fs, t_left: %2.1fm",
                i_ep,
                episodes,
                i_step + 1,
                epsilon_cur[0],
                epsilon_cur[1],
                latest_loss,
                t_ep,
                t_left,
            )
            t_last = time.time()

        epsilon_cur[0] = max(epsilon_end[0], epsilon_cur[0] * epsilon_dec[0])
        epsilon_cur[1] = max(epsilon_end[1], epsilon_cur[1] * epsilon_dec[1])

    t_end = time.time()

    writer.close()
    _LOG.info(
        "Did %d episodes in %2.1fm, saving model to: %s",
        episodes,
        (t_end - t_start) / 60,
        _MODEL_SAVE_PATH,
    )
    torch.save(agent, _MODEL_SAVE_PATH)


class Agent002(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.n_inputs = num_inputs
        self.n_outputs = num_outputs

        # Size of input-state: 6 rows x 7 columns = 42 spots
        # Each spot can have 3 states (None, opponent (-1), owned (+1))
        # If 2 spots, this yields, 00, 01, 10, 11, 0n, n0, nn, n1, 1n -> 2^3
        # If this is correct and we extrapolat, we have 42**3 possible combinations
        # of state -> 74088 (not sure this is completely right though, combinatorial logic is hard)
        # In that case, we might need a bigger neural network... for now lets stay with a dense network
        # and save the CNN-stuff for  agent003.
        n_face = 42
        self.the_stack = torch.nn.Sequential(
            # Hidden layer
            torch.nn.Linear(self.n_inputs, n_face * 2),  # input to internals
            torch.nn.ReLU(),
            # Hidden layer
            torch.nn.Linear(n_face * 2, n_face * 3),
            torch.nn.ReLU(),
            # Hidden layer
            torch.nn.Linear(n_face * 3, 20),
            torch.nn.ReLU(),
            # Output layer (pick an action - drop a coin in a column)
            torch.nn.Linear(20, self.n_outputs),
        )

    def forward(self, board_state):
        q_values = self.the_stack(board_state)
        return q_values


# ######################### Playing #########################
def play_agent_002():
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
                _LOG.debug("\tAgent wins \\o/ !!!")
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
