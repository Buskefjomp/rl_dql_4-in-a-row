#! /usr/bin/env python3
"""The 4-in-a-row board - this essentially holds all game-state."""

import numpy as np


class FiarBoard:
    """The playing-board."""

    def __init__(self, cols=7, rows=6):
        """Initialize."""
        self.cols, self.rows = cols, rows

        # Game-board, columns from the left (0), rows from bottom (0) and up (rows-1)
        # Value is 0 for no coin, otherwise players coin
        self._state = np.zeros((self.cols, self.rows), dtype=int)

    def add_coin(self, player, column):
        """
        Add a coin to the board.

        Return the indexed column and row as a tuple.
        Return None if this is an illegal action.
        """
        # assert isinstance(player, int)
        # assert isinstance(column, int)
        if player <= 0:
            return None
        if not (0 <= column < self.cols):
            return None

        end_row = -1
        for i_row in range(0, self.rows):
            if self._state[column, i_row] == 0:
                end_row = i_row
                break
        if end_row == -1:
            return None

        the_col = column
        the_row = end_row
        self._state[the_col, the_row] = player

        return (column, end_row)

    def get_span(self, player, the_col, the_row, allow_not_owned=False):
        """Return longest spanning line for <player> at <the_col, the_row> or None if illegal."""
        if (the_col < 0) or (self.cols <= the_col):
            return None
        if (the_row < 0) or (self.rows <= the_row):
            return None
        if (not allow_not_owned) and self._state[the_col, the_row] != player:
            return 0

        # Start the search - from the middle and out (one could close the row between two halves)
        # Do left and right, up and down and the two diagonals
        max_len = 0
        cur_len_init = 1
        if allow_not_owned:
            cur_len_init = 0
            assert self._state[the_col, the_row] != player, "Cannot own this... misuse?"

        for dx, dy in ((+1, 0), (0, +1), (+1, +1), (+1, -1)):
            cur_len = 1  # this is the initial spot
            if allow_not_owned:
                cur_len = cur_len_init

            # print("d x-y", dx, dy, "base", the_col, the_row)
            for dir in (-1, 1):
                for i_step in range(1, 4):
                    i_col = the_col + dx * dir * i_step
                    i_row = the_row + dy * dir * i_step
                    # print("\ti_step", i_step, "->", i_col, i_row)
                    if not (0 <= i_col < self.cols):
                        break
                    if not (0 <= i_row < self.rows):
                        break
                    if self._state[i_col, i_row] == player:
                        cur_len += 1
                    else:
                        break
            max_len = max(max_len, cur_len)
        return max_len

    def get_flat_state(self, player):
        """Return the flattened state."""
        ret_state = np.reshape(self._state.copy(), -1)

        for i, v in enumerate(ret_state):
            if v == 0:
                continue
            if v == player:
                ret_state[i] = +1
            else:
                ret_state[i] = -1

        assert len(ret_state.shape) == 1
        return np.reshape(self._state, -1)

    def clear_state(self):
        """Cleanup and start over."""
        self._state = np.zeros((self.cols, self.rows), dtype=int)

    def print_state(self):
        """Print it all nicely."""
        print("<state>  ")
        for i_row in range(self.rows - 1, -1, -1):
            print(f"\t{i_row:2d}: ", end="")
            for i_col in range(0, self.cols):
                print(f" {self._state[i_col, i_row]:2d}", end="")
            print("")
        print("\t-------------------------")
        print("\t -  ", end="")
        for i_col in range(0, self.cols):
            print(f" {i_col:2d}", end="")
        print("")
        print("</state>")
