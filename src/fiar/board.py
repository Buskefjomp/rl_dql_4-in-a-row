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
        assert isinstance(player, int)
        assert isinstance(column, int)
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

    def get_state(self):
        """Return the current state."""
        return self._state

    def clear_state(self):
        """Cleanup and start over."""
        self._state = np.zeros((self.cols, self.rows), dtype=int)

    def print_state(self):
        """Print it all nicely."""
        for i_row in range(self.rows - 1, 0, -1):
            print("\t", end='')
            for i_col in range(0, self.cols):
                print(f" {self._state[i_col, i_row]:2d}", end="")
            print("")