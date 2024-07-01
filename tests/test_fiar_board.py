# import pytest

from fiar.board import FiarBoard


def test_add_count():
    """Check that we can add coins."""
    dut = FiarBoard()

    # Illegal additions
    assert dut.add_coin(0, 0) is None, "Not illegal player?"
    assert dut.add_coin(1, -1) is None, "Not too early column?"
    assert dut.add_coin(1, dut.cols) is None, "Not beyond columns?"

    player = 1
    column = 2
    for i_row in range(dut.rows):
        r = dut.add_coin(player, column)
        assert r == (column, i_row)
    assert dut.add_coin(player, column) is None, "Row is not full?"


if __name__ == "__main__":
    test_add_count()
