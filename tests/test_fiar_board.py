# import pytest

from fiar.board import FiarBoard


def test_add_count():
    """Check that we can add coins."""
    dut = FiarBoard()

    # Illegal additions
    assert dut.add_coin(0, 0) is None, "Not illegal player?"
    assert dut.add_coin(1, -1) is None, "Not too early column?"
    assert dut.add_coin(1, dut.cols) is None, "Not beyond columns?"

    # Fill up a column and test underway
    player = 1
    column = 2
    for i_row in range(dut.rows):
        r = dut.add_coin(player, column)
        assert r == (column, i_row)
    assert dut.add_coin(player, column) is None, "Row is not full?"


def test_get_span():
    """Validate getting spans."""
    dut = FiarBoard()

    s = dut._state  # let's mangle this

    # Test the horizontal
    player1 = 1
    s[0, 0] = player1
    assert dut.get_span(player1, 0, 0) == 1
    s[1, 0] = player1
    assert dut.get_span(player1, 0, 0) == 2
    s[2, 0] = player1
    assert dut.get_span(player1, 0, 0) == 3
    s[3, 0] = player1
    assert dut.get_span(player1, 0, 0) == 4
    assert dut.get_span(player1, 2, 0) == 4

    # Test vertical
    dut.clear_state()
    s = dut._state
    player = 2
    for i_row in range(0, 4):
        s[2, i_row + 1] = player
        assert dut.get_span(player, 2, i_row + 1) == i_row + 1
    assert dut.get_span(player, 2, 3) == 4
    assert dut.get_span(player + 1, 2, 3) == 0

    # Test first diagonal, from upper left and down-right
    dut.clear_state()
    s = dut._state
    player = 2
    for i_step in range(0, 4):
        s[1 + i_step, dut.rows - 1 - i_step] = player
        assert dut.get_span(player, 1, dut.rows - 1) == i_step + 1
    assert dut.get_span(player, 3, dut.rows - 3) == 4

    # Test second diagonal, from lower left and up-right
    dut.clear_state()
    s = dut._state
    player = 4
    for i_step in range(0, 4):
        s[1 + i_step, i_step] = player
        # dut.print_state()
        assert dut.get_span(player, 1, 0) == i_step + 1
    assert dut.get_span(player, 3, 2) == 4
    # dut.print_state()


def test_get_unowned_span():
    """Does it work if we get the unowned span?"""
    dut = FiarBoard()

    s = dut._state  # let's mangle this
    p1 = 1
    p2 = 2

    # Test the horizontal
    s[0, 0] = p1
    assert dut.get_span(p1, 0, 0) == 1
    assert dut.get_span(p1, 1, 0, allow_not_owned=True) == 1, "Block 1 connected?"
    assert dut.get_span(p1, 2, 0, allow_not_owned=True) == 0, "Not blocking anything"
    s[1, 0] = p1
    assert dut.get_span(p1, 0, 0) == 2
    assert dut.get_span(p1, 2, 0, allow_not_owned=True) == 2


if __name__ == "__main__":
    test_add_count()
    test_get_span()
    test_get_unowned_span()
