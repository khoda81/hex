from dataclasses import dataclass
from enum import IntEnum
from string import ascii_lowercase

import numpy as np
from numpy.typing import NDArray

NEIGHBORS = np.array(
    [
        [-1,  0],
        [-1,  1],
        [ 0, -1],
        [ 0,  1],
        [ 1, -1],
        [ 1,  0],
    ]
)


class Cell(IntEnum):
    EMPTY = 0
    BLUE = 1
    RED = 2
    SELECT = 3

    def __neg__(self):
        return (Cell.BLUE if self == Cell.RED else
                Cell.RED if self == Cell.BLUE else Cell.EMPTY)

    def __repr__(self):
        return ".BRS"[self]

    def __str__(self):
        colors = {
            Cell.EMPTY: "\033[0m",
            Cell.BLUE: "\033[94m",
            Cell.RED: "\033[91m",
            Cell.SELECT: "\033[93m",
        }

        return colors[self] + repr(self) + "\033[0m"


@dataclass
class HexState:
    board: NDArray
    is_red: bool

    @classmethod
    def initial_state(cls, n=7, is_red=False):
        return cls(
            board=np.full((n, n), Cell.EMPTY, dtype=np.int8),
            is_red=is_red,
        )

    @property
    def player_color(self) -> Cell:
        return Cell.RED if self.is_red else Cell.BLUE

    def is_losing(self):
        # transpose the board if it's blue's turn
        # so that we can use the same logic for both players
        board = self.board_from_player()

        h, w = board.shape
        # forbidden is a boolean array that is True if the cell is not empty
        # and is not the current player's color
        forbidden = board != -self.player_color

        def dfs(start):
            r, c = start
            if r not in range(h) or c not in range(w) or forbidden[r, c]:
                return False

            if c == w - 1:
                return True

            forbidden[r, c] = True

            return any(map(dfs, NEIGHBORS + start))

        return any(dfs(np.array([r, 0])) for r in range(h))

    def board_from_player(self):
        return self.board if self.is_red else self.board.T

    def place_peg(self, position):
        if position is None:
            return self

        position = tuple(position)
        if self.board[position] == Cell.EMPTY:
            board = self.board.copy()
            board[position] = self.player_color
            return HexState(board, not self.is_red)

        return self

    def __iter__(self):
        rows, cols = np.argwhere(self.board == Cell.EMPTY).T.tolist()
        boards = np.repeat(self.board[np.newaxis, ...], len(rows), axis=0)
        boards[np.arange(len(rows)), rows, cols] = self.player_color

        for action, board in zip(zip(rows, cols), boards):
            yield action, HexState(board, not self.is_red)

    def __str__(self):
        _, w = self.board.shape
        header = f"{self.player_color} " + " ".join(ascii_lowercase[:w])
        rows = [
            f"{'':>{r}} {r} {' '.join(map(lambda i: str(Cell(i)), row))}"
            for r, row in enumerate(self.board)
        ]

        return header + "\n" + "\n".join(rows)
