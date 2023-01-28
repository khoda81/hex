import heapq
import os
import random
from dataclasses import dataclass
from enum import IntEnum
from functools import cache
from math import inf
from string import ascii_letters, ascii_lowercase

import numpy as np
from numpy.typing import NDArray


NEIGHBORS = np.array(
    [
        [-1, 0],
        [-1, 1],
        [0, 1],
        [1, 0],
        [1, -1],
        [0, -1],
    ]
)


class Cell(IntEnum):
    EMPTY = 0
    BLUE = 1
    RED = 2
    SELECT = 3

    def __neg__(self):
        return (
            Cell.BLUE
            if self == Cell.RED
            else Cell.RED
            if self == Cell.BLUE
            else Cell.EMPTY
        )

    def __str__(self):
        colors = {
            Cell.EMPTY: "\033[0m",
            Cell.BLUE: "\033[94m",
            Cell.RED: "\033[91m",
            Cell.SELECT: "\033[93m",
        }

        return colors[self] + repr(self) + "\033[0m"

    def __repr__(self):
        return ".BRS"[self]


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

    def __str__(self):
        _, w = self.board.shape
        header = f"{self.player_color} " + " ".join(ascii_lowercase[:w])
        rows = [
            f"{'':>{r}} {r} {' '.join(map(lambda i: str(Cell(i)), row))}"
            for r, row in enumerate(self.board)
        ]

        return header + "\n" + "\n".join(rows)

    def __iter__(self):
        rows, cols = np.argwhere(self.board == Cell.EMPTY).T.tolist()
        boards = np.repeat(self.board[np.newaxis, ...], len(rows), axis=0)
        boards[np.arange(len(rows)), rows, cols] = self.player_color

        for action, board in zip(zip(rows, cols), boards):
            yield action, HexState(board, not self.is_red)


@cache
def generate_neighbors(row, col, height, width):
    neighbors = []
    for dr, dc in NEIGHBORS:
        nr = row + dr
        if nr not in range(width):
            continue

        nc = col + dc
        if nc not in range(height):
            continue

        neighbors.append((nr, nc))

    return neighbors


def dijkstra(state: HexState):
    board = state.board_from_player()
    h, w = board.shape

    min_dist = [[inf] * w for _ in range(h)]
    board_list = board.tolist()

    heap = []

    nr = 0
    dist = 0
    for nc in range(w):
        if board_list[nr][nc] == Cell.EMPTY:
            new_dist = dist + 1
        elif board_list[nr][nc] == state.player_color:
            new_dist = dist
        else:
            continue

        if min_dist[nr][nc] > new_dist:
            min_dist[nr][nc] = new_dist
            heapq.heappush(heap, (new_dist, (nr, nc)))

    while heap:
        dist, (r, c) = heapq.heappop(heap)

        neighbors = generate_neighbors(r, c, w, h)
        for nr, nc in neighbors:
            if board_list[nr][nc] == Cell.EMPTY:
                new_dist = dist + 1
            elif board_list[nr][nc] == state.player_color:
                new_dist = dist
            else:
                continue

            if min_dist[nr][nc] > new_dist:
                min_dist[nr][nc] = new_dist
                if nr == h - 1:
                    return new_dist

                heapq.heappush(heap, (new_dist, (nr, nc)))

    return min(min_dist[-1])


def evaluate(state: HexState):
    my_dist = dijkstra(state)
    other_dist = dijkstra(HexState(state.board, not state.is_red))
    value = other_dist - my_dist
    return value


def negamax(
    state: HexState,
    alpha: float,
    beta: float,
    depth: int = 0,
) -> tuple[float, list[tuple[int, int]]]:
    if depth <= 0:
        return -evaluate(state), None

    next_action_states = list(state)
    # todo: sort actions by pegs placed close to other pegs
    random.shuffle(next_action_states)

    best_action = None
    value = -inf
    for action, state in next_action_states:
        new_value, _ = negamax(state, -beta, -alpha, depth - 1)
        if best_action is None or new_value > value:
            value = new_value
            best_action = action

        alpha = max(alpha, value)
        if alpha >= beta:
            break

    return -value, best_action


class Negamax:
    def __init__(self, depth=3):
        self.depth = depth

    def act(self, obs: HexState):
        _, action = negamax(obs, -inf, inf, depth=self.depth)
        return action


class Random:
    def act(self, state: HexState):
        action, _ = random.choice(list(state))
        return action


class Human:
    def __init__(self, helper=Random()):
        self.helper = helper

    def act(self, state: HexState):
        row, column = self.helper.act(state)
        user_input = input(f"[{ascii_letters[column]}{row}]> ").strip()

        if user_input:
            row, column = sorted(user_input)
            column = ascii_letters.index(column)
            row = int(row)

        return row, column


def main():
    players = {
        Cell.BLUE: Negamax(depth=4),
        Cell.RED: Negamax(depth=4),
    }

    game_state = HexState.initial_state(is_red=True)

    os.system("cls")
    print(game_state)
    print()
    while not game_state.is_losing():
        action = players[game_state.player_color].act(game_state)
        game_state = game_state.place_peg(action)

        os.system("cls")
        print(game_state)
        print()

    print("WINNER:", -game_state.player_color)
    print("LOSER: ", game_state.player_color)


if __name__ == "__main__":
    main()
