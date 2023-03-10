import heapq
import random
from functools import cache
from math import inf

from hex_game import NEIGHBORS, Cell, HexState

__all__ = ["Negamax", "dijkstra", "generate_neighbors", "negamax"]


@cache
def generate_neighbors(row, col, height, width):
    """Generate all inbound neighbors for a given cell."""
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
    # convert to ints for faster comparisons
    empty = int(Cell.EMPTY)
    my_color = int(state.player_color)

    # get board from player perspective so we can use row 0 as the source
    h, w = state.board.shape
    board = state.board_from_player().tolist()

    min_dist = [[inf] * w for _ in range(h)]
    heap = []

    # initialize heap with all cells in first row
    first_row = board[0]
    first_dist = min_dist[0]
    for nc, cell in enumerate(first_row):
        if cell == empty:
            new_dist = 1  # every empty cell costs 1 to reach
        elif cell == my_color:
            new_dist = 0  # our color is free to move to
        else:
            continue  # other color is not reachable

        first_dist[nc] = new_dist
        heapq.heappush(heap, (new_dist, (0, nc)))

    while heap:
        dist, (r, c) = heapq.heappop(heap)

        neighbors = generate_neighbors(r, c, w, h)
        for nr, nc in neighbors:
            cell = board[nr][nc]
            if cell == empty:
                new_dist = dist + 1
            elif cell == my_color:
                new_dist = dist
            else:
                continue

            if min_dist[nr][nc] > new_dist:
                min_dist[nr][nc] = new_dist
                if nr != h - 1:
                    heapq.heappush(heap, (new_dist, (nr, nc)))

    return min(min_dist[-1])


def negamax(
    state: HexState,
    alpha: float,
    beta: float,
    depth: int = 0,
) -> tuple[float, list[tuple[int, int]]]:
    if depth <= 0:
        my_dist = dijkstra(state)
        other_dist = dijkstra(HexState(state.board, not state.is_red))
        value = other_dist - my_dist
        return -value, None

    # generate all action-state pairs
    next_action_states = list(state)
    # TODO: sort actions by pegs placed close to other pegs
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
