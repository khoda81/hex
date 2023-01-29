import os
import random
from string import ascii_letters

from hex_game import Cell, HexState
from hex_negamax import Negamax


class Random:
    def act(self, state: HexState):
        action, _ = random.choice(list(state))
        return action


class Human:
    def __init__(self, helper=Random()):
        self.helper = helper

    def act(self, state: HexState):
        row, column = self.helper.act(state)
        user_input = input(f"[{ascii_letters[column]}{row}]> ")

        if user_input:
            try:
                user_input = ''.join(sorted(user_input)).strip()
                row, column = user_input
                column = ascii_letters.index(column)
                row = int(row)
            except ValueError:
                return None

        return row, column


def main():
    players = {
        Cell.RED:  Negamax(depth=4),
        Cell.BLUE: Human(helper=Negamax(depth=3)),
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
