from random import random
from string import ascii_letters

from agents.negamax import Negamax
from hex_game import HexState

__all__ = ["Negamax", "Random", "Human"]


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
                user_input = "".join(sorted(user_input)).strip()
                row, column = user_input
                column = ascii_letters.index(column)
                row = int(row)
            except ValueError:
                return None

        return row, column
