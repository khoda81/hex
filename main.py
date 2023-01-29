import os

from agents import *
from hex_game import Cell, HexState


def main():
    players = {
        Cell.RED:  Negamax(depth=3),
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
