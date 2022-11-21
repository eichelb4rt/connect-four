import time
import numpy as np
from enum import Enum
from board import Board, STONE_NAMES, winning_on
from search import SearchTree

LOG = False


class PlayerTypes(Enum):
    PC = 0
    NPC = 1


class Results(Enum):
    PLAYER1_WIN = "Player 1 won."
    PLAYER2_WIN = "Player 2 won."
    DRAW = "Draw."


class Game:
    def __init__(self, width=7, height=6, player1_type=PlayerTypes.PC, player2_type=PlayerTypes.NPC):
        self.width = width
        self.height = height
        self.board = Board(width, height)
        # playing with 1 stones
        self.player1_type = player1_type
        # playing with -1 stones
        self.player2_type = player2_type

    def play(self, starting=1, print_board=True) -> Results:
        if print_board:
            print(self.board)
        player = starting
        while True:
            # get move
            player_type = self.player1_type if player == 1 else self.player2_type
            column = self.get_pc_move(player) if player_type == PlayerTypes.PC else self.get_npc_move(player)
            # get position where the stone should be put
            x, y = column, self.board.top[column]
            position = np.array([y, x])
            # put it there
            self.board.put(player, column)
            # maybe print the new board
            if print_board:
                print(self.board)
            # check if the player won with that move
            if winning_on(self.board, player, position):
                return Results.PLAYER1_WIN if player == 1 else Results.PLAYER2_WIN
            # if the board is full and no one won, that's a draw
            if self.board.all_full():
                return Results.DRAW
            # next turn: players are just 1, and -1, so we can just flip the sign
            player *= -1

    def get_pc_move(self, player: int) -> int:
        """Gets the move a player wants to play from console input. Returns column index."""

        playername = "player 1" if player == 1 else "player 2"
        print(f"It's {playername}s ({STONE_NAMES[player]}) turn. What's your move?")
        # get input until it's a working input
        while True:
            player_input = input()
            correct, msg = self.correct_input(player_input)
            # return the move if it's correct
            if correct:
                return int(player_input)
            # else, print the message and try again
            print(msg)

    def get_npc_move(self, player: int) -> int:
        """Gets the move the npc wants to play."""

        # search game tree for options and maximize evaluation
        start = time.time()
        tree = SearchTree(self.board, player, log=LOG)
        evaluation, best_move = tree.search()
        end = time.time()
        print(f"Best move evaluation: {evaluation}")
        print(f"Time needed: {end - start}")
        return best_move

    def correct_input(self, player_input: str) -> tuple[bool, str]:
        """Checks if player input is actually a correct input. Also returns error message."""

        # if it can't be converted to int, fail
        try:
            column = int(player_input)
        except ValueError:
            return False, "Please enter an integer."
        # if it's out of bounds, fail
        if column < 0 or column >= self.width:
            return False, f"Please enter an integer within the bounds [0, {self.width - 1}]."
        # if the column is full, fail
        if self.board.full(column):
            return False, f"Column {column} is already full."
        return True, ""


def main():
    game = Game()
    result = game.play()
    print(result.value)


if __name__ == "__main__":
    main()
