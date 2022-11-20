import numpy as np
from enum import Enum
import directions
from board import Board, STONE_NAMES


N_CONNECT = 4


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
            if self.winning_on(player, position):
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

        # this sometimes wants to put stuff in full columns, but it will be replaced soon anywaygh
        return np.random.randint(0, self.width)

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

    def winning_on(self, player: int, position: np.ndarray) -> bool:
        # if there's no stone from player on the position, that player is not winning there
        if self.board[*position] != player:
            return False
        # if there are 3 stones pointing into this position and there's a stone on the position, it's winning
        return self.danger_on(player, position) == N_CONNECT - 1

    def danger_on(self, player: int, position: np.ndarray) -> int:
        """The max amount of stones by `player` pointing into `position` in any direction. Capped by `N_CONNECT - 1` (default: `4 - 1 = 3`)"""

        # directions with a horizontal part have to be checked in both ways
        dangers = [self.danger_in_direction(player, position, d) + self.danger_in_direction(player, position, -d) for d in [directions.NE, directions.E, directions.SE]]
        # north does not need to be checked
        dangers.append(self.danger_in_direction(player, position, directions.S))
        return min(max(dangers), N_CONNECT - 1)

    def danger_in_direction(self, player: int, position: np.ndarray, direction: np.ndarray) -> int:
        """The amount of stones by `player` pointing into `position` in `direction`. Capped by `N_CONNECT - 1` (default: `4 - 1 = 3`)"""

        # copy needed or else the position array is changed in place
        current_pos = position.copy()
        for steps in range(1, N_CONNECT):
            # go 1 step in the direction
            current_pos += direction
            # if the board ends here, the line ended at the last step
            if not self.within_bounds(current_pos):
                return steps - 1
            # if there's no stone from the player on that position, the line ended at the last step
            if self.board[*current_pos] != player:
                return steps - 1
        # if we only found stones from the player, then the number of stones was the number of iterations (N_CONNECT - 1)
        return N_CONNECT - 1

    def within_bounds(self, position: np.ndarray) -> bool:
        y, x = position
        return 0 <= x and x < self.width and 0 <= y and y < self.height


def main():
    game = Game()
    result = game.play()
    print(result.value)


if __name__ == "__main__":
    main()
