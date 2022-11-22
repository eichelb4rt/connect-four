import numpy as np
from typing import Self
from tabulate import tabulate

import directions


N_CONNECT = 4

STONE_NAMES = {
    0: "",
    1: "x",
    -1: "o"
}

STONE_MAP = np.vectorize(lambda stone: STONE_NAMES[stone])


class Board:
    def __init__(self, width=7, height=6):
        self.width = width
        self.height = height
        # board itself
        self.stones = np.zeros((self.height, self.width), dtype=np.int8)
        # where the next stone can be put in each column
        self.top = np.zeros(self.width, dtype=np.int8)

    def put(self, stone: int, column: int):
        if column < 0 or column >= self.width:
            raise IndexError(f"Column {column} does not exist.")
        if self.full(column):
            raise IndexError(f"Column {column} is already full.")
        # put it at the top of the column
        x, y = column, self.top[column]
        self.stones[y, x] = stone
        # update the top of the column
        self.top[column] += 1

    def full(self, column: int) -> bool:
        return self.top[column] >= self.height

    def all_full(self) -> bool:
        return np.all(self.top >= self.height)

    def within_bounds(self, position: np.ndarray) -> bool:
        y, x = position
        return 0 <= x and x < self.width and 0 <= y and y < self.height

    def __getitem__(self, item) -> int:
        return self.stones[item]

    def __repr__(self) -> str:
        return f"({','.join([''.join(column) for column in STONE_MAP(self.stones.T)])})"

    def __str__(self) -> str:
        horizontal_line = np.array(["="] * self.width)
        indices = range(self.width)
        contents = STONE_MAP(np.flip(self.stones, axis=0))
        return tabulate(np.vstack([horizontal_line, contents, horizontal_line, indices]), tablefmt='plain')

    def copy(self) -> Self:
        copied = Board(self.width, self.height)
        copied.stones = self.stones.copy()
        copied.top = self.top.copy()
        return copied


def winning_on(board: Board, player: int, position: np.ndarray) -> bool:
    # if there's no stone from player on the position, that player is not winning there
    if board[*position] != player:
        return False
    # if there are 3 stones pointing into this position and there's a stone on the position, it's winning
    return max_stones_into(board, player, position) == N_CONNECT - 1


def max_stones_into(board: Board, player: int, position: np.ndarray) -> int:
    """The max amount of stones by `player` pointing into `position` in any direction. Capped by `N_CONNECT - 1` (default: `4 - 1 = 3`)"""

    # directions with a horizontal part have to be checked in both ways
    dangers = [stones_in_direction(board, player, position, d) + stones_in_direction(board, player, position, -d) for d in [directions.NE, directions.E, directions.SE]]
    # north does not need to be checked
    dangers.append(stones_in_direction(board, player, position, directions.S))
    return min(max(dangers), N_CONNECT - 1)


def stones_in_direction(board: Board, player: int, position: np.ndarray, direction: np.ndarray) -> int:
    """The amount of stones by `player` pointing into `position` in `direction`. Capped by `N_CONNECT - 1` (default: `4 - 1 = 3`)"""

    # copy needed or else the position array is changed in place
    current_pos = position.copy()
    for steps in range(1, N_CONNECT):
        # go 1 step in the direction
        current_pos += direction
        # if the board ends here, the line ended at the last step
        if not board.within_bounds(current_pos):
            return steps - 1
        # if there's no stone from the player on that position, the line ended at the last step
        if board[*current_pos] != player:
            return steps - 1
    # if we only found stones from the player, then the number of stones was the number of iterations (N_CONNECT - 1)
    return N_CONNECT - 1


def evaluate(board: Board, player: int, on_turn: int) -> float:
    """Evaluates the `board` for `player`. `on_turn` is the player that's turn it is currently.

    Returns
    -------
    float
        Number in between -1 and 1: 1 indicates `player` winning, -1 indicates `player` losing, 0 indicates draw.
    """

    enemy = -player
    player_score = 0
    enemy_score = 0
    for column in range(board.width):
        # ignore full columns
        if board.full(column):
            continue
        # get top cell and look how many stones are pointing into it
        x, y = column, board.top[column]
        position = np.array([y, x])
        player_stones = max_stones_into(board, player, position)
        enemy_stones = max_stones_into(board, enemy, position)
        # if we got 3 stones in a row and it's our turn, we're winning
        if player_stones == N_CONNECT - 1 and player == on_turn:
            return 1
        # opposite holds for the enemy
        if enemy_stones == N_CONNECT - 1 and enemy == on_turn:
            return -1
        # add score to the total score
        player_score += player_stones
        enemy_score += enemy_stones
    # 1 if player got all the scores, -1 if enemy got all the scores
    if player_score + enemy_score == 0:
        return 0
    return (player_score - enemy_score) / (player_score + enemy_score)


def main():
    board = Board()
    board.put(1, 0)
    board.put(-1, 1)
    board.put(1, 0)
    print(board)


if __name__ == "__main__":
    main()
