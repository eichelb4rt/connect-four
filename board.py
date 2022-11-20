from typing import Self
import numpy as np
from tabulate import tabulate


STONE_NAMES = {
    0: " ",
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

    def __getitem__(self, item) -> int:
        return self.stones[item]

    def __repr__(self) -> str:
        horizontal_line = np.array(["="] * self.width)
        indices = range(self.width)
        contents = STONE_MAP(np.flip(self.stones, axis=0))
        return tabulate(np.vstack([horizontal_line, contents, horizontal_line, indices]), tablefmt='plain')

    def copy(self) -> Self:
        copied = Board(self.width, self.height)
        copied.stones = self.stones.copy()
        copied.top = self.top.copy()
        return copied


def main():
    board = Board()
    board.put(1, 0)
    board.put(-1, 1)
    board.put(1, 0)
    print(board)


if __name__ == "__main__":
    main()
