import math
import time
import numpy as np
from enum import Enum
import directions
from identifier import Identifier
from board import Board, STONE_NAMES


N_CONNECT = 4
MAX_DEPTH = 4


class PlayerTypes(Enum):
    PC = 0
    NPC = 1


class Results(Enum):
    PLAYER1_WIN = "Player 1 won."
    PLAYER2_WIN = "Player 2 won."
    DRAW = "Draw."


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
    return (player_score - enemy_score) / (player_score + enemy_score)


def search_tree(root_board: Board, player: int) -> tuple[float, int]:
    """Evaluates the `board` for `player`, when it's `player`s turn. Returns evaluation and best move."""

    # we start with depth = 0
    depth = 0
    # stores node ids and their depth in the search tree, and the column that was changed
    # root did not change any position
    node_list: list[tuple[Board, int, int]] = [(root_board, depth, -1)]
    evaluated: set[Board] = set()
    evaluation: dict[Board, float] = {}
    parent: dict[Board, Board] = {}
    while root_board not in evaluated:
        # save previous depth so we know when we go back up
        previous_depth = depth
        # pop top node, depth, changed position
        board, depth, changed_column = node_list[-1]
        # if we just went back up in the search tree, then x has been fully evaluated, check the loop condition again maybe
        if previous_depth > depth:
            evaluated.add(board)
            continue
        # maximizer (even depth): it's my players turn
        # minimizer (odd depth): it's the enemy players turn
        on_turn = player if depth % 2 == 0 else -player
        # if node has an evaluation
        if board in evaluated:
            p = parent[board]
            # x in max node -> parent in min node
            # remove node from list and get a new one
            if (player == on_turn and evaluation[board] < evaluation[p]) or (player != on_turn and evaluation[board] > evaluation[p]):
                evaluation[p] = evaluation[board]
                if p == root_board:
                    best_move = changed_column
            del node_list[-1]
            # also we can forget about the board configuration
            del board
            continue

        # x was not evaluated yet:
        # if x is a leaf, evaluate it
        # root didn't change any column, it can't be a leaf.
        if changed_column != -1:
            x, y = changed_column, board.top[changed_column] - 1
            stone_position = np.array([y, x])
            if winning_on(board, player, stone_position):
                evaluation[board] = 1
                evaluated.add(board)
                continue
            elif winning_on(board, -player, stone_position):
                evaluation[board] = -1
                evaluated.add(board)
                continue

        # if we don't want to expand the node, evaluate it using the evaluation function
        if depth >= MAX_DEPTH:
            evaluation[board] = evaluate(board, player, on_turn)
            evaluated.add(board)
            continue

        # we want to expand the node
        # maximizer: init with -infty
        # minimizer: init with infty
        evaluation[board] = -math.inf if player == on_turn else math.inf
        # append children to node list
        for column in range(root_board.width):
            # skip full columns
            if board.full(column):
                continue
            child_board = board.copy()
            # add a stone by on_turn in the column
            child_board.put(on_turn, column)
            # assign an id to the child board
            parent[child_board] = board
            # add child to node list
            node_list.append((child_board, depth + 1, column))
    return evaluation[root_board], best_move


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
        evaluation, best_move = search_tree(self.board, player)
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
