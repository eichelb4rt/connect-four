import math
import numpy as np

from board import Board, evaluate, winning_on

MAX_DEPTH = 4

IS_MAX_NODE = lambda d: d % 2 == 0
IS_MIN_NODE = lambda d: d % 2 == 1


class SearchTree:
    def __init__(self, root_board: Board, player: int):
        self.root_board = root_board
        self.player = player
        # we start at the root with depth = 0
        self.depth = 0
        self.alpha = -math.inf
        self.beta = math.inf
        # stores node ids and their depth in the search tree, and the column that was changed
        # root did not change any position
        self.worklist_nodes: list[Board] = [root_board]
        self.worklist_depths: list[int] = [0]
        self.worklist_changed: list[int] = [-1]
        self.evaluated: set[Board] = set()
        self.evaluation: dict[Board, float] = {}
        self.parent: dict[Board, Board] = {}
        # depth -> maximum evaluation of min nodes in that depth (only odd depths)
        self.alphas: dict[int, float] = {}
        # depth -> minimum evaluation of max nodes in that depth (only even depths)
        self.betas: dict[int, float] = {0: math.inf}
        # flag to save if something was pruned
        self.just_pruned = False

    def search(self) -> tuple[float, int]:
        while self.root_board not in self.evaluated:
            board, changed_column = self.prepare()
            # if the node was not evaluated yet:
            # - but it's a leaf or we don't want to expand it, evaluate it now
            # - else add children and start over
            if board not in self.evaluated:
                # if x is a leaf, evaluate it and go on
                if self.is_leaf(board, changed_column):
                    continue
                # if we don't want to expand the node, evaluate it using the evaluation function
                if self.depth >= MAX_DEPTH:
                    self.evaluation[board] = evaluate(board, self.player, self.on_turn)
                    self.evaluated.add(board)
                    continue
                # we want to expand
                # maximizer: init with -infty
                # minimizer: init with infty
                self.evaluation[board] = -math.inf if IS_MAX_NODE(self.depth) else math.inf
                self.expand(board)
                continue

            # node has an evaluation
            # remove node from list and get a new one
            # board in max node -> parent in min node
            if IS_MAX_NODE(self.depth):
                self.update_max(board, changed_column)
            # board in min node -> parent in max node
            else:
                self.update_min(board, changed_column)

        return self.evaluation[self.root_board], self.best_move

    def prepare(self) -> tuple[Board, int]:
        # save previous depth so we know when we go back up
        self.previous_depth = self.depth
        # pop top node, depth, changed position
        board = self.worklist_nodes[-1]
        self.depth = self.worklist_depths[-1]
        changed_column = self.worklist_changed[-1]
        # maximizer (even depth): it's my players turn
        # minimizer (odd depth): it's the enemy players turn
        self.on_turn = self.player if self.depth % 2 == 0 else -self.player
        self.combine_alpha_beta()
        # if we just went back up in the search tree, then the node has been fully evaluated, check the loop condition again maybe
        if self.depth < self.previous_depth:
            # if we prune and jump up 2, that means we're done with that list of children
            # if we don't prune, we just jump up 1 and we're also done with that list of children
            if self.children_done():
                self.evaluated.add(board)
            self.just_pruned = False
            self.clean_alpha_beta()
            # if we just changed depth, do it again because we changed some stuff now
            return self.prepare()
        # if we enter a new depth, we'll have to init the new alpha / beta
        elif self.depth > self.previous_depth:
            self.init_alpha_beta()
        return board, changed_column

    def children_done(self) -> bool:
        return not self.just_pruned or (self.just_pruned and self.depth == self.previous_depth - 2)

    def is_leaf(self, board: Board, changed_column: int) -> bool:
        # root didn't change any column, it can't be a leaf.
        if board == self.root_board:
            return False
        # see if board is a leaf. If yes, evaluate it.
        x, y = changed_column, board.top[changed_column] - 1
        stone_position = np.array([y, x])
        if winning_on(board, self.player, stone_position):
            self.evaluation[board] = 1
            self.evaluated.add(board)
            return True
        elif winning_on(board, -self.player, stone_position):
            self.evaluation[board] = -1
            self.evaluated.add(board)
            return True
        return False

    def expand(self, board: Board):
        # generate children and add them to the lists
        for column in range(self.root_board.width):
            # skip full columns
            if board.full(column):
                continue
            child_board = board.copy()
            # add a stone by on_turn in the column
            child_board.put(self.on_turn, column)
            # assign an id to the child board
            self.parent[child_board] = board
            # add child to node list
            self.worklist_nodes.append(child_board)
            self.worklist_depths.append(self.depth + 1)
            self.worklist_changed.append(column)

    def prune(self, board: Board):
        index = self.worklist_nodes.index(board)
        print(f"Pruned: {self.worklist_nodes[index:]}")
        del self.worklist_nodes[index:]
        del self.worklist_depths[index:]
        del self.worklist_changed[index:]
        self.just_pruned = True

    def update_max(self, board: Board, changed_column: int):
        p = self.parent[board]
        # update betas in the depth
        self.betas[self.depth] = min(self.betas[self.depth], self.evaluation[board])
        # if outcome is limited by alpha, cut the whole family
        if self.evaluation[board] >= self.beta:
            print(f"beta: {self.beta}")
            self.prune(p)
            return
        # overwrite parent
        if self.evaluation[board] < self.evaluation[p]:
            self.evaluation[p] = self.evaluation[board]
            # update parent alpha
            self.alphas[self.depth - 1] = max(self.alphas[self.depth - 1], self.evaluation[p])
            if p == self.root_board:
                self.best_move = changed_column
        # remove board from node list
        del self.worklist_nodes[-1]
        del self.worklist_depths[-1]
        del self.worklist_changed[-1]

    def update_min(self, board: Board, changed_column: int):
        p = self.parent[board]
        # update alphas in the depth
        self.alphas[self.depth] = max(self.alphas[self.depth], self.evaluation[board])
        # if outcome is limited by beta, cut the whole family
        if self.evaluation[board] <= self.alpha:
            print(f"alpha: {self.alpha}")
            self.prune(p)
            return
        # overwrite parent
        if self.evaluation[board] > self.evaluation[p]:
            self.evaluation[p] = self.evaluation[board]
            # update parent beta
            self.betas[self.depth - 1] = min(self.betas[self.depth - 1], self.evaluation[p])
            if p == self.root_board:
                self.best_move = changed_column
        # remove board from node list
        del self.worklist_nodes[-1]
        del self.worklist_depths[-1]
        del self.worklist_changed[-1]

    def init_alpha_beta(self):
        if IS_MIN_NODE(self.depth):
            self.alphas[self.depth] = -math.inf
        else:
            self.betas[self.depth] = math.inf

    def combine_alpha_beta(self):
        # if we change the depth in any way, the alphas/betas may have to be updated
        if self.depth != self.previous_depth:
            if IS_MIN_NODE(self.depth):
                self.alpha = self.combine_alphas()
            else:
                self.beta = self.combine_betas()

    def clean_alpha_beta(self):
        # also, we can forget about the alpha/beta 2 depths below
        if IS_MIN_NODE(self.depth) and self.depth + 2 in self.alphas:
            del self.alphas[self.depth + 2]
        elif IS_MAX_NODE(self.depth) and self.depth + 2 in self.betas:
            del self.betas[self.depth + 2]

    def combine_alphas(self) -> float:
        # only odd depths
        alpha = -math.inf
        for depth in range(1, self.depth, 2):
            alpha = max(alpha, self.alphas[depth])
        return alpha

    def combine_betas(self) -> float:
        # only even depths
        beta = math.inf
        for depth in range(0, self.depth, 2):
            beta = min(beta, self.betas[depth])
        return beta
