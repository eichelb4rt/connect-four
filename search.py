import math
import numpy as np

from board import Board, evaluate, winning_on

MAX_DEPTH = 5

IS_MAX_NODE = lambda d: d % 2 == 0
IS_MIN_NODE = lambda d: d % 2 == 1
# maximizer: init with -infty
# minimizer: init with infty
INIT_NODE = lambda d: -math.inf if IS_MAX_NODE(d) else math.inf


class SearchTree:
    def __init__(self, root_board: Board, player: int, log=False):
        self.log = log
        self.root_board = root_board
        self.player = player
        # we start at the root with depth = 0
        self.depth = 0
        # max over min nodes
        self.alpha = -math.inf
        # min over max nodes
        self.beta = math.inf
        # stores node ids and their depth in the search tree, and the column that was changed
        # root did not change any position
        self.worklist_nodes: list[Board] = [root_board]
        self.worklist_depths: list[int] = [0]
        self.worklist_changed: list[int] = [-1]
        self.evaluated: set[Board] = set()
        self.evaluation: dict[Board, float] = {}
        self.parent: dict[Board, Board] = {}
        # depth -> current siblings on that depth
        self.siblings: dict[int, list[Board]] = {0: [root_board]}
        # flag to save if something was pruned
        self.just_pruned_parents = False

    def search(self) -> tuple[float, int]:
        """Evaluates the position. Returns evaluation and best move."""

        # get the next board as long as the root board isn't evaluated
        while (board := self.get_next()) and self.root_board not in self.evaluated:
            # if the node was not evaluated yet:
            # - but it's a leaf or we don't want to expand it, evaluate it now
            # - else add children and start over
            if board not in self.evaluated:
                # if x is a leaf, evaluate it and go on
                if self.is_leaf(board):
                    continue
                # if we don't want to expand the node, evaluate it using the evaluation function
                if self.depth >= MAX_DEPTH:
                    self.evaluation[board] = evaluate(board, self.player, self.on_turn)
                    self.evaluated.add(board)
                    continue
                # we want to expand
                self.evaluation[board] = INIT_NODE(self.depth)
                self.expand(board)
                continue

            # node has an evaluation
            # maybe prune
            self.just_pruned_parents = False
            if self.prunable(board):
                if repr(board) == "(ooox,,,x,xx,,)" or repr(self.parent[board]) == "(ooox,,,x,xx,,)":
                    self.collect_alpha_beta(board)
                self.prune(self.parent[board])
                self.just_pruned_parents = True
                continue
            # update parent
            parent = self.parent[board]
            overwritten = self.update_parent(board, parent)
            # if parent was overwritten, update alpha/beta
            if overwritten:
                # if root was overwritten, consider it as the best move
                if parent == self.root_board:
                    best_move = self.changed_column
            # remove node from list and get a new one
            self.prune_last()

        return self.evaluation[self.root_board], best_move

    def get_next(self) -> Board:
        """Gets next node in list and prepares the loop."""

        # save previous depth so we know when we go back up
        self.previous_depth = self.depth
        # pop top node, depth, changed position
        board = self.worklist_nodes[-1]
        self.depth = self.worklist_depths[-1]
        self.changed_column = self.worklist_changed[-1]
        # maximizer (even depth): it's my players turn
        # minimizer (odd depth): it's the enemy players turn
        self.on_turn = self.player if self.depth % 2 == 0 else -self.player
        # if we change the depth in any way, the alphas/betas may have to be updated
        if self.depth != self.previous_depth:
            self.collect_alpha_beta(board)
        # if all the children were pruned, also prune this node and get a new one
        if self.all_children_pruned(board):
            self.prune_last()
            # the parents weren't pruned
            self.just_pruned_parents = False
            return self.get_next()
        # if we're done with all the children, then the node has been fully evaluated
        if self.all_children_done():
            self.evaluated.add(board)
        # if we go back up, we can remove some alphas/betas
        if self.depth < self.previous_depth:
            self.clean_siblings()
        # update if we just pruned
        return board

    def all_children_pruned(self, board: Board) -> bool:
        """Checks if all children of the current board were pruned."""

        # if all the children are done but the parent node wasn't overwritten, then all children were pruned
        return self.all_children_done() and self.evaluation[board] == INIT_NODE(self.depth)

    def all_children_done(self) -> bool:
        """Checks if we just completed visiting all the children."""

        # if we prune and jump up 2, that means we're done with that list of children
        # if we don't prune, we just jump up 1 and we're also done with that list of children
        return (not self.just_pruned_parents and self.depth == self.previous_depth - 1) \
            or (self.just_pruned_parents and self.depth == self.previous_depth - 2)

    def is_leaf(self, board: Board) -> bool:
        """Checks if a board is a leaf. If it is, it is immediately evaluated."""

        # root didn't change any column, it can't be a leaf.
        if board == self.root_board:
            return False
        # see if board is a leaf. If yes, evaluate it.
        x, y = self.changed_column, board.top[self.changed_column] - 1
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
        """Adds all possible children to node list."""

        # generate children and add them to the lists]
        self.siblings[self.depth + 1] = []
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
            self.siblings[self.depth + 1].append(child_board)
            self.worklist_nodes.append(child_board)
            self.worklist_depths.append(self.depth + 1)
            self.worklist_changed.append(column)

    def prunable(self, board: Board) -> bool:
        """Checks if board (and therefore its parent as well) is alpha/beta prunable."""

        if IS_MAX_NODE(self.depth):
            return self.evaluation[board] <= self.alpha
        else:
            return self.evaluation[board] >= self.beta

    def prune_last(self):
        """Remove the last node from the node list, so we can visit a new one."""

        del self.worklist_nodes[-1]
        del self.worklist_depths[-1]
        del self.worklist_changed[-1]

    def prune(self, board: Board):
        """Prunes a board and all its children from the node list."""

        index = self.worklist_nodes.index(board)
        if self.log:
            print(f"Pruned: {self.worklist_nodes[index:]}")
        del self.worklist_nodes[index:]
        del self.worklist_depths[index:]
        del self.worklist_changed[index:]

    def update_parent(self, board: Board, parent: Board) -> bool:
        """Updates parent of any node. Returns `True` if it was overwritten."""

        if IS_MAX_NODE(self.depth):
            return self.update_max(board, parent)
        else:
            return self.update_min(board, parent)

    def update_max(self, board: Board, parent: Board) -> bool:
        """Updates the parent of a max node. Returns `True` if it was overwritten."""

        # board in max node -> parent in min node
        if self.evaluation[board] < self.evaluation[parent]:
            # eval[p] = min(eval[p], eval[board])
            self.evaluation[parent] = self.evaluation[board]
            return True
        return False

    def update_min(self, board: Board, parent: Board) -> bool:
        """Updates the parent of a min node. Returns `True` if it was overwritten."""

        # board in min node -> parent in max node
        if self.evaluation[board] > self.evaluation[parent]:
            # eval[p] = max(eval[p], eval[board])
            self.evaluation[parent] = self.evaluation[board]
            return True
        return True

    def ancestors(self, board: Board) -> list[Board]:
        """Calculates the ancestors of a board (this is also the root path)."""

        ancestors: list[Board] = []
        node = board
        # while the node has a parent
        while node in self.parent:
            ancestors.append(node)
            node = self.parent[node]
        # all ancestors but root appended. root is ancestor to all.
        ancestors.append(self.root_board)
        return ancestors

    def clean_siblings(self):
        """Cleans up siblings that are no longer needed."""

        del self.siblings[self.depth + 1]
        # if we just pruned, then we might have jumped.
        if self.depth + 2 in self.siblings:
            del self.siblings[self.depth + 2]

    def collect_alpha_beta(self, board: Board):
        """Combines alpha/beta of the layers above into 1 value."""

        if IS_MAX_NODE(self.depth):
            self.alpha = self.collect_alpha(board)
        else:
            self.beta = self.collect_beta(board)

    def collect_alpha(self, board: Board) -> float:
        ancestors = self.ancestors(board)
        # only odd depths
        alpha = -math.inf
        for depth in range(1, self.depth, 2):
            siblings = self.siblings[depth]
            for sibling in siblings:
                # not a sibling, but an ancestor
                if sibling in ancestors:
                    continue
                # not a useful evaluation yet
                if sibling not in self.evaluation:
                    continue
                if self.evaluation[sibling] == INIT_NODE(depth):
                    continue
                # actual evaluation and actual sibling
                alpha = max(alpha, self.evaluation[sibling])
        return alpha

    def collect_beta(self, board: Board) -> float:
        ancestors = self.ancestors(board)
        # only even depths
        beta = math.inf
        for depth in range(0, self.depth, 2):
            for sibling in self.siblings[depth]:
                # not a sibling, but an ancestor
                if sibling in ancestors:
                    continue
                # not a useful evaluation yet
                if sibling not in self.evaluation:
                    continue
                if self.evaluation[sibling] == INIT_NODE(depth):
                    continue
                # actual evaluation and actual sibling
                beta = min(beta, self.evaluation[sibling])
        return beta
