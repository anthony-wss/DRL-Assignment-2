import random
import numpy as np
import sys

class TreeNode:
    def __init__(self, grid, parent, player):
        self.grid = grid
        self.parent = parent
        self.player = player
        self.children = {}
        self.grid_size = grid.shape[0]
        self.available_moves = self.get_relevant_pos()
        self.visits = 0
        self.value = 0.0
    
    def get_relevant_pos(self):
        dir = [
            (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
            (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
            (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
            (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
            (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)
        ]
        relevant_pos = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] != 0:
                    continue
                for d in dir:
                    ri = r + d[0]
                    ci = c + d[1]
                    if ri >= 0 and ri < self.grid_size and ci >= 0 and ci < self.grid_size:
                        if self.grid[ri, ci] != 0:
                            relevant_pos.add((r, c))
                            break
        relevant_pos = list(relevant_pos)
        relevant_pos = list(zip(relevant_pos, relevant_pos[1:]))
        return relevant_pos
    
    def select_child(self):
        max_ucb = float('-inf')
        best_child = None
        for child in self.children.values():
            ucb = child.value + 2 * np.sqrt(np.log(self.visits) / child.visits)
            if ucb > max_ucb:
                max_ucb = ucb
                best_child = child
        return best_child
    
    def fully_expanded(self):
        return len(self.available_moves) == len(self.children)
    
    def expand(self):
        for move in self.available_moves:
            if move not in self.children:
                new_board = self.grid.copy()
                new_board[move[0][0], move[0][1]], new_board[move[1][0], move[1][1]] = self.player, self.player
                new_node = TreeNode(new_board, self, 3-self.player)
                self.children[move] = new_node
                return new_node
    
    def simulate(self):
        return random.choice([1, 0, -1])
    
    def backpropagate(self, sim_value):
        self.visits += 1
        self.value += sim_value
        if self.parent:
            self.parent.backpropagate(-sim_value)

def mct_search(board, player, iterations=10, debug=False):
    root = TreeNode(board, None, player)

    for _ in range(iterations):
        if debug:
            print("Iteration", _, file=sys.stderr, flush=True)
        node = root

        # Selection
        while node.fully_expanded():
            node = node.select_child()
        
        if debug:
            print("selection done", file=sys.stderr, flush=True)
        
        # Expansion
        if not node.fully_expanded():
            node = node.expand()

        if debug:
            print("expansion done", file=sys.stderr, flush=True)

        # Simulation
        sim_value = node.simulate()

        if debug:
            print("simulation done", file=sys.stderr, flush=True)


        # Backpropagation
        node.backpropagate(sim_value)

        if debug:
            print("backpropagation done", file=sys.stderr, flush=True)

    # Return the move of the most visited child
    max_visits = -1
    best_move = None
    for move, node in root.children.items():
        if node.visits > max_visits:
            max_visits = node.visits
            best_move = move
    return best_move, root