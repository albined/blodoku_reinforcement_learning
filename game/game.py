from game.grid import Grid
from game.block import Block, random_block
import numpy as np

class Game:
    def __init__(self, width=12, height=10):
        self.grid = Grid(width, height)
        self.regenerate_block_queue()
        self.score = 0
        self.done = False
        
    def regenerate_block_queue(self):
        self.block_queue = [random_block() for _ in range(3)]
        
    def step(self, action):
        block_index, x, y = action
        if self.done:
            raise Exception("game is already over")
        
        block = self.block_queue.pop(block_index)
        # Regenerate list if empty
        if len(self.block_queue) == 0:
            self.regenerate_block_queue()
        
        if not self.grid.is_valid_placement(block, x ,y):
            return -1000, True
        
        reward = self.grid.place_and_score(block, x, y)
        self.score += reward
        
        if not self.any_valid_moves():
            self.done = True
    
        return self.get_state(), reward, self.done, {}
    
    def any_valid_moves(self):
        for idx, block in enumerate(self.block_queue):
            valid = self.grid.get_all_valid_placements(block)
            if np.any(valid):
                return True
        return False

    def reset(self):
        self.grid = Grid(self.grid.grid.shape[1], self.grid.grid.shape[0])
        self.regenerate_block_queue()
        self.score = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        return {
            "grid": self.grid.grid.copy(),
            "blocks": [block.grid() for block in self.block_queue],
        }