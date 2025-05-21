from game.grid import Grid
from game.block import EmptyBlock, PaddedTypeBlock, random_block
import numpy as np

INVALID_MOVE_PENALTY = -100

class Game:
        
    def __init__(self, width=12, height=10, block_size=4):
        self.grid = Grid(width, height, block_size)
        self.block_size = block_size
        self.regenerate_block_queue()
        self.score = 0
        self.done = False
        
    def regenerate_block_queue(self):
        self.block_queue = [random_block() for _ in range(3)]
        # self.block_queue = [PaddedTypeBlock(1, 0, 4) for _ in range(3)]
        self.block_queue_size = 3
        
    def step(self, action):
        block_index, x, y = action
        if self.done:
            raise Exception("game is already over")
        
        block = self.block_queue[block_index]
        if isinstance(block, EmptyBlock):
            return self.get_state(), INVALID_MOVE_PENALTY, True, {}
        self.block_queue[block_index] = EmptyBlock(self.block_size)
        self.block_queue_size -= 1
        
        # Regenerate list if empty
        if self.block_queue_size == 0:
            self.regenerate_block_queue()
        
        if not self.grid.is_valid_placement(block, x ,y):
            return self.get_state(), INVALID_MOVE_PENALTY, True, {}

        reward = self.grid.place_and_score(block, x, y)
        self.score += 10
        
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
        self.grid = Grid(self.grid.get_game_grid().shape[1], self.grid.get_game_grid().shape[0], self.block_size)
        self.regenerate_block_queue()
        self.score = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        return {
            "grid": self.grid.get_game_grid().copy(),
            "blocks": [block.grid() for block in self.block_queue],
        }