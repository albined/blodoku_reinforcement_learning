from game.grid import Grid
from game.block import EmptyBlock, PaddedTypeBlock, random_block, random_block_encoded
import numpy as np

INVALID_MOVE_PENALTY = -0.01
GAME_END_PENALTY = -10

class Game:
        
    def __init__(self, width=12, height=10, block_size=4, block_function=random_block_encoded, compute_action_mask=False):
        self.grid = Grid(width, height, block_size)
        self.block_size = block_size
        self.block_function = block_function
        self.regenerate_block_queue()
        self.encoding_shape = self.block_queue[0].get_encoding().shape
        self.score = 0
        self.move_count = 0
        self.invalid_moves = 0
        self.cleared_lines = 0
        self.done = False
        self.compute_action_mask_flag = compute_action_mask

    def regenerate_block_queue(self):
        self.block_queue = [self.block_function() for _ in range(3)]
        self.block_queue_size = 3
        
    def step(self, action):
        block_index, x, y = action
        if self.done:
            raise Exception("game is already over")
        
        block = self.block_queue[block_index]
        if not self.grid.is_valid_placement(block, x ,y):
            self.invalid_moves += 1
            action_mask = self.compute_action_mask() if self.compute_action_mask_flag else None
            return (self.get_state(), action_mask), INVALID_MOVE_PENALTY, False, self.get_info_dict()

        if isinstance(block, EmptyBlock):
            self.invalid_moves += 1
            action_mask = self.compute_action_mask() if self.compute_action_mask_flag else None
            return (self.get_state(), action_mask), INVALID_MOVE_PENALTY, False, self.get_info_dict()
        self.block_queue[block_index] = EmptyBlock(self.block_size)
        self.block_queue_size -= 1
        
        # Regenerate list if empty
        if self.block_queue_size == 0:
            self.regenerate_block_queue()

        reward, cleared_lines = self.grid.place_and_score(block, x, y)
        self.move_count += 1
        self.cleared_lines += cleared_lines
        
        action_mask = self.compute_action_mask()
        
        if not action_mask.any():
            self.done = True
            reward = GAME_END_PENALTY

        self.score += reward
        return (self.get_state(), action_mask), reward, self.done, self.get_info_dict()

    def get_info_dict(self):
        return {
            'move_count': self.move_count,
            'invalid_moves': self.invalid_moves,
            'cleared_lines': self.cleared_lines,
        }

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
        self.move_count = 0
        self.invalid_moves = 0
        self.cleared_lines = 0
        self.done = False
        return self.get_state(), self.compute_action_mask()

    def get_state(self):
        return {
            "grid": self.grid.get_game_grid().copy(),
            "blocks": [block.get_encoding() if not isinstance(block, EmptyBlock) else np.zeros(self.encoding_shape) for block in self.block_queue],
            }
        
    def compute_action_mask(self):
        grid = self.grid
        blocks = self.block_queue
        num_blocks = len(blocks)
        
        grid_height, grid_width = grid.get_game_grid().shape
        mask = np.zeros((num_blocks * grid_height * grid_width,), dtype=bool)

        for block_idx, block in enumerate(blocks):
            if isinstance(block, EmptyBlock):
                continue

            valid_locs = np.argwhere(grid.get_all_valid_placements(block))  # shape (N, 2), each row is [y, x]
            for (y, x) in valid_locs:
                flat_action = block_idx * (grid_width * grid_height) + y * grid_width + x
                mask[flat_action] = True

        return mask  # shape: (num_blocks * H * W,)