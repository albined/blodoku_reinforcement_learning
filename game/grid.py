import numpy as np
from game.block import Block
from numpy.lib.stride_tricks import sliding_window_view

class Grid:
    def __init__(self, width, height):
        self.grid = np.zeros((height, width), dtype=int)
        # Maybe add combo here later if I can get it to work normally first
    
    def is_valid_placement(self, block: Block, x: int, y: int) -> bool:
        block_array = block.grid()
        h, w = block_array.shape

        # Check bounds
        if x + h > self.grid.shape[0] or y + w > self.grid.shape[1]:
            return False

        # Check for overlap
        region = self.grid[x:x+h, y:y+w]
        return np.all(region + block_array <= 1)
    
    def get_all_valid_placements(self, block: Block, x, y):
        h, w = block.shape
        windows = sliding_window_view(self.grid, (h, w))  # shape: (H-h+1, W-w+1, h, w)
        overlaps = windows + block  # broadcast block to all windows
        valid = np.all(overlaps <= 1, axis=(-2, -1))  # shape: (H-h+1, W-w+1)
        return valid  # Boolean mask of valid positions
    
    def place_block(self, block: Block, x: int, y: int):
        block_array = block.grid()
        h, w = block_array.shape
        self.grid[x:x+h, y:y+w] += block_array

    def clear_lines(self) -> int:
        """Clears full rows and columns. Returns number of cleared lines."""
        full_rows = np.all(self.grid == 1, axis=1)
        full_cols = np.all(self.grid == 1, axis=0)

        cleared = 0

        # Clear rows
        if np.any(full_rows):
            self.grid[full_rows, :] = 0
            cleared += np.sum(full_rows)

        # Clear columns
        if np.any(full_cols):
            self.grid[:, full_cols] = 0
            cleared += np.sum(full_cols)

        return cleared
    
    def place_and_score(self, block: Block, x: int, y: int) -> int:
        """
        Places block, clears lines, and returns total reward.
        +1 per block cell placed
        + combo multiplier for line clears
        """
        if not self.is_valid_placement(block, x, y):
            return -1000  # or some penalty for illegal moves if needed

        block_array = block.grid()
        num_cells = np.sum(block_array)

        self.place_block(block, x, y)
        lines_cleared = self.clear_lines()

        return num_cells + lines_cleared*10
