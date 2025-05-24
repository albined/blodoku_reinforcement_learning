import numpy as np
from game.block import Block
from numpy.lib.stride_tricks import sliding_window_view

class Grid:
    def __init__(self, width, height, block_size):
        self.block_size = block_size
        self.grid_ = np.zeros((height, width), dtype=int)
        self.grid_ = np.pad(self.grid_, ((0, block_size), (0, block_size)), 'constant', constant_values=1)
        self.combo = 0
    
    def get_padded_grid(self):
        return self.grid_
    
    def get_game_grid(self):
        return self.grid_[:-self.block_size, :-self.block_size]
    
    def is_valid_placement(self, block: Block, x: int, y: int) -> bool:
        block_array = block.grid()
        h, w = block_array.shape
        region = self.get_padded_grid()[x:x+h, y:y+w]
        return np.all(region + block_array <= 1)
    
    def get_all_valid_placements(self, block: Block):
        h, w = block.shape
        windows = sliding_window_view(self.get_padded_grid(), (h, w))  # shape: (H-h+1, W-w+1, h, w)
        overlaps = windows + block.grid()  # broadcast block to all windows
        valid = np.all(overlaps <= 1, axis=(-2, -1))  # shape: (H-h+1, W-w+1)
        return valid  # Boolean mask of valid positions
    
    def place_block(self, block: Block, x: int, y: int):
        block_array = block.grid()
        h, w = block_array.shape
        self.get_padded_grid()[x:x+h, y:y+w] += block_array

    def clear_lines(self) -> int:
        """Clears full rows and columns. Returns number of cleared lines."""
        full_rows = np.all(self.get_game_grid() == 1, axis=1)
        full_cols = np.all(self.get_game_grid() == 1, axis=0)

        cleared = 0

        # Clear rows
        if np.any(full_rows):
            self.get_game_grid()[full_rows, :] = 0
            cleared += np.sum(full_rows)

        # Clear columns
        if np.any(full_cols):
            self.get_game_grid()[:, full_cols] = 0
            cleared += np.sum(full_cols)

        return cleared

    def compute_circumference_area_ratio(self, grid: np.ndarray) -> float:
        h, w = grid.shape
        area = np.sum(grid)
        
        if area == 0:
            return float('inf')

        inner_circumference = 0

        # Check each filled cell
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if grid[i, j] == 1:
                    # Count adjacent empty cells (up, down, left, right)
                    neighbors = [
                        grid[i - 1, j],
                        grid[i + 1, j],
                        grid[i, j - 1],
                        grid[i, j + 1]
                    ]
                    inner_circumference += sum(1 for n in neighbors if n == 0)

        return inner_circumference / area
    
    def place_and_score(self, block: Block, x: int, y: int) -> int:
        """
        Places block, clears lines, and returns total reward.
        +1 per block cell placed
        + combo multiplier for line clears
        """
        if not self.is_valid_placement(block, x, y):
            print("Invalid placement!!!")
            return -1000  # or some penalty for illegal moves if needed

        # block_array = block.grid()
        # num_cells = np.sum(block_array)

        # before_ratio = self.compute_circumference_area_ratio(self.get_game_grid())
        self.place_block(block, x, y)
        # after_ratio = self.compute_circumference_area_ratio(self.get_game_grid())
        # delta_ratio = before_ratio - after_ratio
        # delta_ratio = np.clip(delta_ratio * 1, -0.3, 0.3)
        
        lines_cleared = self.clear_lines()
        if lines_cleared > 0:
            self.combo += lines_cleared
        else:
            self.combo = 0

        combo_multiplier_dict = {0: 0, 1: 1, 2: 3, 3: 5}
        combo_multiplier = combo_multiplier_dict.get(self.combo, 5)
        
        return 0.01 + lines_cleared*combo_multiplier*1, lines_cleared