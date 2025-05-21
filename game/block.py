import numpy as np

class Block:
    def __init__(self, block_grid):
        self.block_grid = block_grid
        self.shape = block_grid.shape
    
    def grid(self):
        return self.block_grid

class TypeBlock(Block):

    block_type_lookup_dict = {
        0: np.array([[1,],]),
        1: np.array([[1, 1], [0, 0]]),
        2: np.array([[1, 1], [1, 0]]),
        3: np.array([[1, 1], [1, 1]]),
        4: np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]]),
        5: np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]]),
        6: np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]]),
        7: np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]]),
        8: np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]]),
        9: np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]]),
        10: np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    }

    def __init__(self, block_type, rotation):
        block_grid = self.block_type_lookup_dict[block_type]
        block_grid = np.rot90(block_grid, k=rotation % 4)
        block_grid = self.shift_to_top_left(block_grid)
        super().__init__(block_grid)

    def shift_to_top_left(self, grid):
        n = grid.shape[0]
        rows = np.any(grid, axis=1)
        cols = np.any(grid, axis=0)

        # bounding‑box of the “1”s
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        return np.pad(sub, ((0, n - sub.shape[0]), (0, n - sub.shape[1])), 'constant')
        
class PaddedTypeBlock(TypeBlock):
    def __init__(self, block_type, rotation, size):
        super().__init__(block_type, rotation)
        self.block_grid = np.pad(self.block_grid, ((0, size - self.block_grid.shape[0]), (0, size - self.block_grid.shape[1])), 'constant')
        self.shape = self.block_grid.shape
        
class EmptyBlock(Block):
    def __init__(self, size):
        super().__init__(np.zeros((size, size), dtype=int))

def random_block():
    block_type = np.random.choice(list(TypeBlock.block_type_lookup_dict.keys()))
    block_rotation = np.random.randint(0, 4)
    return PaddedTypeBlock(block_type, block_rotation, size=4)
