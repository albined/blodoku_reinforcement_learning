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
    
    def get_encoding(self):
        return self.grid()
        
class EmptyBlock(Block):
    def __init__(self, size):
        super().__init__(np.zeros((size, size), dtype=int))

def random_block():
    block_type = np.random.choice(list(TypeBlock.block_type_lookup_dict.keys()))
    block_rotation = np.random.randint(0, 4)
    return PaddedTypeBlock(block_type, block_rotation, size=4)

class EncodedOrPaddedTypeBlock(PaddedTypeBlock):
    _canonical_blocks = None
    _block_encoding_lookup = None

    def __init__(self, block_type, rotation, size=4):
        super().__init__(block_type, rotation, size)
        self.init_canonical_encodings()

    @staticmethod
    def shift_to_top_left_static(grid):
        rows = np.any(grid, axis=1)
        cols = np.any(grid, axis=0)

        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]

        sub = grid[r0:r1 + 1, c0:c1 + 1]
        return sub

    @classmethod
    def init_canonical_encodings(cls):
        if cls._canonical_blocks is not None:
            return

        seen = set()
        canonical_blocks = []
        encoding_lookup = {}

        for block_array in TypeBlock.block_type_lookup_dict.values():
            for k in range(4):  # Try all 4 rotations
                rotated = np.rot90(block_array, k)
                shifted = cls.shift_to_top_left_static(rotated)
                padded = np.pad(shifted, ((0, 4 - shifted.shape[0]), (0, 4 - shifted.shape[1])), 'constant')
                flat_tuple = tuple(padded.flatten())

                if flat_tuple not in seen:
                    seen.add(flat_tuple)
                    idx = len(canonical_blocks)
                    canonical_blocks.append(padded)
                    encoding_lookup[flat_tuple] = idx

        cls._canonical_blocks = canonical_blocks
        cls._block_encoding_lookup = encoding_lookup

    def get_encoding(self):
        flat_tuple = tuple(self.block_grid.flatten())
        index = self._block_encoding_lookup[flat_tuple]
        one_hot = np.zeros(len(self._canonical_blocks), dtype=np.float32)
        one_hot[index] = 1.0
        return one_hot

def random_block_encoded():
    block_type = np.random.choice(list(TypeBlock.block_type_lookup_dict.keys()))
    block_rotation = np.random.randint(0, 4)
    return EncodedOrPaddedTypeBlock(block_type, block_rotation, size=4)
