import numpy as np
from game.block import EmptyBlock

def encode_state(state):
    """
    Takes in a state dictionary with 'grid' and 'blocks' and returns a flat tensor.
    """
    grid = state["grid"].flatten() / 1.0
    blocks = np.array(state["blocks"]).flatten() / 1.0
    return np.concatenate([grid, blocks])

# More advanced encoding for CNN, not finished yet, maybe complete later 
def shift_array(arr, dx=0, dy=0):
    """
    Shifts array arr to the right by dx and down by dy.
    Pads with zeros. Cuts off overflow.
    """

    if len(arr.shape) == 2:
        H, W = arr.shape
        padded = np.pad(arr, ((dy, 0), (dx, 0)), mode='constant')
        return padded[:H, :W]
    elif len(arr.shape) == 3:
        C, H, W = arr.shape
        padded = np.pad(arr, ((0, 0), (dy, 0), (dx, 0)), mode='constant')
        return padded[:, :H, :W]

def generate_shifted_channels(arr, x_shifts=[1, 2, 3], y_shifts=[1, 2, 3]):
    shifted = [shift_array(arr, dx=dx, dy=dy) for dx, dy in zip(x_shifts, y_shifts)]
    return np.stack(shifted, axis=0)  # shape: (num_shifts, H, W)

def encode_state_cnn(state):
    """
    Takes in a state dictionary with 'grid' and 'blocks' and returns a flat tensor.
    """
    grid = state["grid"]
    blocks = np.array(state["blocks"])
    
    # Create tiled channels from the blocks
    gw, gh = grid.shape
    bw, bh = blocks.shape[1:3]
    base_tile = np.tile(blocks, (1, gw//bw, gh//bh))
    # Zero pad the base_tile to match the grid size
    pad_h = grid.shape[0] - base_tile.shape[1]
    pad_w = grid.shape[1] - base_tile.shape[2]
    base_tile = np.pad(base_tile, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
    x_shifts = range(bw)
    y_shifts = range(bh)
    X, Y = np.meshgrid(x_shifts, y_shifts)
    X = X.flatten()
    Y = Y.flatten()
    shifted_tiles = generate_shifted_channels(base_tile, x_shifts=X, y_shifts=Y)
    shifted_tiles = shifted_tiles.reshape(-1, *shifted_tiles.shape[2:])  # Flatten the first dimension
    stacked = np.concatenate([grid[None, :, :], shifted_tiles], axis=0)
    stacked = np.astype(stacked, np.uint8)
    # stacked = np.pad(stacked, ((0, 0), (0, 64 - stacked.shape[1]), (0, 64 - stacked.shape[2])), mode='constant')
    return stacked.transpose(1, 2, 0)  # shape: (H, W, num_channels)

def encode_state_cnn_modern(state):
    # Should return a grid of shape (H, W) and then one hot blocks of shape (num_blocks, one_hot_size)
    grid = state["grid"]
    blocks = np.array(state["blocks"])
    
    return grid, blocks