import numpy as np


def encode_state(state):
    """
    Takes in a state dictionary with 'grid' and 'blocks' and returns a flat tensor.
    """
    grid = state["grid"].flatten() / 1.0
    blocks = np.array(state["blocks"]).flatten() / 1.0
    return np.concatenate([grid, blocks])
