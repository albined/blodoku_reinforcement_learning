import matplotlib.pyplot as plt
import numpy as np

def plot_grid(grid: np.ndarray):
    """Plot the current game grid in black and white."""
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='Greys', origin='upper', extent=[0, grid.shape[1], grid.shape[0], 0])
    ax.set_xticks(np.arange(0, grid.shape[1]+1, 1))
    ax.set_yticks(np.arange(0, grid.shape[0]+1, 1))
    ax.grid(color='black', linewidth=1)
    ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()

def plot_block(block_array: np.ndarray):
    """Plot a standalone block in black and white."""
    fig, ax = plt.subplots()
    ax.imshow(block_array, cmap='Greys', origin='upper', extent=[0, block_array.shape[1], block_array.shape[0], 0])
    ax.set_xticks(np.arange(0, block_array.shape[1]+1, 1))
    ax.set_yticks(np.arange(0, block_array.shape[0]+1, 1))
    ax.grid(color='black', linewidth=1)
    ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()

def plot_placement(grid: np.ndarray, block_array: np.ndarray, x: int, y: int):
    """Plot the grid and overlay a proposed block placement in red."""
    overlay = np.zeros_like(grid, dtype=float)

    h, w = block_array.shape
    # Overlay red where the block would go
    overlay[x:x+h, y:y+w] = block_array

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='Greys', origin='upper', extent=[0, grid.shape[1], grid.shape[0], 0])
    ax.imshow(overlay, cmap='Reds', alpha=0.5, origin='upper', extent=[0, grid.shape[1], grid.shape[0], 0])
    ax.set_xticks(np.arange(0, grid.shape[1]+1, 1))
    ax.set_yticks(np.arange(0, grid.shape[0]+1, 1))
    ax.grid(color='black', linewidth=1)
    ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()

def render_text(grid: np.ndarray):
    """Render the grid using ASCII characters."""
    filled = '■'
    empty = '□'
    lines = []
    for row in grid:
        line = ' '.join(filled if cell else empty for cell in row)
        lines.append(line)
    print('\n'.join(lines))
