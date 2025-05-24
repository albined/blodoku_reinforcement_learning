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
    bad = 'x'
    lines = []
    for row in grid:
        line = ' '.join(filled if cell == 1 else empty if cell == 0 else bad for cell in row)
        lines.append(line)
    print('\n'.join(lines))

def render_text_with_blocks(grid: np.ndarray, blocks: list):
    """Render the grid using ASCII characters, with blocks shown to the right of the grid, separated by a | column."""
    filled = '■'
    empty = '□'
    bad = 'x'

    grid_lines = []
    for row in grid:
        line = ' '.join(filled if cell == 1 else empty if cell == 0 else bad for cell in row)
        grid_lines.append(line)

    # Prepare block lines (all blocks, padded to same height)
    block_heights = [block.shape[0] for block in blocks]
    max_block_height = max(block_heights) if blocks else 0
    block_lines = []
    for i in range(max_block_height):
        line_parts = []
        for block in blocks:
            if i < block.shape[0]:
                line = ' '.join(filled if cell == 1 else empty for cell in block[i])
            else:
                line = ' ' * (2 * block.shape[1] - 1)
            line_parts.append(line)
        block_lines.append('  '.join(line_parts))

    # Pad grid lines to match block lines if needed
    total_lines = max(len(grid_lines), len(block_lines))
    grid_lines += [' ' * len(grid_lines[0])] * (total_lines - len(grid_lines))
    block_lines += [''] * (total_lines - len(block_lines))

    # Print combined lines with separator
    for g, b in zip(grid_lines, block_lines):
        if b:
            print(f"{g} | {b}")
        else:
            print(f"{g} |")