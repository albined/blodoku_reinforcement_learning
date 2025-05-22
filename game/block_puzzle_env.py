from gymnasium import spaces, Env
import numpy as np
from game.game import Game
from game.plot_game import render_text
from game.block import random_block, random_block_encoded

class BlockPuzzleEnv(Env):
    def __init__(self, width=12, height=10, num_blocks=3, block_function=random_block_encoded):
        super(BlockPuzzleEnv, self).__init__()
        self.game = Game(width, height, block_size=4, block_function=block_function)

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(width, height), dtype=np.int32),
            "blocks": spaces.Box(low=0, high=1, shape=(num_blocks, 4, 4), dtype=np.int32),
        })
        self.action_space = spaces.MultiDiscrete([num_blocks, height, width])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state, action_mask = self.game.reset()
        return self._format_state(state), action_mask
    
    def step(self, action):
        (state, action_mask), reward, done, _ = self.game.step(action)
        return (self._format_state(state), action_mask), reward, done, False, {}
    
    def _format_state(self, state):
        # Convert underlying state dict to observation format
        grid = state["grid"]
        blocks = np.stack(state["blocks"], axis=0)
        return {"grid": grid, "blocks": blocks}

    def render(self, mode="human"):
        if mode == "human":
            render_text(self.game.grid.get_game_grid())
        else:
            super().render(mode=mode)
