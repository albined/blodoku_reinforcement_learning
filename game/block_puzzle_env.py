from gymnasium import spaces, Env
import numpy as np
from game.game import Game
from plot_game import render_text

class BlockPuzzleEnv(Env):
    def __init__(self, width=12, height=10, num_blocks=3):
        super(BlockPuzzleEnv, self).__init__()
        self.game = Game(width, height)
        
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(width, height), dtype=np.int32),
            "blocks": spaces.Box(low=0, high=1, shape=(num_blocks, 4, 4), dtype=np.int32),
        })
        self.action_space = spaces.MultiDiscrete([num_blocks, height, width])
    
    def reset(self):
        state = self.game.reset()
        return self._format_state(state)
    
    def step(self, action):
        state, reward, done, _ = self.game.step(action)
        return self._format_state(state), reward, done, {}
    
    def _format_state(self, state):
        grid = state["grid"]
        blocks = np.array([block for block in state["blocks"]])
        return {
            "grid": grid,
            "blocks": blocks,
        }
    
    def render(self, mode="human"):
        if mode == "human":
            render_text(self.game.grid.grid)
            
