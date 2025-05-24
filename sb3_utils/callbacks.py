import os
from datetime import datetime
from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class AveragedMetricsCallback(BaseCallback):
    def __init__(self, metrics_to_track=None, verbose=0):
        super().__init__(verbose)
        self.metrics_to_track = metrics_to_track or ['cleared_lines', 'invalid_moves', 'move_count']
        self.metric_buffers = {metric: deque(maxlen=100) for metric in self.metrics_to_track}

    def _on_step(self):
        infos = self.locals["infos"]
        for info in infos:
            if "episode" in info:  # Only log at the end of episodes
                for metric in self.metrics_to_track:
                    if metric in info:
                        self.metric_buffers[metric].append(info[metric])

        for metric, buffer in self.metric_buffers.items():
            if buffer:  # Avoid empty
                avg_value = np.mean(buffer)
                self.logger.record(f"custom/{metric}_avg", avg_value)

        return True
    
class SaveEveryNTimestepsCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=0, name="save_model", with_time=True, with_timesteps=True):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name = name
        self.with_time = with_time
        self.with_timesteps = with_timesteps
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            day_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if self.with_time else ""
            timesteps_string = f"_{self.num_timesteps}" if self.with_timesteps else ""
            save_file = os.path.join(self.save_path, f"{self.name}_{day_time_string}_{timesteps_string}.zip")
            self.model.save(save_file)
            if self.verbose:
                print(f"Saved model to {save_file}")
        return True
    
class UnfreezeCallback(BaseCallback):
    def __init__(self, unfreeze_step, unfreeze_layers, verbose=0):
        super().__init__(verbose)
        self.unfreeze_step = unfreeze_step
        self.unfreeze_layers = unfreeze_layers

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.unfreeze_step:
            for layer in self.unfreeze_layers:
                for p in getattr(self.model.policy.features_extractor, layer).parameters():
                    p.requires_grad = True
            return False  # remove this callback
        return True
